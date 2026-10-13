"""Explicit local-controller execution without physical or storage ownership.

The coordinator owns the physical program, trial evidence and accepted-iterate
publication. This driver keeps only controller coordinates and detached actual
observations; SciPy termination is not design acceptance or resumable curvature.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ..canonical import float64_from_hex


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class LocalUnavailable:
    """An actual physical/derivative unavailability, not a penalty gradient."""

    code: str
    message: str
    evidence: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "evidence", _freeze(self.evidence))


@dataclass(frozen=True, slots=True)
class LocalEvaluation:
    """One recorded physical trial returned by the coordinator."""

    point: Mapping[str, object]
    locator: Mapping[str, object] | None
    cost: float | None
    gradient: tuple[float, ...] | None
    unavailable: LocalUnavailable | None = None

    def __post_init__(self):
        object.__setattr__(self, "point", _freeze(self.point))
        if self.locator is not None:
            object.__setattr__(self, "locator", _freeze(self.locator))
        if self.gradient is not None:
            object.__setattr__(self, "gradient", tuple(self.gradient))


@dataclass(frozen=True, slots=True)
class LocalOptimizationOutcome:
    """Actual controller facts; stage winner selection remains coordinator-owned."""

    termination: str
    message: str
    coordinates: tuple[float, ...]
    cost: float | None
    gradient: tuple[float, ...] | None
    evaluation_requests: int
    accepted_iterations: int
    scipy_success: bool | None
    scipy_status: int | None
    scipy_nfev: int | None
    scipy_njev: int | None
    scipy_nit: int | None
    last_evaluation: LocalEvaluation | None
    unavailable: LocalUnavailable | None


class _StageUnavailable(Exception):
    def __init__(self, coordinates, evaluation):
        self.coordinates = coordinates
        self.evaluation = evaluation


class _CoordinatorFailure(Exception):
    def __init__(self, original):
        self.original = original
        self.traceback = original.__traceback__


def run_local_optimizer(*, initial: Sequence[float], bounds,
                        controls: Mapping[str, object],
                        evaluate: Callable[..., LocalEvaluation],
                        on_accepted: Callable[[tuple[float, ...]], object]) -> LocalOptimizationOutcome:
    """Dispatch the named local method using the owner's actual evaluations.

    ``initial`` and ``bounds`` are already mapped controller coordinates. The
    driver neither chooses coordinate scales nor reconstructs physical points.
    ``on_accepted`` is invoked only for valid accepted iterations; its owner
    performs exact trial lookup and durable publication. Its errors are fatal.
    """
    if controls['type'] == 'projected_lbfgs':
        return run_projected_local_optimizer(initial=initial, bounds=bounds, controls=controls,
                                             evaluate=evaluate, on_accepted=on_accepted)
    if controls['type'] != 'lbfgsb':
        raise ValueError('unknown local optimizer strategy')
    from scipy.optimize import minimize

    evaluations = 0
    accepted = 0
    last = None

    def value_and_gradient(x):
        nonlocal evaluations, last
        coordinates = tuple(x)
        evaluations += 1
        observed = evaluate(coordinates, need_gradient=True)
        if not isinstance(observed, LocalEvaluation):
            raise TypeError("evaluate must return LocalEvaluation")
        last = observed
        if observed.unavailable is not None:
            raise _StageUnavailable(coordinates, observed)
        if observed.cost is None or observed.gradient is None:
            raise TypeError("an available LocalEvaluation requires its actual cost and gradient")
        return observed.cost, observed.gradient

    def accepted_iterate(xk):
        nonlocal accepted
        accepted += 1
        try:
            on_accepted(tuple(xk))
        except BaseException as error:
            # SciPy treats callback StopIteration as a normal stop. The
            # coordinator's durable callback/storage failures remain fatal.
            raise _CoordinatorFailure(error) from error

    try:
        result = minimize(
            value_and_gradient, tuple(initial), method="L-BFGS-B", jac=True,
            bounds=bounds, callback=accepted_iterate,
            options={"maxfun": controls["max_evaluations"],
                     "maxiter": controls["max_iterations"],
                     "ftol": float64_from_hex(controls["ftol_f64"]),
                     "gtol": float64_from_hex(controls["gtol_f64"]),
                     "maxls": controls["max_line_search"]},
        )
    except _CoordinatorFailure as failed:
        raise failed.original.with_traceback(failed.traceback) from None
    except _StageUnavailable as stopped:
        observation = stopped.evaluation
        return LocalOptimizationOutcome(
            termination=observation.unavailable.code,
            message=observation.unavailable.message,
            coordinates=stopped.coordinates, cost=observation.cost,
            gradient=observation.gradient, evaluation_requests=evaluations,
            accepted_iterations=accepted, scipy_success=None, scipy_status=None,
            scipy_nfev=None, scipy_njev=None, scipy_nit=None,
            last_evaluation=observation, unavailable=observation.unavailable,
        )
    status = int(result.status)
    termination = ("optimizer_converged" if result.success else
                   "optimizer_budget_or_iteration_limit" if status == 1 else
                   "optimizer_line_search_failed" if status == 2 else "optimizer_stopped")
    return LocalOptimizationOutcome(
        termination=termination, message=str(result.message),
        coordinates=tuple(result.x), cost=result.fun, gradient=tuple(result.jac),
        evaluation_requests=evaluations, accepted_iterations=accepted,
        scipy_success=bool(result.success), scipy_status=status,
        scipy_nfev=int(result.nfev), scipy_njev=int(result.njev), scipy_nit=int(result.nit),
        last_evaluation=last, unavailable=None,
    )


def run_projected_local_optimizer(*, initial: Sequence[float], bounds,
                                  controls: Mapping[str, object],
                                  evaluate: Callable[..., LocalEvaluation],
                                  on_accepted: Callable[[tuple[float, ...]], object]) -> LocalOptimizationOutcome:
    """Monotone projected L-BFGS with explicit rejected-trial ownership.

    This is not L-BFGS-B: it has no generalized Cauchy/subspace minimization.
    The CONVERGING procedure uses memory10, unit controller step and halving;
    only actual strict cost decrease plus an available real gradient accepts a
    trial. No Armijo constant, scientific threshold, fabricated gradient or
    penalized physical objective participates. The chosen coordinate map is raw SI
    when no scale is declared. Supplied budgets/ftol/gtol apply.
    """
    import numpy as np

    x = np.asarray(initial, dtype=np.float64)
    lower = np.asarray([float('-inf') if row[0] is None else row[0] for row in bounds], dtype=np.float64)
    upper = np.asarray([float('inf') if row[1] is None else row[1] for row in bounds], dtype=np.float64)
    # Same bound projection as SciPy initialization; the actual projected point
    # is passed explicitly to evaluate and retained by the coordinator.
    x = np.clip(x, lower, upper)
    evaluations = accepted = 0
    last = None
    pairs = []
    ftol = float64_from_hex(controls['ftol_f64'])
    gtol = float64_from_hex(controls['gtol_f64'])

    def request(point):
        nonlocal evaluations, last
        evaluations += 1
        observed = evaluate(tuple(point), need_gradient=True)
        if not isinstance(observed, LocalEvaluation):
            raise TypeError('evaluate must return LocalEvaluation')
        last = observed
        if observed.unavailable is None and (observed.cost is None or observed.gradient is None):
            raise TypeError('an available LocalEvaluation requires actual cost and gradient')
        return observed

    def finish(reason, message, cost, gradient, *, unavailable=None):
        return LocalOptimizationOutcome(termination=reason, message=message,
            coordinates=tuple(x), cost=cost, gradient=None if gradient is None else tuple(gradient),
            evaluation_requests=evaluations, accepted_iterations=accepted,
            scipy_success=None, scipy_status=None, scipy_nfev=None, scipy_njev=None,
            scipy_nit=None, last_evaluation=last, unavailable=unavailable)

    initial_observation = request(x)
    if initial_observation.unavailable is not None:
        unavailable = initial_observation.unavailable
        return finish(unavailable.code, unavailable.message, initial_observation.cost, None,
                      unavailable=unavailable)
    cost = initial_observation.cost
    gradient = np.asarray(initial_observation.gradient, dtype=np.float64)
    while accepted < controls['max_iterations'] and evaluations < controls['max_evaluations']:
        projected_gradient = x - np.clip(x - gradient, lower, upper)
        if np.max(np.abs(projected_gradient)) <= gtol:
            return finish('optimizer_converged', 'declared projected-gradient tolerance reached', cost, gradient)
        q = gradient.copy()
        alphas = []
        for displacement, change, rho in reversed(pairs):
            alpha = rho * np.dot(displacement, q)
            q -= alpha * change
            alphas.append(alpha)
        scale = (np.dot(pairs[-1][0], pairs[-1][1]) / np.dot(pairs[-1][1], pairs[-1][1])) if pairs else 1.
        direction = scale * q
        for (displacement, change, rho), alpha in zip(pairs, reversed(alphas), strict=True):
            direction += displacement * (alpha - rho * np.dot(change, direction))
        direction = -direction
        outward = ((x <= lower) & (direction < 0)) | ((x >= upper) & (direction > 0))
        direction[outward] = 0.
        # Backtrack one feasible displacement, rather than repeatedly clipping
        # a raw-SI direction to the same bound at every step size.
        direction = np.clip(x + direction, lower, upper) - x
        if not np.dot(gradient, direction) < 0:
            direction = -projected_gradient
        step = 1.
        chosen = None
        for _ in range(controls['max_line_search']):
            if evaluations >= controls['max_evaluations']:
                break
            trial = x + step * direction
            if np.array_equal(trial, x):
                return finish('optimizer_line_search_failed', 'no distinct representable projected trial', cost, gradient)
            observation = request(trial)
            # The actual primal cost is preserved even if its derivative is
            # unavailable. Such a trial supplies no gradient/curvature update.
            if observation.unavailable is None and observation.cost < cost:
                chosen = trial, observation
                break
            step *= .5
        if chosen is None:
            if evaluations >= controls['max_evaluations']:
                return finish('optimizer_budget_or_iteration_limit', 'declared evaluation budget reached', cost, gradient)
            return finish('optimizer_line_search_failed', 'declared line-search budget exhausted without a valid decreasing trial', cost, gradient)
        trial, observation = chosen
        next_gradient = np.asarray(observation.gradient, dtype=np.float64)
        displacement, change = trial - x, next_gradient - gradient
        curvature = np.dot(displacement, change)
        # Positive curvature is the inverse-Hessian representation's domain;
        # no epsilon cutoff or inferred physical conditioning rule is used.
        if curvature > 0:
            pairs.append((displacement.copy(), change.copy(), 1. / curvature))
            if len(pairs) > 10:
                del pairs[0]
        previous_cost = cost
        x, cost, gradient = trial, observation.cost, next_gradient
        accepted += 1
        on_accepted(tuple(x))
        if previous_cost - cost <= ftol * max(1., abs(previous_cost), abs(cost)):
            return finish('optimizer_converged', 'declared relative cost-decrease tolerance reached', cost, gradient)
    return finish('optimizer_budget_or_iteration_limit', 'declared evaluation or iteration budget reached', cost, gradient)
