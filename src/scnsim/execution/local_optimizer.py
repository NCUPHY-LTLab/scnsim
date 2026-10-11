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
    """Run L-BFGS-B using explicit controls and the owner's actual evaluations.

    ``initial`` and ``bounds`` are already mapped controller coordinates. The
    driver neither chooses coordinate scales nor reconstructs physical points.
    ``on_accepted`` is invoked only for SciPy accepted iterations; its owner
    performs exact trial lookup and durable publication. Its errors are fatal.
    """
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
