"""CMA coordinate mapping and post-tell state, independent of execution.

The coordinator owns budgets, ordered physical evaluations, winner selection,
callbacks and durable publication. This controller owns only the pinned CMA
instance and its exact RNG/state representation; it never chooses a stopping
condition. Hybrid stage entry supplies actual resolved starting coordinates.
"""
from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
import math

import numpy as np

from ..canonical import float64_from_hex, float64_hex
from ..compilation.compiler import parameter_key
from ..compilation.mesh import quantity
from ..errors import (
    CompilerInvariantError, InvalidCandidatePhysicalParameter, InvalidOptimizationSpec,
    RuntimePreparationError,
)
from ..numeric_encoding import array_from_record, array_record, bits, unbits

__all__ = (
    "CMAController", "linquad", "domain_bounds", "latent_initial",
    "mapped_values", "cma_snapshot", "restore_cma", "CMA_FIELDS",
    "physical_seed_coordinates",
)


def linquad(x: float, lower: float, upper: float) -> float:
    al = 1 if not math.isfinite(lower) else min((upper - lower) / 2, (1 + abs(lower)) / 20)
    au = 1 if not math.isfinite(upper) else min((upper - lower) / 2, (1 + abs(upper)) / 20)
    if x < lower - 2 * al - (upper - lower) / 2 or x > upper + 2 * au + (upper - lower) / 2:
        period = 2 * (upper - lower + al + au)
        x -= period * math.floor((x - (lower - 2 * al - (upper - lower) / 2)) / period)
    if x > upper + au:
        x -= 2 * (x - upper - au)
    if x < lower - al:
        x += 2 * (lower - al - x)
    if x < lower + al:
        return lower + (x - (lower - al)) ** 2 / (4 * al)
    if x < upper - au:
        return x
    return upper - (x - (upper + au)) ** 2 / (4 * au)


def domain_bounds(variable: dict, base: dict) -> tuple[float, float]:
    domain = variable.get("domain")
    if domain == "NONNEGATIVE":
        return -base[parameter_key(variable["parameter"])] / quantity(variable["scale"]), math.inf
    if domain == "NONPOSITIVE":
        return -math.inf, -base[parameter_key(variable["parameter"])] / quantity(variable["scale"])
    return (-math.inf, math.inf) if domain is not None else (0, 1)


def latent_initial(spec: dict, base: dict) -> np.ndarray:
    result = []
    for variable, encoded in zip(spec["variables"], spec["optimizer"]["baseline_optimizer_coordinates_f64"]):
        y = float64_from_hex(encoded)
        if variable.get("domain") not in ("UNBOUNDED", "POSITIVE", "NEGATIVE"):
            lower, upper = domain_bounds(variable, base)
            al = 1 if not math.isfinite(lower) else min((upper - lower) / 2, (1 + abs(lower)) / 20)
            au = 1 if not math.isfinite(upper) else min((upper - lower) / 2, (1 + abs(upper)) / 20)
            y = lower - al + 2 * math.sqrt(al * (y - lower)) if y < lower + al else y if y < upper - au else upper + au - 2 * math.sqrt(au * (upper - y))
        result.append(y)
    return np.array(result)


def mapped_values(spec: dict, base: dict, latent: np.ndarray) -> dict:
    values = base.copy()
    for variable, x in zip(spec["variables"], latent):
        key = parameter_key(variable["parameter"])
        domain = variable.get("domain")
        if not math.isfinite(float(x)):
            raise InvalidCandidatePhysicalParameter("CMA candidate coordinate is non-finite", stage="unit_map")
        if domain is None:
            z = linquad(float(x), 0, 1)
            low, high = quantity(variable["lower"]), quantity(variable["upper"])
            values[key] = low + z * (high - low) if variable["transform"] == "linear" else low * (high / low) ** z
        elif domain in ("POSITIVE", "NEGATIVE"):
            with np.errstate(over="ignore", under="ignore"):
                values[key] = base[key] * float(np.exp(float(x)))
        else:
            lower, upper = domain_bounds(variable, base)
            coordinate = linquad(float(x), lower, upper) if domain != "UNBOUNDED" else float(x)
            values[key] = base[key] + quantity(variable["scale"]) * coordinate
        value = values[key]
        if (not math.isfinite(value) or
                (domain in ("POSITIVE", "NEGATIVE") and (value == 0 or math.copysign(1, value) != math.copysign(1, base[key]))) or
                (domain == "NONNEGATIVE" and value < 0) or (domain == "NONPOSITIVE" and value > 0)):
            raise InvalidCandidatePhysicalParameter("candidate mapping left its physical domain", stage="unit_map")
    return values


CMA_FIELDS = (
    "_n_dim", "_popsize", "_mu", "_mu_eff", "_cc", "_c1", "_cmu", "_c_sigma", "_d_sigma", "_cm", "_chi_n",
    "_weights", "_p_sigma", "_pc", "_mean", "_C", "_sigma", "_D", "_B", "_bounds", "_n_max_resampling", "_g",
    "_lr_adapt", "_alpha", "_beta_mean", "_beta_Sigma", "_gamma", "_Emean", "_ESigma", "_Vmean", "_VSigma",
    "_eta_mean", "_eta_Sigma", "_tolx", "_tolxup", "_tolfun", "_tolconditioncov", "_funhist_term", "_funhist_values",
)


def cma_snapshot(optimizer) -> dict:
    state = {}
    for name in CMA_FIELDS:
        value = getattr(optimizer, name)
        state[name] = {"array": array_record(value)} if isinstance(value, np.ndarray) else {"float_f64": bits(value)} if isinstance(value, (float, np.floating)) else value
    algorithm, keys, position, has_gaussian, cached = optimizer._rng.get_state()
    return {"schema": "scnsim.python_cma_state", "schema_version": 1, "fields": state,
            "rng": {"algorithm": algorithm, "keys": keys.tolist(), "position": position,
                    "has_gaussian": has_gaussian, "cached_gaussian_f64": bits(cached)}}


def restore_cma(optimizer, record: dict) -> None:
    if set(record["fields"]) != set(CMA_FIELDS) or record["rng"]["algorithm"] != "MT19937":
        raise CompilerInvariantError("checkpoint optimizer state layout differs", stage="benchmark_checkpoint")
    for name, value in record["fields"].items():
        value = array_from_record(value["array"]).copy() if isinstance(value, dict) and "array" in value else unbits(value["float_f64"]) if isinstance(value, dict) else value
        setattr(optimizer, name, value)
    rng = record["rng"]
    optimizer._rng.set_state(("MT19937", np.asarray(rng["keys"], dtype=np.uint32), rng["position"],
                              rng["has_gaussian"], float64_from_hex(rng["cached_gaussian_f64"])))


def physical_seed_coordinates(spec, values):
    """Resolve the existing unit-box/domain coordinates at an actual stage seed.

    Declaration encoding remains untouched. Domain reference zero is the
    actual supplied physical point, not a prior stage's controller coordinate.
    """
    result = []
    for variable in spec["variables"]:
        value = values[parameter_key(variable["parameter"])]
        domain = variable.get("domain")
        if domain is not None:
            if domain in {"POSITIVE", "NEGATIVE"} and (
                value == 0.0 or (value > 0.0) != (domain == "POSITIVE")
            ):
                raise InvalidOptimizationSpec(
                    "signed log domain requires a strictly signed initial value",
                    stage="spec_validation",
                )
            if domain == "NONNEGATIVE" and value < 0.0 or domain == "NONPOSITIVE" and value > 0.0:
                raise InvalidOptimizationSpec(
                    "initial value is outside optimization domain", stage="spec_validation"
                )
            if variable.get("scale") is not None and domain != "UNBOUNDED" and not math.isfinite(
                    value / quantity(variable["scale"])):
                raise InvalidOptimizationSpec(
                    "domain scale must be finite, positive, and same-unit", stage="spec_validation"
                )
            result.append(float64_hex(0.0))
            continue
        low, high = quantity(variable["lower"]), quantity(variable["upper"])
        if not low <= value <= high:
            raise InvalidOptimizationSpec(
                "sealed baseline must lie within resolved variable bounds",
                stage="spec_validation",
            )
        coordinate = ((value - low) / (high - low)
                      if variable["transform"] == "linear"
                      else math.log(value / low) / math.log(high / low))
        if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
            raise InvalidOptimizationSpec(
                "sealed baseline has no finite unit-box coordinate", stage="spec_validation"
            )
        result.append(float64_hex(coordinate))
    return tuple(result)


class CMAController:
    """One real stage optimizer, restored only from a post-tell snapshot."""

    __slots__ = ("_spec", "_base", "_optimizer")

    def __init__(self, spec, base_values, *, restore_state=None):
        from cmaes import CMA

        if version("cmaes") != "0.13.1":
            raise RuntimePreparationError(
                "benchmark CMA state requires cmaes==0.13.1",
                stage="benchmark_prepare",
            )
        self._spec = deepcopy(spec)
        self._base = deepcopy(base_values)
        controls = self._spec["optimizer"]
        optimizer = CMA(
            mean=latent_initial(self._spec, self._base),
            sigma=float64_from_hex(controls["initial_sigma_f64"]),
            population_size=controls["resolved_population_size"],
            bounds=None, seed=0,
        )
        seed = controls["seed"] & ((1 << 64) - 1)
        optimizer._rng.seed(
            np.array([seed & 0xffffffff, seed >> 32], dtype=np.uint32)
        )
        # The dependency allocates unused history with empty(); retain the
        # existing zeroing so snapshots never expose allocator contents.
        optimizer._funhist_values.fill(0)
        if restore_state is not None:
            restore_cma(optimizer, restore_state)
        self._optimizer = optimizer

    @classmethod
    def from_physical_seed(cls, spec, physical_values):
        """Start a new stage from physical values, never restore old curvature.

        A private resolved copy supplies the actual mean; the sealed request's
        null transition seed and deferred coordinates are not mutated.
        """
        resolved = deepcopy(spec)
        resolved["optimizer"]["baseline_optimizer_coordinates_f64"] = list(
            physical_seed_coordinates(spec, physical_values)
        )
        return cls(resolved, physical_values)

    @property
    def generation(self):
        return self._optimizer.generation

    @property
    def population_size(self):
        return self._optimizer.population_size

    def ask_population(self):
        return tuple(
            self._optimizer.ask() for _ in range(self._optimizer.population_size)
        )

    def map_coordinates(self, raw):
        return mapped_values(self._spec, self._base, raw)

    def tell(self, raw, costs):
        self._optimizer.tell([
            (value.copy(), cost) for value, cost in zip(raw, costs)
        ])

    def snapshot(self):
        return cma_snapshot(self._optimizer)
