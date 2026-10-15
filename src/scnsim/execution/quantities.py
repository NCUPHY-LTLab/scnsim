"""Pure quantity records, declared expressions and root-continuation policy.

Physical computation belongs to the shared QuantityProgram. This module never
builds backend jobs or owns an evaluator/cache. The existing baseline-anchored
continuation rule consumes caller-supplied actual branch evaluations; projection
and passive diagonal policy have their unique authority in quantity_policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from hashlib import sha256
import math

import numpy as np

from ..canonical import canonical_json_bytes
from ..numerics.models import (
    EvaluationResult,
    NumericalFailure,
)
from ..numeric_encoding import array_record, record_bytes, record_document


_ARRAY_FIELDS = ("S", "Y", "Z", "operator_values", "capacitance_values", "null_vector", "branch_roots_rad_s")
_SCALAR_FIELDS = (
    "response_value", "root_omega_rad_s", "root_slope", "numerator_slope", "denominator",
    "residue_a", "residue_b", "coupling_rad_s", "evaluation_omega_rad_s",
)
_ROOT_KINDS = frozenset(("diagonal_root", "operator_element_root", "hybridized_pole", "transfer_zero"))
_ROOT_SELECTORS = frozenset((
    "diagonal_root_projection", "operator_element_root_projection", "hybridized_pole_projection",
    "transfer_zero_projection",
))
_LEAF_SELECTORS = _ROOT_SELECTORS | frozenset(("response_element_projection", "residue_coupling_projection"))


def quantity_record(result: EvaluationResult) -> dict[str, object]:
    """Encode the single numerical result body without widening its dtype."""
    record: dict[str, object] = {
        "id": result.id,
        "status": "failure" if result.failure is not None else "success",
        "evidence": record_document(result.evidence_bytes),
    }
    if result.failure is not None:
        record["failure"] = {
            "kind": result.failure.kind,
            "stage": result.failure.stage,
            "detail": result.failure.detail,
            "evidence": record_document(result.failure.evidence_bytes),
        }
        return record
    for name in _ARRAY_FIELDS:
        value = getattr(result, name)
        if value is not None:
            record[name] = array_record(value)
    for name in _SCALAR_FIELDS:
        value = getattr(result, name)
        if value is not None:
            record[name] = array_record(np.asarray(value))
    return record




def quantity_body_id(result: EvaluationResult) -> tuple[str, dict[str, object]]:
    body = quantity_record(result)
    return sha256(record_bytes(body)).hexdigest(), body


def root_frequency_linewidth(root_omega_rad_s: object) -> tuple[float, float]:
    """Project a certified omega into public Hz fields in declaration units."""
    omega = complex(root_omega_rad_s)
    return omega.real / (2.0 * math.pi), -omega.imag / math.pi


def expression_leaves(expression: dict) -> list[dict]:
    """Flatten scalar expression leaves in their declared evaluation order."""
    kind = expression["type"]
    if kind == "quantity_sum":
        return [leaf for term in expression["terms"] for leaf in expression_leaves(term)]
    if kind == "quantity_difference":
        return expression_leaves(expression["left"]) + expression_leaves(expression["right"])
    if kind == "quantity_absolute":
        return expression_leaves(expression["operand"])
    if kind not in _LEAF_SELECTORS:
        raise NotImplementedError(f"benchmark quantity {kind!r} is unsupported")
    return [expression]








class EvaluationFailure(Exception):
    """A structured backend failure encountered while resolving a dependency."""

    def __init__(self, failure: NumericalFailure, *, result: EvaluationResult | None = None,
                 dependencies: Mapping[str, dict[str, object]] | None = None):
        self.failure = failure
        self.result = result
        self.dependencies = dict(dependencies or {})
        super().__init__(failure.detail)


def continue_root(*, baseline: dict, values: dict, baseline_root: complex,
                  endpoint_view, initial_result: EvaluationResult,
                  candidate_view, make_job, evaluate, identity: str, observe=None,
                  continuation_step_scope=None):
    """The existing baseline-anchored dyadic continuation for every root kind."""
    def evaluate_intermediate_step(left_root, right_t):
        # A candidate actor may need to bind its current mutable state around
        # continuation preparation, evaluation and evidence observation. The
        # typed result is immutable, so it can safely leave that scope before
        # recursion.
        scope = (nullcontext() if continuation_step_scope is None
                 else continuation_step_scope())
        with scope:
            point = {}
            for parameter, base in baseline.items():
                candidate = values[parameter]
                if isinstance(base, dict):
                    if canonical_json_bytes(base) != canonical_json_bytes(candidate):
                        from ..errors import CompilerInvariantError
                        raise CompilerInvariantError("root continuation cannot interpolate RLGC", stage="root_continuation")
                    point[parameter] = base
                else:
                    point[parameter] = base + right_t * (candidate - base)
            view = candidate_view(point)
            result = evaluate((make_job(f"continuation:{identity}:{right_t.hex()}", view, left_root),))[0]
            if observe is not None:
                observe(point, view, result, right_t)
            return result

    def advance(left_t, left_root, right_t, depth, result=None):
        if right_t == 1.0:
            point, view = values, endpoint_view
            if result is None:
                result = evaluate((make_job(f"continuation:{identity}:{right_t.hex()}", view, left_root),))[0]
            if observe is not None:
                observe(point, view, result, right_t)
        else:
            result = evaluate_intermediate_step(left_root, right_t)
        if result.failure is None:
            return result
        if result.failure.kind != "numerical_resolution_unresolved" or depth >= 32:
            raise EvaluationFailure(result.failure, result=result)
        midpoint = (left_t + right_t) / 2
        middle = advance(left_t, left_root, midpoint, depth + 1)
        return advance(midpoint, middle.root_omega_rad_s, right_t, depth + 1)

    return advance(0.0, baseline_root, 1.0, 0, initial_result)
