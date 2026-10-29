"""Ordered declared objective arithmetic for the checked physical graph.

This module owns loss formulas only. Expression lowering supplies scalar SI
operands and explicit smooth-square eligibility. The checked graph/coordinator
owns numerical unavailability and existing nonfinite-objective failures; this
helper neither replaces them by finite penalties nor chooses a derivative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, NamedTuple, Sequence

from ..canonical import float64_from_hex


OperandMode = Literal["value", "squared_magnitude_zero_target"]


class ObjectiveLossPayload(NamedTuple):
    """Actual ordered diagnostic values and arithmetic cost, with no live owner.

    Magnitudes in the smooth-square mode are reporting auxiliaries: they do
    not become a differentiable standalone abs-at-zero observable. The graph
    differentiates total_cost, whose square is formed directly from components.
    """

    values: tuple[Any, ...]
    normalized_residuals: tuple[Any, ...]
    contributions: tuple[Any, ...]
    total_cost: Any


@dataclass(frozen=True, slots=True)
class _LossObjective:
    id: str
    comparison: Literal["target", "at_least"]
    target: float
    scale: float | None
    weight: float
    operand_mode: OperandMode
    si_unit: str


@dataclass(frozen=True, slots=True)
class ObjectiveLossProgram:
    """One stable callable bound to an immutable ordered declaration."""

    objectives: tuple[_LossObjective, ...]
    _output_pullback: Any = field(init=False, repr=False, compare=False)
    _algebra: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        ordered = self.structural_record()
        def algebra(operands, coefficients):
            return _ordered_loss_payload(ordered, operands, coefficients)
        object.__setattr__(self, '_algebra', algebra)
        def function(output_values, runtime_coefficients):
            import jax
            import jax.numpy as jnp
            def scalar(values):
                payload = algebra(values, runtime_coefficients)
                return payload.total_cost, payload
            total, pullback, payload = jax.vjp(scalar, output_values, has_aux=True)
            return payload, pullback(jnp.ones_like(total))[0]
        object.__setattr__(self, '_output_pullback', function)

    def algebra_function(self):
        """Coefficient-free callable; captures only ordered static loss algebra."""
        return self._algebra

    def output_pullback_function(self):
        """Stable pure loss producer shared by warmup and bounded response AD."""
        return self._output_pullback

    @property
    def objective_ids(self) -> tuple[str, ...]:
        return tuple(objective.id for objective in self.objectives)

    def structural_record(self):
        """Static ordered algebra; numeric coefficients remain runtime operands."""
        return tuple((row.id, row.comparison, row.operand_mode, row.si_unit)
                     for row in self.objectives)

    def runtime_operands(self):
        """Use execution-finalized scales; never manufacture a baseline scale."""
        if any(row.scale is None for row in self.objectives):
            raise RuntimeError('Objective runtime scales must be finalized by execution')
        return (tuple(row.target for row in self.objectives),
                tuple(row.scale for row in self.objectives),
                tuple(row.weight for row in self.objectives))

    def __call__(self, operands: Sequence[Any], *, coefficients=None) -> ObjectiveLossPayload:
        coefficients = self.runtime_operands() if coefficients is None else coefficients
        return self._algebra(operands, coefficients)


def _ordered_loss_payload(objectives, operands, coefficients):
    """Single ordered loss formula for value and transformed output pullback."""
    from jax import lax
    import jax.numpy as jnp
    targets, scales, weights = coefficients
    values, residuals, contributions = [], [], []
    for ordinal, (objective, operand) in enumerate(zip(objectives, operands, strict=True)):
        # Round the public scalar operand before residual arithmetic can fuse
        # with its producer; preserve the selected real or complex dtype.
        operand = lax.optimization_barrier(jnp.asarray(operand))
        real = jnp.real(operand)
        target = jnp.asarray(targets[ordinal], dtype=real.dtype)
        scale = jnp.asarray(scales[ordinal], dtype=real.dtype)
        weight = jnp.asarray(weights[ordinal], dtype=real.dtype)
        if objective[2] == "squared_magnitude_zero_target":
            value = jnp.abs(operand)
            residual = value / scale
            # No abs/sqrt node lies on the cost path. In particular, a
            # zero raw operand has the derivative of this smooth square.
            real_scaled = real / scale
            imag_scaled = jnp.imag(operand) / scale
            contribution = weight * (
                real_scaled * real_scaled + imag_scaled * imag_scaled
            )
        else:
            value = operand
            residual = (value - target) / scale
            if objective[1] == "at_least":
                residual = jnp.maximum(jnp.zeros_like(residual), -residual)
            contribution = weight * (residual * residual)
        values.append(value)
        residuals.append(residual)
        contributions.append(contribution)
    # Preserve declared accumulation order rather than introducing a
    # reduction-tree-dependent cost or another aggregation authority.
    total = contributions[0]
    for contribution in contributions[1:]:
        total = total + contribution
    return ObjectiveLossPayload(
        tuple(values), tuple(residuals), tuple(contributions), total
    )


def compile_objective_loss(
    objectives: Sequence[Mapping[str, object]],
    *,
    operand_modes: Sequence[OperandMode],
    require_resolved_scales: bool = True,
) -> ObjectiveLossProgram:
    """Bind finalized request records; do not resolve targets/scales again.

    ``value`` consumes the actual real scalar expression value. The explicit
    smooth-square mode consumes its raw abs operand and is only the algebraic
    target-zero square; expression lowering, not this helper, authorizes that
    transformation. Physical units have already been lowered to the recorded SI
    unit. Existing canonical token decoding retains its native integrity errors.
    """
    declarations = []
    for record, mode in zip(objectives, operand_modes, strict=True):
        comparison = record["comparison"]
        target = float64_from_hex(record["target"]["si_value_f64"])
        if require_resolved_scales:
            scale = float64_from_hex(record["resolved_scale"]["si_value_f64"])
        else:
            resolved = record.get("resolved_scale")
            scale = None if resolved is None else float64_from_hex(resolved["si_value_f64"])
        weight = float64_from_hex(record["weight_f64"])
        if comparison not in {"target", "at_least"}:
            raise ValueError("objective comparison must be target or at_least")
        if mode not in {"value", "squared_magnitude_zero_target"}:
            raise ValueError("unknown objective operand mode")
        if mode == "squared_magnitude_zero_target" and (
            comparison != "target" or target != 0.0
        ):
            raise ValueError("smooth magnitude square requires an exact zero target")
        declarations.append(_LossObjective(
            record["id"], comparison, target, scale, weight, mode, record["target"]["si_unit"]
        ))
    return ObjectiveLossProgram(tuple(declarations))
