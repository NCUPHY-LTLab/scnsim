"""Transient maintainer diagnostics; never solve status or persisted evidence.

The caller supplies the residual and scale from its actual postverification.
This module imports no numerical backend and owns no solver or tolerance.
"""
from __future__ import annotations
from typing import Any, NamedTuple


class ComponentwiseDiagnostics(NamedTuple):
    per_rhs: Any
    performed: bool
    unavailable_reason: str | None


class SolveWithDiagnostics(NamedTuple):
    outcome: Any
    diagnostics: ComponentwiseDiagnostics


def componentwise_per_rhs(error, scale, *, xp):
    """Return max_i |r_ij|/s_ij for each RHS; no eligibility decision.

    A vector is one RHS. Exact 0/0 is zero, nonzero/0 and nonfinite operands
    produce infinity. The supplied scale is |A||X|+|B|, not a reduced maximum.
    """
    absolute = xp.abs(error)
    zero = scale == 0
    safe_scale = xp.where(zero, xp.ones_like(scale), scale)
    ratios = absolute / safe_scale
    infinity = xp.full_like(ratios, xp.inf)
    ratios = xp.where(zero, xp.where(absolute == 0, xp.zeros_like(ratios), infinity), ratios)
    ratios = xp.where(xp.isfinite(absolute) & xp.isfinite(scale), ratios, infinity)
    per_rhs = xp.max(ratios, axis=0, initial=0)
    if error.ndim == 1:
        per_rhs = xp.reshape(per_rhs, (1,))
    return ComponentwiseDiagnostics(per_rhs, True, None)


def unperformed_diagnostics():
    """An early-returned equation has no postverification arrays to diagnose."""
    return ComponentwiseDiagnostics(None, False, 'equation did not reach postverification')
