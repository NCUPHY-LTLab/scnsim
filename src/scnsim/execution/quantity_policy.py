"""Pure projections and existing diagonal-root publication policy.

Consumes detached actual numerical results only. Physical compilation, graph
execution, continuation and persistence have separate owners; this module runs
none of them and changes no root certificate or numerical precision policy.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from ..numerics.models import EvaluationResult, NumericalFailure
from ..numeric_encoding import record_bytes, record_document

_ROOT_SELECTORS = frozenset((
    'diagonal_root_projection', 'operator_element_root_projection',
    'hybridized_pole_projection', 'transfer_zero_projection',
))


def diagonal_passive_policy(result: EvaluationResult) -> EvaluationResult:
    """Preserve the existing positive-imaginary diagonal-root failure contract."""
    if result.failure is None and result.root_omega_rad_s.imag > 0:
        from ..canonical import float64_hex
        evidence = record_document(result.evidence_bytes)
        omega = complex(result.root_omega_rad_s)
        evidence['root_omega_rad_s'] = {
            'real_f64': float64_hex(omega.real), 'imag_f64': float64_hex(omega.imag),
        }
        return replace(result, root_omega_rad_s=None, root_slope=None,
                       failure=NumericalFailure(
                           'numerical_resolution_unresolved', 'newton_certificate',
                           'diagonal root violates passive imaginary-root policy', record_bytes(evidence)))
    return result


def project_quantity(selector: Mapping[str, object], result: EvaluationResult) -> float:
    """Present one declared scalar selector from its actual detached primal body."""
    # These common body/failure utilities remain independent of any evaluator.
    # Import at the call boundary so quantities may consume this policy module.
    from .quantities import EvaluationFailure, root_frequency_linewidth
    if result.failure is not None:
        raise EvaluationFailure(result.failure, result=result)
    kind = selector['type']
    projection = selector.get('projection')
    if kind in _ROOT_SELECTORS:
        frequency, linewidth = root_frequency_linewidth(result.root_omega_rad_s)
        values = {'frequency': frequency, 'linewidth': linewidth}
        return float(values[projection])
    if kind == 'response_element_projection':
        value = result.response_value
        values = {'real': float(np.real(value)), 'imag': float(np.imag(value)),
                  'magnitude': float(abs(value))}
        return values[projection]
    if kind == 'residue_coupling_projection':
        value = result.coupling_rad_s
        values = {'real': float(np.real(value)), 'imag': float(np.imag(value)),
                  'magnitude': float(abs(value))}
        return values[projection]
    raise NotImplementedError(f'quantity selector {kind!r} is unsupported')
