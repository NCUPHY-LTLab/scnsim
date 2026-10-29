"""Canonical numerical request selectors for the single physical frontend.

Ordinary execution and explicit warmup share this declaration-only translator.
It chooses no point, frequency, View or physical route: every operand comes
from the exact prepared request. Numerical lowering remains QuantityProgram's
sole authority, and persistence/body encoding belongs to Execution/Results.
"""
from __future__ import annotations


def quantity_declarations(spec, view):
    """Ordered canonical Quantity mapping for one exact prepared declaration."""
    kind = spec['type']
    if kind == 'direct_solve':
        return {family: {'type': 'matrix_projection', 'projection': family,
                         'spec': spec, 'view': view} for family in ('S', 'Y', 'Z')}
    if kind == 'static_capacitance':
        selector, projection = 'static_capacitance_projection', 'matrix'
    elif kind == 'static_capacitance_element':
        selector, projection = 'static_capacitance_element_projection', 'value'
    elif kind == 'operator':
        selector, projection = 'operator_projection', 'matrix'
    elif kind == 'response_element':
        selector, projection = 'response_element_projection', None
    elif kind in ('diagonal_root', 'operator_element_root', 'hybridized_pole', 'transfer_zero'):
        selector = {'diagonal_root': 'diagonal_root_projection',
                    'operator_element_root': 'operator_element_root_projection',
                    'hybridized_pole': 'hybridized_pole_projection',
                    'transfer_zero': 'transfer_zero_projection'}[kind]
        projection = None
    elif kind == 'residue_normalized_coupling':
        selector, projection = 'residue_coupling_projection', None
    else:
        raise TypeError(f'unsupported physical declaration {kind}')
    return {'value': {'type': selector, 'projection': projection,
                      'spec': spec, 'view': view}}
