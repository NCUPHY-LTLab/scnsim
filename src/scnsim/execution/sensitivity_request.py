"""Pure finalized Sensitivity declarations, independent of execution ownership.

The facade resolves exact Plan parameters, coordinate aliases and owned View
leaves first. This handoff retains canonical bytes and scalar identities only;
it cannot execute, publish evidence or retain a Run, callback or factor.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from .. import units
from ..canonical import canonical_json_bytes
from .identity import _jax_runtime_identity
from .prepared import PreparedAnalysis


@dataclass(frozen=True, slots=True)
class PreparedSensitivity:
    analysis: PreparedAnalysis
    effective_point_bytes: bytes
    active_keys: tuple[tuple[str, str], ...]
    direction_bytes: bytes | None
    weights_bytes: bytes | None
    scales_bytes: bytes | None
    leaf_view_bytes: tuple[bytes, ...]
    output_selection_bytes: tuple[bytes, ...]
    functional_unit: str | None

    @property
    def request_bytes(self) -> bytes:
        return self.analysis.request_bytes

    @property
    def request_sha256(self) -> str:
        return self.analysis.request_sha256

    def request(self) -> dict[str, object]:
        return self.analysis.request()

    def source_units(self) -> tuple[dict[str, object], ...]:
        return self.analysis.source_units()

    def effective_point(self) -> dict[str, object]:
        return json.loads(self.effective_point_bytes)


def _leaves(expression):
    kind = expression['type']
    if kind == 'quantity_sum':
        for term in expression['terms']:
            yield from _leaves(term)
    elif kind == 'quantity_difference':
        yield from _leaves(expression['left'])
        yield from _leaves(expression['right'])
    elif kind == 'quantity_absolute':
        yield from _leaves(expression['operand'])
    else:
        yield expression


def _output_unit(expression):
    kind = expression['type']
    if kind == 'quantity_sum':
        return _output_unit(expression['terms'][0])
    if kind == 'quantity_difference':
        return _output_unit(expression['left'])
    if kind == 'quantity_absolute':
        return _output_unit(expression['operand'])
    if kind == 'matrix_projection':
        return {'S': 'dimensionless', 'Y': 'siemens', 'Z': 'ohm'}[expression['projection']]
    if kind == 'operator_projection':
        return 'siemens / second'
    if kind == 'response_element_projection':
        return {'S': 'dimensionless', 'Y': 'siemens', 'Z': 'ohm'}[expression['spec']['family']]
    if kind == 'residue_coupling_projection':
        return 'radian / second'
    if kind in ('diagonal_root_projection', 'operator_element_root_projection',
                'hybridized_pole_projection', 'transfer_zero_projection'):
        return 'hertz'
    raise TypeError(f'unsupported Sensitivity output declaration {kind}')


def finalize_sensitivity_spec(encoded_spec: Mapping[str, object]) -> dict[str, object]:
    """Bind the declared cotangent unit without a parameter point or runtime."""
    output_units = {row['id']: _output_unit(row['quantity']) for row in encoded_spec['quantities']}
    derivative = encoded_spec['derivative']
    functional_unit = None
    if derivative['type'] == 'vjp':
        for row in derivative['weights']:
            label = row['output']['quantity_label']
            product = units.registry.Quantity(1, row['value']['si_unit']) * units.registry.Quantity(1, output_units[label])
            if functional_unit is None:
                functional_unit = str(product.to_base_units().units)
            else:
                product.to(functional_unit)  # Pint owns incompatible cotangent units.
    return {**encoded_spec, 'functional_unit': functional_unit}


def prepare_sensitivity(*, plan_sha256: str, encoded_spec: Mapping[str, object],
                        parameter_record: Mapping[str, object], runtime_base,
                        precision: str, source_units=()) -> PreparedSensitivity:
    """Bind finalized physical data without lowering a numerical model."""
    quantities = encoded_spec['quantities']
    active_keys = tuple((ref['definitions_id'], ref['parameter_id']) for ref in encoded_spec['wrt'])
    leaf_views = []
    seen_views = set()
    selections = []
    output_units = {}
    for quantity in quantities:
        label, expression = quantity['id'], quantity['quantity']
        output_units[label] = _output_unit(expression)
        selections.append(canonical_json_bytes({'id': label, 'quantity': expression,
                                                'si_unit': output_units[label]}))
        for leaf in _leaves(expression):
            encoded_view = canonical_json_bytes(leaf['view'])
            if encoded_view not in seen_views:
                seen_views.add(encoded_view)
                leaf_views.append(encoded_view)
    derivative = encoded_spec['derivative']
    spec = finalize_sensitivity_spec(encoded_spec)
    functional_unit = spec['functional_unit']
    runtime = _jax_runtime_identity(runtime_base, precision=precision)
    runtime['algorithm_id'] = 'scnsim.jax.physical_program.sensitivity.v1'
    analysis = PreparedAnalysis.create(
        plan_sha256=plan_sha256, operation='sensitivity',
        view={'type': 'network_view_set', 'views': [json.loads(value) for value in leaf_views]},
        spec=spec, parameter_source={'kind': 'point', 'parameters': parameter_record},
        runtime_semantic=runtime, source_units=source_units,
    )
    return PreparedSensitivity(
        analysis=analysis, effective_point_bytes=canonical_json_bytes(parameter_record),
        active_keys=active_keys,
        direction_bytes=None if derivative['type'] != 'jvp' else canonical_json_bytes(derivative['direction']),
        weights_bytes=None if derivative['type'] != 'vjp' else canonical_json_bytes(derivative['weights']),
        scales_bytes=None if spec['scales'] is None else canonical_json_bytes(spec['scales']),
        leaf_view_bytes=tuple(leaf_views), output_selection_bytes=tuple(selections),
        functional_unit=functional_unit,
    )
