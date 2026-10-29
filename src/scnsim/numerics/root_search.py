"""Discrete declared candidate selection over the one checked root authority.

Discovery is host-owned and never differentiated. It retains every source
candidate and correction outcome; selected working packets belong to the
current candidate only. No region, tolerance, deduplication or retry defaults
are supplied here. Local implicit AD consumes the selected certified packet.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from ..compilation.mesh import quantity
from ..canonical import float64_from_hex
from ..numeric_encoding import record_document
from .evidence import evidence_bytes
from .models import EvaluationResult, NumericalFailure


def complex_frequency(value):
    if value['type'] == 'complex_quantity_f64':
        return complex(float64_from_hex(value['real_si_f64']),
                       float64_from_hex(value['imag_si_f64']))
    return complex(quantity(value))


def resolve_region(search, authored_hint):
    """Resolve relative extent from the original declaration, never an anchor."""
    declaration = search['region']
    if 'hint_frequency_factor_f64' in declaration:
        real = (0.0, authored_hint.real *
                float64_from_hex(declaration['hint_frequency_factor_f64']))
    else:
        real = tuple(quantity(value) for value in declaration['real_frequency_interval'])
    imag = (None if 'imag_frequency_interval' not in declaration else
            tuple(quantity(value) for value in declaration['imag_frequency_interval']))
    return real, imag


def select_declared_root(search, authored_hint, candidates, correct, *, catalogue=None):
    """Correct declared candidates in source order and return one actual packet.

    ``correct(omega_rad_s)`` is the original checked factory and projector.
    Unexpected exceptions escape unchanged. Failure of one proposed candidate
    is kept as data and does not change another candidate's equation prior.
    """
    real, imag = resolve_region(search, authored_hint)
    def contains(frequency):
        return (real[0] <= frequency.real <= real[1]
                and (imag is None or imag[0] <= frequency.imag <= imag[1]))
    records, accepted = [], []
    def compact_facts(value):
        if isinstance(value, dict):
            return {key:compact_facts(item) for key,item in value.items()
                    if key != 'selected_state'}
        if isinstance(value, list):
            return [compact_facts(item) for item in value]
        return value
    for ordinal, omega in enumerate(candidates):
        if omega is None:
            records.append({'source_index': ordinal, 'defined': False})
            continue
        frequency = complex(omega) / (2 * np.pi)
        row = {'source_index': ordinal, 'defined': True,
               'proposed_omega_rad_s': complex(omega), 'inside_region': contains(frequency)}
        records.append(row)
        if not row['inside_region']:
            continue
        result, packet = correct(omega)
        row.update(result={
            'id': result.id,
            'root_omega_rad_s': result.root_omega_rad_s,
            'failure': None if result.failure is None else {
                'kind': result.failure.kind, 'stage': result.failure.stage,
                'detail': result.failure.detail,
                'evidence': compact_facts(record_document(result.failure.evidence_bytes)),
            },
            'evidence': compact_facts(record_document(result.evidence_bytes)),
        }, corrected_inside_region=False)
        if result.failure is not None:
            continue
        corrected = complex(np.asarray(packet.omega).item()) / (2 * np.pi)
        row['corrected_inside_region'] = contains(corrected)
        if row['corrected_inside_region']:
            distance = abs(corrected - authored_hint)
            row['distance_hz'] = distance
            accepted.append((distance, ordinal, result, packet))
    metadata = {'declaration': search, 'authored_hint_hz': authored_hint,
                'resolved_real_frequency_interval_hz': real,
                'resolved_imag_frequency_interval_hz': imag,
                'catalogue': catalogue, 'candidates': tuple(records),
                'selected_source_index': None}
    if accepted:
        nearest = min(row[0] for row in accepted)
        matches = [row for row in accepted if row[0] == nearest]
        if len(matches) == 1 or search['selection']['tie'] == 'source_order':
            _, ordinal, result, packet = matches[0]
            metadata['selected_source_index'] = ordinal
            facts = record_document(result.evidence_bytes)
            facts['root_search'] = metadata
            return replace(result, evidence_bytes=evidence_bytes(facts)), packet, metadata
        detail = 'Declared root search has an exact nearest-Hint selection tie'
    else:
        detail = 'Declared root search returned no certified candidate inside its region'
    failure = NumericalFailure('numerical_resolution_unresolved', 'root_search',
                               detail, evidence_bytes(metadata))
    return EvaluationResult('root_search', failure=failure), None, metadata
