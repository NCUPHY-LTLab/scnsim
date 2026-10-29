"""Immutable numerical call records shared by native producers and projections.

These records own declarations or call-local arrays only. They retain no Run,
Workspace, SQL connection, backend, factor or AD interpreter. Producers own
numerical execution; host projection owns certification and public Results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple


class ProgramInputs(NamedTuple):
    theta: Any
    omega_rad_s: Any
    branch_starts_rad_s: Any



@dataclass(frozen=True, slots=True)
class ObservableDescriptor:
    kind: str
    view_index: int
    frequency_slice: tuple[int, int]
    indices: tuple[int, ...] = ()
    branch_indices: tuple[int, ...] = ()
    projection: str | None = None
    quantity_key: str = ''
    si_unit: str = ''
    axes: tuple[tuple[str, tuple[str, ...]], ...] = ()
    dependency_indices: tuple[int, ...] = ()
    frequency_mode: str | None = None
    zero_charge_indices: tuple[int, ...] = ()
    retained_indices: tuple[int, ...] = ()



class ProgramAux(NamedTuple):
    status: Any
    failed_site: Any
    certificates: tuple
    derivative_available: Any
    derivative_reasons: tuple[str, ...]
    realized_capabilities: tuple[bool, ...]
    observable_available: tuple
    observable_reasons: tuple[tuple[str, ...], ...]
    branch_results: tuple = ()
    quantity_results: tuple = ()
    observable_values: tuple = ()
    observable_raw_values: tuple = ()
    static_facts: tuple = ()



class ProgramOutcome(NamedTuple):
    values: tuple
    aux: ProgramAux



class JacobianPayload(NamedTuple):
    values: Any
    row_indices: tuple[int, ...]
    shape: tuple[int, int]



class DerivativeOutcome(NamedTuple):
    primal: ProgramOutcome
    derivative: Any
    aux: ProgramAux



@dataclass(frozen=True, slots=True)
class RootDeclaration:
    kind: str
    start_index: int
    quantity_key: str
    coordinate_index: int | None = None
    row_index: int | None = None
    column_index: int | None = None
    family: str | None = None
    output_index: int | None = None
    input_index: int | None = None

