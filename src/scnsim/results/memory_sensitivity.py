"""Detached Sensitivity Results for all-storage-off JAX execution.

The execution adapter returns synchronized host blocks and their shared
certificate authority. This module binds those actual values to the existing
unit-aware Sensitivity projections without a Workspace reader, receipt
identity, or numeric-array codec.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import numpy as np
from pint import Quantity

from .. import units
from ..authoring import ParameterRef, ParameterSet
from ..construction import unavailable
from ..diagnostics.memory_operation import InMemoryBenchmarkResult
from ..errors import SCNSimStateError
from ..specs import OutputCoordinate
from ..value_storage import immutable_array
from ..visualization.presentation import Theme
from .base import Result, _sha256
from .decode import VerifiedResultDecoder
from .decode_sensitivity import (
    _axes,
    _decode_quantity,
    _output_coordinate,
    _quantity_output_coordinates,
    _request_derivative_diagnostics,
)
from .factory import _MEMORY_RESULT_TOKEN, _verified_result
from .memory_optimization import _parameter_set
from .sensitivity import (
    SensitivityJacobianBlock,
    SensitivityParameterBlock,
    SensitivityProjection,
    SensitivityQuantity,
    SensitivityResult,
    SensitivityValueBlock,
    _NormalizedJacobian,
)

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


@dataclass(frozen=True, slots=True)
class InMemorySensitivityResult(Result):
    """Materialized checked derivative output with execution-local provenance.

    It intentionally has no AnalysisResult/receipt identity. Every public
    block is detached before return; no JAX residual, selected root state,
    frame owner, or derivative closure survives in this value.
    """

    plan_sha256: str
    request_sha256: str
    execution_id: str
    persistence: Literal["memory"]
    operation: str
    status: Literal["complete", "partial"]
    quantities: tuple[SensitivityQuantity, ...]
    primal: tuple[SensitivityValueBlock, ...]
    operations: Mapping[str, SensitivityProjection]
    parameters: ParameterSet
    quantity_metadata: tuple[Mapping[str, object], ...]
    observable_metadata: tuple[Mapping[str, object], ...]
    output_layout: Mapping[str, object]
    certificates: tuple[object, ...]
    summary: Mapping[str, object]
    _presentation: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    _benchmark: InMemoryBenchmarkResult | None = field(default=None, repr=False, compare=False)
    _memory_report_token: object = field(default=None, init=False, repr=False, compare=False)

    def __init__(self) -> None:
        unavailable("InMemorySensitivityResult construction")

    @property
    def persistence_kind(self) -> Literal["memory"]:
        return "memory"

    @property
    def benchmark(self) -> InMemoryBenchmarkResult:
        benchmark = object.__getattribute__(self, "_benchmark")
        if benchmark is None:
            raise RuntimeError("in-memory Sensitivity result finalization is incomplete")
        return benchmark

    def __repr__(self) -> str:
        operation_names = ", ".join(self.operations)
        return (
            "InMemorySensitivityResult("
            f"execution_id={self.execution_id!r}, status={self.status!r}, "
            f"quantities={len(self.quantities)}, operations=({operation_names}))"
        )

    def value(self, block_id: str) -> SensitivityValueBlock:
        for block in self.primal:
            if block.block_id == block_id:
                return block
        raise KeyError(block_id)

    def _operation(self, kind: Literal["jacobian", "jvp", "vjp"]) -> SensitivityProjection:
        try:
            return self.operations[kind]
        except KeyError as error:
            raise SCNSimStateError(
                f"{kind} was not requested in this Sensitivity Result",
                stage="result_access",
                evidence={"requested_operations": tuple(self.operations)},
            ) from error

    @property
    def jacobian(self) -> SensitivityProjection:
        return self._operation("jacobian")

    @property
    def jvp(self) -> SensitivityProjection:
        return self._operation("jvp")

    @property
    def vjp(self) -> SensitivityProjection:
        return self._operation("vjp")

    @property
    def normalized_jacobian(self) -> Mapping[str, np.ndarray]:
        scales = self._presentation.get("normalization_scales")
        if not isinstance(scales, Mapping):
            raise SCNSimStateError(
                "normalized_jacobian requires explicit scales in the Sensitivity request",
                stage="result_access",
                evidence={"operation": "jacobian", "scales_available": False},
            )
        inputs, outputs = scales.get("inputs"), scales.get("outputs")
        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
            raise SCNSimStateError(
                "normalized_jacobian requires explicit input and output scales",
                stage="result_access",
                evidence={"operation": "jacobian", "scales_available": False},
            )
        return _NormalizedJacobian(self.jacobian, inputs, outputs)

    def plot(
        self,
        *,
        kind: Literal["jacobian", "jvp", "vjp"],
        block_id: str,
        style: Literal["auto", "table", "bar", "heatmap"] = "auto",
        output_unit: str | None = None,
        output_units: Mapping[OutputCoordinate, str] | None = None,
        parameter_units: Mapping[ParameterRef | tuple[str, str], str] | None = None,
        parameter_labels: Mapping[ParameterRef | tuple[str, str], str] | None = None,
        normalized: bool = False,
        input_scales: Mapping[ParameterRef | tuple[str, str], Quantity] | None = None,
        output_scales: Mapping[OutputCoordinate, Quantity] | None = None,
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        return SensitivityResult.plot(
            self, kind=kind, block_id=block_id, style=style,
            output_unit=output_unit, output_units=output_units,
            parameter_units=parameter_units, parameter_labels=parameter_labels,
            normalized=normalized, input_scales=input_scales,
            output_scales=output_scales, theme=theme,
        )

    def show(
        self,
        *,
        kind: Literal["jacobian", "jvp", "vjp"] | None = None,
        block_id: str | None = None,
        style: Literal["auto", "table", "bar", "heatmap"] = "auto",
        output_unit: str | None = None,
        output_units: Mapping[OutputCoordinate, str] | None = None,
        parameter_units: Mapping[ParameterRef | tuple[str, str], str] | None = None,
        parameter_labels: Mapping[ParameterRef | tuple[str, str], str] | None = None,
        normalized: bool = False,
        input_scales: Mapping[ParameterRef | tuple[str, str], Quantity] | None = None,
        output_scales: Mapping[OutputCoordinate, Quantity] | None = None,
        theme: Theme = Theme.AUTO,
    ) -> object:
        return SensitivityResult.show(
            self, kind=kind, block_id=block_id, style=style,
            output_unit=output_unit, output_units=output_units,
            parameter_units=parameter_units, parameter_labels=parameter_labels,
            normalized=normalized, input_scales=input_scales,
            output_scales=output_scales, theme=theme,
        )


def _parameter_refs(
    rows: object,
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
) -> tuple[ParameterRef, ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("in-memory Sensitivity parameter layout must be ordered")
    references = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("in-memory Sensitivity parameter identity is malformed")
        references.append(parameter_lookup[(row["definitions_id"], row["parameter_id"])])
    return tuple(references)


def _parameter_layout_refs(
    rows: object,
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
) -> tuple[ParameterRef, ...]:
    """Resolve the producer's compact input-axis records in their declared order."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("in-memory Sensitivity parameter layout must be ordered")
    references = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row["parameter"], Mapping):
            raise TypeError("in-memory Sensitivity parameter layout row is malformed")
        record = row["parameter"]
        references.append(parameter_lookup[(record["definitions_id"], record["parameter_id"])])
    return tuple(references)


def _normalization_scales(
    spec: Mapping[str, object],
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
) -> Mapping[str, object]:
    scales_record = spec.get("scales")
    if scales_record is None:
        return {}
    if not isinstance(scales_record, Mapping):
        raise TypeError("in-memory Sensitivity scales must be a mapping")
    decoder = object.__new__(VerifiedResultDecoder)
    object.__setattr__(decoder, "_parameter_lookup", parameter_lookup)
    inputs: dict[tuple[str, str], Quantity] = {}
    for row in scales_record["inputs"]:
        if not isinstance(row, Mapping):
            raise TypeError("in-memory Sensitivity input scale is malformed")
        reference = decoder._decode_parameter_ref(row["parameter"])
        inputs[(reference.definitions_id, reference.id)] = _decode_quantity(row["value"])
    outputs: dict[OutputCoordinate, Quantity] = {}
    for row in scales_record["outputs"]:
        if not isinstance(row, Mapping):
            raise TypeError("in-memory Sensitivity output scale is malformed")
        outputs[_output_coordinate(row["output"])] = _decode_quantity(row["value"])
    return {"normalization_scales": {"inputs": inputs, "outputs": outputs}}


def _block_map(
    rows: object,
    *,
    quantity_metadata: Sequence[Mapping[str, object]],
    output_layout: Mapping[str, object],
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
    parameter_refs: tuple[ParameterRef, ...],
) -> dict[str, Mapping[str, object]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("in-memory Sensitivity host blocks must be ordered")
    layout_blocks = output_layout["blocks"]
    if not isinstance(layout_blocks, Sequence) or isinstance(layout_blocks, (str, bytes)):
        raise TypeError("in-memory Sensitivity output blocks are malformed")
    by_key = {row["quantity_key"]: row for row in layout_blocks}
    if len(by_key) != len(layout_blocks):
        raise ValueError("in-memory Sensitivity output layout repeats a quantity block")
    expected_keys = tuple(row["key"] for row in quantity_metadata)
    result: dict[str, Mapping[str, object]] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("in-memory Sensitivity host block is malformed")
        key = row["quantity_key"]
        if key in result:
            raise ValueError("in-memory Sensitivity host block is duplicated")
        if ordinal >= len(expected_keys) or key != expected_keys[ordinal]:
            raise ValueError("in-memory Sensitivity host block order differs from declarations")
        layout = by_key[key]
        quantity = quantity_metadata[ordinal]
        if (
            row["observable_index"] != quantity["observable_index"]
            or layout["observable_index"] != quantity["observable_index"]
            or row["si_unit"] != quantity["si_unit"]
            or tuple(row["axes"]) != tuple(quantity["axes"])
        ):
            raise ValueError("in-memory Sensitivity host block differs from its declared output")
        block_parameters = _parameter_refs(row["parameter_keys"], parameter_lookup)
        if block_parameters != parameter_refs:
            raise ValueError("in-memory Sensitivity host parameter order differs from its request")
        result[key] = row
    if tuple(result) != expected_keys:
        raise ValueError("in-memory Sensitivity host blocks omit a requested quantity")
    return result


def _availability(row: Mapping[str, object]) -> tuple[bool, Mapping[str, object] | None]:
    available = row["available"]
    reason = row["reason"]
    if not isinstance(available, bool):
        raise TypeError("in-memory Sensitivity availability must be boolean")
    if available and reason is not None:
        raise TypeError("available in-memory Sensitivity block cannot carry an unavailable reason")
    if not available and not isinstance(reason, Mapping):
        raise TypeError("unavailable in-memory Sensitivity block must preserve its reason")
    return available, reason


def _public_array(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"in-memory Sensitivity {name} shape differs from its declared axes")
    return immutable_array(array)


def build_memory_sensitivity_result(
    *,
    payload: Mapping[str, object],
    request: Mapping[str, object],
    plan_sha256: str,
    request_sha256: str,
    execution_id: str,
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
    benchmark: InMemoryBenchmarkResult | None = None,
) -> InMemorySensitivityResult:
    """Bind the actual host-projected derivative blocks to public Results."""

    if not isinstance(payload, Mapping) or payload.get("persistence") != "memory":
        raise TypeError("in-memory Sensitivity payload has no memory persistence authority")
    if request.get("operation") != "sensitivity" or payload.get("operation") != "sensitivity":
        raise TypeError("in-memory Sensitivity requires its exact prepared operation")
    if payload["execution_id"] != execution_id:
        raise ValueError("in-memory Sensitivity payload belongs to another execution")
    _sha256(plan_sha256, name="plan_sha256")
    _sha256(request_sha256, name="request_sha256")
    UUID(execution_id)
    if not isinstance(parameter_lookup, Mapping):
        raise TypeError("in-memory Sensitivity requires the captured parameter lookup")
    spec = request["spec"]
    if not isinstance(spec, Mapping):
        raise TypeError("in-memory Sensitivity request Spec is malformed")
    derivative = spec["derivative"]
    if not isinstance(derivative, Mapping):
        raise TypeError("in-memory Sensitivity derivative declaration is malformed")
    derivative_kind = derivative["type"]
    if payload["derivative_kind"] != derivative_kind:
        raise ValueError("in-memory Sensitivity payload derivative differs from its request")
    quantity_metadata = payload["quantity_metadata"]
    observable_metadata = payload["observable_metadata"]
    output_layout = payload["output_layout"]
    if not isinstance(quantity_metadata, Sequence) or isinstance(quantity_metadata, (str, bytes)):
        raise TypeError("in-memory Sensitivity quantity metadata must be ordered")
    if not isinstance(observable_metadata, Sequence) or isinstance(observable_metadata, (str, bytes)):
        raise TypeError("in-memory Sensitivity observable metadata must be ordered")
    if not isinstance(output_layout, Mapping):
        raise TypeError("in-memory Sensitivity output layout is malformed")
    metadata = tuple(quantity_metadata)
    if not all(isinstance(row, Mapping) for row in metadata):
        raise TypeError("in-memory Sensitivity quantity metadata row is malformed")
    if tuple(row["key"] for row in metadata) != tuple(row["id"] for row in spec["quantities"]):
        raise ValueError("in-memory Sensitivity metadata order differs from its request")
    parameter_refs = _parameter_layout_refs(payload["parameter_layout"], parameter_lookup)
    requested_refs = _parameter_refs(spec["wrt"], parameter_lookup)
    if parameter_refs != requested_refs:
        raise ValueError("in-memory Sensitivity parameter order differs from its request")
    block_by_key = _block_map(
        payload["blocks"], quantity_metadata=metadata, output_layout=output_layout,
        parameter_lookup=parameter_lookup, parameter_refs=parameter_refs,
    )

    quantities = tuple(
        _verified_result(
            SensitivityQuantity,
            id=row["key"], label=row["key"], unit=row["si_unit"],
            axes=_axes(row["axes"]),
        )
        for row in metadata
    )
    primal_blocks: list[SensitivityValueBlock] = []
    for row in metadata:
        host = block_by_key[row["key"]]
        output_axes = _axes(row["axes"], omit_component=True)
        expected_shape = tuple(len(axis.coordinates) for axis in output_axes)
        value = _public_array(host["primal"], shape=expected_shape, name="primal")
        primal_blocks.append(_verified_result(
            SensitivityValueBlock,
            block_id=f"primal:{len(primal_blocks)}:{row['key']}",
            quantity_id=row["key"],
            status="complete",
            axes=output_axes,
            value=units.registry.Quantity(value, row["si_unit"]),
            failure=None,
        ))

    public_blocks: list[object] = []
    if derivative_kind in {"jacobian", "jvp"}:
        for row in metadata:
            host = block_by_key[row["key"]]
            available, reason = _availability(host)
            output_axes = _axes(row["axes"], omit_component=True)
            expected_shape = tuple(len(axis.coordinates) for axis in output_axes)
            coordinates = _quantity_output_coordinates({
                "label": row["key"], "axes": row["axes"],
            })
            if derivative_kind == "jacobian":
                values = None
                if available:
                    values = _public_array(
                        host["derivative"],
                        shape=(len(coordinates), len(parameter_refs)),
                        name="Jacobian block",
                    )
                public_blocks.append(_verified_result(
                    SensitivityJacobianBlock,
                    block_id=f"derivative:{row['key']}",
                    quantity_id=row["key"],
                    quantity_label=row["key"],
                    status="complete" if available else "unavailable",
                    output_axes=output_axes,
                    output_coordinates=coordinates,
                    row_units=(row["si_unit"],) * len(coordinates),
                    parameters=parameter_refs,
                    output_unit=row["si_unit"],
                    values=values,
                    failure=reason,
                ))
            else:
                value = None
                if available:
                    value = units.registry.Quantity(
                        _public_array(host["derivative"], shape=expected_shape,
                                      name="JVP block"), row["si_unit"],
                    )
                public_blocks.append(_verified_result(
                    SensitivityValueBlock,
                    block_id=f"derivative:{row['key']}",
                    quantity_id=row["key"],
                    status="complete" if available else "unavailable",
                    axes=output_axes,
                    value=value,
                    failure=reason,
                ))
    elif derivative_kind == "vjp":
        functional = payload["functional"]
        if not isinstance(functional, Mapping):
            raise TypeError("in-memory VJP payload lacks its aggregate functional projection")
        available, reason = _availability(functional)
        functional_unit = spec["functional_unit"]
        functional_parameters = _parameter_refs(functional["parameter_keys"], parameter_lookup)
        if functional_parameters != parameter_refs:
            raise ValueError("in-memory VJP parameter order differs from its request")
        values = None
        if available:
            values = _public_array(functional["derivative"], shape=(len(parameter_refs),),
                                   name="VJP functional")
        public_blocks.append(_verified_result(
            SensitivityParameterBlock,
            block_id="derivative:vjp",
            quantity_id=None,
            status="complete" if available else "unavailable",
            functional_unit=functional_unit,
            parameters=parameter_refs,
            units=tuple(str(units.registry.Unit(functional_unit) / units.registry.Unit(
                parameter.spec.si_unit
            )) for parameter in parameter_refs),
            values=values,
            failure=reason,
        ))
    else:
        raise ValueError(f"unsupported in-memory Sensitivity derivative kind: {derivative_kind}")

    derivative_statuses = tuple(block.status for block in public_blocks)
    projection_status = (
        "complete" if all(status == "complete" for status in derivative_statuses)
        else "unavailable" if all(status == "unavailable" for status in derivative_statuses)
        else "partial"
    )
    decoder = object.__new__(VerifiedResultDecoder)
    object.__setattr__(decoder, "_parameter_lookup", parameter_lookup)
    # Request diagnostics use the same typed quantity-layout rows as saved
    # Results; memory quantity_metadata instead carries compiler-local keys.
    quantity_by_id = {
        row["id"]: row for row in payload["manifest"]["quantity_layout"]
    }
    diagnostics = _request_derivative_diagnostics(
        decoder, derivative_kind, derivative, quantity_by_id,
    )
    projection = _verified_result(
        SensitivityProjection,
        kind=derivative_kind,
        status=projection_status,
        blocks=tuple(public_blocks),
        diagnostics=diagnostics,
    )
    aux = payload["aux"]
    certificates = tuple(aux.certificates)
    outcome = payload["outcome"]
    if outcome not in {"success", "partial"}:
        raise TypeError("in-memory Sensitivity outcome is malformed")
    presentation = {
        "spec": spec,
        "view": request.get("view"),
        **_normalization_scales(spec, parameter_lookup),
    }
    result = _verified_result(
        InMemorySensitivityResult,
        plan_sha256=plan_sha256,
        request_sha256=request_sha256,
        execution_id=execution_id,
        persistence="memory",
        operation="sensitivity",
        status="complete" if outcome == "success" else "partial",
        quantities=quantities,
        primal=tuple(primal_blocks),
        operations={derivative_kind: projection},
        parameters=_parameter_set(payload["parameters"], parameter_lookup),
        quantity_metadata=metadata,
        observable_metadata=tuple(observable_metadata),
        output_layout=output_layout,
        certificates=certificates,
        summary=payload["summary"],
        _presentation=presentation,
        _benchmark=benchmark,
    )
    object.__setattr__(result, "_memory_report_token", _MEMORY_RESULT_TOKEN)
    return result


def attach_memory_sensitivity_benchmark(
    result: InMemorySensitivityResult,
    benchmark: InMemoryBenchmarkResult,
) -> InMemorySensitivityResult:
    if type(result) is not InMemorySensitivityResult:
        raise TypeError("benchmark can be attached only to an in-memory Sensitivity Result")
    if not isinstance(benchmark, InMemoryBenchmarkResult):
        raise TypeError("in-memory Sensitivity requires its operation benchmark")
    if object.__getattribute__(result, "_benchmark") is not None:
        raise RuntimeError("in-memory Sensitivity benchmark was already finalized")
    object.__setattr__(result, "_benchmark", benchmark)
    return result


__all__ = [
    "InMemorySensitivityResult",
    "attach_memory_sensitivity_benchmark",
    "build_memory_sensitivity_result",
]
