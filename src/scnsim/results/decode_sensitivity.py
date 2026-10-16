"""Decode fixed Sensitivity blocks into detached, unit-aware Results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

from .. import units
from ..canonical import (
    complex_quantity_from_envelope,
    float64_from_hex,
    quantity_from_envelope,
)
from ..errors import EvidenceIntegrityError
from ..numeric_encoding import array_from_record
from ..specs import OutputCoordinate
from .factory import _verified_result
from .sensitivity import (
    SensitivityAxis,
    SensitivityAxisCoordinate,
    SensitivityBlockSequence,
    SensitivityJacobianBlock,
    SensitivityParameterBlock,
    SensitivityProjection,
    SensitivityQuantity,
    SensitivityResult,
    SensitivityValueBlock,
)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceIntegrityError(
            f"Sensitivity {field} is malformed", stage="result_decode"
        )
    return value


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvidenceIntegrityError(
            f"Sensitivity {field} is malformed", stage="result_decode"
        )
    return value


def _axes(value: object, *, omit_component: bool = False) -> tuple[SensitivityAxis, ...]:
    axes = []
    for raw_axis in _sequence(value, "axes"):
        axis = _mapping(raw_axis, "axis")
        identifier, label = axis.get("id"), axis.get("label")
        if not isinstance(identifier, str) or not isinstance(label, str):
            raise EvidenceIntegrityError(
                "Sensitivity axis identity is malformed", stage="result_decode"
            )
        if omit_component and identifier == "component":
            continue
        coordinates = []
        for raw_coordinate in _sequence(axis.get("coordinates"), "axis coordinates"):
            coordinate = _mapping(raw_coordinate, "axis coordinate")
            coordinate_id, coordinate_label = coordinate.get("id"), coordinate.get("label")
            if not isinstance(coordinate_id, str) or not isinstance(coordinate_label, str):
                raise EvidenceIntegrityError(
                    "Sensitivity coordinate identity is malformed", stage="result_decode"
                )
            component = coordinate.get("component")
            if component not in (None, "real", "imag"):
                raise EvidenceIntegrityError(
                    "Sensitivity coordinate component is malformed", stage="result_decode"
                )
            coordinates.append(_verified_result(
                SensitivityAxisCoordinate,
                id=coordinate_id,
                label=coordinate_label,
                value=coordinate.get("value"),
                unit=coordinate.get("unit"),
                component=component,
                metadata={key: item for key, item in coordinate.items()
                          if key not in {"id", "label", "value", "unit", "component"}},
            ))
        unit = axis.get("unit")
        if unit is not None and not isinstance(unit, str):
            raise EvidenceIntegrityError(
                "Sensitivity axis unit is malformed", stage="result_decode"
            )
        axes.append(_verified_result(
            SensitivityAxis,
            id=identifier,
            label=label,
            coordinates=tuple(coordinates),
            unit=unit,
        ))
    return tuple(axes)


def _decode_quantity(value: object):
    record = _mapping(value, "physical quantity")
    unit = record.get("si_unit")
    dimensionality = record.get("dimensionality")
    if not isinstance(unit, str) or not isinstance(dimensionality, str):
        raise EvidenceIntegrityError(
            "Sensitivity physical quantity unit is malformed", stage="result_decode"
        )
    # First use the closed canonical vocabulary. Sensitivity request values
    # also admit derived Pint units, so decode that exact scalar envelope when
    # its unit is outside the primitive set.
    try:
        if record.get("type") == "complex_quantity_f64":
            return complex_quantity_from_envelope(record, registry=units.registry)
        return quantity_from_envelope(record, registry=units.registry)
    except EvidenceIntegrityError as canonical_error:
        kind = record.get("type")
        fields = (
            {"type", "real_si_f64", "imag_si_f64", "si_unit", "dimensionality"}
            if kind == "complex_quantity_f64"
            else {"type", "si_value_f64", "si_unit", "dimensionality"}
        )
        if set(record) != fields or kind not in {"quantity_f64", "complex_quantity_f64"}:
            raise
        try:
            parsed_unit = units.registry.Unit(unit)
            if str(parsed_unit.dimensionality) != dimensionality:
                raise EvidenceIntegrityError(
                    "Sensitivity derived quantity dimensionality differs from its unit",
                    stage="result_decode",
                    evidence={"si_unit": unit, "dimensionality": dimensionality},
                )
            if kind == "complex_quantity_f64":
                magnitude = complex(
                    float64_from_hex(record["real_si_f64"]),
                    float64_from_hex(record["imag_si_f64"]),
                )
            else:
                magnitude = float64_from_hex(record["si_value_f64"])
            return units.registry.Quantity(magnitude, parsed_unit)
        except EvidenceIntegrityError:
            raise
        except Exception as error:
            raise EvidenceIntegrityError(
                "Sensitivity derived quantity is malformed",
                stage="result_decode",
                evidence={"si_unit": unit},
            ) from error


def _request_derivative_diagnostics(
    decoder,
    derivative_kind: str,
    derivative: Mapping[str, object],
    quantity_by_id: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    """Retain the exact typed direction or cotangent already sealed in request."""

    if derivative_kind == "jvp":
        direction = {}
        for raw_row in _sequence(derivative.get("direction"), "JVP direction"):
            row = _mapping(raw_row, "JVP direction row")
            parameter = decoder._decode_parameter_ref(row.get("parameter"))
            direction[parameter] = _decode_quantity(row.get("value"))
        return {"direction": direction}
    if derivative_kind == "vjp":
        weights = {}
        for raw_row in _sequence(derivative.get("weights"), "VJP weights"):
            row = _mapping(raw_row, "VJP weight row")
            coordinate = _output_coordinate(row.get("output"))
            quantity = quantity_by_id.get(coordinate.quantity_label)
            if quantity is None or coordinate not in _quantity_output_coordinates(quantity):
                raise EvidenceIntegrityError(
                    "Sensitivity VJP weight is outside the requested output layout",
                    stage="result_decode",
                    evidence={"quantity_id": coordinate.quantity_label},
                )
            weights[coordinate] = _decode_quantity(row.get("value"))
        return {"weights": weights}
    return {}


def _output_coordinate(record: object) -> OutputCoordinate:
    output = _mapping(record, "output coordinate")
    label = output.get("quantity_label")
    component = output.get("component")
    index = output.get("index")
    if (
        not isinstance(label, str)
        or component not in {"real", "imag"}
        or not isinstance(index, Sequence)
        or isinstance(index, (str, bytes))
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in index)
    ):
        raise EvidenceIntegrityError(
            "Sensitivity output coordinate is malformed", stage="result_decode"
        )
    return OutputCoordinate(label, component, tuple(index))


def _quantity_output_coordinates(quantity: Mapping[str, object]) -> tuple[OutputCoordinate, ...]:
    axes = _axes(quantity.get("axes"))
    component_axis = next((axis for axis in axes if axis.id == "component"), None)
    physical_axes = tuple(axis for axis in axes if axis.id != "component")
    physical_coordinates = tuple(product(*(range(len(axis.coordinates)) for axis in physical_axes)))
    components = (
        tuple(coordinate.component or coordinate.id
              for coordinate in component_axis.coordinates)
        if component_axis is not None else ("real",)
    )
    if any(component not in {"real", "imag"} for component in components):
        raise EvidenceIntegrityError(
            "Sensitivity component axis must use real/imag coordinates",
            stage="result_decode",
        )
    label = quantity.get("label")
    if not isinstance(label, str):
        raise EvidenceIntegrityError(
            "Sensitivity quantity label is malformed", stage="result_decode"
        )
    return tuple(
        OutputCoordinate(label, component, tuple(index))
        for index in physical_coordinates
        for component in components
    )


def _block_status(record: Mapping[str, object], *, kind: str) -> str:
    field = "primal_status" if kind == "primal" else "derivative_status"
    status = record.get(field)
    if status not in {"complete", "unavailable"}:
        raise EvidenceIntegrityError(
            "Sensitivity block status is malformed",
            stage="result_decode",
            evidence={"block_id": record.get("block_id"), "field": field},
        )
    return status


def _payload_values(record: Mapping[str, object], *, status: str):
    payload = _mapping(record.get("payload"), "block payload")
    if "values" not in payload:
        raise EvidenceIntegrityError(
            "Sensitivity block payload lacks its values member",
            stage="result_decode",
            evidence={"block_id": record.get("block_id")},
        )
    encoded = payload["values"]
    if status == "unavailable":
        if encoded is not None or not isinstance(payload.get("reason"), Mapping):
            raise EvidenceIntegrityError(
                "Unavailable Sensitivity block must preserve its reason without values",
                stage="result_decode",
                evidence={"block_id": record.get("block_id")},
            )
        return None, payload["reason"]
    if not isinstance(encoded, Mapping):
        raise EvidenceIntegrityError(
            "Complete Sensitivity block has no encoded values",
            stage="result_decode",
            evidence={"block_id": record.get("block_id")},
        )
    return array_from_record(encoded), None


def _slice_bounds(value: object, field: str) -> tuple[int, int] | None:
    if value is None:
        return None
    record = _mapping(value, field)
    start, stop = record.get("start"), record.get("stop")
    if (
        not isinstance(start, int) or isinstance(start, bool) or start < 0
        or not isinstance(stop, int) or isinstance(stop, bool) or stop < start
    ):
        raise EvidenceIntegrityError(
            f"Sensitivity {field} bounds are malformed", stage="result_decode"
        )
    return start, stop


def _selected_output_rows(
    quantity: Mapping[str, object],
    bounds: tuple[int, int],
    *,
    block_id: object,
) -> tuple[OutputCoordinate, ...]:
    """Select saved output coordinates without splitting complex value pairs."""

    rows = _quantity_output_coordinates(quantity)
    start, stop = bounds
    if stop > len(rows):
        raise EvidenceIntegrityError(
            "Sensitivity output slice exceeds its declared quantity layout",
            stage="result_decode",
            evidence={"block_id": block_id, "output_slice": bounds},
        )
    selected = rows[start:stop]
    expected: dict[tuple[int, ...], set[str]] = {}
    actual: dict[tuple[int, ...], set[str]] = {}
    for coordinate in rows:
        expected.setdefault(coordinate.index, set()).add(coordinate.component)
    for coordinate in selected:
        actual.setdefault(coordinate.index, set()).add(coordinate.component)
    if any(
        len(components) == 2 and actual.get(index, set()) != components
        for index, components in expected.items()
    ):
        raise EvidenceIntegrityError(
            "Sensitivity value slice splits a complex real/imaginary coordinate pair",
            stage="result_decode",
            evidence={"block_id": block_id, "output_slice": bounds},
        )
    return selected


def _validate_payload_shape(
    values: object | None,
    axes: Sequence[SensitivityAxis],
    *,
    expected: tuple[int, ...] | None = None,
    block_id: object,
) -> None:
    if values is None:
        return
    actual = getattr(values, "shape", None)
    expected_shape = (
        tuple(len(axis.coordinates) for axis in axes)
        if expected is None else expected
    )
    if actual is None or tuple(actual) != expected_shape:
        raise EvidenceIntegrityError(
            "Sensitivity block array shape differs from its selected coordinate axes",
            stage="result_decode",
            evidence={"block_id": block_id, "expected_shape": expected_shape,
                      "actual_shape": None if actual is None else tuple(actual)},
        )


def _parameter_refs(decoder, records: object) -> tuple[object, ...]:
    refs = []
    for record in _sequence(records, "parameter metadata"):
        selected = _mapping(record, "parameter metadata")
        reference = {
            "definitions_id": selected.get("definitions_id"),
            "parameter_id": selected.get("parameter_id"),
        }
        refs.append(decoder._decode_parameter_ref(reference))
    return tuple(refs)


def _decode_block(
    decoder,
    record: Mapping[str, object],
    index_row: Mapping[str, object],
    *,
    quantity_by_id: Mapping[str, Mapping[str, object]],
    kind: str,
    derivative_layout: Mapping[str, object],
    all_primal_ids: Sequence[str],
    rows_by_id: Mapping[str, Mapping[str, object]],
):
    block_id = record.get("block_id")
    quantity_id = index_row.get("quantity_id")
    summary = _mapping(index_row.get("summary_json"), "block summary")
    status = _block_status(record, kind=kind)
    values, reason = _payload_values(record, status=status)
    # The component axis expands real-coordinate Jacobian rows. Primal and
    # JVP arrays retain their physical complex shape, so it is metadata only.
    block_axes = _axes(summary.get("axes", ()), omit_component=True)
    if kind == "primal":
        if not isinstance(quantity_id, str) or quantity_id not in quantity_by_id:
            raise EvidenceIntegrityError(
                "Primal Sensitivity block has no declared quantity",
                stage="result_decode",
                evidence={"block_id": block_id, "quantity_id": quantity_id},
            )
        unit = summary.get("value_unit")
        if (
            not isinstance(unit, str)
            or unit != quantity_by_id[quantity_id].get("unit")
        ):
            raise EvidenceIntegrityError(
                "Primal Sensitivity block unit differs from its declared quantity",
                stage="result_decode",
                evidence={"block_id": block_id, "unit": unit},
            )
        output_slice = _slice_bounds(index_row.get("output_slice"), "output slice")
        if output_slice is None:
            raise EvidenceIntegrityError(
                "Primal Sensitivity block has no output slice", stage="result_decode"
            )
        _selected_output_rows(
            quantity_by_id[quantity_id], output_slice, block_id=block_id
        )
        _validate_payload_shape(values, block_axes, block_id=block_id)
        if index_row.get("parameter_slice") is not None:
            raise EvidenceIntegrityError(
                "Primal Sensitivity block has a parameter slice",
                stage="result_decode",
                evidence={"block_id": block_id},
            )
        value = None if values is None else units.registry.Quantity(values, unit)
        return _verified_result(
            SensitivityValueBlock,
            block_id=block_id,
            quantity_id=quantity_id,
            status=status,
            axes=block_axes,
            value=value,
            failure=reason,
        )

    derivative_kind = derivative_layout.get("kind")
    if derivative_kind not in {"jacobian", "jvp", "vjp"}:
        raise EvidenceIntegrityError(
            "Sensitivity derivative kind is malformed", stage="result_decode"
        )
    parameters = _parameter_refs(decoder, summary.get("parameters", ()))
    full_parameters = _parameter_refs(decoder, derivative_layout.get("parameters", ()))
    parameter_slice = _slice_bounds(index_row.get("parameter_slice"), "parameter slice")
    links = index_row.get("primal_blocks")
    if not isinstance(links, Sequence) or isinstance(links, (str, bytes)):
        raise EvidenceIntegrityError(
            "Sensitivity derivative primal links are malformed", stage="result_decode"
        )
    linked_ids = tuple(
        _mapping(link, "primal block link").get("block_id") for link in links
    )
    if any(not isinstance(item, str) for item in linked_ids):
        raise EvidenceIntegrityError(
            "Sensitivity derivative primal link identity is malformed", stage="result_decode"
        )
    if derivative_kind == "vjp":
        if index_row.get("output_slice") is not None:
            raise EvidenceIntegrityError(
                "VJP block must be aggregate across output coordinates",
                stage="result_decode",
                evidence={"block_id": block_id},
            )
        if tuple(linked_ids) != tuple(all_primal_ids):
            raise EvidenceIntegrityError(
                "VJP block does not bind all requested primal blocks in order",
                stage="result_decode",
                evidence={"block_id": block_id},
            )
        if parameter_slice is None or parameter_slice[1] > len(full_parameters):
            raise EvidenceIntegrityError(
                "VJP parameter slice exceeds its declared input layout",
                stage="result_decode",
                evidence={"block_id": block_id, "parameter_slice": parameter_slice},
            )
        if parameters != full_parameters[parameter_slice[0]:parameter_slice[1]]:
            raise EvidenceIntegrityError(
                "VJP block parameters differ from its selected input slice",
                stage="result_decode",
                evidence={"block_id": block_id},
            )
    else:
        # JVP blocks have no input-column axis. Jacobian column slices are
        # required and validated against the declared parameter layout below.
        if derivative_kind == "jvp" and parameter_slice is not None:
            raise EvidenceIntegrityError(
                "jvp block must not expose a parameter-axis slice",
                stage="result_decode",
                evidence={"block_id": block_id},
            )
        linked_quantities = {
            rows_by_id[linked_id].get("quantity_id")
            for linked_id in linked_ids
            if linked_id in rows_by_id and rows_by_id[linked_id].get("kind") == "primal"
        }
        if (
            not linked_ids
            or len(linked_quantities) != 1
            or linked_quantities != {quantity_id}
        ):
            raise EvidenceIntegrityError(
                f"{derivative_kind} block does not bind its requested primal quantity",
                stage="result_decode",
                evidence={"block_id": block_id, "quantity_id": quantity_id},
            )
    if derivative_kind == "jacobian":
        if not isinstance(quantity_id, str) or quantity_id not in quantity_by_id:
            raise EvidenceIntegrityError(
                "Jacobian block has no declared quantity",
                stage="result_decode",
                evidence={"block_id": block_id, "quantity_id": quantity_id},
            )
        declared = quantity_by_id[quantity_id]
        full_rows = _quantity_output_coordinates(declared)
        output_slice = _slice_bounds(index_row.get("output_slice"), "output slice")
        if output_slice is None or parameter_slice is None:
            raise EvidenceIntegrityError(
                "Jacobian block requires output and parameter slices",
                stage="result_decode",
                evidence={"block_id": block_id},
            )
        output_start, output_stop = output_slice
        parameter_start, parameter_stop = parameter_slice
        if output_stop > len(full_rows):
            raise EvidenceIntegrityError(
                "Jacobian output slice exceeds its declared quantity layout",
                stage="result_decode",
                evidence={"block_id": block_id, "output_slice": output_slice},
            )
        if parameter_stop > len(full_parameters):
            raise EvidenceIntegrityError(
                "Jacobian parameter slice exceeds its declared input layout",
                stage="result_decode",
                evidence={"block_id": block_id, "parameter_slice": parameter_slice},
            )
        selected_rows = full_rows[output_start:output_stop]
        selected_parameters = full_parameters[parameter_start:parameter_stop]
        if parameters != selected_parameters:
            raise EvidenceIntegrityError(
                "Jacobian parameters differ from its selected input slice",
                stage="result_decode",
                evidence={"block_id": block_id},
            )
        _validate_payload_shape(
            values,
            block_axes,
            expected=(len(selected_rows), len(selected_parameters)),
            block_id=block_id,
        )
        row_units = (declared["unit"],) * len(selected_rows)
        value_unit = declared.get("unit")
        if (
            not isinstance(value_unit, str)
            or summary.get("value_unit") != value_unit
        ):
            raise EvidenceIntegrityError(
                "Jacobian block unit differs from its declared quantity",
                stage="result_decode",
                evidence={"block_id": block_id, "unit": summary.get("value_unit")},
            )
        return _verified_result(
            SensitivityJacobianBlock,
            block_id=block_id,
            quantity_id=quantity_id,
            quantity_label=declared["label"],
            status=status,
            output_axes=block_axes,
            output_coordinates=selected_rows,
            row_units=row_units,
            parameters=selected_parameters,
            output_unit=value_unit,
            values=values,
            failure=reason,
        )
    if derivative_kind == "jvp":
        if not isinstance(quantity_id, str) or quantity_id not in quantity_by_id:
            raise EvidenceIntegrityError(
                "JVP block has no declared quantity",
                stage="result_decode",
                evidence={"block_id": block_id, "quantity_id": quantity_id},
            )
        unit = summary.get("value_unit")
        if (
            not isinstance(unit, str)
            or unit != quantity_by_id[quantity_id].get("unit")
        ):
            raise EvidenceIntegrityError(
                "JVP block unit differs from its declared quantity",
                stage="result_decode",
                evidence={"block_id": block_id, "unit": unit},
            )
        output_slice = _slice_bounds(index_row.get("output_slice"), "output slice")
        if output_slice is None:
            raise EvidenceIntegrityError(
                "JVP block has no output slice", stage="result_decode",
                evidence={"block_id": block_id},
            )
        _selected_output_rows(
            quantity_by_id[quantity_id], output_slice, block_id=block_id
        )
        _validate_payload_shape(values, block_axes, block_id=block_id)
        value = None if values is None else units.registry.Quantity(values, unit)
        return _verified_result(
            SensitivityValueBlock,
            block_id=block_id,
            quantity_id=quantity_id,
            status=status,
            axes=block_axes,
            value=value,
            failure=reason,
        )

    if quantity_id is not None:
        raise EvidenceIntegrityError(
            "VJP block must be an aggregate across requested outputs",
            stage="result_decode",
            evidence={"block_id": block_id, "quantity_id": quantity_id},
        )
    functional_unit = derivative_layout.get("functional_unit")
    if not isinstance(functional_unit, str):
        raise EvidenceIntegrityError(
            "VJP layout has no functional unit", stage="result_decode"
        )
    units_by_parameter = tuple(
        str(units.registry.Unit(functional_unit) / units.registry.Unit(parameter.spec.si_unit))
        for parameter in parameters
    )
    _validate_payload_shape(
        values, (), expected=(len(parameters),), block_id=block_id
    )
    return _verified_result(
        SensitivityParameterBlock,
        block_id=block_id,
        status=status,
        functional_unit=functional_unit,
        parameters=parameters,
        units=units_by_parameter,
        values=values,
        failure=reason,
        quantity_id=None,
    )


def decode_sensitivity_operation(
    decoder,
    *,
    identity,
    request: Mapping[str, object],
    projection: Mapping[str, object],
    fixed_reader: object,
):
    """Decode compact Sensitivity indexes and selected value blocks lazily."""

    manifest = _mapping(projection.get("manifest"), "manifest")
    terminal = _mapping(projection.get("terminal"), "terminal")
    if terminal.get("outcome") not in {"success", "partial"}:
        raise EvidenceIntegrityError(
            "Sensitivity terminal outcome is unsupported", stage="result_decode"
        )
    spec = _mapping(request.get("spec"), "request specification")
    derivative_layout = _mapping(manifest.get("derivative_layout"), "derivative layout")
    quantity_layout = _sequence(manifest.get("quantity_layout"), "quantity layout")
    request_quantities = _sequence(spec.get("quantities"), "request quantities")
    request_quantity_ids = []
    for raw_quantity in request_quantities:
        declared = _mapping(raw_quantity, "request quantity")
        identifier = declared.get("id")
        if not isinstance(identifier, str):
            raise EvidenceIntegrityError(
                "Sensitivity request quantity identity is malformed",
                stage="result_decode",
            )
        request_quantity_ids.append(identifier)
    quantities = []
    quantity_by_id = {}
    for raw in quantity_layout:
        item = _mapping(raw, "quantity layout entry")
        identifier, label, unit = item.get("id"), item.get("label"), item.get("unit")
        if (
            not isinstance(identifier, str)
            or not isinstance(label, str)
            or label != identifier
            or not isinstance(unit, str)
        ):
            raise EvidenceIntegrityError(
                "Sensitivity quantity identity or unit is malformed", stage="result_decode"
            )
        if identifier in quantity_by_id:
            raise EvidenceIntegrityError(
                "Sensitivity quantity identity is duplicated", stage="result_decode"
            )
        quantity_by_id[identifier] = item
        quantities.append(_verified_result(
            SensitivityQuantity,
            id=identifier,
            label=label,
            unit=unit,
            axes=_axes(item.get("axes", ())),
        ))
    if tuple(quantity_by_id) != tuple(request_quantity_ids):
        raise EvidenceIntegrityError(
            "Sensitivity quantity layout differs from request declaration order",
            stage="result_decode",
            evidence={"request_quantity_ids": tuple(request_quantity_ids)},
        )

    request_derivative = _mapping(spec.get("derivative"), "requested derivative")
    request_kind = request_derivative.get("type")
    derivative_kind = derivative_layout.get("kind")
    if request_kind != derivative_kind:
        raise EvidenceIntegrityError(
            "Sensitivity derivative layout differs from the requested operation",
            stage="result_decode",
            evidence={"requested_kind": request_kind, "saved_kind": derivative_kind},
        )
    request_parameters = _parameter_refs(decoder, spec.get("wrt"))
    layout_parameters = _parameter_refs(
        decoder, derivative_layout.get("parameters", ())
    )
    if request_parameters != layout_parameters:
        raise EvidenceIntegrityError(
            "Sensitivity derivative parameter layout differs from the request",
            stage="result_decode",
        )
    if derivative_kind == "vjp" and derivative_layout.get("functional_unit") != spec.get("functional_unit"):
        raise EvidenceIntegrityError(
            "Sensitivity VJP functional unit differs from the request",
            stage="result_decode",
        )

    rows = fixed_reader.project("blocks")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise EvidenceIntegrityError(
            "Sensitivity block metadata is malformed", stage="result_decode"
        )
    rows_by_id = {}
    primal_ids = []
    derivative_ids = []
    for row in rows:
        item = _mapping(row, "block index row")
        block_id, kind = item.get("block_id"), item.get("kind")
        if not isinstance(block_id, str) or block_id in rows_by_id or kind not in {"primal", "derivative"}:
            raise EvidenceIntegrityError(
                "Sensitivity block identity or kind is malformed", stage="result_decode"
            )
        rows_by_id[block_id] = item
        (primal_ids if kind == "primal" else derivative_ids).append(block_id)

    def decode_primal(record):
        row = rows_by_id.get(record.get("block_id"))
        if row is None or row.get("kind") != "primal":
            raise EvidenceIntegrityError(
                "Selected primal block differs from its sealed index", stage="result_decode"
            )
        return _decode_block(
            decoder, record, row, quantity_by_id=quantity_by_id, kind="primal",
            derivative_layout=derivative_layout, all_primal_ids=primal_ids,
            rows_by_id=rows_by_id,
        )

    primal = SensitivityBlockSequence(fixed_reader, primal_ids, decode_primal)
    if derivative_kind not in {"jacobian", "jvp", "vjp"} or not derivative_ids:
        raise EvidenceIntegrityError(
            "Sensitivity result lacks its requested derivative blocks", stage="result_decode"
        )

    derivative_rows = [rows_by_id[block_id] for block_id in derivative_ids]
    statuses = [row.get("derivative_status") for row in derivative_rows]
    if any(status not in {"complete", "unavailable"} for status in statuses):
        raise EvidenceIntegrityError(
            "Sensitivity derivative block status is malformed", stage="result_decode"
        )
    status = (
        "complete" if all(value == "complete" for value in statuses)
        else "unavailable" if all(value == "unavailable" for value in statuses)
        else "partial"
    )

    def decode_derivative(record):
        row = rows_by_id.get(record.get("block_id"))
        if row is None or row.get("kind") != "derivative":
            raise EvidenceIntegrityError(
                "Selected derivative block differs from its sealed index", stage="result_decode"
            )
        return _decode_block(
            decoder, record, row, quantity_by_id=quantity_by_id, kind="derivative",
            derivative_layout=derivative_layout, all_primal_ids=primal_ids,
            rows_by_id=rows_by_id,
        )

    derivative = _verified_result(
        SensitivityProjection,
        kind=derivative_kind,
        status=status,
        blocks=SensitivityBlockSequence(
            fixed_reader, derivative_ids, decode_derivative,
            quantity_blocks={identifier: tuple(block_id for block_id in derivative_ids
                                              if rows_by_id[block_id].get("quantity_id") == identifier)
                             for identifier in quantity_by_id},
        ),
        diagnostics=_request_derivative_diagnostics(
            decoder, derivative_kind, request_derivative, quantity_by_id
        ),
    )
    operations = {derivative_kind: derivative}

    scales_record = spec.get("scales")
    presentation = {}
    if scales_record is not None:
        scales = _mapping(scales_record, "normalization scales")
        inputs = {}
        for row in _sequence(scales.get("inputs"), "input normalization scales"):
            item = _mapping(row, "input normalization scale")
            parameter = decoder._decode_parameter_ref(item.get("parameter"))
            inputs[(parameter.definitions_id, parameter.id)] = _decode_quantity(item.get("value"))
        outputs = {}
        for row in _sequence(scales.get("outputs"), "output normalization scales"):
            item = _mapping(row, "output normalization scale")
            outputs[_output_coordinate(item.get("output"))] = _decode_quantity(item.get("value"))
        presentation["normalization_scales"] = {"inputs": inputs, "outputs": outputs}

    return _verified_result(
        SensitivityResult,
        identity=identity,
        status="complete" if terminal["outcome"] == "success" else "partial",
        quantities=tuple(quantities),
        primal=primal,
        operations=operations,
        _presentation=presentation,
    )


__all__ = ["decode_sensitivity_operation"]
