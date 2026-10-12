"""Plot already-saved Sensitivity blocks without requesting new derivatives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any, Literal

import numpy as np
from pint import Quantity

from ... import units
from ...errors import SCNSimStateError
from ...results.sensitivity import (
    SensitivityJacobianBlock,
    SensitivityParameterBlock,
    SensitivityValueBlock,
    _require_angular_unit_convention,
)
from ...specs import OutputCoordinate
from ..presentation import Theme
from .common import _display_axis, _plotly, _presentation_source, _style


_STYLE = Literal["auto", "table", "bar", "heatmap"]
_ParameterKey = tuple[str, str]


def _parameter_key(value: object) -> _ParameterKey:
    from ...authoring import ParameterRef

    if isinstance(value, ParameterRef):
        return value.definitions_id, value.id
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, str) for item in value)
    ):
        return value
    raise TypeError("parameter display keys must be ParameterRef or (definitions_id, parameter_id)")


def _keyed_parameter_options(
    options: Mapping[object, object] | None,
    *,
    parameters: Sequence[object],
    name: str,
    value_type: type,
) -> dict[_ParameterKey, object]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError(f"{name} must be a mapping")
    known = {(item.definitions_id, item.id) for item in parameters}
    selected: dict[_ParameterKey, object] = {}
    for raw_key, value in options.items():
        key = _parameter_key(raw_key)
        if key not in known:
            raise ValueError(f"{name} contains a parameter outside the selected block: {key!r}")
        if key in selected:
            raise ValueError(f"{name} contains the same parameter identity more than once")
        if not isinstance(value, value_type):
            raise TypeError(f"{name} values must be {value_type.__name__}")
        if value_type is Quantity and value._REGISTRY is not units.registry:
            raise TypeError(f"{name} quantities must use scnsim.units")
        selected[key] = value
    return selected


def _coordinate_key(coordinate: OutputCoordinate) -> tuple[str, str, tuple[int, ...]]:
    return coordinate.quantity_label, coordinate.component, coordinate.index


def _coordinate_record(coordinate: OutputCoordinate) -> dict[str, object]:
    return {
        "quantity_label": coordinate.quantity_label,
        "component": coordinate.component,
        "index": list(coordinate.index),
    }


def _convert_values(value: object, source_unit: object, target_unit: str) -> np.ndarray:
    return np.asarray(
        units.registry.Quantity(value, source_unit).to(target_unit).magnitude
    )


def _coordinate_label(axis: object, coordinate: object) -> str:
    """Show stored physical coordinates in display units, retaining labels otherwise."""

    value = coordinate.value
    unit = coordinate.unit or axis.unit
    if axis.id != "frequency" or value is None or unit is None:
        return coordinate.label
    quantity = units.registry.Quantity(
        float.fromhex(value), units.registry.Unit(unit)
    )
    displayed, display_unit = _display_axis(quantity)
    return f"{float(displayed):.9g} {display_unit}"


def _axis_records(axes: Sequence[object]) -> list[dict[str, object]]:
    """Keep exact saved coordinate identities alongside their display labels."""

    return [
        {
            "id": axis.id,
            "label": axis.label,
            "unit": axis.unit,
            "coordinates": [
                {
                    "id": coordinate.id,
                    "label": coordinate.label,
                    "value": coordinate.value,
                    "unit": coordinate.unit,
                    "component": coordinate.component,
                }
                for coordinate in axis.coordinates
            ],
        }
        for axis in axes
    ]


def _coordinate_labels(axes: Sequence[object]) -> list[str]:
    labels = [""]
    for axis in axes:
        labels = [
            f"{prefix} / {_coordinate_label(axis, coordinate)}"
            if prefix else _coordinate_label(axis, coordinate)
            for prefix in labels
            for coordinate in axis.coordinates
        ]
    return labels


def _output_quantity(result: Any, block: SensitivityJacobianBlock) -> Any:
    quantity = next(
        (item for item in result.quantities if item.id == block.quantity_id), None
    )
    if quantity is None:
        raise SCNSimStateError(
            "the selected Jacobian quantity is absent from its fixed result layout",
            stage="result_access",
            evidence={"quantity_id": block.quantity_id},
        )
    return quantity


def _output_coordinate_labels(
    quantity: Any, block: SensitivityJacobianBlock
) -> list[str]:
    axes = tuple(axis for axis in quantity.axes if axis.id != "component")
    labels = []
    for coordinate in block.output_coordinates:
        selected = []
        for axis, index in zip(axes, coordinate.index, strict=True):
            if index >= len(axis.coordinates):
                raise SCNSimStateError(
                    "the selected Jacobian coordinate is outside its fixed output layout",
                    stage="result_access",
                    evidence={"quantity_id": block.quantity_id, "index": coordinate.index},
                )
            selected.append(_coordinate_label(axis, axis.coordinates[index]))
        location = " / ".join(selected)
        labels.append(
            f"{coordinate.quantity_label}"
            + (f" / {location}" if location else "")
            + f" / {coordinate.component}"
        )
    return labels


def _failure_text(block: object) -> str:
    reason = getattr(block, "failure", None)
    if not isinstance(reason, Mapping):
        return ""
    kind = reason.get("kind")
    stage = reason.get("stage")
    message = reason.get("message")
    if message is None:
        message = reason.get("detail")
    location = " @ ".join(str(item) for item in (kind, stage) if item is not None)
    if message is None:
        return location
    return f"{location}: {message}" if location else str(message)


def _required_values(block: object) -> np.ndarray:
    values = getattr(block, "values", None)
    if values is None:
        value = getattr(block, "value", None)
        if value is not None:
            values = value.magnitude
    if values is None:
        raise SCNSimStateError(
            "the selected Sensitivity block has no saved values",
            stage="result_access",
            evidence={
                "quantity_id": getattr(block, "quantity_id", None),
                "status": getattr(block, "status", None),
            },
        )
    return np.asarray(values)


def _parameter_labels(
    parameters: Sequence[object],
    labels: Mapping[_ParameterKey, object],
    units_by_parameter: Sequence[str],
) -> list[str]:
    return [
        (
            f"{escape(str(labels.get((parameter.definitions_id, parameter.id), parameter.id)))}\n"
            f"{units_by_parameter[index]}"
            if units_by_parameter[index]
            else escape(str(labels.get((parameter.definitions_id, parameter.id), parameter.id)))
        )
        for index, parameter in enumerate(parameters)
    ]


def _requested_normalization(
    result: Any,
    *,
    parameters: Sequence[object],
    coordinates: Sequence[OutputCoordinate],
    input_scales: Mapping[object, object] | None,
    output_scales: Mapping[OutputCoordinate, object] | None,
) -> tuple[Mapping[_ParameterKey, Quantity], Mapping[OutputCoordinate, Quantity]]:
    if (input_scales is None) != (output_scales is None):
        raise ValueError("input_scales and output_scales must be supplied together")
    if input_scales is None:
        saved = result._presentation.get("normalization_scales")
        if not isinstance(saved, Mapping):
            raise SCNSimStateError(
                "normalized display requires explicit input and output scales",
                stage="result_access",
                evidence={"operation": "jacobian", "scales_available": False},
            )
        raw_inputs = saved.get("inputs")
        raw_outputs = saved.get("outputs")
        if not isinstance(raw_inputs, Mapping) or not isinstance(raw_outputs, Mapping):
            raise SCNSimStateError(
                "normalized display requires explicit input and output scales",
                stage="result_access",
                evidence={"operation": "jacobian", "scales_available": False},
            )
        selected_inputs = dict(raw_inputs)
        selected_outputs = dict(raw_outputs)
    else:
        if not isinstance(output_scales, Mapping):
            raise TypeError("output_scales must be a mapping")
        selected_inputs = _keyed_parameter_options(
            input_scales,
            parameters=parameters,
            name="input_scales",
            value_type=Quantity,
        )
        selected_outputs = dict(output_scales)
        if any(not isinstance(key, OutputCoordinate) for key in selected_outputs):
            raise TypeError("output_scales keys must be saved OutputCoordinate values")
        if any(
            not isinstance(value, Quantity) or value._REGISTRY is not units.registry
            for value in selected_outputs.values()
        ):
            raise TypeError("output_scales values must be scnsim.units quantities")
    active_parameters = {
        (parameter.definitions_id, parameter.id) for parameter in parameters
    }
    input_by_key = {
        _parameter_key(key): value for key, value in selected_inputs.items()
    }
    missing_inputs = active_parameters - set(input_by_key)
    if missing_inputs:
        raise ValueError(f"input_scales is missing selected parameters: {sorted(missing_inputs)!r}")
    coordinate_keys = {_coordinate_key(item) for item in coordinates}
    output_by_key = {
        _coordinate_key(key): (key, value)
        for key, value in selected_outputs.items()
        if isinstance(key, OutputCoordinate)
    }
    missing_outputs = coordinate_keys - set(output_by_key)
    if missing_outputs:
        raise ValueError("output_scales is missing selected output coordinates")
    return (
        {key: value for key, value in input_by_key.items()},
        {output_by_key[key][0]: output_by_key[key][1] for key in coordinate_keys},
    )


def _normalization_records(
    scales: tuple[Mapping[_ParameterKey, Quantity], Mapping[OutputCoordinate, Quantity]],
) -> dict[str, object]:
    inputs, outputs = scales
    return {
        "inputs": [
            {
                "definitions_id": key[0],
                "parameter_id": key[1],
                "value": str(value),
            }
            for key, value in inputs.items()
        ],
        "outputs": [
            {"coordinate": _coordinate_record(key), "value": str(value)}
            for key, value in outputs.items()
        ],
    }


def _block_identity_metadata(block: object, kind: str) -> dict[str, object]:
    if kind == "jacobian":
        return {
            "quantity_id": block.quantity_id,
            "output_coordinates": [
                _coordinate_record(item) for item in block.output_coordinates
            ],
            "parameter_identities": [
                {"definitions_id": item.definitions_id, "parameter_id": item.id}
                for item in block.parameters
            ],
        }
    if kind == "vjp":
        return {
            "quantity_id": block.quantity_id,
            "parameter_identities": [
                {"definitions_id": item.definitions_id, "parameter_id": item.id}
                for item in block.parameters
            ],
        }
    return {"quantity_id": block.quantity_id}


def _diagnostic_metadata(projection: object, kind: str) -> dict[str, object]:
    diagnostics = projection.diagnostics
    if kind == "jvp":
        direction = diagnostics.get("direction", {})
        return {
            "direction": [
                {
                    "definitions_id": parameter.definitions_id,
                    "parameter_id": parameter.id,
                    "value": str(value),
                }
                for parameter, value in direction.items()
            ]
        }
    if kind == "vjp":
        weights = diagnostics.get("weights", {})
        return {
            "weights": [
                {"output": _coordinate_record(coordinate), "value": str(value)}
                for coordinate, value in weights.items()
            ]
        }
    return {}


def _table_figure(
    go: Any,
    headers: Sequence[str],
    columns: Sequence[Sequence[object]],
    meta: Mapping[str, object],
) -> Any:
    return go.Figure(data=[go.Table(
        header={"values": list(headers), "align": "left"},
        cells={"values": [list(column) for column in columns], "align": "left"},
        meta={"scnsim": dict(meta)},
    )])


def _value_table(
    go: Any,
    *,
    block: object,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    values: np.ndarray | None,
    cell_units: Sequence[Sequence[str]],
    meta: Mapping[str, object],
) -> Any:
    if values is None:
        return _table_figure(
            go,
            ("block", "status", "failure"),
            ([getattr(block, "block_id")], [getattr(block, "status")], [_failure_text(block)]),
            meta,
        )
    if values.ndim == 1:
        labels = list(column_labels) or list(row_labels)
        flat = values.reshape(-1)
        units_flat = [unit for row in cell_units for unit in row]
        return _table_figure(
            go,
            ("coordinate", "value", "unit"),
            (labels, [f"{item}" for item in flat], units_flat),
            meta,
        )
    rows = values.reshape((len(row_labels), len(column_labels)))
    row_units = [list(row) for row in cell_units]
    columns: list[list[object]] = [[*row_labels]]
    headers = ["output coordinate"]
    for column, label in enumerate(column_labels):
        headers.append(label)
        columns.append([
            f"{rows[row, column]} {row_units[row][column]}"
            for row in range(rows.shape[0])
        ])
    return _table_figure(go, headers, columns, meta)


def sensitivity_plot(
    result: Any,
    *,
    kind: str,
    block_id: str,
    style: _STYLE = "auto",
    output_unit: str | None = None,
    output_units: Mapping[OutputCoordinate, str] | None = None,
    parameter_units: Mapping[object, str] | None = None,
    parameter_labels: Mapping[object, str] | None = None,
    normalized: bool = False,
    input_scales: Mapping[object, Quantity] | None = None,
    output_scales: Mapping[OutputCoordinate, Quantity] | None = None,
    theme: Theme = Theme.AUTO,
) -> Any:
    """Render exactly one selected saved block from the requested operation."""

    if kind not in {"jacobian", "jvp", "vjp"}:
        raise ValueError("kind must be 'jacobian', 'jvp', or 'vjp'")
    if style not in {"auto", "table", "bar", "heatmap"}:
        raise ValueError("style must be 'auto', 'table', 'bar', or 'heatmap'")
    if output_unit is not None and output_units is not None:
        raise ValueError("output_unit and output_units are mutually exclusive")
    if output_units is not None and not isinstance(output_units, Mapping):
        raise TypeError("output_units must be a mapping")
    if normalized and any(
        value is not None for value in (output_unit, output_units, parameter_units)
    ):
        raise ValueError("normalized display cannot also select physical display units")
    if not normalized and (input_scales is not None or output_scales is not None):
        raise ValueError("normalization scales require normalized=True")
    if kind != "jacobian" and normalized:
        raise ValueError("normalized display applies only to a Jacobian block")
    if kind == "vjp" and (output_unit is not None or output_units is not None):
        raise ValueError("output units apply only to Jacobian or JVP blocks")
    if kind == "jvp" and (output_units is not None or parameter_units is not None or parameter_labels is not None):
        raise ValueError("row and parameter display maps do not apply to a JVP block")
    if kind != "jacobian" and output_units is not None:
        raise ValueError("output_units apply only to a Jacobian block")

    go, _, _ = _plotly()
    operation = getattr(result, kind)
    block = operation.block(block_id)  # one fixed-reader body access
    source = _presentation_source(result)
    if kind == "jacobian" and not isinstance(block, SensitivityJacobianBlock):
        raise TypeError("Jacobian projection contains a non-Jacobian block")
    if kind == "jvp" and not isinstance(block, SensitivityValueBlock):
        raise TypeError("JVP projection contains a non-value block")
    if kind == "vjp" and not isinstance(block, SensitivityParameterBlock):
        raise TypeError("VJP projection contains a non-parameter block")
    if kind == "jacobian" and output_units is not None:
        unknown = set(output_units) - set(block.output_coordinates)
        if unknown:
            raise ValueError("output_units contains an output outside the selected block")
    normalized_scales = None
    if normalized:
        normalized_scales = _requested_normalization(
            result,
            parameters=block.parameters,
            coordinates=block.output_coordinates,
            input_scales=input_scales,
            output_scales=output_scales,
        )

    parameters = block.parameters if kind in {"jacobian", "vjp"} else ()
    parameter_units_by_key = _keyed_parameter_options(
        parameter_units,
        parameters=parameters,
        name="parameter_units",
        value_type=str,
    )
    parameter_labels_by_key = _keyed_parameter_options(
        parameter_labels,
        parameters=parameters,
        name="parameter_labels",
        value_type=str,
    )
    for selected_unit in parameter_units_by_key.values():
        units.registry.Unit(selected_unit)
    if output_unit is not None:
        units.registry.Unit(output_unit)
    if output_units is not None:
        if any(not isinstance(key, OutputCoordinate) for key in output_units):
            raise TypeError("output_units keys must be saved OutputCoordinate values")
        if any(not isinstance(value, str) for value in output_units.values()):
            raise TypeError("output_units values must be unit strings")
        for selected_unit in output_units.values():
            units.registry.Unit(selected_unit)
    full_parameter_labels = {
        (parameter.definitions_id, parameter.id): (
            f"{parameter_labels_by_key.get((parameter.definitions_id, parameter.id), parameter.id)}"
        )
        for parameter in parameters
    }

    failure = _failure_text(block)
    if block.status != "complete":
        if style not in {"auto", "table"}:
            raise SCNSimStateError(
                "numeric plotting is unavailable for this Sensitivity block",
                stage="result_access",
                evidence={"block_id": block_id, "status": block.status, "failure": failure},
            )
        figure = _table_figure(
            go,
            ("block", "operation", "status", "failure"),
            ([block_id], [kind], [block.status], [failure]),
            {
                "kind": f"sensitivity_{kind}",
                "block_id": block_id,
                "status": block.status,
                "failure": failure,
                "normalized": normalized,
                **_block_identity_metadata(block, kind),
                **_diagnostic_metadata(operation, kind),
                **(
                    {"normalization_scales": _normalization_records(normalized_scales)}
                    if normalized_scales is not None
                    else {}
                ),
                "source": source,
            },
        )
        return _style(figure, theme, title=f"Sensitivity {kind}: {block.status}")

    if kind == "jacobian":
        quantity = _output_quantity(result, block)
        values = _required_values(block).reshape((-1, len(block.parameters)))
        row_labels = _output_coordinate_labels(quantity, block)
        if normalized:
            values = block.normalized(
                input_scales=normalized_scales[0],
                output_scales=normalized_scales[1],
            ).reshape((-1, len(block.parameters)))
        row_units = []
        parameter_si_units = [parameter.spec.si_unit for parameter in block.parameters]
        parameter_display_units = [
            parameter_units_by_key.get(
                (parameter.definitions_id, parameter.id), parameter.spec.si_unit
            )
            for parameter in block.parameters
        ]
        if normalized:
            display_values = values
            row_units = ["dimensionless"] * len(block.output_coordinates)
            cell_units = [["dimensionless"] * len(block.parameters) for _ in row_labels]
            output_display_units = ["dimensionless"] * len(row_labels)
            parameter_display_units = ["normalized input"] * len(block.parameters)
        else:
            display_values = np.empty_like(values, dtype=np.result_type(values, np.float64))
            output_display_units = []
            cell_units = []
            for row, coordinate in enumerate(block.output_coordinates):
                source_output_unit = block.row_units[row] if block.row_units else block.output_unit
                target_output_unit = (
                    output_units.get(coordinate, source_output_unit)
                    if output_units is not None
                    else output_unit or source_output_unit
                )
                _require_angular_unit_convention(
                    source_output_unit,
                    target_output_unit,
                    context="Jacobian output display unit",
                )
                output_display_units.append(target_output_unit)
                row_cell_units = []
                for column, parameter in enumerate(block.parameters):
                    target_parameter_unit = parameter_display_units[column]
                    _require_angular_unit_convention(
                        parameter_si_units[column],
                        target_parameter_unit,
                        context="Jacobian parameter display unit",
                    )
                    source_ratio = (
                        units.registry.Unit(source_output_unit)
                        / units.registry.Unit(parameter_si_units[column])
                    )
                    target_ratio = (
                        units.registry.Unit(target_output_unit)
                        / units.registry.Unit(target_parameter_unit)
                    )
                    converted = units.registry.Quantity(values[row, column], source_ratio).to(target_ratio)
                    display_values[row, column] = converted.magnitude
                    row_cell_units.append(f"{target_ratio:~P}")
                cell_units.append(row_cell_units)
                row_units.append(target_output_unit)
        column_labels = _parameter_labels(
            block.parameters,
            parameter_labels_by_key,
            parameter_display_units,
        )
        entry_unit_labels = {
            "output_coordinates": [_coordinate_record(item) for item in block.output_coordinates],
            "output_units": output_display_units,
            "parameter_identities": [
                {"definitions_id": item.definitions_id, "parameter_id": item.id}
                for item in block.parameters
            ],
            "parameter_labels": [full_parameter_labels[(item.definitions_id, item.id)] for item in block.parameters],
            "parameter_units": parameter_display_units,
            "entry_units": cell_units,
        }
        if normalized_scales is not None:
            entry_unit_labels["normalization_scales"] = _normalization_records(
                normalized_scales
            )
        common_units = {unit for row in cell_units for unit in row}
        common_unit = len(common_units) == 1
        selected_style = style
        if selected_style == "auto":
            selected_style = (
                "bar" if display_values.shape[0] == 1 and common_unit
                else "heatmap" if common_unit
                else "table"
            )
        meta = {
            "kind": "sensitivity_jacobian",
            "block_id": block_id,
            "quantity_id": block.quantity_id,
            "status": block.status,
            "normalized": normalized,
            **entry_unit_labels,
            "source": source,
            **_diagnostic_metadata(operation, kind),
        }
        if selected_style == "table":
            figure = _value_table(
                go,
                block=block,
                row_labels=row_labels,
                column_labels=column_labels,
                values=display_values,
                cell_units=cell_units,
                meta=meta,
            )
        elif selected_style == "bar":
            if not common_unit:
                raise ValueError("bar display requires homogeneous Jacobian entry units")
            flat_values = display_values.reshape(-1)
            labels = [
                f"{row_labels[row]} / {column_labels[column]}"
                for row in range(display_values.shape[0])
                for column in range(display_values.shape[1])
            ]
            figure = go.Figure(data=[go.Bar(
                x=labels,
                y=np.real(flat_values),
                customdata=np.asarray([
                    [
                        (
                            f"{block.output_coordinates[row].quantity_label}/"
                            f"{block.output_coordinates[row].component}/"
                            f"{block.output_coordinates[row].index}"
                        ),
                        (
                            f"{block.parameters[column].definitions_id}."
                            f"{block.parameters[column].id}"
                        ),
                        cell_units[row][column],
                    ]
                    for row in range(display_values.shape[0])
                    for column in range(display_values.shape[1])
                ], dtype=object),
                hovertemplate=(
                    "output=%{customdata[0]}<br>parameter=%{customdata[1]}<br>"
                    "value=%{y}<br>unit=%{customdata[2]}<extra></extra>"
                ),
                meta={"scnsim": meta},
            )])
            if np.iscomplexobj(flat_values):
                figure.data[0].name = "real"
                figure.add_trace(go.Bar(x=labels, y=np.imag(flat_values), name="imag"))
            figure.update_layout(barmode="group")
            figure.update_xaxes(title_text="output / parameter coordinate")
            figure.update_yaxes(
                title_text=(
                    "normalized Jacobian"
                    if normalized
                    else f"Jacobian ({cell_units[0][0]})"
                )
            )
        else:
            if not common_unit:
                raise ValueError("heatmap display requires homogeneous Jacobian entry units")
            if np.iscomplexobj(display_values):
                raise ValueError("a complex Jacobian requires table or bar display")
            cell_customdata = np.asarray([
                [
                    cell_units[row][column],
                    f"{block.output_coordinates[row].quantity_label}/"
                    f"{block.output_coordinates[row].component}/"
                    f"{block.output_coordinates[row].index}",
                    f"{block.parameters[column].definitions_id}."
                    f"{block.parameters[column].id}",
                ]
                for row in range(display_values.shape[0])
                for column in range(display_values.shape[1])
            ], dtype=object).reshape((*display_values.shape, 3))
            figure = go.Figure(data=[go.Heatmap(
                z=display_values,
                x=column_labels,
                y=row_labels,
                colorbar={"title": "dimensionless" if normalized else cell_units[0][0]},
                customdata=cell_customdata,
                colorscale="RdBu",
                reversescale=True,
                zmid=0,
                hovertemplate=(
                    "output=%{customdata[1]}<br>parameter=%{customdata[2]}<br>"
                    "value=%{z}<br>unit=%{customdata[0]}<extra></extra>"
                ),
                meta={"scnsim": meta},
            )])
            figure.update_xaxes(title_text="parameter coordinates")
            figure.update_yaxes(title_text="saved output coordinates")
        title = f"Sensitivity Jacobian: {escape(block.quantity_id)}"
    elif kind == "jvp":
        quantity_value = block.value
        if quantity_value is None:
            _required_values(block)
        values = np.asarray(quantity_value.magnitude).reshape(-1)
        target_unit = output_unit or str(quantity_value.units)
        _require_angular_unit_convention(
            quantity_value.units,
            target_unit,
            context="JVP output display unit",
        )
        values = _convert_values(values, quantity_value.units, target_unit)
        labels = _coordinate_labels(block.axes)
        selected_style = "line" if style == "auto" else style
        meta = {
            "kind": "sensitivity_jvp",
            "block_id": block_id,
            "quantity_id": block.quantity_id,
            "status": block.status,
            "output_unit": target_unit,
            "axes": _axis_records(block.axes),
            "coordinate_labels": labels,
            "source": source,
            **_diagnostic_metadata(operation, kind),
        }
        unit_text = f"{units.registry.Unit(target_unit):~P}"
        if selected_style == "table":
            components = (["real", "imag"] if np.iscomplexobj(values) else ["real"])
            columns = [[block.quantity_id] * len(labels), labels]
            headers = ["output quantity", "coordinate"]
            for component in components:
                headers.append(component)
                columns.append([
                    str(item) for item in (np.real(values) if component == "real" else np.imag(values))
                ])
            columns.append([unit_text] * len(labels))
            headers.append("unit")
            figure = _table_figure(go, headers, columns, meta)
        elif selected_style == "bar":
            figure = go.Figure()
            figure.add_trace(go.Bar(
                x=labels,
                y=np.real(values),
                name="real" if np.iscomplexobj(values) else escape(block.quantity_id),
                customdata=np.asarray([
                    [block.quantity_id, "real" if np.iscomplexobj(values) else "value"]
                    for _ in labels
                ], dtype=object),
                hovertemplate=(
                    "quantity=%{customdata[0]}<br>coordinate=%{x}<br>"
                    "component=%{customdata[1]}<br>value=%{y}<br>"
                    f"unit={escape(unit_text)}<extra></extra>"
                ),
            ))
            if np.iscomplexobj(values):
                figure.add_trace(go.Bar(
                    x=labels,
                    y=np.imag(values),
                    name="imag",
                    customdata=np.asarray([[block.quantity_id, "imag"] for _ in labels], dtype=object),
                    hovertemplate=(
                        "quantity=%{customdata[0]}<br>coordinate=%{x}<br>"
                        "component=%{customdata[1]}<br>value=%{y}<br>"
                        f"unit={escape(unit_text)}<extra></extra>"
                    ),
                ))
            figure.update_layout(barmode="group")
            figure.update_xaxes(title_text="saved output coordinates")
            figure.update_yaxes(title_text=f"JVP ({unit_text})")
            figure.data[0].meta = {"scnsim": meta}
        elif selected_style == "heatmap":
            rows = np.vstack((np.real(values), np.imag(values))) if np.iscomplexobj(values) else np.real(values)[None, :]
            figure = go.Figure(data=[go.Heatmap(
                z=rows,
                x=labels,
                y=["real", "imag"] if np.iscomplexobj(values) else [escape(block.quantity_id)],
                colorbar={"title": unit_text},
                colorscale="RdBu",
                reversescale=True,
                zmid=0,
                customdata=np.asarray([
                    [block.quantity_id, component]
                    for component in (["real", "imag"] if np.iscomplexobj(values) else ["value"])
                    for _ in labels
                ], dtype=object).reshape((*rows.shape, 2)),
                hovertemplate=(
                    "quantity=%{customdata[0]}<br>coordinate=%{x}<br>"
                    "component=%{customdata[1]}<br>value=%{z}<br>"
                    f"unit={escape(unit_text)}<extra></extra>"
                ),
                meta={"scnsim": meta},
            )])
            figure.update_xaxes(title_text="saved output coordinates")
            figure.update_yaxes(title_text="component")
        else:
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=labels,
                y=np.real(values),
                mode="lines+markers",
                name="real" if np.iscomplexobj(values) else escape(block.quantity_id),
                customdata=np.asarray([
                    [block.quantity_id, "real" if np.iscomplexobj(values) else "value"]
                    for _ in labels
                ], dtype=object),
                meta={"scnsim": meta},
                hovertemplate=(
                    "quantity=%{customdata[0]}<br>coordinate=%{x}<br>"
                    "component=%{customdata[1]}<br>value=%{y}<extra></extra>"
                ),
            ))
            if np.iscomplexobj(values):
                figure.add_trace(go.Scatter(
                    x=labels,
                    y=np.imag(values),
                    mode="lines+markers",
                    name="imag",
                    customdata=np.asarray([
                        [block.quantity_id, "imag"] for _ in labels
                    ], dtype=object),
                    hovertemplate=(
                        "quantity=%{customdata[0]}<br>coordinate=%{x}<br>"
                        "component=%{customdata[1]}<br>value=%{y}<extra></extra>"
                    ),
                ))
            figure.update_xaxes(title_text="saved output coordinates")
            figure.update_yaxes(title_text=f"JVP ({unit_text})")
        title = f"Sensitivity JVP: {escape(block.quantity_id)}"
    else:
        values = _required_values(block).reshape(-1)
        derivative_source_units = list(block.units)
        parameter_display_units = [
            parameter_units_by_key.get(
                (parameter.definitions_id, parameter.id), parameter.spec.si_unit
            )
            for parameter in block.parameters
        ]
        for parameter, target_unit in zip(
            block.parameters, parameter_display_units, strict=True
        ):
            _require_angular_unit_convention(
                parameter.spec.si_unit,
                target_unit,
                context="VJP parameter display unit",
            )
        derivative_display_units = [
            str(units.registry.Unit(block.functional_unit) / units.registry.Unit(parameter_unit))
            for parameter_unit in parameter_display_units
        ]
        display_values = np.asarray([
            units.registry.Quantity(value, source_unit).to(target_unit).magnitude
            for value, source_unit, target_unit in zip(
                values, derivative_source_units, derivative_display_units, strict=True
            )
        ])
        labels = _parameter_labels(
            block.parameters,
            parameter_labels_by_key,
            parameter_display_units,
        )
        common_unit = bool(derivative_display_units) and all(
            unit == derivative_display_units[0] for unit in derivative_display_units
        )
        selected_style = ("bar" if common_unit else "table") if style == "auto" else style
        meta = {
            "kind": "sensitivity_vjp",
            "block_id": block_id,
            "quantity_id": block.quantity_id,
            "status": block.status,
            "functional_unit": block.functional_unit,
            "parameter_identities": [
                {"definitions_id": item.definitions_id, "parameter_id": item.id}
                for item in block.parameters
            ],
            "parameter_labels": [
                full_parameter_labels[(item.definitions_id, item.id)]
                for item in block.parameters
            ],
            "parameter_units": parameter_display_units,
            "derivative_units": derivative_display_units,
            "source": source,
            **_diagnostic_metadata(operation, kind),
        }
        if selected_style == "table":
            columns = [
                [f"{item.definitions_id}.{item.id}" for item in block.parameters],
                labels,
                [str(item) for item in display_values],
                derivative_display_units,
            ]
            figure = _table_figure(
                go,
                ("parameter identity", "label", "value", "derivative unit"),
                columns,
                meta,
            )
        elif selected_style == "bar":
            if not common_unit:
                raise ValueError("bar display requires homogeneous VJP parameter units")
            figure = go.Figure(data=[go.Bar(
                x=labels,
                y=np.real(display_values),
                meta={"scnsim": meta},
                customdata=np.asarray([
                    [
                        f"{parameter.definitions_id}.{parameter.id}",
                        derivative_display_units[index],
                    ]
                    for index, parameter in enumerate(block.parameters)
                ], dtype=object),
                hovertemplate=(
                    "parameter=%{customdata[0]}<br>value=%{y}<br>"
                    "unit=%{customdata[1]}<extra></extra>"
                ),
            )])
            if np.iscomplexobj(display_values):
                figure.data[0].name = "real"
                figure.add_trace(go.Bar(
                    x=labels,
                    y=np.imag(display_values),
                    name="imag",
                    customdata=np.asarray([
                        [
                            f"{parameter.definitions_id}.{parameter.id}",
                            derivative_display_units[index],
                        ]
                        for index, parameter in enumerate(block.parameters)
                    ], dtype=object),
                    hovertemplate=(
                        "parameter=%{customdata[0]}<br>value=%{y}<br>"
                        "unit=%{customdata[1]}<extra></extra>"
                    ),
                ))
            figure.update_layout(barmode="group")
            figure.update_xaxes(title_text="parameter coordinates")
            figure.update_yaxes(title_text=f"VJP ({derivative_display_units[0]})")
        else:
            if np.iscomplexobj(display_values):
                raise ValueError("a complex VJP requires table or bar display")
            if not common_unit:
                raise ValueError("heatmap display requires homogeneous VJP parameter units")
            figure = go.Figure(data=[go.Heatmap(
                z=np.real(display_values)[None, :],
                x=labels,
                y=["VJP"],
                colorbar={"title": derivative_display_units[0]},
                colorscale="RdBu",
                reversescale=True,
                zmid=0,
                customdata=np.asarray([[
                    [
                        f"{parameter.definitions_id}.{parameter.id}",
                        derivative_display_units[index],
                    ]
                    for index, parameter in enumerate(block.parameters)
                ]], dtype=object),
                hovertemplate=(
                    "parameter=%{customdata[0]}<br>value=%{z}<br>"
                    "unit=%{customdata[1]}<extra></extra>"
                ),
                meta={"scnsim": meta},
            )])
            figure.update_xaxes(title_text="parameter coordinates")
            figure.update_yaxes(title_text="VJP")
        title = f"Sensitivity VJP ({escape(block.functional_unit)})"

    figure.update_layout(meta={"scnsim": meta})
    return _style(figure, theme, title=title)
