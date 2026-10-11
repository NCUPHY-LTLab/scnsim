"""Plot already-saved Sensitivity blocks without requesting new derivatives."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any

import numpy as np

from ... import units
from ...errors import SCNSimStateError
from ...results.sensitivity import (
    SensitivityJacobianBlock,
    SensitivityParameterBlock,
    SensitivityValueBlock,
)
from ..presentation import Theme
from .common import _display_axis, _plotly, _presentation_source, _style


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
        coordinates = axis.coordinates
        labels = [
            f"{prefix} / {_coordinate_label(axis, coordinate)}"
            if prefix else _coordinate_label(axis, coordinate)
            for prefix in labels
            for coordinate in coordinates
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
        suffix = coordinate.component
        labels.append(
            f"{coordinate.quantity_label}"
            + (f" / {location}" if location else "")
            + f" / {suffix}"
        )
    return labels


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


def sensitivity_plot(
    result: Any,
    *,
    kind: str,
    block_id: str,
    theme: Theme = Theme.AUTO,
) -> Any:
    """Render exactly one selected saved block from the requested operation."""

    go, _, _ = _plotly()
    operation = getattr(result, kind)
    block = operation.block(block_id)
    source = _presentation_source(result)

    if kind == "jacobian":
        if not isinstance(block, SensitivityJacobianBlock):
            raise TypeError("Jacobian projection contains a non-Jacobian block")
        quantity = _output_quantity(result, block)
        matrix = _required_values(block).reshape((-1, len(block.parameters)))
        row_labels = _output_coordinate_labels(quantity, block)
        parameter_labels = [
            f"{parameter.definitions_id}.{parameter.id}"
            for parameter in block.parameters
        ]
        entry_units = [
            [str(block.entry_unit(parameter, row)) for parameter in block.parameters]
            for row in range(matrix.shape[0])
        ]
        figure = go.Figure(data=[go.Heatmap(
            z=matrix,
            x=parameter_labels,
            y=row_labels,
            colorbar={"title": "numeric value"},
            customdata=np.asarray(entry_units, dtype=object),
            hovertemplate=(
                "output=%{y}<br>parameter=%{x}<br>value=%{z}<br>"
                "unit=%{customdata}<extra></extra>"
            ),
            meta={"scnsim": {
                "kind": "sensitivity_jacobian",
                "block_id": block_id,
                "quantity_id": block.quantity_id,
                "output_unit": block.output_unit,
                "entry_units_by_output_row": entry_units,
                "axes": _axis_records(quantity.axes),
                "source": source,
            }},
        )])
        figure.update_xaxes(title_text="parameter coordinates")
        figure.update_yaxes(title_text="saved output coordinates")
        title = f"Sensitivity Jacobian: {escape(block.quantity_id)}"
    elif kind == "jvp":
        if not isinstance(block, SensitivityValueBlock):
            raise TypeError("JVP projection contains a non-value block")
        quantity_value = block.value
        if quantity_value is None:
            _required_values(block)
        values = np.asarray(quantity_value.magnitude).reshape(-1)
        labels = _coordinate_labels(block.axes)
        figure = go.Figure()
        if np.iscomplexobj(values):
            figure.add_trace(go.Scatter(
                x=labels, y=np.real(values), mode="lines+markers", name="real",
            ))
            figure.add_trace(go.Scatter(
                x=labels, y=np.imag(values), mode="lines+markers", name="imag",
            ))
        else:
            figure.add_trace(go.Scatter(
                x=labels, y=values, mode="lines+markers", name=escape(block.quantity_id),
            ))
        figure.update_xaxes(title_text="saved output coordinates")
        figure.update_yaxes(title_text=f"JVP ({escape(str(quantity_value.units))})")
        figure.update_layout(meta={"scnsim": {
            "kind": "sensitivity_jvp", "block_id": block_id,
            "quantity_id": block.quantity_id,
            "axes": _axis_records(block.axes),
            "complex_components": ["real", "imag"] if np.iscomplexobj(values) else None,
            "source": source,
        }})
        title = f"Sensitivity JVP: {escape(block.quantity_id)}"
    elif kind == "vjp":
        if not isinstance(block, SensitivityParameterBlock):
            raise TypeError("VJP projection contains a non-parameter block")
        values = _required_values(block).reshape(-1)
        labels = [
            f"{parameter.definitions_id}.{parameter.id}"
            for parameter in block.parameters
        ]
        figure = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            customdata=np.asarray(block.units, dtype=object),
            hovertemplate="parameter=%{x}<br>value=%{y}<br>unit=%{customdata}<extra></extra>",
            meta={"scnsim": {
                "kind": "sensitivity_vjp", "block_id": block_id,
                "quantity_id": block.quantity_id,
                "parameter_units": list(block.units), "source": source,
            }},
        )])
        figure.update_xaxes(title_text="parameter coordinates")
        figure.update_yaxes(title_text="VJP value (units vary by parameter)")
        title = f"Sensitivity VJP ({escape(block.functional_unit)})"
    else:
        raise ValueError("kind must be 'jacobian', 'jvp', or 'vjp'")

    return _style(figure, theme, title=title)

