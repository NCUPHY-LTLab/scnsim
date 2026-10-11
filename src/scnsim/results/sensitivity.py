"""Detached Sensitivity values and lazy access to sealed derivative blocks.

Results in this module own immutable coordinates and a fixed, read-only
Sensitivity reader. They never retain a Run, native factor, AD tape, or
derivative closure; reading a block consumes only that block's sealed body.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from operator import index as integer_index
from typing import TYPE_CHECKING, Literal

import numpy as np
from pint import Quantity

from .. import units
from ..authoring import ParameterRef
from ..construction import unavailable
from ..errors import SCNSimStateError
from ..specs import OutputCoordinate
from ..value_storage import immutable_array, immutable_quantity, quantity_view
from ..visualization.presentation import Theme
from .base import AnalysisResult, _freeze, _fresh_quantity_attribute

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


SensitivityOperationKind = Literal["jacobian", "jvp", "vjp"]
SensitivityStatus = Literal["complete", "partial", "unavailable"]


@dataclass(frozen=True, slots=True)
class SensitivityAxisCoordinate:
    """One named coordinate in a stored observable or parameter axis."""

    id: str
    label: str
    value: object | None = None
    unit: str | None = None
    component: Literal["real", "imag"] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __init__(self) -> None:
        unavailable("SensitivityAxisCoordinate construction")


@dataclass(frozen=True, slots=True)
class SensitivityAxis:
    """Ordered, named coordinates for one stored array dimension."""

    id: str
    label: str
    coordinates: tuple[SensitivityAxisCoordinate, ...]
    unit: str | None = None

    def __init__(self) -> None:
        unavailable("SensitivityAxis construction")


@dataclass(frozen=True, slots=True)
class SensitivityQuantity:
    """Saved identity, units, and ordered axes for one named observable."""

    id: str
    label: str
    unit: str
    axes: tuple[SensitivityAxis, ...]

    def __init__(self) -> None:
        unavailable("SensitivityQuantity construction")


@dataclass(frozen=True, slots=True)
class SensitivityValueBlock:
    """A detached primal or JVP value block with homogeneous physical units."""

    block_id: str
    quantity_id: str
    status: Literal["complete", "unavailable"]
    axes: tuple[SensitivityAxis, ...]
    value: Quantity | None
    failure: Mapping[str, object] | None = None
    _quantity_fields = frozenset({"value"})

    def __init__(self) -> None:
        unavailable("SensitivityValueBlock construction")

    def __getattribute__(self, name: str) -> object:
        return _fresh_quantity_attribute(self, name)


@dataclass(frozen=True, slots=True)
class SensitivityJacobianBlock:
    """One real-coordinate Jacobian block; units are row-unit / parameter-unit."""

    block_id: str
    quantity_id: str
    quantity_label: str
    status: Literal["complete", "unavailable"]
    output_axes: tuple[SensitivityAxis, ...]
    output_coordinates: tuple[OutputCoordinate, ...]
    row_units: tuple[str, ...]
    parameters: tuple[ParameterRef, ...]
    output_unit: str
    values: np.ndarray | None
    failure: Mapping[str, object] | None = None

    def __init__(self) -> None:
        unavailable("SensitivityJacobianBlock construction")

    def entry_unit(self, parameter: ParameterRef, row: int = 0) -> object:
        """Return a saved row's physical derivative unit for one input column."""

        if parameter not in self.parameters:
            raise KeyError(parameter)
        unit = self.row_units[row] if self.row_units else self.output_unit
        return units.registry.Unit(unit) / units.registry.Unit(
            parameter.spec.si_unit
        )

    def normalized(
        self,
        *,
        input_scales: Mapping[tuple[str, str], Quantity],
        output_scales: Mapping[OutputCoordinate, Quantity],
    ) -> np.ndarray:
        """Return a dimensionless block using each explicit output-row scale."""

        if self.status != "complete" or self.values is None:
            raise SCNSimStateError(
                "cannot normalize an unavailable Jacobian block",
                stage="result_access",
                evidence={"quantity_id": self.quantity_id, "status": self.status},
            )
        column_scales = np.asarray(
            [
                float(
                    input_scales[(parameter.definitions_id, parameter.id)]
                    .to(parameter.spec.si_unit)
                    .magnitude
                )
                for parameter in self.parameters
            ],
            dtype=np.float64,
        )
        values = np.asarray(self.values)
        rows = values.reshape((-1, len(self.parameters)))
        coordinates = self.output_coordinates
        if len(coordinates) != rows.shape[0]:
            raise ValueError("Jacobian output coordinates do not match its saved rows")
        units_by_row = self.row_units or (self.output_unit,) * rows.shape[0]
        normalized_rows = np.empty_like(rows, dtype=np.result_type(rows, np.float64))
        for index, (coordinate, output_unit) in enumerate(
            zip(coordinates, units_by_row, strict=True)
        ):
            output_magnitude = float(
                output_scales[coordinate].to(output_unit).magnitude
            )
            normalized_rows[index] = rows[index] * column_scales / output_magnitude
        normalized = normalized_rows.reshape(values.shape)
        return immutable_array(normalized)


@dataclass(frozen=True, slots=True)
class SensitivityParameterBlock:
    """A VJP block in declared parameter order with one unit per parameter."""

    block_id: str
    quantity_id: str | None
    status: Literal["complete", "unavailable"]
    functional_unit: str
    parameters: tuple[ParameterRef, ...]
    units: tuple[str, ...]
    values: np.ndarray | None
    failure: Mapping[str, object] | None = None

    def __init__(self) -> None:
        unavailable("SensitivityParameterBlock construction")


class SensitivityBlockSequence(Sequence[object]):
    """Read a Sensitivity block only when that exact block is selected.

    The block identifiers and decoder are metadata-only. No value cache is
    retained, so repeated access revalidates and reads the selected sealed
    block through the fixed Workspace reader.
    """

    __slots__ = ("_reader", "_identifiers", "_decode")

    def __init__(
        self,
        reader: object,
        identifiers: Sequence[str],
        decode: Callable[[Mapping[str, object]], object],
    ) -> None:
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_identifiers", tuple(identifiers))
        object.__setattr__(self, "_decode", decode)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Sensitivity block sequences are immutable")

    def __len__(self) -> int:
        return len(self._identifiers)

    def _index(self, selected: int | slice) -> int | range:
        if isinstance(selected, slice):
            return range(*selected.indices(len(self)))
        index = integer_index(selected)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Sensitivity block index out of range")
        return index

    def __getitem__(self, selected: int | slice) -> object:
        index = self._index(selected)
        if isinstance(index, range):
            return tuple(self[item] for item in index)
        identifier = self._identifiers[index]
        record = self._reader.read_block(identifier)
        return _freeze(self._decode(record))

    def by_id(self, identifier: str) -> object:
        """Read one block by its request-owned identifier."""
        try:
            index = self._identifiers.index(identifier)
        except ValueError as error:
            raise KeyError(identifier) from error
        return self[index]


@dataclass(frozen=True, slots=True)
class SensitivityProjection:
    """One requested derivative operation and its lazy detached blocks."""

    kind: SensitivityOperationKind
    status: SensitivityStatus
    blocks: Sequence[object]
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __init__(self) -> None:
        unavailable("SensitivityProjection construction")

    def block(self, identifier: str) -> object:
        blocks = self.blocks
        if isinstance(blocks, SensitivityBlockSequence):
            return blocks.by_id(identifier)
        for block in blocks:
            if getattr(block, "block_id", None) == identifier:
                return block
        raise KeyError(identifier)


class _NormalizedJacobian(Mapping[str, np.ndarray]):
    """Lazy mapping of dimensionless blocks using scales sealed in the request."""

    __slots__ = ("_projection", "_input_scales", "_output_scales")

    def __init__(
        self,
        projection: SensitivityProjection,
        input_scales: Mapping[tuple[str, str], Quantity],
        output_scales: Mapping[OutputCoordinate, Quantity],
    ) -> None:
        object.__setattr__(self, "_projection", projection)
        object.__setattr__(self, "_input_scales", input_scales)
        object.__setattr__(self, "_output_scales", output_scales)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("normalized Jacobian projections are immutable")

    def __iter__(self):
        blocks = self._projection.blocks
        if isinstance(blocks, SensitivityBlockSequence):
            return iter(blocks._identifiers)
        return iter(tuple(getattr(block, "block_id") for block in blocks))

    def __len__(self) -> int:
        return len(self._projection.blocks)

    def __getitem__(self, block_id: str) -> np.ndarray:
        block = self._projection.block(block_id)
        if not isinstance(block, SensitivityJacobianBlock):
            raise TypeError("Jacobian projection contains a non-Jacobian block")
        return block.normalized(
            input_scales=self._input_scales,
            output_scales=self._output_scales,
        )


@dataclass(frozen=True, slots=True)
class SensitivityResult(AnalysisResult):
    """Fixed-identity primal and requested derivative values.

    Derivative properties expose only operations present in the saved request.
    A Result never computes a missing derivative and keeps only a fixed reader
    plus detached declaration metadata.
    """

    status: Literal["complete", "partial"]
    quantities: tuple[SensitivityQuantity, ...]
    primal: Sequence[object]
    operations: Mapping[str, SensitivityProjection]
    _presentation: Mapping[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __init__(self) -> None:
        unavailable("SensitivityResult construction")

    def __repr__(self) -> str:
        operation_names = ", ".join(self.operations)
        return (
            "SensitivityResult("
            f"request={self.identity.request_sha256[:12]}, "
            f"status={self.status!r}, "
            f"quantities={len(self.quantities)}, "
            f"operations=({operation_names}))"
        )

    def value(self, block_id: str) -> object:
        values = self.primal
        if isinstance(values, SensitivityBlockSequence):
            return values.by_id(block_id)
        for item in values:
            if getattr(item, "block_id", None) == block_id:
                return item
        raise KeyError(block_id)

    def _operation(
        self, kind: SensitivityOperationKind
    ) -> SensitivityProjection:
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

    def plot(
        self,
        *,
        kind: SensitivityOperationKind,
        block_id: str,
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ..visualization.plots.sensitivity import sensitivity_plot

        return sensitivity_plot(self, kind=kind, block_id=block_id, theme=theme)

    def show(
        self,
        *,
        kind: SensitivityOperationKind | None = None,
        block_id: str | None = None,
        theme: Theme = Theme.AUTO,
    ) -> object:
        if kind is None:
            if block_id is not None:
                raise ValueError("block_id requires a Sensitivity operation kind")
            from ..visualization.quantity_html import sensitivity_summary

            return sensitivity_summary(self, theme=theme)
        if block_id is None:
            raise ValueError("block_id is required to plot saved Sensitivity data")
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(kind=kind, block_id=block_id, theme=theme))

    @property
    def normalized_jacobian(self) -> Mapping[str, np.ndarray]:
        scales = self._presentation.get("normalization_scales")
        if not isinstance(scales, Mapping):
            raise SCNSimStateError(
                "normalized_jacobian requires explicit scales in the Sensitivity request",
                stage="result_access",
                evidence={"operation": "jacobian", "scales_available": False},
            )
        input_scales = scales.get("inputs")
        output_scales = scales.get("outputs")
        if not isinstance(input_scales, Mapping) or not isinstance(output_scales, Mapping):
            raise SCNSimStateError(
                "normalized_jacobian requires explicit input and output scales",
                stage="result_access",
                evidence={"operation": "jacobian", "scales_available": False},
            )
        return _NormalizedJacobian(
            self.jacobian,
            input_scales,
            output_scales,
        )

