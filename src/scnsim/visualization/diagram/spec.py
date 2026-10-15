"""Diagram-only request declaration, isolated from numerical spec imports."""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import SCNSimValidationError
from ..presentation import Theme, _require_theme
from ..schematic import DiagramUnit, SchematicLayout


@dataclass(frozen=True, slots=True)
class CircuitDiagramSpec:
    """Immutable manual-authoring request; numerical compilation is separate."""

    theme: Theme = Theme.AUTO
    show_parameter_values: bool = True
    show_provenance: bool = True
    drawing_unit: DiagramUnit | None = None
    layout: SchematicLayout | None = None

    def __init__(
        self,
        *,
        theme: Theme = Theme.AUTO,
        show_parameter_values: bool = True,
        show_provenance: bool = True,
        drawing_unit: DiagramUnit | None = None,
        layout: SchematicLayout | None = None,
    ) -> None:
        if drawing_unit is not None and not isinstance(drawing_unit, DiagramUnit):
            raise TypeError("drawing_unit must be DiagramUnit or None")
        object.__setattr__(self, "drawing_unit", drawing_unit)
        object.__setattr__(self, "theme", _require_theme(theme))
        object.__setattr__(self, "show_parameter_values", show_parameter_values)
        object.__setattr__(self, "show_provenance", show_provenance)
        if layout is not None and not isinstance(layout, SchematicLayout):
            raise SCNSimValidationError(
                "layout must be a SchematicLayout or None",
                stage="schematic_layout",
            )
        object.__setattr__(self, "layout", layout)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.show_parameter_values, bool):
            raise TypeError("show_parameter_values must be bool")
        if not isinstance(self.show_provenance, bool):
            raise TypeError("show_provenance must be bool")

    @property
    def representation(self) -> str:
        """Fixed internal identity tag, not a public representation selector."""
        return "authoring"
