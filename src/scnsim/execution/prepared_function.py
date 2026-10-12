"""Detached acquisition layouts, without Run, Workspace or publication authority.

Compilation owns the transformable callable. AOT handles are terminal execution
artifacts; raw values/status do not constitute a certified Result.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze_layout(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_layout(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_layout(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PreparedFunction:
    """Callable(theta_SI, omega_rad_per_s, seeds) -> values and fixed-shape aux.

    Layouts bind ordered physical coordinates, units, axes and components.
    Caller ownership is independent of Run.close(); no mutable point or factor
    belongs to this record.
    """
    numeric_function: Callable[..., Any]
    input_layout: Mapping[str, object]
    output_layout: Mapping[str, object]
    aux_layout: Mapping[str, object]
    structural_identity: str
    fixed_mesh: Mapping[str, object]
    branch_controls: Mapping[str, object]
    precision: str
    _compile_core: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for name in ("input_layout", "output_layout", "aux_layout", "fixed_mesh", "branch_controls"):
            object.__setattr__(self, name, _freeze_layout(getattr(self, name)))
