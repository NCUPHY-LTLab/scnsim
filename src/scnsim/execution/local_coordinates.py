"""Explicit smooth local coordinates, separate from CMA boundary folding.

The stage owner supplies finalized variables and the actual full SI starting
point. Units come from recorded physical envelopes or explicit captured
si_unit metadata, never from numeric magnitude. This helper owns no solver,
request, Workspace or controller state.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from ..canonical import float64_from_hex, float64_hex
from ..errors import InvalidCandidatePhysicalParameter


def _key(record):
    return record["definitions_id"], record["parameter_id"]


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class _Coordinate:
    key: tuple[str, str]
    unit: str
    domain: str | None
    mapping: str
    reference: float
    scale: float | None
    scale_source: str
    exponent_scale: float | None
    physical_bounds: tuple[float | None, float | None]
    bounds: tuple[float | None, float | None]

    def inverse(self, value):
        if self.mapping == "log":
            return math.log(value / self.reference) / self.exponent_scale
        if self.mapping == "affine":
            return (value - self.reference) / self.scale
        return value

    def value(self, coordinate):
        if self.mapping == "log":
            try:
                return self.reference * math.exp(self.exponent_scale * coordinate)
            except OverflowError as error:
                raise InvalidCandidatePhysicalParameter(
                    "candidate mapping left its physical domain", stage="unit_map"
                ) from error
        if self.mapping == "affine":
            return self.reference + self.scale * coordinate
        return coordinate

    def derivative(self, coordinate):
        if self.mapping == "log":
            return self.exponent_scale * self.value(coordinate)
        return self.scale if self.mapping == "affine" else 1.0


@dataclass(frozen=True, slots=True, init=False)
class LocalCoordinateMap:
    """Detached stage map; controller coordinates never replace physical points.

    ``coordinate_scales`` consumes finalized LB rows ``{parameter, value}``.
    Unscaled signed-log variable rows require captured ``si_unit`` metadata.
    Physical gradient sequences follow variable order; keyed gradients use the
    exact (definitions_id, parameter_id) keys from the same declaration.
    """

    _axes: tuple[_Coordinate, ...]
    _base: Mapping

    def __init__(self, variables, base_values, *, coordinate_scales=None):
        scales = {} if coordinate_scales is None else {
            _key(row["parameter"]): row["value"] for row in coordinate_scales
        }
        axes = []
        for variable in variables:
            key = _key(variable["parameter"])
            reference = base_values[key]
            domain = variable.get("domain")
            supplied = scales.get(key)
            declared = variable.get("scale")
            envelope = supplied if supplied is not None else declared
            source = "explicit" if supplied is not None else "domain" if declared is not None else "none"
            scale = None if envelope is None else float64_from_hex(envelope["si_value_f64"])
            unit_record = envelope if envelope is not None else variable.get("lower")
            unit = unit_record["si_unit"] if unit_record is not None else variable["si_unit"]
            if domain is None:
                low = float64_from_hex(variable["lower"]["si_value_f64"])
                high = float64_from_hex(variable["upper"]["si_value_f64"])
            else:
                low = 0.0 if domain == "NONNEGATIVE" else None
                high = 0.0 if domain == "NONPOSITIVE" else None
            is_log = variable["transform"] == "log"
            mapping = "log" if is_log else "affine" if scale is not None else "raw_si"
            a = (scale / abs(reference) if scale is not None else 1.0) if is_log else None
            axis = _Coordinate(key, unit, domain, mapping, reference, scale,
                               source, a, (low, high), (None, None))
            mapped_low = None if low is None else axis.inverse(low)
            mapped_high = None if high is None else axis.inverse(high)
            # Negative explicit scales reverse orientation, not physical bounds.
            orientation = a if is_log else scale if scale is not None else 1.0
            bounds = ((mapped_low, mapped_high) if orientation >= 0
                      else (mapped_high, mapped_low))
            axes.append(_Coordinate(key, unit, domain, mapping, reference,
                                    scale, source, a, (low, high), bounds))
        object.__setattr__(self, "_axes", tuple(axes))
        object.__setattr__(self, "_base", _freeze(base_values))

    @property
    def bounds(self):
        return tuple(axis.bounds for axis in self._axes)

    def physical_values(self, coordinates):
        values = _thaw(self._base)
        for axis, coordinate in zip(self._axes, coordinates, strict=True):
            value = axis.value(float(coordinate))
            if (not math.isfinite(value) or
                    (axis.domain in {"POSITIVE", "NEGATIVE"} and
                     (value == 0 or math.copysign(1, value) != math.copysign(1, axis.reference))) or
                    (axis.domain == "NONNEGATIVE" and value < 0) or
                    (axis.domain == "NONPOSITIVE" and value > 0)):
                raise InvalidCandidatePhysicalParameter(
                    "candidate mapping left its physical domain", stage="unit_map"
                )
            values[axis.key] = value
        return values

    def coordinates(self, values):
        return tuple(axis.inverse(values[axis.key]) for axis in self._axes)

    def gradient(self, physical_gradient, coordinates):
        ordered = (tuple(physical_gradient[axis.key] for axis in self._axes)
                   if isinstance(physical_gradient, Mapping) else physical_gradient)
        return tuple(value * axis.derivative(coordinate)
                     for axis, value, coordinate in zip(
                         self._axes, ordered, coordinates, strict=True))

    def canonical_record(self):
        def bound(value):
            return None if value is None else float64_hex(value)
        return {
            "type": "local_coordinate_map",
            "coordinates": [
                {"parameter": {"definitions_id": axis.key[0], "parameter_id": axis.key[1]},
                 "si_unit": axis.unit, "domain": axis.domain, "mapping": axis.mapping,
                 "reference_f64": float64_hex(axis.reference),
                 "scale_f64": bound(axis.scale), "scale_source": axis.scale_source,
                 "exponent_scale_f64": bound(axis.exponent_scale),
                 "coordinate_unit": axis.unit if axis.mapping == "raw_si" else "dimensionless",
                 "physical_bounds_f64": [bound(value) for value in axis.physical_bounds],
                 "controller_bounds_f64": [bound(value) for value in axis.bounds]}
                for axis in self._axes
            ],
        }
