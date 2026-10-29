"""Physical values for structured authoring; independent of Plan/runtime state."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
import unicodedata
import numpy as np
from ..value_storage import immutable_quantity, quantity_view
from ..errors import SCNSimValidationError
from ..units import Quantity, registry, require_positive_quantity, require_quantity


def identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    value = unicodedata.normalize("NFC", value)
    if not value or any(ord(c) < 32 or ord(c) == 127 or c in "/\\" for c in value):
        raise ValueError(f"{field} must be a nonempty portable identifier")
    return value


def quantity_record(value: Quantity, unit: str) -> dict[str, object]:
    from ..canonical import quantity_envelope

    return quantity_envelope(value, si_unit=unit, registry=registry)


def unit_name(value: object) -> str:
    if getattr(value, "_REGISTRY", None) is not registry:
        raise TypeError("unit must use scnsim.units.registry")
    for unit in (
        "farad",
        "henry",
        "ohm",
        "siemens",
        "hertz",
        "meter",
        "weber",
        "ampere",
        "volt",
        "dimensionless",
    ):
        try:
            require_quantity(registry.Quantity(1, value), unit, name="unit")
        except ValueError:
            continue
        return unit
    raise ValueError("unit is outside SCNSim's supported physical vocabulary")


@dataclass(frozen=True, slots=True, kw_only=True)
class ParameterSpec:
    unit: object
    si_unit: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "si_unit", unit_name(self.unit))

    def _record(self) -> dict[str, object]:
        return {
            "kind": "scalar",
            "si_unit": self.si_unit,
            "dimensionality": quantity_record(
                registry.Quantity(1, self.si_unit), self.si_unit
            )["dimensionality"],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RLGCParameterSpec:
    conductors: tuple[str, ...]
    reference_conductor: str

    def __post_init__(self) -> None:
        c = tuple(identifier(x, field="conductor") for x in self.conductors)
        r = identifier(self.reference_conductor, field="reference_conductor")
        if not c or len(set(c)) != len(c) or r in c:
            raise ValueError("RLGC conductor basis is invalid")
        object.__setattr__(self, "conductors", c)
        object.__setattr__(self, "reference_conductor", r)

    def _record(self) -> dict[str, object]:
        return {
            "kind": "rlgc",
            "conductors": list(self.conductors),
            "reference_conductor": self.reference_conductor,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ElectricalResolution:
    """Length-dependent Pi resolution for an explicitly lossless RLGC line."""

    max_frequency: Quantity = 10 * registry.GHz
    sections_per_wavelength: int = 40

    def __post_init__(self) -> None:
        frequency = require_positive_quantity(
            self.max_frequency, "hertz", name="max_frequency"
        )
        count = self.sections_per_wavelength
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("sections_per_wavelength must be a positive integer")
        if count <= 0:
            raise ValueError("sections_per_wavelength must be a positive integer")
        object.__setattr__(self, "max_frequency", immutable_quantity(frequency))

    def __getattribute__(self, name: str) -> object:
        if name == "max_frequency":
            from ..value_storage import quantity_view

            return quantity_view(object.__getattribute__(self, name))
        return object.__getattribute__(self, name)

    def _record(self) -> dict[str, object]:
        return {
            "kind": "electrical_resolution",
            "max_frequency": quantity_record(self.max_frequency, "hertz"),
            "sections_per_wavelength": self.sections_per_wavelength,
        }


def _matrix(
    value: object, unit: str, n: int, name: str
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Quantity) or value._REGISTRY is not registry:
        raise TypeError(f"{name} must be a SCNSim Quantity matrix")
    a = np.asarray(value.to(unit).magnitude, dtype=np.float64)
    if a.shape != (n, n):
        raise SCNSimValidationError(
            f"{name} must be a {n} by {n} matrix", stage="authoring"
        )
    if not np.isfinite(a).all():
        raise ValueError(f"{name} entries must be finite")
    return tuple(tuple(float(x) for x in row) for row in a)


def _validate_rlgc_matrices(
    r: tuple[tuple[float, ...], ...],
    l: tuple[tuple[float, ...], ...],
    g: tuple[tuple[float, ...], ...],
    c: tuple[tuple[float, ...], ...],
) -> None:
    """Retain the accepted passive per-length-matrix contract before capture."""
    matrices = {
        "resistance_per_length": r,
        "inductance_per_length": l,
        "conductance_per_length": g,
        "capacitance_per_length": c,
    }
    arrays = {
        name: np.asarray(value, dtype=np.float64) for name, value in matrices.items()
    }
    for name, matrix in arrays.items():
        if not np.array_equal(matrix.view(np.uint64), matrix.T.view(np.uint64)):
            raise SCNSimValidationError(
                f"{name} must be exactly symmetric", stage="authoring"
            )
    for name in ("capacitance_per_length", "conductance_per_length"):
        matrix = arrays[name]
        if np.any(matrix[~np.eye(len(matrix), dtype=bool)] > 0):
            raise SCNSimValidationError(
                f"{name} off-diagonal entries must be nonpositive", stage="authoring"
            )
        if np.any(matrix.sum(axis=1) < 0):
            raise SCNSimValidationError(
                f"{name} row sums must be nonnegative", stage="authoring"
            )
    for name in ("capacitance_per_length", "inductance_per_length"):
        try:
            np.linalg.cholesky(arrays[name])
        except np.linalg.LinAlgError as exc:
            raise SCNSimValidationError(
                f"{name} must be positive definite", stage="authoring"
            ) from exc
    for name in ("conductance_per_length", "resistance_per_length"):
        if float(np.linalg.eigvalsh(arrays[name])[0]) < 0:
            raise SCNSimValidationError(
                f"{name} must be positive semidefinite", stage="authoring"
            )


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType({str(k): freeze(v) for k, v in item.items()})
        if isinstance(item, (tuple, list)):
            return tuple(freeze(v) for v in item)
        if isinstance(item, np.ndarray):
            copied = np.array(item, copy=True)
            copied.setflags(write=False)
            return copied
        if isinstance(item, Quantity):
            magnitude = np.array(item.magnitude, copy=True)
            magnitude.setflags(write=False)
            return registry.Quantity(magnitude, item.units)
        return item

    return MappingProxyType({str(k): freeze(v) for k, v in value.items()})


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class RLGC:
    """Immutable, copied RLGC physical value with accepted passivity checks."""

    __slots__ = (
        "conductors",
        "reference_conductor",
        "_m",
        "_frequency",
        "source",
        "_source_quantities",
    )

    def __init__(
        self,
        *,
        conductors: Sequence[str],
        reference_conductor: str,
        resistance_per_length: Quantity,
        inductance_per_length: Quantity,
        conductance_per_length: Quantity,
        capacitance_per_length: Quantity,
        extraction_frequency: Quantity | None = None,
        source: Mapping[str, object] | None = None,
    ) -> None:
        # The public physical value retains its typed authoring failures even
        # though parameter specifications share the basis validation below.
        if isinstance(conductors, (str, bytes)) or not isinstance(conductors, Sequence):
            raise TypeError("conductors must be a nonempty sequence of identifiers")
        names = tuple(identifier(name, field="conductor") for name in conductors)
        if not names:
            raise ValueError("RLGC requires at least one conductor")
        if len(set(names)) != len(names):
            raise SCNSimValidationError(
                "RLGC conductor names must be unique", stage="authoring"
            )
        reference = identifier(reference_conductor, field="reference_conductor")
        if reference in names:
            raise SCNSimValidationError(
                "reference_conductor cannot occur in the RLGC conductor basis",
                stage="authoring",
            )
        spec = RLGCParameterSpec(
            conductors=names, reference_conductor=reference
        )
        n = len(spec.conductors)
        matrices = (
            _matrix(resistance_per_length, "ohm / meter", n, "resistance_per_length"),
            _matrix(inductance_per_length, "henry / meter", n, "inductance_per_length"),
            _matrix(
                conductance_per_length, "siemens / meter", n, "conductance_per_length"
            ),
            _matrix(
                capacitance_per_length, "farad / meter", n, "capacitance_per_length"
            ),
        )
        _validate_rlgc_matrices(*matrices)
        object.__setattr__(self, "conductors", spec.conductors)
        object.__setattr__(self, "reference_conductor", spec.reference_conductor)
        object.__setattr__(self, "_m", matrices)
        frequency = (
            None
            if extraction_frequency is None
            else require_positive_quantity(
                extraction_frequency, "hertz", name="extraction_frequency"
            )
        )
        object.__setattr__(
            self,
            "_frequency",
            None
            if frequency is None
            else registry.Quantity(float(frequency.to("hertz").magnitude), "hertz"),
        )
        object.__setattr__(
            self, "source", _freeze_mapping(source or {"source_kind": "manual"})
        )

        def source_copy(quantity):
            magnitude = np.array(quantity.magnitude, copy=True)
            magnitude.setflags(write=False)
            return registry.Quantity(magnitude, quantity.units)

        sources = {
            "resistance_per_length": source_copy(resistance_per_length),
            "inductance_per_length": source_copy(inductance_per_length),
            "conductance_per_length": source_copy(conductance_per_length),
            "capacitance_per_length": source_copy(capacitance_per_length),
        }
        if extraction_frequency is not None:
            sources["extraction_frequency"] = source_copy(extraction_frequency)
        object.__setattr__(self, "_source_quantities", MappingProxyType(sources))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RLGC values are immutable")

    @classmethod
    def _from_source(cls, *, source: Mapping[str, object], **kwargs: object) -> "RLGC":
        return cls(source=source, **kwargs)

    @property
    def resistance_per_length(self) -> Quantity:
        return registry.Quantity(
            np.asarray(self._m[0], dtype=np.float64), "ohm / meter"
        )

    @property
    def inductance_per_length(self) -> Quantity:
        return registry.Quantity(
            np.asarray(self._m[1], dtype=np.float64), "henry / meter"
        )

    @property
    def conductance_per_length(self) -> Quantity:
        return registry.Quantity(
            np.asarray(self._m[2], dtype=np.float64), "siemens / meter"
        )

    @property
    def capacitance_per_length(self) -> Quantity:
        return registry.Quantity(
            np.asarray(self._m[3], dtype=np.float64), "farad / meter"
        )

    @property
    def extraction_frequency(self) -> Quantity | None:
        return (
            None
            if self._frequency is None
            else registry.Quantity(
                float(self._frequency.to("hertz").magnitude), "hertz"
            )
        )

    def _record(self) -> dict[str, object]:
        result = {
            "type": "rlgc",
            "conductors": list(self.conductors),
            "reference_conductor": self.reference_conductor,
            "orientation": "extractor_positive_z_is_head_to_tail",
            "source": _thaw(self.source),
        }
        for name, unit, m in zip(
            (
                "resistance_per_length",
                "inductance_per_length",
                "conductance_per_length",
                "capacitance_per_length",
            ),
            ("ohm / meter", "henry / meter", "siemens / meter", "farad / meter"),
            self._m,
        ):
            result[name] = {
                "type": "quantity_matrix_f64",
                "shape": [len(m), len(m)],
                "values_f64": [
                    quantity_record(registry.Quantity(x, unit), unit)["si_value_f64"]
                    for row in m
                    for x in row
                ],
                "si_unit": unit,
                "dimensionality": quantity_record(registry.Quantity(1, unit), unit)[
                    "dimensionality"
                ],
            }
        result["extraction_frequency"] = (
            None
            if self.extraction_frequency is None
            else quantity_record(self.extraction_frequency, "hertz")
        )
        return result


def _checked_field_baseline(value, unit, name, positive=False, nonnegative=False):
    """Apply the native scalar field domain at a factory boundary."""
    quantity = require_quantity(value, unit, name=name)
    magnitude = quantity.to(unit).magnitude
    if positive and not magnitude > 0:
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and magnitude < 0:
        raise ValueError(f"{name} must be nonnegative")
    return quantity


def _retained_literal_field(value, unit, name, positive=False, nonnegative=False):
    """Validate and detach one literal retained by a component body."""

    return immutable_quantity(
        _checked_field_baseline(value, unit, name, positive, nonnegative)
    )


def _validate_captured_field(value, unit, name, positive=False, nonnegative=False):
    """Apply the same field domain with selected-point validation errors."""
    if isinstance(value, RLGC):
        return value
    try:
        return _checked_field_baseline(value, unit, name, positive, nonnegative)
    except (TypeError, ValueError) as exc:
        raise SCNSimValidationError(
            f"selected {name} is outside its physical field domain",
            stage="authoring",
        ) from exc


class AffineMap:
    """The sole accepted scalar derived-field binding."""

    __slots__ = (
        "input",
        "slope",
        "intercept",
        "support",
        "_source_quantities",
        "_output_unit",
        "_slope_unit",
    )

    def __init__(
        self,
        *,
        input: object,
        slope: object,
        intercept: object,
        support: tuple[object, object],
    ) -> None:
        # Avoid an import cycle while retaining a closed public type contract.
        from .parameters import ParameterRef

        if not isinstance(input, ParameterRef):
            raise TypeError("input must be a ParameterRef")
        if not isinstance(input.spec, ParameterSpec):
            raise TypeError("AffineMap input must be scalar")
        if not isinstance(support, tuple) or len(support) != 2:
            raise TypeError("support must be a two-Quantity tuple")
        if not isinstance(intercept, Quantity) or intercept._REGISTRY is not registry:
            raise TypeError("intercept must be a SCNSim Quantity")

        def source_copy(quantity: Quantity) -> Quantity:
            magnitude = np.array(quantity.magnitude, copy=True)
            magnitude.setflags(write=False)
            return registry.Quantity(magnitude, quantity.units)

        self.input = input
        self._output_unit = unit_name(intercept.units)
        self.intercept = require_quantity(
            intercept, self._output_unit, name="intercept"
        )
        slope_dim = (
            registry.Unit(self._output_unit) / registry.Unit(input.spec.si_unit)
        ).dimensionality
        # This is the accepted canonical vocabulary used by the prior
        # authoring/canonical boundary, including per-length coefficients.
        candidates = (
            "farad",
            "henry",
            "ohm",
            "siemens",
            "hertz",
            "radian / second",
            "meter",
            "weber",
            "ampere",
            "volt",
            "ohm / meter",
            "henry / meter",
            "siemens / meter",
            "farad / meter",
            "siemens / second",
            "dimensionless",
        )
        self._slope_unit = next(
            (u for u in candidates if registry.Unit(u).dimensionality == slope_dim),
            None,
        )
        if self._slope_unit is None:
            raise ValueError(
                "slope dimensionality is outside SCNSim's canonical unit vocabulary"
            )
        self.slope = require_quantity(slope, self._slope_unit, name="slope")
        self.support = (
            require_quantity(support[0], input.spec.si_unit, name="support[0]"),
            require_quantity(support[1], input.spec.si_unit, name="support[1]"),
        )
        object.__setattr__(
            self,
            "_source_quantities",
            MappingProxyType(
                {
                    "slope": source_copy(slope),
                    "intercept": source_copy(intercept),
                    "support_lower": source_copy(support[0]),
                    "support_upper": source_copy(support[1]),
                }
            ),
        )
        if (
            self.support[0].to(input.spec.si_unit).magnitude
            >= self.support[1].to(input.spec.si_unit).magnitude
        ):
            raise ValueError("AffineMap support must be strictly ordered")
        value = input.baseline.to(input.spec.si_unit).magnitude
        if (
            not self.support[0].to(input.spec.si_unit).magnitude
            <= value
            <= self.support[1].to(input.spec.si_unit).magnitude
        ):
            raise ValueError("AffineMap input baseline must lie inside support")

    def value_at(self, value: Quantity, *, unit: str, name: str) -> Quantity:
        return require_quantity(self.slope * value + self.intercept, unit, name=name)

    def _record(self) -> dict[str, object]:
        return {
            "kind": "affine",
            "input": self.input._key_record(),
            "slope": quantity_record(self.slope, self._slope_unit),
            "intercept": quantity_record(self.intercept, self._output_unit),
            "support": [
                quantity_record(x, self.input.spec.si_unit) for x in self.support
            ],
        }


_RLGC_SCALAR_FIELDS = (
    ("resistance", "resistance_per_length", "ohm / meter"),
    ("inductance", "inductance_per_length", "henry / meter"),
    ("conductance", "conductance_per_length", "siemens / meter"),
    ("capacitance", "capacitance_per_length", "farad / meter"),
)


@dataclass(frozen=True, slots=True, init=False)
class RLGCScalarTerm:
    """A scalar source times fixed matrices; only realized totals are passive."""

    _input: object
    _affine: tuple | None
    _source_unit: str
    _basis: tuple
    _source_basis: tuple

    def __init__(self, source, *, resistance=None, inductance=None, conductance=None, capacitance=None) -> None:
        from .parameters import ParameterRef
        ref = source.input if isinstance(source, AffineMap) else source
        if not isinstance(ref, ParameterRef) or not isinstance(ref.spec, ParameterSpec):
            raise TypeError("RLGC scalar source must be a scalar ParameterRef or AffineMap")
        affine = None
        source_unit = ref.spec.si_unit
        if isinstance(source, AffineMap):
            source_unit = source._output_unit
            affine = (immutable_quantity(source.slope), immutable_quantity(source.intercept), tuple(immutable_quantity(value) for value in source.support), source._slope_unit)
        basis = []
        source_basis = []
        for name, field, unit in _RLGC_SCALAR_FIELDS:
            value = locals()[name]
            source_basis.append(None if value is None else immutable_quantity(value))
            if value is not None:
                if not isinstance(value, Quantity) or value._REGISTRY is not registry:
                    raise TypeError(f"{name} basis must use scnsim.units")
                basis_unit = registry.Unit(unit) / registry.Unit(source_unit)
                magnitude = np.asarray(value.to(basis_unit).magnitude, dtype=np.float64)
                if magnitude.ndim != 2 or magnitude.shape[0] != magnitude.shape[1]:
                    raise ValueError(f"{name} basis must be a square matrix")
                value = immutable_quantity(registry.Quantity(magnitude, basis_unit))
            basis.append(value)
        if all(value is None for value in basis):
            raise ValueError("an RLGC scalar term requires at least one basis matrix")
        object.__setattr__(self, "_input", ref)
        object.__setattr__(self, "_affine", affine)
        object.__setattr__(self, "_source_unit", source_unit)
        object.__setattr__(self, "_basis", tuple(basis))
        object.__setattr__(self, "_source_basis", tuple(source_basis))

    @property
    def source(self):
        if self._affine is None:
            return self._input
        slope, intercept, support, _ = self._affine
        return AffineMap(input=self._input, slope=quantity_view(slope), intercept=quantity_view(intercept), support=tuple(quantity_view(value) for value in support))

    @property
    def resistance(self):
        return None if self._basis[0] is None else quantity_view(self._basis[0])

    @property
    def inductance(self):
        return None if self._basis[1] is None else quantity_view(self._basis[1])

    @property
    def conductance(self):
        return None if self._basis[2] is None else quantity_view(self._basis[2])

    @property
    def capacitance(self):
        return None if self._basis[3] is None else quantity_view(self._basis[3])

    def _value_at(self, value, *, authorized=False):
        candidate = require_quantity(value, self._input.spec.si_unit, name="RLGC scalar source")
        if self._affine is None:
            return candidate
        slope, intercept, support, _ = self._affine
        lower, upper = support
        magnitude = candidate.to(self._input.spec.si_unit).magnitude
        if not authorized and not lower.to(self._input.spec.si_unit).magnitude <= magnitude <= upper.to(self._input.spec.si_unit).magnitude:
            raise SCNSimValidationError("AffineMap value lies outside support", stage="authoring")
        return slope * candidate + intercept

    def _record(self):
        result = {"source": {"kind": "ref", "parameter": self._input._key_record()} if self._affine is None else self.source._record()}
        for (name, _, unit), basis in zip(_RLGC_SCALAR_FIELDS, self._basis):
            if basis is None:
                result[name] = None
                continue
            basis_unit = str(registry.Unit(unit) / registry.Unit(self._source_unit))
            from ..canonical import derived_quantity_matrix_envelope
            result[name] = derived_quantity_matrix_envelope(basis, si_unit=basis_unit, registry=registry)
        return result


@dataclass(frozen=True, slots=True, init=False)
class ScalarRLGCMap:
    """Immutable fixed conductor basis with explicit scalar physical terms."""

    base: RLGC
    terms: tuple[RLGCScalarTerm, ...]

    def __init__(self, base: RLGC, *, terms: Sequence[RLGCScalarTerm]) -> None:
        if not isinstance(base, RLGC):
            raise TypeError("base must be RLGC")
        terms = tuple(terms)
        if not terms or any(not isinstance(term, RLGCScalarTerm) for term in terms):
            raise TypeError("terms must contain RLGCScalarTerm values")
        shape = (len(base.conductors), len(base.conductors))
        if any(np.shape(value.magnitude) != shape for term in terms for value in term._basis if value is not None):
            raise ValueError("RLGC basis matrices must match base conductor shape")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "terms", terms)
        # Construction applies the existing physical validator to the total,
        # without imposing positivity on individual linear basis matrices.
        self.value_at({term._input: term._input.baseline for term in terms})

    @property
    def parameters(self):
        return tuple(dict.fromkeys(term._input for term in self.terms))

    @property
    def baseline(self):
        return self.value_at({ref: ref.baseline for ref in self.parameters})

    def value_at(self, values: Mapping, *, authorized=frozenset()):
        matrices = [getattr(self.base, field) for _, field, _ in _RLGC_SCALAR_FIELDS]
        for term in self.terms:
            value = term._value_at(values[term._input], authorized=term._input in authorized)
            for index, basis in enumerate(term._basis):
                if basis is not None:
                    matrices[index] = matrices[index] + basis * value
        return RLGC(conductors=self.base.conductors, reference_conductor=self.base.reference_conductor,
            resistance_per_length=matrices[0], inductance_per_length=matrices[1],
            conductance_per_length=matrices[2], capacitance_per_length=matrices[3],
            extraction_frequency=self.base.extraction_frequency, source=self.base.source)

    def _record(self):
        return {"kind": "scalar_rlgc", "base": self.base._record(), "terms": [term._record() for term in self.terms]}
