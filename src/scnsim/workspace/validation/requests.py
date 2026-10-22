"""Request, operation-specification, and lineage verification."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from itertools import product

from ...canonical import canonical_json_bytes as _canonical_bytes, sha256_hex as _sha256
from ...errors import UnsupportedEvidenceVersionError
from .common import (
    _IDENTIFIER,
    _SHA256,
    _f64_value,
    _finite_f64,
    _identifiers,
    _integrity,
    _parameter_key_integrity,
    _valid_sha,
    _verify_bounds,
    _verify_branch_refs,
    _verify_parameter_set_document,
    _verify_parameter_value,
    _verify_quantity_compatible,
    _verify_quantity_role,
)

_NATIVE_OPTIMIZATION_ALGORITHM_ID = (
    "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v9"
)
_JULIA_RUNTIME_SEMANTIC_FIELDS = frozenset({
    "algorithm_id",
    "python_source_sha256",
    "julia_source_sha256",
    "julia_version",
    "project_sha256",
    "manifest_sha256",
})
_JAX_RUNTIME_SEMANTIC_FIELDS = frozenset({
    "backend", "precision", "python_source_sha256", "scipy_version",
    "jax_version", "jaxlib_version", "cmaes_version", "resources",
})
_JAX_ALGORITHMS = {
    "solve_direct": "scnsim.jax.native_physical.direct.stage_bundle.v1",
    "evaluate_direct": "scnsim.jax.native_physical.evaluate.stage_bundle.v1",
    "optimize_direct": "scnsim.jax.native_physical.optimize.stage_bundle.v1",
    "sensitivity": "scnsim.jax.native_physical.sensitivity.stage_bundle.v1",
}
_JAX_RESOURCE_FIELDS = frozenset({
    "cpu_threads", "pjrt_nproc", "allocation_policy", "optimization_workers",
    "optimization_execution_model", "optimization_numeric_pool_location",
    "optimization_pjrt_nproc", "optimization_blas_threads",
})


def _is_current_native_optimization_request(
    request: Mapping[str, object],
) -> bool:
    """Recognize the exact current Julia Optimization request identity."""
    spec = request.get("spec")
    runtime = request.get("runtime_semantic")
    return (
        request.get("operation") == "optimize_direct"
        and isinstance(spec, Mapping)
        and spec.get("type") == "optimization"
        and isinstance(runtime, Mapping)
        and set(runtime) == _JULIA_RUNTIME_SEMANTIC_FIELDS
        and runtime.get("algorithm_id") == _NATIVE_OPTIMIZATION_ALGORITHM_ID
    )


def _verify_request_document(
    request: Mapping[str, object],
    plan_sha256: str,
    plan: Mapping[str, object],
) -> None:
    operation = request.get("operation")
    spec = request.get("spec")
    runtime = request.get("runtime_semantic")
    julia_algorithms = {
        "solve_direct": {"direct_solve": "scnsim.direct_response.v1"},
        "solve_hb": {"hb_solve": "scnsim.hb_response.josephsoncircuits.v1"},
        "evaluate_direct": {
            "diagonal_root": "scnsim.diagonal_root.newton32.v2",
            "operator_element_root": "scnsim.operator_element_root.newton32.v1",
            "hybridized_pole": "scnsim.hybridized_pole.newton32.v1",
            "transfer_zero": "scnsim.transfer_zero.newton32.v4",
            "residue_normalized_coupling": "scnsim.residue_normalized_coupling.v2",
            "response_element": "scnsim.response_element.v1",
            "operator": "scnsim.direct_operator.v1",
        },
        "optimize_direct": {"optimization": _NATIVE_OPTIMIZATION_ALGORITHM_ID},
    }
    is_jax = isinstance(runtime, dict) and runtime.get("backend") == "jax"
    if is_jax:
        expected_algorithm = _JAX_ALGORITHMS.get(operation)
        expected_spec = {
            "solve_direct": "direct_solve",
            "evaluate_direct": "diagonal_root",  # checked against its closed evaluation catalog below
            "optimize_direct": "optimization",
            "sensitivity": "sensitivity",
        }.get(operation)
        if operation == "evaluate_direct":
            expected_spec = spec.get("type") if isinstance(spec, dict) else None
            if expected_spec not in {*julia_algorithms["evaluate_direct"], "static_capacitance", "static_capacitance_element"}:
                expected_spec = None
    else:
        expected_algorithm = julia_algorithms.get(operation, {}).get(spec.get("type") if isinstance(spec, dict) else None)
        # The same closed operation/spec catalog owns both discriminators.
        expected_spec = spec.get("type") if expected_algorithm is not None else None
    if (
        set(request) != {"schema", "schema_version", "plan_sha256", "operation", "view", "spec", "parameter_source", "runtime_semantic"}
        or request.get("schema") != "scnsim.request"
        or request.get("schema_version") != 2
        or request.get("plan_sha256") != plan_sha256
        or expected_algorithm is None
        or expected_spec is None
        or not isinstance(spec, dict)
        or spec.get("type") != expected_spec
        or any(not isinstance(request.get(field), dict) for field in ("view", "spec", "parameter_source", "runtime_semantic"))
    ):
        raise _integrity("Stored request envelope is open or inconsistent.")
    if is_jax:
        _verify_jax_runtime_semantic(runtime)
    else:
        if set(runtime) != _JULIA_RUNTIME_SEMANTIC_FIELDS or not isinstance(runtime.get("julia_version"), str) or not runtime["julia_version"]:
            raise _integrity("Stored runtime semantic identity is open or malformed.")
        for field in ("python_source_sha256", "julia_source_sha256", "project_sha256", "manifest_sha256"):
            _valid_sha(runtime.get(field))
    algorithm_id = runtime.get("algorithm_id")
    historical_optimization_algorithms = {
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v3",
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v4",
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v5",
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v6",
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v7",
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v8",
    }
    if operation == "optimize_direct" and algorithm_id in historical_optimization_algorithms:
        raise UnsupportedEvidenceVersionError(
            "Stored optimization evidence requires its original source-bound runtime.",
            stage="workspace",
            evidence={
                "operation": "optimize_direct",
                "expected_algorithm_id": expected_algorithm,
                "observed_algorithm_id": algorithm_id,
            },
        )
    if operation == "evaluate_direct" and isinstance(spec, dict) and spec.get("type") == "diagonal_root" and algorithm_id == "scnsim.diagonal_root.newton32.v1":
        raise UnsupportedEvidenceVersionError(
            "Stored diagonal-root evidence requires its original source-bound runtime.",
            stage="workspace",
            evidence={
                "operation": "evaluate_direct",
                "expected_algorithm_id": expected_algorithm,
                "observed_algorithm_id": algorithm_id,
            },
        )
    if algorithm_id != expected_algorithm:
        raise _integrity("Stored request runtime identity is inconsistent.")
    _verify_parameter_source(request["parameter_source"], plan)
    if operation == "sensitivity":
        _verify_v1_sensitivity_request(request["view"], spec, request["parameter_source"], plan, plan_sha256)
        return
    static_capacitance = spec.get("type") in {"static_capacitance", "static_capacitance_element"}
    if static_capacitance and (not is_jax or request["parameter_source"]["kind"] != "point"):
        raise _integrity("Static capacitance requires a JAX single-point request.")
    terminal, port_realizable = _verify_view_declaration(request["view"], plan,
                                                        static_capacitance=static_capacitance)
    if operation == "solve_direct":
        _verify_v1_direct_spec(spec, terminal, port_realizable)
    elif operation == "solve_hb":
        logical_ports = [port["id"] for port in plan["connectivity"]["ports"]]
        _verify_v1_hb_spec(spec, terminal, port_realizable, logical_ports)
    elif operation == "evaluate_direct":
        _verify_v1_evaluation_spec(spec, terminal, port_realizable,
            static_retained=static_capacitance and request["view"]["retain"] is not None)
    else:
        if is_jax:
            _verify_v1_jax_optimization_spec(
                spec, plan, request["parameter_source"].get("parameters")
            )
        else:
            _verify_v1_optimization_spec(spec, plan, request["parameter_source"].get("parameters"))
        leaves = _optimization_selector_leaves(spec)
        if not leaves or request["view"] != leaves[0].get("view"):
            raise _integrity("Optimization primary View is not its first normalized selector View.")
        parameters = request["parameter_source"].get("parameters")
        if not isinstance(parameters, Mapping):
            raise _integrity("Optimization request requires one complete parameter point.")
        active = {_parameter_key_integrity(item["parameter"]) for item in spec["variables"]}
        declared = {_parameter_key_integrity(item) for item in spec["allow_extrapolation"]}
        if declared != ({_parameter_key_integrity(item) for item in parameters["allow_extrapolation"]} & active):
            raise _integrity("Optimization request has inconsistent active extrapolation authorities.")


def _verify_jax_runtime_semantic(runtime: Mapping[str, object]) -> None:
    """Verify the closed JAX identity without initializing a numerical backend."""
    if set(runtime) != _JAX_RUNTIME_SEMANTIC_FIELDS:
        raise _integrity("Stored JAX runtime identity is open or malformed.")
    if runtime.get("backend") != "jax" or runtime.get("precision") not in {"float32", "float64"}:
        raise _integrity("Stored JAX backend or precision identity is malformed.")
    _valid_sha(runtime.get("python_source_sha256"))
    for field in ("scipy_version", "jax_version", "jaxlib_version", "cmaes_version"):
        value = runtime.get(field)
        if not isinstance(value, str) or not value:
            raise _integrity("Stored JAX package identity is malformed.", field=field)
    resources = runtime.get("resources")
    if not isinstance(resources, Mapping) or set(resources) != _JAX_RESOURCE_FIELDS:
        raise _integrity("Stored JAX resource identity is open or malformed.")
    cpu_threads = resources.get("cpu_threads")
    workers = resources.get("optimization_workers")
    if cpu_threads is not None and (
        not isinstance(cpu_threads, int) or isinstance(cpu_threads, bool) or cpu_threads < 1
    ):
        raise _integrity("Stored JAX thread declaration is malformed.")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise _integrity("Stored JAX optimization worker declaration is malformed.")
    for field in ("pjrt_nproc", "optimization_pjrt_nproc"):
        if resources.get(field) is not None and not isinstance(resources[field], str):
            raise _integrity("Stored JAX PJRT resource identity is malformed.", field=field)
    if resources.get("allocation_policy") != "scnsim.candidate-thread-ready-batch.v1":
        raise _integrity("Stored JAX allocation policy is unsupported.")
    if resources.get("optimization_execution_model") not in {"serial", "threads", "processes"}:
        raise _integrity("Stored JAX execution model is malformed.")
    if resources.get("optimization_numeric_pool_location") not in {"parent", "child"}:
        raise _integrity("Stored JAX numerical pool location is malformed.")
    blas_threads = resources.get("optimization_blas_threads")
    if blas_threads is not None and (
        not isinstance(blas_threads, int) or isinstance(blas_threads, bool) or blas_threads < 1
    ):
        raise _integrity("Stored JAX BLAS thread declaration is malformed.")


def _verify_v1_sensitivity_request(
    view_set: object,
    spec: Mapping[str, object],
    parameter_source: Mapping[str, object],
    plan: Mapping[str, object],
    plan_sha256: str,
) -> None:
    """Close the physical Sensitivity declaration and its ordered View set."""
    if set(spec) != {
        "type", "quantities", "wrt", "derivative", "discretization", "scales",
        "functional_unit",
    } or spec.get("type") != "sensitivity":
        raise _integrity("Sensitivity request Spec is open or malformed.")
    if not isinstance(view_set, Mapping) or set(view_set) != {"type", "views"} or view_set.get("type") != "network_view_set":
        raise _integrity("Sensitivity View set is malformed.")
    views = view_set.get("views")
    if not isinstance(views, list) or not views:
        raise _integrity("Sensitivity View set must contain its selected Views.")

    parameter_document = parameter_source.get("parameters")
    if parameter_source.get("kind") != "point" or not isinstance(parameter_document, Mapping):
        raise _integrity("Sensitivity request requires one complete parameter point.")
    parameter_bindings = {
        _parameter_key_integrity(binding["parameter"]): binding["value"]
        for binding in parameter_document.get("bindings", ())
        if isinstance(binding, Mapping) and isinstance(binding.get("parameter"), Mapping)
    }
    definitions = plan.get("parameter_closure", {}).get("definitions")
    if not isinstance(definitions, list):
        raise _integrity("Plan parameter definitions are malformed.")
    definitions_by_key = {
        _parameter_key_integrity({
            "definitions_id": row.get("definitions_id"),
            "parameter_id": row.get("parameter_id"),
        }): row
        for row in definitions if isinstance(row, Mapping)
    }

    def parameter_role(reference: object) -> tuple[tuple[str, str], str, str]:
        key = _parameter_key_integrity(reference)
        definition = definitions_by_key.get(key)
        if not isinstance(definition, Mapping) or key not in parameter_bindings:
            raise _integrity("Sensitivity parameter is absent from its Plan-bound point.", parameter=key)
        parameter_spec = definition.get("spec")
        if not isinstance(parameter_spec, Mapping) or parameter_spec.get("kind") != "scalar":
            raise _integrity("Sensitivity input must bind a scalar physical parameter.", parameter=key)
        unit, dimension = parameter_spec.get("si_unit"), parameter_spec.get("dimensionality")
        if not isinstance(unit, str) or not unit or not isinstance(dimension, str) or not dimension:
            raise _integrity("Sensitivity parameter physical role is malformed.", parameter=key)
        return key, unit, dimension

    wrt = spec.get("wrt")
    if not isinstance(wrt, list) or not wrt:
        raise _integrity("Sensitivity WRT declaration is malformed.")
    wrt_keys = [parameter_role(reference)[0] for reference in wrt]
    if len(set(wrt_keys)) != len(wrt_keys):
        raise _integrity("Sensitivity WRT declaration repeats a parameter.")

    quantities = spec.get("quantities")
    if not isinstance(quantities, list) or not quantities:
        raise _integrity("Sensitivity output declaration is malformed.")
    output_roles: dict[str, tuple[str, str, tuple[int, ...], bool]] = {}
    encountered_views: list[dict[str, object]] = []

    def checked_leaf_view(value: object, *, static_capacitance=False) -> tuple[list[str], bool]:
        if not isinstance(value, Mapping) or set(value) != {
            "type", "plan_sha256", "ptc", "transforms", "retain",
        } or value.get("type") != "network_view" or value.get("plan_sha256") != plan_sha256:
            raise _integrity("Sensitivity leaf View is not bound to this Plan.")
        declaration = {key: value[key] for key in ("type", "ptc", "transforms", "retain")}
        full_declaration = dict(value)
        if not any(_canonical_bytes(item) == _canonical_bytes(full_declaration) for item in encountered_views):
            encountered_views.append(full_declaration)
        return _verify_view_declaration(declaration, plan, static_capacitance=static_capacitance)

    def frequency_grid(value: object) -> list[object]:
        if not isinstance(value, Mapping) or set(value) != {"type", "shape", "values"} or value.get("type") != "quantity_array":
            raise _integrity("Sensitivity frequency grid is malformed.")
        shape, values = value.get("shape"), value.get("values")
        if not isinstance(shape, list) or len(shape) != 1 or not isinstance(values, list) or shape != [len(values)]:
            raise _integrity("Sensitivity frequency grid shape is malformed.")
        for item in values:
            _verify_quantity_role(item, complex_value=False, unit="hertz", dimensionality="inverse_time")
        return values

    def expression(value: object) -> tuple[str, str, tuple[int, ...], bool]:
        if not isinstance(value, Mapping) or not isinstance(value.get("type"), str):
            raise _integrity("Sensitivity quantity expression is malformed.")
        kind = value["type"]
        if kind == "quantity_sum":
            if set(value) != {"type", "terms"} or not isinstance(value.get("terms"), list) or not value["terms"]:
                raise _integrity("Sensitivity QuantitySum is malformed.")
            roles = [expression(term) for term in value["terms"]]
            if any(role[1] != roles[0][1] or role[2] != roles[0][2] for role in roles[1:]):
                raise _integrity("Sensitivity QuantitySum terms have incompatible layouts.")
            return roles[0]
        if kind == "quantity_difference":
            if set(value) != {"type", "left", "right"}:
                raise _integrity("Sensitivity QuantityDifference is malformed.")
            left, right = expression(value["left"]), expression(value["right"])
            if left[1] != right[1] or left[2] != right[2]:
                raise _integrity("Sensitivity QuantityDifference operands have incompatible layouts.")
            return left
        if kind == "quantity_absolute":
            if set(value) != {"type", "operand"}:
                raise _integrity("Sensitivity QuantityAbsolute is malformed.")
            unit, dimension, shape, _ = expression(value["operand"])
            return unit, dimension, shape, False

        if set(value) != {"type", "spec", "projection", "view"} or not isinstance(value.get("spec"), Mapping):
            raise _integrity("Sensitivity quantity selector is malformed.")
        selected_spec = value["spec"]
        is_static = kind in {"static_capacitance_projection", "static_capacitance_element_projection"}
        terminal, port_realizable = checked_leaf_view(value.get("view"), static_capacitance=is_static)
        if is_static:
            matrix = kind == "static_capacitance_projection"
            if (value.get("projection") != ("matrix" if matrix else "value")
                    or selected_spec.get("type") != ("static_capacitance" if matrix else "static_capacitance_element")):
                raise _integrity("Sensitivity static capacitance projection is malformed.")
            _verify_v1_evaluation_spec(selected_spec, terminal, False,
                static_retained=value["view"]["retain"] is not None)
            # Anonymous finite-mesh node axes are compiled OutputLayout facts,
            # not Port/public-coordinate counts. Check rank here; the compiler
            # and detached result projection own the actual extents.
            return "farad", "capacitance", (None, None) if matrix else (), False
        if kind == "matrix_projection":
            if value.get("projection") not in {"S", "Y", "Z"} or selected_spec.get("type") != "direct_solve":
                raise _integrity("Sensitivity matrix projection is malformed.")
            normalized_spec = dict(selected_spec)
            normalized_spec["frequencies"] = frequency_grid(selected_spec.get("frequencies"))
            _verify_v1_direct_spec(normalized_spec, terminal, port_realizable)
            role = {
                "S": ("dimensionless", "dimensionless"),
                "Y": ("siemens", "conductance"),
                "Z": ("ohm", "resistance"),
            }[value["projection"]]
            return role[0], role[1], (len(normalized_spec["frequencies"]), len(terminal), len(terminal)), True
        if kind == "operator_projection":
            if value.get("projection") != "matrix" or selected_spec.get("type") != "operator":
                raise _integrity("Sensitivity operator projection is malformed.")
            frequencies = frequency_grid(selected_spec.get("frequencies"))
            _verify_v1_evaluation_spec(
                {"type": "operator", "frequencies": frequencies}, terminal, port_realizable,
            )
            return "siemens / second", "conductance_per_time", (len(frequencies), len(terminal), len(terminal)), True
        projections = {
            "diagonal_root_projection": ("diagonal_root", {"frequency", "linewidth"}, ("hertz", "inverse_time"), False),
            "operator_element_root_projection": ("operator_element_root", {"frequency"}, ("hertz", "inverse_time"), False),
            "hybridized_pole_projection": ("hybridized_pole", {"frequency", "linewidth"}, ("hertz", "inverse_time"), False),
            "transfer_zero_projection": ("transfer_zero", {"frequency"}, ("hertz", "inverse_time"), False),
            "residue_coupling_projection": ("residue_normalized_coupling", {"real", "imag", "magnitude"}, ("radian / second", "inverse_time"), False),
            "response_element_projection": ("response_element", {"magnitude", "real", "imag"}, None, False),
        }
        expected = projections.get(kind)
        if expected is None or value.get("projection") not in expected[1] or selected_spec.get("type") != expected[0]:
            raise _integrity("Sensitivity quantity selector is outside the Direct catalog.")
        _verify_v1_evaluation_spec(selected_spec, terminal, port_realizable)
        role = expected[2]
        if role is None:
            family = selected_spec.get("family")
            role = {
                "S": ("dimensionless", "dimensionless"),
                "Y": ("siemens", "conductance"),
                "Z": ("ohm", "resistance"),
            }.get(family)
            if role is None:
                raise _integrity("Sensitivity response selector family is malformed.")
        return role[0], role[1], (), False

    labels: list[str] = []
    for row in quantities:
        if not isinstance(row, Mapping) or set(row) != {"id", "quantity"}:
            raise _integrity("Sensitivity quantity row is open or malformed.")
        label = row.get("id")
        if not isinstance(label, str) or _IDENTIFIER.fullmatch(label) is None or label in output_roles:
            raise _integrity("Sensitivity quantity label is malformed or duplicated.")
        labels.append(label)
        output_roles[label] = expression(row.get("quantity"))

    if views != encountered_views:
        raise _integrity("Sensitivity View set differs from its ordered leaf declarations.")

    derivative = spec.get("derivative")
    if not isinstance(derivative, Mapping) or not isinstance(derivative.get("type"), str):
        raise _integrity("Sensitivity derivative declaration is malformed.")

    def derived_scalar(value: object, *, expected_role: tuple[str, str] | None = None) -> tuple[str, str]:
        if not isinstance(value, Mapping):
            raise _integrity("Sensitivity physical value is malformed.")
        kind = value.get("type")
        if kind == "quantity_array":
            if set(value) != {"type", "shape", "values"}:
                raise _integrity("Sensitivity physical array is open or malformed.")
            shape, values = value.get("shape"), value.get("values")
            if not isinstance(shape, list) or not isinstance(values, list):
                raise _integrity("Sensitivity physical array shape or values are malformed.")
            count = 1
            for dimension in shape:
                if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
                    raise _integrity("Sensitivity physical array dimension is malformed.")
                count *= dimension
            if count != len(values):
                raise _integrity("Sensitivity physical array shape differs from its values.")
            roles = [derived_scalar(item, expected_role=expected_role) for item in values]
            if roles and any(role != roles[0] for role in roles[1:]):
                raise _integrity("Sensitivity physical array mixes unit roles.")
            return roles[0] if roles else (expected_role or ("", ""))
        if kind not in {"quantity_f64", "complex_quantity_f64"}:
            raise _integrity("Sensitivity value is outside the derived-quantity vocabulary.")
        complex_value = kind == "complex_quantity_f64"
        fields = {"type", "si_unit", "dimensionality", "real_si_f64", "imag_si_f64"} if complex_value else {"type", "si_unit", "dimensionality", "si_value_f64"}
        if set(value) != fields:
            raise _integrity("Sensitivity scalar quantity is open or malformed.")
        unit, dimension = value.get("si_unit"), value.get("dimensionality")
        numeric = (value.get("real_si_f64"), value.get("imag_si_f64")) if complex_value else (value.get("si_value_f64"),)
        if not isinstance(unit, str) or not unit or not isinstance(dimension, str) or not dimension or any(not _finite_f64(token) for token in numeric):
            raise _integrity("Sensitivity scalar quantity unit or Float64 encoding is malformed.")
        if expected_role is not None and (unit, dimension) != expected_role:
            raise _integrity("Sensitivity scalar quantity has the wrong physical role.", expected=expected_role)
        return unit, dimension

    def output_coordinate(value: object) -> tuple[str, str, tuple[int, ...], bool]:
        if not isinstance(value, Mapping) or set(value) != {"quantity_label", "component", "index"}:
            raise _integrity("Sensitivity output coordinate is malformed.")
        label, component, index = value.get("quantity_label"), value.get("component"), value.get("index")
        if (not isinstance(label, str) or label not in output_roles
                or component not in {"real", "imag"} or not isinstance(index, list)):
            raise _integrity("Sensitivity output coordinate is outside its declared quantity layout.")
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in index):
            raise _integrity("Sensitivity output coordinate index is malformed.")
        unit, dimension, shape, complex_output = output_roles[label]
        if (len(index) != len(shape)
                or any(extent is not None and coordinate >= extent for coordinate, extent in zip(index, shape))
                or (component == "imag" and not complex_output)):
            raise _integrity("Sensitivity output coordinate is outside its declared quantity axes.")
        return unit, dimension, shape, complex_output

    derivative_kind = derivative.get("type")
    if derivative_kind == "jacobian":
        if set(derivative) != {"type"}:
            raise _integrity("Sensitivity Jacobian declaration is malformed.")
    elif derivative_kind == "jvp":
        rows = derivative.get("direction")
        if set(derivative) != {"type", "direction"} or not isinstance(rows, list):
            raise _integrity("Sensitivity JVP declaration is malformed.")
        observed: set[tuple[str, str]] = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"parameter", "value"}:
                raise _integrity("Sensitivity JVP direction row is malformed.")
            key, unit, dimension = parameter_role(row.get("parameter"))
            if key not in wrt_keys or key in observed:
                raise _integrity("Sensitivity JVP direction does not match its WRT declaration.")
            derived_scalar(row.get("value"), expected_role=(unit, dimension))
            observed.add(key)
    elif derivative_kind == "vjp":
        rows = derivative.get("weights")
        if set(derivative) != {"type", "weights"} or not isinstance(rows, list):
            raise _integrity("Sensitivity VJP declaration is malformed.")
        observed_outputs: set[bytes] = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"output", "value"}:
                raise _integrity("Sensitivity VJP weight row is malformed.")
            output_coordinate(row.get("output"))
            key = _canonical_bytes(row["output"])
            if key in observed_outputs:
                raise _integrity("Sensitivity VJP repeats an output coordinate.")
            derived_scalar(row.get("value"))
            observed_outputs.add(key)
    else:
        raise _integrity("Sensitivity derivative operation is unknown.")

    if derivative_kind == "vjp":
        if spec.get("functional_unit") is not None and (not isinstance(spec["functional_unit"], str) or not spec["functional_unit"]):
            raise _integrity("Sensitivity VJP functional unit is malformed.")
    elif spec.get("functional_unit") is not None:
        raise _integrity("Only VJP requests may bind a functional unit.")

    discretization = spec.get("discretization")
    if discretization is not None:
        if not isinstance(discretization, Mapping) or set(discretization) != {"type", "resolution", "sections"} or discretization.get("type") != "fixed_mesh":
            raise _integrity("Sensitivity fixed-mesh declaration is malformed.")
        resolution, sections = discretization.get("resolution"), discretization.get("sections")
        if (resolution is None) == (sections is None):
            raise _integrity("Sensitivity fixed mesh must declare one resolution form.")
        if resolution is not None:
            if (not isinstance(resolution, Mapping)
                    or set(resolution) != {"kind", "max_frequency", "sections_per_wavelength"}
                    or resolution.get("kind") != "electrical_resolution"):
                raise _integrity("Sensitivity electrical resolution is malformed.")
            _verify_quantity_role(resolution.get("max_frequency"), complex_value=False, unit="hertz", dimensionality="inverse_time")
            count = resolution.get("sections_per_wavelength")
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise _integrity("Sensitivity electrical resolution count is malformed.")
        if sections is not None:
            if not isinstance(sections, list):
                raise _integrity("Sensitivity fixed sections are malformed.")
            prior_path = None
            for row in sections:
                if (not isinstance(row, Mapping) or set(row) != {"path", "count"}
                        or not isinstance(row.get("path"), list) or not row["path"]
                        or any(not isinstance(part, str) or _IDENTIFIER.fullmatch(part) is None for part in row["path"])
                        or not isinstance(row.get("count"), int) or isinstance(row["count"], bool) or row["count"] < 1):
                    raise _integrity("Sensitivity fixed section row is malformed.")
                path = tuple(row["path"])
                if prior_path is not None and path <= prior_path:
                    raise _integrity("Sensitivity fixed section paths are not ordered and unique.")
                prior_path = path

    scales = spec.get("scales")
    if scales is not None:
        if not isinstance(scales, Mapping) or set(scales) != {"inputs", "outputs"} or not isinstance(scales.get("inputs"), list) or not isinstance(scales.get("outputs"), list):
            raise _integrity("Sensitivity normalization scales are malformed.")
        seen_inputs: set[tuple[str, str]] = set()
        for row in scales["inputs"]:
            if not isinstance(row, Mapping) or set(row) != {"parameter", "value"}:
                raise _integrity("Sensitivity input scale row is malformed.")
            key, unit, dimension = parameter_role(row.get("parameter"))
            if key not in wrt_keys or key in seen_inputs:
                raise _integrity("Sensitivity input scale does not match its WRT declaration.")
            derived_scalar(row.get("value"), expected_role=(unit, dimension))
            seen_inputs.add(key)
        seen_outputs: set[bytes] = set()
        for row in scales["outputs"]:
            if not isinstance(row, Mapping) or set(row) != {"output", "value"}:
                raise _integrity("Sensitivity output scale row is malformed.")
            unit, dimension, _, _ = output_coordinate(row.get("output"))
            key = _canonical_bytes(row["output"])
            if key in seen_outputs:
                raise _integrity("Sensitivity output scales repeat a coordinate.")
            derived_scalar(row.get("value"), expected_role=(unit, dimension))
            seen_outputs.add(key)

def _verify_parameter_source(source: object, plan: Mapping[str, object]) -> None:
    if not isinstance(source, dict):
        raise _integrity("Parameter source is malformed.")
    try:
        from ...execution.request import canonical_parameter_source

        if canonical_parameter_source(source) != source:
            raise ValueError("noncanonical parameter source")
    except Exception as error:
        raise _integrity("Parameter source is open or noncanonical.") from error
    definitions = plan.get("parameter_closure", {}).get("definitions")
    if not isinstance(definitions, list):
        raise _integrity("Plan parameter closure is malformed.")
    expected = {
        _parameter_key_integrity({
            "definitions_id": item.get("definitions_id"),
            "parameter_id": item.get("parameter_id"),
        })
        for item in definitions
        if isinstance(item, Mapping)
    }
    if len(expected) != len(definitions):
        raise _integrity("Plan parameter definitions are malformed.")

    def parameter_set(value: object, *, complete: bool) -> None:
        _verify_parameter_set_document(value)
        keys = {_parameter_key_integrity(item["parameter"]) for item in value["bindings"]}
        if not keys <= expected or (complete and keys != expected):
            raise _integrity("Parameter source does not bind the Plan's consumed definitions.")

    kind = source["kind"]
    if kind == "point":
        parameter_set(source["parameters"], complete=True)
    elif kind == "grid":
        parameter_set(source["base_parameters"], complete=True)
        axis_keys = []
        for axis in source["axes"]:
            key = _parameter_key_integrity(axis["parameter"])
            if key not in expected:
                raise _integrity("Grid axis does not bind a consumed Plan parameter.")
            axis_keys.append(key)
            for value in axis["values"]:
                _verify_parameter_value(value)
        if len(set(axis_keys)) != len(axis_keys):
            raise _integrity("Grid axes repeat a parameter.")
    elif kind == "points":
        parameter_set(source["baseline_parameters"], complete=True)
        for point in source["points"]:
            parameter_set(point, complete=False)

def _verify_view_declaration(
    view: object,
    plan: Mapping[str, object],
    *, static_capacitance: bool = False,
) -> tuple[list[str], bool]:
    if not isinstance(view, dict) or set(view) != {"type", "ptc", "transforms", "retain"} or view.get("type") != "network_view":
        raise _integrity("Declarative View is malformed.")
    _, public = _plan_coordinates(plan)
    available = list(sorted(public))
    connectivity = plan.get("connectivity")
    ports = connectivity.get("ports") if isinstance(connectivity, Mapping) else None
    if not isinstance(ports, list):
        raise _integrity("Plan Port inventory is malformed.")
    port_ids = [port.get("id") for port in ports]
    if any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in port_ids) or len(set(port_ids)) != len(port_ids):
        raise _integrity("Plan Port IDs are malformed.")
    net_to_compiler = {
        node["final_net"]: node["compiler_node_id"]
        for node in connectivity["node_coordinates"]
    }
    port_coordinates = {
        net_to_compiler[port["net"]]
        for port in ports
        if port.get("net") in net_to_compiler
    }
    ptc = view.get("ptc")
    if ptc is not None:
        if not isinstance(ptc, dict) or set(ptc) != {"selected_ports"}:
            raise _integrity("View PTC declaration is malformed.")
        selected = ptc["selected_ports"]
        if not isinstance(selected, list) or not selected or selected != [item for item in port_ids if item in selected] or len(set(selected)) != len(selected):
            raise _integrity("View PTC Port order is malformed.")
        by_id = {port["id"]: port for port in ports}
        if any(by_id[item].get("role") != "nonloading_probe" for item in selected):
            raise _integrity("View PTC selects a loading Port.")
    transforms = view.get("transforms")
    if not isinstance(transforms, list):
        raise _integrity("View transforms are malformed.")
    transform_ids: set[str] = set()
    for transform in transforms:
        if not isinstance(transform, dict) or set(transform) != {"id", "input_coordinates", "output_coordinates"}:
            raise _integrity("View transform is malformed.")
        identifier = transform.get("id")
        inputs = transform.get("input_coordinates")
        outputs = transform.get("output_coordinates")
        expected_outputs = [f"{identifier}.common", f"{identifier}.differential"]
        if (
            not isinstance(identifier, str)
            or _IDENTIFIER.fullmatch(identifier) is None
            or identifier in transform_ids
            or not isinstance(inputs, list)
            or len(inputs) != 2
            or len(set(inputs)) != 2
            or any(item not in available for item in inputs)
            or outputs != expected_outputs
            or any(item in available for item in expected_outputs)
        ):
            raise _integrity("View transform basis transition is malformed.")
        transform_ids.add(identifier)
        index = min(available.index(item) for item in inputs)
        both_ports = all(item in port_coordinates for item in inputs)
        available = [item for item in available if item not in inputs]
        available[index:index] = expected_outputs
        port_coordinates.difference_update(inputs)
        if both_ports:
            port_coordinates.update(expected_outputs)
    retain = view.get("retain")
    if retain is None:
        return (available, False) if static_capacitance else (list(port_ids), bool(port_ids))
    if not isinstance(retain, dict) or set(retain) != {"retained_coordinates"}:
        raise _integrity("View retain declaration is malformed.")
    retained = retain["retained_coordinates"]
    if not isinstance(retained, list) or not retained or len(set(retained)) != len(retained) or any(item not in available for item in retained):
        raise _integrity("View retained basis is malformed.")
    return list(retained), all(item in port_coordinates for item in retained)

def _verify_v1_lineage(lineage: object, plan: Mapping[str, object] | None) -> tuple[list[str], bool]:
    """Close the full dev5 View grammar without reimplementing compilation.

    Matrix bytes are compiler-owned evidence; the workspace binds their hashes,
    ordering and applicability rather than manufacturing a second compiler in
    Python.
    """

    if not isinstance(lineage, dict) or set(lineage) != {
        "type", "original", "ptc", "transforms", "retain", "terminal_coordinates", "port_realizable", "lineage_sha256",
    } or lineage.get("type") != "network_view_lineage":
        raise _integrity("View lineage envelope is open or malformed.")
    if lineage.get("lineage_sha256") != _sha256(_canonical_bytes({key: value for key, value in lineage.items() if key != "lineage_sha256"})):
        raise _integrity("View lineage hash does not bind its contents.")
    original = lineage.get("original")
    if not isinstance(original, dict) or set(original) != {"type", "compiled_graph_sha256", "coordinate_order", "port_order", "port_realizable"} or original.get("type") != "original":
        raise _integrity("Original View lineage is malformed.")
    _valid_sha(original.get("compiled_graph_sha256"))
    original_coordinates = _identifiers(original.get("coordinate_order"), field="Original coordinate order")
    if plan is not None and original_coordinates != _plan_coordinates(plan)[0]:
        raise _integrity("Original View coordinate order disagrees with the sealed Plan.")
    connectivity = plan.get("connectivity") if plan is not None else None
    plan_ports = connectivity.get("ports") if isinstance(connectivity, Mapping) else None
    if plan is not None and not isinstance(plan_ports, list):
        raise _integrity("Sealed Plan ports are malformed.")
    expected_ports = [port.get("id") for port in plan_ports if isinstance(port, dict)] if isinstance(plan_ports, list) else None
    port_roles = {
        port.get("id"): port.get("role")
        for port in plan_ports or ()
        if isinstance(port, dict)
    }
    port_order = _identifiers(original.get("port_order"), field="Original Port order", nonempty=False)
    if (expected_ports is not None and port_order != expected_ports) or not isinstance(original.get("port_realizable"), bool):
        raise _integrity("Original View Port identity disagrees with the sealed Plan.")
    coordinates = list(original_coordinates)
    ptc = lineage.get("ptc")
    if ptc is not None:
        if not isinstance(ptc, dict) or set(ptc) != {"type", "selected_ports", "load_mask_sha256", "loads", "reconstruction_residual_f64", "output_coordinate_order", "evidence_sha256"} or ptc.get("type") != "ptc":
            raise _integrity("PTC lineage step is malformed.")
        selected = _identifiers(ptc.get("selected_ports"), field="PTC selected Ports")
        if any(port not in port_order for port in selected) or ptc.get("output_coordinate_order") != coordinates:
            raise _integrity("PTC lineage does not preserve the original coordinate basis.")
        if any(port_roles.get(port) != "nonloading_probe" for port in selected):
            raise _integrity("PTC selects a Port that is not a nonloading probe.")
        if not isinstance(ptc.get("loads"), list) or [item.get("port_id") if isinstance(item, dict) else None for item in ptc["loads"]] != selected:
            raise _integrity("PTC load evidence does not match its selected Port order.")
        for item in ptc["loads"]:
            if not isinstance(item, dict) or set(item) != {"port_id", "reference_impedance", "before", "after"} or item.get("before") != "raw" or item.get("after") != "compensated":
                raise _integrity("PTC load evidence is malformed.")
            _verify_quantity_role(item.get("reference_impedance"), complex_value=False, unit="ohm", dimensionality="resistance")
        _valid_sha(ptc.get("load_mask_sha256")); _valid_sha(ptc.get("evidence_sha256")); _f64_value(ptc.get("reconstruction_residual_f64"))
    transforms = lineage.get("transforms")
    if not isinstance(transforms, list):
        raise _integrity("Transform lineage must be an array.")
    for step in transforms:
        fields = {"type", "input_coordinates", "weights_f64", "differential_id", "common_id", "included_external_cut_branches", "excluded_direct_mutual_branches", "reference_matrix", "principal_root", "reconstruction_residual_f64", "output_coordinate_order", "evidence_sha256"}
        if not isinstance(step, dict) or set(step) != fields or step.get("type") != "transform_pair":
            raise _integrity("Transform lineage step is malformed.")
        pair = _identifiers(step.get("input_coordinates"), field="Transform input coordinates")
        if len(pair) != 2 or any(item not in coordinates for item in pair):
            raise _integrity("Transform input coordinates are not an ordered current pair.")
        weights = step.get("weights_f64")
        if not isinstance(weights, list) or len(weights) != 2 or any(not _finite_f64(item) for item in weights):
            raise _integrity("Transform weights are malformed.")
        common, differential = step.get("common_id"), step.get("differential_id")
        if any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in (common, differential)) or differential == common or differential in coordinates or common in coordinates:
            raise _integrity("Transform output coordinate identity is malformed.")
        expected = [item for item in coordinates if item not in pair] + [common, differential]
        if step.get("output_coordinate_order") != expected:
            raise _integrity("Transform output ordering is not canonical.")
        for matrix in ("reference_matrix", "principal_root"):
            evidence = step.get(matrix)
            if not isinstance(evidence, dict) or set(evidence) != {"rows", "columns", "sha256"} or any(not isinstance(evidence.get(field), int) or evidence[field] < 0 for field in ("rows", "columns")):
                raise _integrity("Transform matrix evidence is malformed.")
            _valid_sha(evidence.get("sha256"))
        _verify_branch_refs(step.get("included_external_cut_branches"), field="Transform included cut branches", nonempty=True)
        _verify_branch_refs(step.get("excluded_direct_mutual_branches"), field="Transform excluded direct-mutual branches", nonempty=False)
        _valid_sha(step.get("evidence_sha256")); _f64_value(step.get("reconstruction_residual_f64"))
        coordinates = expected
    retain = lineage.get("retain")
    if retain is not None:
        fields = {"type", "retained_coordinates", "eliminated_coordinates", "output_coordinate_order", "a_matrix", "b_matrix", "r_matrix", "d_matrix", "q_matrix", "selected_projector", "omitted_projector", "omitted_matched_loads", "source_boundary_sha256", "deembedding_evidence_sha256"}
        if not isinstance(retain, dict) or set(retain) != fields or retain.get("type") != "retain":
            raise _integrity("Retain lineage step is malformed.")
        retained = _identifiers(retain.get("retained_coordinates"), field="Retained coordinates")
        eliminated = _identifiers(retain.get("eliminated_coordinates"), field="Eliminated coordinates", nonempty=False)
        if set(retained) | set(eliminated) != set(coordinates) or set(retained) & set(eliminated) or retain.get("output_coordinate_order") != retained:
            raise _integrity("Retain lineage is not an exact partition of its input basis.")
        for matrix in ("a_matrix", "b_matrix", "r_matrix", "d_matrix", "q_matrix", "selected_projector", "omitted_projector", "omitted_matched_loads"):
            evidence = retain.get(matrix)
            if not isinstance(evidence, dict) or set(evidence) != {"rows", "columns", "sha256"}:
                raise _integrity("Retain matrix evidence is malformed.")
            _valid_sha(evidence.get("sha256"))
        _valid_sha(retain.get("source_boundary_sha256")); _valid_sha(retain.get("deembedding_evidence_sha256"))
        coordinates = retained
    terminal = _identifiers(lineage.get("terminal_coordinates"), field="Terminal channel order")
    port_realizable = lineage.get("port_realizable")
    # The compiler's original basis is physical-node ordered, whereas the raw
    # public Direct boundary is the declared logical-Port order.  Transforms
    # alter quantity coordinates but do not themselves create a wave boundary;
    # only terminal retain() selects transformed channel IDs.
    expected_terminal = coordinates if retain is not None else port_order
    if terminal != expected_terminal or not isinstance(port_realizable, bool):
        raise _integrity("Terminal View capability disagrees with its lineage.")
    return terminal, port_realizable

def _verify_v1_direct_spec(spec: object, terminal: list[str], port_realizable: bool) -> None:
    if not port_realizable:
        raise _integrity("Direct S/Y/Z request is not Port-realizable.")
    if not isinstance(spec, dict) or set(spec) != {"type", "frequencies", "traces"} or spec.get("type") != "direct_solve":
        raise _integrity("Direct solve Spec is malformed.")
    frequencies = spec.get("frequencies")
    if not isinstance(frequencies, list) or not frequencies:
        raise _integrity("Direct solve frequency grid is malformed.")
    for frequency in frequencies:
        _verify_quantity_role(frequency, complex_value=False, unit="hertz", dimensionality="inverse_time")
        value = _f64_value(frequency["si_value_f64"])
        if value <= 0.0:
            raise _integrity("Direct solve frequency must be strictly positive.")
    traces = spec.get("traces")
    if not isinstance(traces, list):
        raise _integrity("Direct trace declarations are malformed.")
    trace_ids: set[str] = set()
    for trace in traces:
        if (
            not isinstance(trace, dict)
            or set(trace) != {"id", "input_port", "input_mode", "output_port", "output_mode"}
            or not isinstance(trace.get("id"), str)
            or _IDENTIFIER.fullmatch(trace["id"]) is None
            or trace["id"] in trace_ids
            or trace.get("input_port") not in terminal
            or trace.get("output_port") not in terminal
            or trace.get("input_mode") != []
            or trace.get("output_mode") != []
        ):
            raise _integrity("Direct trace declaration is malformed.")
        trace_ids.add(trace["id"])

def _verify_v1_hb_spec(
    spec: object,
    terminal: list[str],
    port_realizable: bool,
    logical_ports: list[str],
) -> None:
    """Close the declared HB request before workspace evidence is allocated.

    Lattice realization remains Julia-owned, but a sealed request must already
    bind a complete, unique authoring declaration to a port-realizable View.
    """

    if not port_realizable:
        raise _integrity("HB S/Y/Z request is not Port-realizable.")
    expected = {
        "type", "pump_axes", "drives", "frequencies", "cases", "truncation",
        "traces", "allow_driven_ptc",
    }
    if not isinstance(spec, dict) or set(spec) != expected or spec.get("type") != "hb_solve":
        raise _integrity("HB solve Spec is malformed.")
    axes = spec.get("pump_axes")
    drives = spec.get("drives")
    cases = spec.get("cases")
    traces = spec.get("traces")
    frequencies = spec.get("frequencies")
    truncation = spec.get("truncation")
    if not isinstance(axes, list) or not isinstance(drives, list) or not isinstance(cases, list) or not cases or not isinstance(traces, list) or not isinstance(frequencies, list) or not frequencies or not isinstance(truncation, dict) or not isinstance(spec.get("allow_driven_ptc"), bool):
        raise _integrity("HB solve Spec has incomplete declarations.")
    axis_ids: set[str] = set()
    for axis in axes:
        if not isinstance(axis, dict) or set(axis) != {"id", "frequency"} or not isinstance(axis.get("id"), str) or _IDENTIFIER.fullmatch(axis["id"]) is None or axis["id"] in axis_ids:
            raise _integrity("HB pump-axis declaration is malformed.")
        _verify_quantity_role(axis.get("frequency"), complex_value=False, unit="hertz", dimensionality="inverse_time")
        if _f64_value(axis["frequency"]["si_value_f64"]) <= 0.0:
            raise _integrity("HB pump-axis frequency is not strictly positive.")
        axis_ids.add(axis["id"])
    previous = 0.0
    for frequency in frequencies:
        _verify_quantity_role(frequency, complex_value=False, unit="hertz", dimensionality="inverse_time")
        value = _f64_value(frequency["si_value_f64"])
        if value <= previous:
            raise _integrity("HB response frequency grid is not strictly positive and increasing.")
        previous = value
    drive_ids: set[str] = set()
    for drive in drives:
        if (
            not isinstance(drive, dict)
            or set(drive) != {"id", "port_id", "mode", "orientation"}
            or not isinstance(drive.get("id"), str)
            or _IDENTIFIER.fullmatch(drive["id"]) is None
            or drive["id"] in drive_ids
            or drive.get("port_id") not in logical_ports
            or drive.get("orientation") != "port_node_to_reference"
            or not _valid_mode_tuple(drive.get("mode"), len(axis_ids))
        ):
            raise _integrity("HB current-drive declaration is malformed.")
        drive_ids.add(drive["id"])
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "currents"} or not isinstance(case.get("id"), str) or _IDENTIFIER.fullmatch(case["id"]) is None or case["id"] in case_ids or not isinstance(case.get("currents"), list):
            raise _integrity("HB case declaration is malformed.")
        current_ids: set[str] = set()
        for current in case["currents"]:
            if (
                not isinstance(current, dict)
                or set(current) != {"drive_id", "coefficient", "coefficient_convention"}
                or current.get("drive_id") not in drive_ids
                or current["drive_id"] in current_ids
                or current.get("coefficient_convention") != "exp_minus_i_m_dot_omega_t_fourier_coefficient"
            ):
                raise _integrity("HB case current binding is malformed.")
            _verify_quantity_role(current.get("coefficient"), complex_value=True, unit="ampere", dimensionality="current")
            current_ids.add(current["drive_id"])
        case_ids.add(case["id"])
    expected_truncation = {"pump_harmonics", "modulation_harmonics", "max_intermodulation_order", "three_wave_mixing", "four_wave_mixing"}
    if set(truncation) != expected_truncation or not isinstance(truncation.get("pump_harmonics"), list) or not isinstance(truncation.get("modulation_harmonics"), list) or len(truncation["pump_harmonics"]) != len(axis_ids) or len(truncation["modulation_harmonics"]) != len(axis_ids) or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in [*truncation["pump_harmonics"], *truncation["modulation_harmonics"]]) or (truncation.get("max_intermodulation_order") is not None and (not isinstance(truncation["max_intermodulation_order"], int) or isinstance(truncation["max_intermodulation_order"], bool) or truncation["max_intermodulation_order"] < 0)) or not isinstance(truncation.get("three_wave_mixing"), bool) or not isinstance(truncation.get("four_wave_mixing"), bool):
        raise _integrity("HB truncation declaration is malformed.")
    trace_ids: set[str] = set()
    for trace in traces:
        if (
            not isinstance(trace, dict)
            or set(trace) != {"id", "input_port", "input_mode", "output_port", "output_mode"}
            or not isinstance(trace.get("id"), str)
            or _IDENTIFIER.fullmatch(trace["id"]) is None
            or trace["id"] in trace_ids
            or trace.get("input_port") not in terminal
            or trace.get("output_port") not in terminal
            or not _valid_mode_tuple(trace.get("input_mode"), len(axis_ids))
            or not _valid_mode_tuple(trace.get("output_mode"), len(axis_ids))
        ):
            raise _integrity("HB trace declaration is malformed.")
        trace_ids.add(trace["id"])

def _valid_mode_tuple(value: object, rank: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == rank
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )

def _hb_operating_lattice_is_vacuous(spec: Mapping[str, object]) -> bool:
    """Reproduce the pinned JC empty operating-basis condition from the request.

    This delegates to the full pinned ordering reconstruction below, so its
    vacuity rule cannot drift from the actual RFFT/parity/crop construction.
    """

    return not _hb_declared_modes_from_spec(spec, response=False)

def _hb_declared_modes_from_spec(spec: Mapping[str, object], *, response: bool) -> list[list[int]]:
    """Mirror the pinned JC 0.5.4 Fourier construction and its ordering.

    The receipt does not merely attest that a returned set fits the requested
    bounds.  `calcfreqsrdft`/`calcfreqsdft`, `truncfreqs`, and (for the
    operating basis) `removeconjfreqs` determine the ordered public channel
    basis.  Reconstructing it here closes result reuse against a backend that
    has silently permuted an otherwise valid lattice.
    """

    axes = spec.get("pump_axes")
    truncation = spec.get("truncation")
    drives = spec.get("drives")
    if not isinstance(axes, list) or not isinstance(truncation, Mapping) or not isinstance(drives, list):
        raise _integrity("HB request cannot reproduce its pinned JC lattice.")
    rank = len(axes)
    limits_key = "modulation_harmonics" if response else "pump_harmonics"
    limits = truncation.get(limits_key)
    if (
        not isinstance(limits, list)
        or len(limits) != rank
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in limits)
        or not isinstance(truncation.get("three_wave_mixing"), bool)
        or not isinstance(truncation.get("four_wave_mixing"), bool)
    ):
        raise _integrity("HB request has an invalid JC lattice truncation.")
    crop = truncation.get("max_intermodulation_order")
    if crop is not None and (not isinstance(crop, int) or isinstance(crop, bool) or crop < 0):
        raise _integrity("HB request has an invalid JC intermodulation crop.")
    declared_dc = any(
        isinstance(drive, Mapping)
        and isinstance(drive.get("mode"), list)
        and len(drive["mode"]) == rank
        and all(value == 0 for value in drive["mode"])
        for drive in drives
    )
    if rank == 0:
        # This is SCNSim's documented private JC adapter: the backend uses an
        # inert `(0,)`, while the public rank-zero basis is `()`.
        return [[]] if response or declared_dc else []

    # Julia CartesianIndices is column-major: the first pump axis advances
    # first.  Iterate reversed Python products to preserve that exact order.
    dimensions = [2 * limit + 1 for limit in limits] if response else [limits[0] + 1, *[2 * limit + 1 for limit in limits[1:]]]
    modes: list[tuple[int, ...]] = []
    for reversed_indices in product(*(range(1, dimension + 1) for dimension in reversed(dimensions))):
        indices = tuple(reversed(reversed_indices))
        mode = tuple(
            index - 1 if index <= limit + 1 else -dimension + index - 1
            for index, limit, dimension in zip(indices, limits, dimensions)
        )
        absolute_order = sum(abs(value) for value in mode)
        criterion = (
            (response and all(value == 0 for value in mode))
            or ((truncation["four_wave_mixing"] if response else truncation["three_wave_mixing"]) and absolute_order > 0 and absolute_order % 2 == 0)
            or ((truncation["three_wave_mixing"] if response else truncation["four_wave_mixing"]) and absolute_order % 2 == 1)
            or (not response and declared_dc and all(value == 0 for value in mode))
        )
        if criterion and (sum(value != 0 for value in mode) == 1 or crop is None or absolute_order <= crop):
            modes.append(mode)
    if response:
        return [list(mode) for mode in modes]

    # `removeconjfreqs` removes the lexicographically greater coordinate of
    # every JC-conjugate pair and retains original Cartesian order.
    nw = tuple(dimensions)
    nt = (2 * nw[0] - 1, *nw[1:])
    removed: set[tuple[int, ...]] = set()
    for reversed_indices in product(*(range(1, dimension + 1) for dimension in reversed(nw))):
        coordinate = tuple(reversed(reversed_indices))
        target = tuple((length - (index - 1)) % length + 1 for index, length in zip(coordinate, nt))
        if coordinate != target and all(index <= dimension for index, dimension in zip(target, nw)):
            removed.add(max(coordinate, target))
    retained: list[list[int]] = []
    for reversed_indices in product(*(range(1, dimension + 1) for dimension in reversed(nw))):
        coordinate = tuple(reversed(reversed_indices))
        if coordinate in removed:
            continue
        mode = tuple(
            index - 1 if index <= limit + 1 else -dimension + index - 1
            for index, limit, dimension in zip(coordinate, limits, dimensions)
        )
        if mode in modes:
            retained.append(list(mode))
    return retained

def _verify_root_search(search: object) -> None:
    """Validate closed caller discovery facts, not numerical root success."""
    if not isinstance(search, dict):
        raise _integrity("Root search declaration is malformed.")
    method = search.get("method")
    if method not in {"polynomial_transfer_zero", "finite_seeds"}:
        raise _integrity("Root search method is malformed.")
    fields = {"method", "region", "selection", "tracking"}
    if method == "finite_seeds":
        fields.add("seeds")
    if set(search) != fields:
        raise _integrity("Root search declaration is open or malformed.")
    region = search["region"]
    if not isinstance(region, dict) or not set(region) <= {"real_frequency_interval", "hint_frequency_factor_f64", "imag_frequency_interval"}:
        raise _integrity("Root search region is malformed.")
    if ("real_frequency_interval" in region) == ("hint_frequency_factor_f64" in region):
        raise _integrity("Root search requires exactly one real-region mode.")
    if "hint_frequency_factor_f64" in region:
        factor = region["hint_frequency_factor_f64"]
        if not _finite_f64(factor) or _f64_value(factor) <= 0:
            raise _integrity("Root search Hint factor encoding is malformed.")
    for axis in ("real_frequency_interval", "imag_frequency_interval"):
        if axis not in region:
            continue
        interval = region[axis]
        if not isinstance(interval, list) or len(interval) != 2:
            raise _integrity("Root search interval is malformed.", axis=axis)
        for value in interval:
            _verify_quantity_role(value, complex_value=False, unit="hertz", dimensionality="inverse_time")
        if _f64_value(interval[0]["si_value_f64"]) > _f64_value(interval[1]["si_value_f64"]):
            raise _integrity("Root search interval is reversed.", axis=axis)
    if method == "finite_seeds":
        seeds = search["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise _integrity("Root search trial seeds are malformed.")
        for value in seeds:
            _verify_quantity_role(value, complex_value=True, unit="hertz", dimensionality="inverse_time")
    selection = search["selection"]
    if (not isinstance(selection, dict) or set(selection) != {"kind", "metric", "tie"}
            or selection["kind"] != "nearest_hint" or selection["metric"] != "complex_frequency_euclidean"
            or selection["tie"] not in {"unavailable", "source_order"}):
        raise _integrity("Root search selection is malformed.")
    if search["tracking"] != {"kind": "baseline_anchor"}:
        raise _integrity("Root search tracking is malformed.")


def _verify_v1_evaluation_spec(
    spec: object,
    terminal: list[str],
    port_realizable: bool,
    *,
    residue_branch: bool = False,
    static_retained: bool = False,
) -> None:
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
        raise _integrity("Direct evaluation Spec is malformed.")
    kind = spec["type"]
    if kind in {"diagonal_root", "operator_element_root", "hybridized_pole", "transfer_zero"} and "search" in spec:
        _verify_root_search(spec["search"])
    if kind == "diagonal_root":
        coordinate = spec.get("coordinate")
        invalid = coordinate not in terminal or (residue_branch and len(terminal) < 2)
        if set(spec) - {"search"} != {"type", "coordinate", "root_hint"} or invalid:
            raise _integrity("Diagonal-root Spec is incompatible with its final View.")
        _verify_frequency_anchor(spec.get("root_hint"))
        hint = spec["root_hint"]
        real_key = "si_value_f64" if hint["type"] == "quantity_f64" else "real_si_f64"
        if _f64_value(hint[real_key]) <= 0.0:
            raise _integrity("Diagonal-root hint must have a positive real part.")
    elif kind == "operator_element_root":
        if set(spec) - {"search"} != {"type", "row", "column", "root_hint"} or spec.get("row") not in terminal or spec.get("column") not in terminal:
            raise _integrity("Operator-element-root Spec is incompatible with its final View.")
        _verify_quantity_role(spec.get("root_hint"), complex_value=False, unit="hertz", dimensionality="inverse_time")
        if _f64_value(spec["root_hint"]["si_value_f64"]) <= 0.0:
            raise _integrity("Operator-element-root hint must be positive.")
    elif kind == "hybridized_pole":
        if set(spec) - {"search"} != {"type", "coordinates", "anchor"} or len(terminal) < 2 or _identifiers(spec.get("coordinates"), field="Hybridized-pole coordinates") != terminal:
            raise _integrity("Hybridized-pole coordinates must equal the complete retained View.")
        _verify_frequency_anchor(spec.get("anchor"))
    elif kind == "transfer_zero":
        if set(spec) - {"search"} != {"type", "anchor", "family", "input_coordinate", "output_coordinate"} or spec.get("family") not in {"S", "Y", "Z"} or spec.get("input_coordinate") not in terminal or spec.get("output_coordinate") not in terminal:
            raise _integrity("Transfer-zero Spec is malformed.")
        if spec.get("family") == "S" and not port_realizable:
            raise _integrity("S-family transfer-zero evaluation is not Port-realizable.")
        _verify_frequency_anchor(spec.get("anchor"))
    elif kind == "residue_normalized_coupling":
        branches = (spec.get("branch_a"), spec.get("branch_b"))
        if (
            set(spec) != {"type", "branch_a", "branch_b", "frequency"}
            or any(not isinstance(branch, dict) or branch.get("type") not in {"diagonal_root", "hybridized_pole"} for branch in branches)
        ):
            raise _integrity("Residue-normalized coupling Spec is malformed.")
        _verify_v1_evaluation_spec(branches[0], terminal, port_realizable, residue_branch=True)
        _verify_v1_evaluation_spec(branches[1], terminal, port_realizable, residue_branch=True)
        frequency = spec.get("frequency")
        if frequency == "complex_root_midpoint":
            pass
        else:
            _verify_quantity_role(frequency, complex_value=False, unit="hertz", dimensionality="inverse_time")
            if _f64_value(frequency["si_value_f64"]) <= 0.0:
                raise _integrity("Residue coupling frequency must be positive.")
    elif kind == "response_element":
        if set(spec) != {"type", "family", "input_coordinate", "output_coordinate", "frequency"} or spec.get("family") not in {"S", "Y", "Z"} or spec.get("input_coordinate") not in terminal or spec.get("output_coordinate") not in terminal:
            raise _integrity("Response-element Spec is malformed.")
        if spec.get("family") == "S" and not port_realizable:
            raise _integrity("S-family response evaluation is not Port-realizable.")
        _verify_quantity_role(spec.get("frequency"), complex_value=False, unit="hertz", dimensionality="inverse_time")
    elif kind in {"static_capacitance", "static_capacitance_element"}:
        fields = {"type"}
        if kind == "static_capacitance_element":
            fields |= {"row", "column"}
        if set(spec) != fields:
            raise _integrity("Static capacitance Spec is open or malformed.")
        # An explicit retained View already owns a closed public-coordinate
        # boundary. Full-View finite-mesh membership remains compiler-owned.
        if kind == "static_capacitance_element":
            for name in ("row", "column"):
                _identifiers([spec[name]], field=f"Static capacitance {name}")
                if static_retained and spec[name] not in terminal:
                    raise _integrity("Static capacitance element is outside its retained View.")
    elif kind == "operator":
        if set(spec) != {"type", "frequencies"}:
            raise _integrity("Operator Spec is malformed.")
        _verify_v1_direct_spec({"type": "direct_solve", "frequencies": spec.get("frequencies"), "traces": []}, terminal, True)
    else:
        raise _integrity("Direct evaluation Spec is outside dev5.")

def _verify_frequency_anchor(value: object) -> None:
    if isinstance(value, dict) and value.get("type") == "quantity_f64":
        _verify_quantity_role(value, complex_value=False, unit="hertz", dimensionality="inverse_time")
    else:
        _verify_quantity_role(value, complex_value=True, unit="hertz", dimensionality="inverse_time")

def _verify_v1_optimization_spec(spec: object, plan: Mapping[str, object], initial_parameters: object) -> None:
    if not isinstance(spec, dict) or set(spec) != {"type", "variables", "objectives", "optimizer", "allow_extrapolation"} or spec.get("type") != "optimization":
        raise _integrity("Optimization Spec is malformed.")
    variables, objectives, optimizer, authorizations = spec.get("variables"), spec.get("objectives"), spec.get("optimizer"), spec.get("allow_extrapolation")
    if not isinstance(variables, list) or not variables or not isinstance(objectives, list) or not objectives or not isinstance(optimizer, dict) or not isinstance(authorizations, list):
        raise _integrity("Optimization Spec has malformed collections.")
    variable_keys: list[tuple[tuple[str, ...], str]] = []
    initial = {_parameter_key_integrity(binding["parameter"]): binding["value"]
        for binding in initial_parameters.get("bindings", [])} if isinstance(initial_parameters, Mapping) else {}
    for variable in variables:
        if not isinstance(variable, dict):
            raise _integrity("Optimization variable is malformed.")
        key = _parameter_key_integrity(variable.get("parameter")); variable_keys.append(key)
        if "domain" in variable:
            domain = variable.get("domain")
            linear = domain in {"UNBOUNDED", "NONNEGATIVE", "NONPOSITIVE"}
            signed = domain in {"POSITIVE", "NEGATIVE"}
            if set(variable) != {"parameter", "domain", "transform", "scale"} or not (linear or signed) or variable.get("transform") != ("linear" if linear else "log"):
                raise _integrity("Optimization domain variable is malformed.")
            scale = variable.get("scale")
            value = initial.get(key)
            if not isinstance(value, Mapping):
                raise _integrity("Optimization domain initial parameter is absent.")
            x0 = _f64_value(value.get("si_value_f64"))
            if (domain == "POSITIVE" and x0 <= 0.0 or domain == "NEGATIVE" and x0 >= 0.0 or
                domain == "NONNEGATIVE" and x0 < 0.0 or domain == "NONPOSITIVE" and x0 > 0.0):
                raise _integrity("Optimization domain initial value is outside its domain.")
            if linear:
                _verify_quantity_compatible(scale, value)
                magnitude = _f64_value(scale["si_value_f64"])
                if magnitude <= 0.0 or (domain != "UNBOUNDED" and not math.isfinite(x0 / magnitude)):
                    raise _integrity("Optimization domain scale must be positive.")
            elif scale is not None:
                raise _integrity("Signed log domain cannot have physical scale.")
            continue
        if set(variable) != {"parameter", "model_default_bounds", "consumer_override_bounds", "lower", "upper", "transform"} or variable.get("transform") not in {"linear", "log"}:
            raise _integrity("Optimization bounded variable is malformed.")
        for name in ("model_default_bounds", "consumer_override_bounds"):
            bounds = variable.get(name)
            if bounds is None and name == "consumer_override_bounds":
                continue
            _verify_bounds(bounds)
        _verify_quantity_compatible(variable.get("lower"), variable.get("upper"))
        if variable.get("consumer_override_bounds") is None:
            if variable.get("model_default_bounds") != [variable.get("lower"), variable.get("upper")]:
                raise _integrity("Optimization resolved bounds do not preserve model defaults.")
        elif variable.get("consumer_override_bounds") != [variable.get("lower"), variable.get("upper")]:
            raise _integrity("Optimization resolved bounds do not match consumer override.")
    if len(set(variable_keys)) != len(variable_keys):
        raise _integrity("Optimization variables are not unique.")
    authorization_keys = [_parameter_key_integrity(item) for item in authorizations]
    if authorization_keys != sorted(set(authorization_keys)) or any(key not in variable_keys for key in authorization_keys):
        raise _integrity("Optimization extrapolation authorization is not sorted active variables.")
    objective_ids: set[str] = set()
    for objective in objectives:
        if not isinstance(objective, dict) or set(objective) != {"id", "quantity", "comparison", "target", "weight_f64", "resolved_scale", "scale_source"} or not isinstance(objective.get("id"), str) or _IDENTIFIER.fullmatch(objective["id"]) is None or objective["id"] in objective_ids:
            raise _integrity("Optimization objective is malformed.")
        objective_ids.add(objective["id"])
        role = _verify_selector(objective.get("quantity"), plan)
        _verify_quantity_role(objective.get("target"), complex_value=False, unit=role[0], dimensionality=role[1])
        _verify_quantity_role(objective.get("resolved_scale"), complex_value=False, unit=role[0], dimensionality=role[1])
        resolved_scale = objective["resolved_scale"]
        weight = objective.get("weight_f64")
        if (
            not _finite_f64(weight)
            or _f64_value(weight) <= 0.0
            or _f64_value(resolved_scale["si_value_f64"]) <= 0.0
            or objective.get("comparison") not in {"target", "at_least"}
            or objective.get("scale_source") not in {"relative_target", "dimensionless_unity", "explicit"}
        ):
            raise _integrity("Optimization objective scale is malformed.")
    required_optimizer = {"type", "seed", "max_evaluations", "population_size", "resolved_population_size", "initial_sigma_f64", "baseline_optimizer_coordinates_f64", "box_transform_id", "complete_generations", "unused_evaluations", "hidden_stops"}
    expected_map = "cmaes-jl-0.2.6-native-domains.v1" if any("domain" in v for v in variables) else "cmaes-jl-0.2.6-linquad-unit-box.v1"
    if set(optimizer) != required_optimizer or optimizer.get("type") != "cma_es" or optimizer.get("box_transform_id") != expected_map or optimizer.get("hidden_stops") != "disabled":
        raise _integrity("Optimization controls are malformed.")
    baseline_coordinates = optimizer.get("baseline_optimizer_coordinates_f64")
    if (
        not isinstance(baseline_coordinates, list)
        or len(baseline_coordinates) != len(variables)
        or any(not _finite_f64(value) or (
            _f64_value(value) != 0.0 if "domain" in variable
            else not 0.0 <= _f64_value(value) <= 1.0
        ) for value, variable in zip(baseline_coordinates, variables))
    ):
        raise _integrity("Optimization baseline coordinates are malformed.")


def _verify_v1_jax_optimization_spec(
    spec: object, plan: Mapping[str, object], initial_parameters: object
) -> None:
    """Verify the current physical CMA/L-BFGS-B/Hybrid request grammar.

    The Julia optimization validator remains a separate source-owned contract.
    This validates the closed request serialization only; it does not initialize
    the numerical backend or derive candidate values.
    """
    if (not isinstance(spec, dict)
            or set(spec) not in (
                {"type", "variables", "objectives", "optimizer", "allow_extrapolation"},
                {"type", "variables", "objectives", "optimizer", "discretization", "allow_extrapolation"},
            )
            or spec.get("type") != "optimization"):
        raise _integrity("Physical Optimization Spec is open or malformed.")
    variables, objectives = spec.get("variables"), spec.get("objectives")
    optimizer, authorizations = spec.get("optimizer"), spec.get("allow_extrapolation")
    if (not isinstance(variables, list) or not variables
            or not isinstance(objectives, list) or not objectives
            or not isinstance(optimizer, dict) or not isinstance(authorizations, list)):
        raise _integrity("Physical Optimization Spec has malformed collections.")

    initial = ({_parameter_key_integrity(binding["parameter"]): binding["value"]
                for binding in initial_parameters.get("bindings", [])}
               if isinstance(initial_parameters, Mapping) else {})
    variable_keys = []
    for variable in variables:
        if not isinstance(variable, dict):
            raise _integrity("Physical Optimization variable is malformed.")
        key = _parameter_key_integrity(variable.get("parameter"))
        variable_keys.append(key)
        if "domain" in variable:
            if set(variable) != {"parameter", "domain", "transform", "scale"}:
                raise _integrity("Physical Optimization domain variable is open.")
            domain = variable.get("domain")
            linear = domain in {"UNBOUNDED", "NONNEGATIVE", "NONPOSITIVE"}
            signed = domain in {"POSITIVE", "NEGATIVE"}
            if not (linear or signed) or variable.get("transform") != ("linear" if linear else "log"):
                raise _integrity("Physical Optimization domain transform is malformed.")
            value = initial.get(key)
            if not isinstance(value, Mapping):
                raise _integrity("Physical Optimization initial parameter is absent.")
            x0 = _f64_value(value.get("si_value_f64"))
            if ((domain == "POSITIVE" and x0 <= 0.0)
                    or (domain == "NEGATIVE" and x0 >= 0.0)
                    or (domain == "NONNEGATIVE" and x0 < 0.0)
                    or (domain == "NONPOSITIVE" and x0 > 0.0)):
                raise _integrity("Physical Optimization initial value is outside its domain.")
            scale = variable.get("scale")
            if linear and scale is not None:
                _verify_quantity_compatible(scale, value)
                if _f64_value(scale["si_value_f64"]) <= 0.0:
                    raise _integrity("Physical Optimization domain scale is not positive.")
            elif not linear and scale is not None:
                raise _integrity("Signed log domain cannot carry a physical scale.")
            continue
        fields = {"parameter", "model_default_bounds", "consumer_override_bounds",
                  "lower", "upper", "transform"}
        if set(variable) != fields or variable.get("transform") not in {"linear", "log"}:
            raise _integrity("Physical Optimization bounded variable is malformed.")
        _verify_bounds(variable.get("model_default_bounds"))
        override = variable.get("consumer_override_bounds")
        if override is not None:
            _verify_bounds(override)
        _verify_quantity_compatible(variable.get("lower"), variable.get("upper"))
        if (override is None and variable.get("model_default_bounds") !=
                [variable.get("lower"), variable.get("upper")]):
            raise _integrity("Physical Optimization bounds differ from model defaults.")
        if (override is not None and override !=
                [variable.get("lower"), variable.get("upper")]):
            raise _integrity("Physical Optimization bounds differ from consumer override.")
    if len(set(variable_keys)) != len(variable_keys):
        raise _integrity("Physical Optimization variables are not unique.")
    active = set(variable_keys)
    auth_keys = [_parameter_key_integrity(item) for item in authorizations]
    if auth_keys != sorted(set(auth_keys)) or any(key not in active for key in auth_keys):
        raise _integrity("Physical Optimization extrapolation authorization is malformed.")

    objective_ids = set()
    for objective in objectives:
        required = {"id", "quantity", "comparison", "target", "weight_f64",
                    "resolved_scale", "scale_source"}
        if (not isinstance(objective, dict) or set(objective) != required
                or not isinstance(objective.get("id"), str)
                or _IDENTIFIER.fullmatch(objective["id"]) is None
                or objective["id"] in objective_ids):
            raise _integrity("Physical Optimization objective is malformed.")
        objective_ids.add(objective["id"])
        unit, dimensionality = _verify_selector(objective.get("quantity"), plan)
        _verify_quantity_role(objective.get("target"), complex_value=False,
                              unit=unit, dimensionality=dimensionality)
        _verify_quantity_role(objective.get("resolved_scale"), complex_value=False,
                              unit=unit, dimensionality=dimensionality)
        if (objective.get("comparison") not in {"target", "at_least"}
                or objective.get("scale_source") not in {
                    "relative_target", "dimensionless_unity", "explicit"}
                or not _finite_f64(objective.get("weight_f64"))
                or _f64_value(objective["weight_f64"]) <= 0.0
                or _f64_value(objective["resolved_scale"]["si_value_f64"]) <= 0.0):
            raise _integrity("Physical Optimization objective scale is malformed.")

    def verify_mesh(value):
        if value is None:
            return
        if (not isinstance(value, Mapping)
                or set(value) != {"type", "resolution", "sections"}
                or value.get("type") != "fixed_mesh"):
            raise _integrity("Physical Optimization fixed mesh is malformed.")
        resolution, sections = value.get("resolution"), value.get("sections")
        if (resolution is None) == (sections is None):
            raise _integrity("Physical Optimization mesh must declare one resolution form.")
        if resolution is not None:
            if (not isinstance(resolution, Mapping)
                    or set(resolution) != {"kind", "max_frequency", "sections_per_wavelength"}
                    or resolution.get("kind") != "electrical_resolution"):
                raise _integrity("Physical Optimization electrical resolution is malformed.")
            _verify_quantity_role(resolution.get("max_frequency"), complex_value=False,
                                  unit="hertz", dimensionality="inverse_time")
            count = resolution.get("sections_per_wavelength")
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise _integrity("Physical Optimization section resolution is malformed.")
        if sections is not None:
            if not isinstance(sections, list):
                raise _integrity("Physical Optimization fixed sections are malformed.")
            prior_path = None
            for row in sections:
                if (not isinstance(row, Mapping) or set(row) != {"path", "count"}
                        or not isinstance(row.get("path"), list) or not row["path"]
                        or any(not isinstance(part, str) or _IDENTIFIER.fullmatch(part) is None
                               for part in row["path"])
                        or not isinstance(row.get("count"), int)
                        or isinstance(row.get("count"), bool) or row["count"] < 1):
                    raise _integrity("Physical Optimization section entry is malformed.")
                path = tuple(row["path"])
                if prior_path is not None and path <= prior_path:
                    raise _integrity("Physical Optimization section paths are not ordered and unique.")
                prior_path = path

    if "discretization" in spec:
        verify_mesh(spec["discretization"])

    def verify_controls(control):
        if not isinstance(control, Mapping):
            raise _integrity("Physical Optimization stage controls are malformed.")
        kind = control.get("type")
        if kind == "cma_es":
            required = {"type", "seed", "max_evaluations", "population_size",
                        "resolved_population_size", "initial_sigma_f64",
                        "baseline_optimizer_coordinates_f64", "box_transform_id",
                        "complete_generations", "unused_evaluations", "hidden_stops"}
            if set(control) != required:
                raise _integrity("Physical CMA controls are open or malformed.")
            seed = control.get("seed")
            budget = control.get("max_evaluations")
            declared_pop = control.get("population_size")
            population = control.get("resolved_population_size")
            generations, unused = control.get("complete_generations"), control.get("unused_evaluations")
            expected_population = (declared_pop if declared_pop is not None else
                                  4 + math.floor(3 * math.log(len(variables))))
            if (not isinstance(seed, int) or isinstance(seed, bool) or not -(2**63) <= seed < 2**63
                    or not isinstance(budget, int) or isinstance(budget, bool) or budget < 1
                    or (declared_pop is not None and
                        (not isinstance(declared_pop, int) or isinstance(declared_pop, bool) or declared_pop < 2))
                    or population != expected_population
                    or not isinstance(population, int) or isinstance(population, bool)
                    or population < 2
                    or generations != (budget - 1) // population
                    or unused != budget - (1 + generations * population)
                    or not isinstance(generations, int) or isinstance(generations, bool) or generations < 0
                    or not isinstance(unused, int) or isinstance(unused, bool) or unused < 0
                    or control.get("hidden_stops") != "disabled"
                    or control.get("box_transform_id") != (
                        "scnsim-python-cmaes-0.13.1-native-domains.v1"
                        if any("domain" in variable for variable in variables)
                        else "scnsim-python-cmaes-0.13.1-linquad-unit-box.v1")):
                raise _integrity("Physical CMA controls are inconsistent.")
            sigma = control.get("initial_sigma_f64")
            if not _finite_f64(sigma) or _f64_value(sigma) <= 0.0:
                raise _integrity("Physical CMA sigma is malformed.")
            coordinates = control.get("baseline_optimizer_coordinates_f64")
            if (not isinstance(coordinates, list) or len(coordinates) != len(variables)
                    or any(not _finite_f64(value) for value in coordinates)):
                raise _integrity("Physical CMA baseline coordinates are malformed.")
            for coordinate, variable in zip(coordinates, variables):
                value = _f64_value(coordinate)
                if ("domain" in variable and value != 0.0) or ("domain" not in variable and not 0.0 <= value <= 1.0):
                    raise _integrity("Physical CMA baseline coordinate is outside its transform.")
            return
        if kind in {"lbfgsb", "projected_lbfgs"}:
            required = {"type", "max_evaluations", "max_iterations", "ftol_f64", "gtol_f64",
                        "max_line_search", "commit_every_iterations", "coordinate_scales"}
            if set(control) != required:
                raise _integrity("Physical L-BFGS-B controls are open or malformed.")
            for name in ("max_evaluations", "max_iterations", "max_line_search", "commit_every_iterations"):
                value = control.get(name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise _integrity("Physical L-BFGS-B integer control is malformed.", field=name)
            for name in ("ftol_f64", "gtol_f64"):
                value = control.get(name)
                if not _finite_f64(value) or _f64_value(value) < 0.0:
                    raise _integrity("Physical L-BFGS-B tolerance encoding is malformed.", field=name)
            scales = control.get("coordinate_scales")
            if scales is not None:
                if not isinstance(scales, list) or len(scales) != len(variables):
                    raise _integrity("Physical L-BFGS-B coordinate scales are malformed.")
                for scale, variable in zip(scales, variables):
                    if (not isinstance(scale, Mapping) or set(scale) != {"parameter", "value"}
                            or scale.get("parameter") != variable.get("parameter")):
                        raise _integrity("Physical L-BFGS-B coordinate scale order differs from variables.")
                    value = scale.get("value")
                    _verify_quantity_role(
                        value, complex_value=False,
                        unit=variable["lower"]["si_unit"] if "lower" in variable else initial[variable_keys[variables.index(variable)]]["si_unit"],
                        dimensionality=variable["lower"]["dimensionality"] if "lower" in variable else initial[variable_keys[variables.index(variable)]]["dimensionality"],
                    )
            return
        raise _integrity("Physical Optimization strategy is unsupported.", strategy=kind)

    optimizer_type = optimizer.get("type")
    if optimizer_type in {"cma_es", "lbfgsb", "projected_lbfgs"}:
        verify_controls(optimizer)
    elif optimizer_type == "hybrid":
        if set(optimizer) != {"type", "stages", "comparison_stage_id"}:
            raise _integrity("Physical Hybrid controls are open or malformed.")
        stages = optimizer.get("stages")
        stage_ids = []
        if not isinstance(stages, list) or not stages:
            raise _integrity("Physical Hybrid stages are malformed.")
        for stage in stages:
            if (not isinstance(stage, Mapping)
                    or set(stage) != {"id", "optimizer", "discretization", "seed"}
                    or not isinstance(stage.get("id"), str)
                    or _IDENTIFIER.fullmatch(stage["id"]) is None):
                raise _integrity("Physical Hybrid stage is malformed.")
            stage_ids.append(stage["id"])
            verify_controls(stage["optimizer"])
            verify_mesh(stage["discretization"])
            if stage.get("seed") is not None:
                _verify_parameter_set_document(stage["seed"])
        if (len(set(stage_ids)) != len(stage_ids)
                or optimizer.get("comparison_stage_id") not in stage_ids):
            raise _integrity("Physical Hybrid comparison stage is malformed.")
    else:
        raise _integrity("Physical Optimization strategy is unsupported.", strategy=optimizer_type)

def _optimization_selector_leaves(spec: object) -> list[Mapping[str, object]]:
    if not isinstance(spec, Mapping) or not isinstance(spec.get("objectives"), list):
        return []
    leaves: list[Mapping[str, object]] = []
    for objective in spec["objectives"]:
        quantity = objective.get("quantity") if isinstance(objective, Mapping) else None
        if isinstance(quantity, Mapping):
            leaves.extend(_selector_terms(quantity))
    return leaves

def _optimization_leaf_catalog(objectives: object) -> list[tuple[dict[str, object], Mapping[str, object]]]:
    if not isinstance(objectives, list):
        raise _integrity("Optimization objectives are malformed.")
    leaves: list[tuple[dict[str, object], Mapping[str, object]]] = []
    for objective in objectives:
        if not isinstance(objective, Mapping) or not isinstance(objective.get("id"), str):
            raise _integrity("Optimization objective identity is malformed.")
        for ordinal, selector in enumerate(_selector_terms(objective.get("quantity")), 1):
            leaves.append(({"objective_id": objective["id"], "term_ordinal": ordinal}, selector))
    return leaves

def _optimization_dependency(selector: Mapping[str, object], *, kind: str = "quantity") -> dict[str, object]:
    view = selector.get("view")
    if not isinstance(view, Mapping):
        raise _integrity("Optimization selector View is malformed.")
    view_sha = _sha256(_canonical_bytes(view))
    selector_type = selector.get("type")
    if selector_type in {"diagonal_root_projection", "residue_diagonal_root_projection", "operator_element_root_projection"}:
        spec = selector.get("spec")
        if not isinstance(spec, Mapping):
            raise _integrity("Optimization root selector is malformed.")
        row = spec.get("row") if selector_type == "operator_element_root_projection" else spec.get("coordinate")
        column = spec.get("column") if selector_type == "operator_element_root_projection" else spec.get("coordinate")
        record = {"type": "selected_element_root", "view": view, "row": row, "column": column, "root_hint": spec.get("root_hint")}
    else:
        record = {"type": selector_type, "spec": selector.get("spec"), "view": view}
    dependency_sha = view_sha if kind == "view" else _sha256(_canonical_bytes(record))
    return {"kind": kind, "view_sha256": view_sha, "dependency_sha256": dependency_sha}

def _verify_optimization_context_shape(value: object) -> Mapping[str, object]:
    required = {"schema", "schema_version", "phase", "candidate", "owner", "affected_leaves"}
    if not isinstance(value, Mapping) or not required.issubset(value) or not set(value).issubset(required | {"dependency", "aggregation_witness"}):
        raise _integrity("Optimization failure context is open or malformed.")
    if value.get("schema") != "scnsim.optimization_failure_context" or value.get("schema_version") != 1 or value.get("phase") not in {
        "candidate_prepare", "candidate_compile", "view_realization", "baseline_root_anchor",
        "quantity_evaluation", "objective_aggregation", "total_aggregation",
    }:
        raise _integrity("Optimization failure phase is malformed.")
    candidate = value.get("candidate")
    if not isinstance(candidate, Mapping) or set(candidate) != {"evaluation_ordinal", "origin", "generation", "population_column"}:
        raise _integrity("Optimization failure candidate position is malformed.")
    ordinal, origin, generation, column = (candidate.get(key) for key in ("evaluation_ordinal", "origin", "generation", "population_column"))
    integer = lambda item: isinstance(item, int) and not isinstance(item, bool)
    if not integer(ordinal) or not integer(generation) or ordinal < 0 or generation < 0:
        raise _integrity("Optimization failure candidate ordinals are malformed.")
    if origin == "baseline":
        if (ordinal, generation, column) != (0, 0, None):
            raise _integrity("Optimization baseline failure position is inconsistent.")
    elif origin == "population":
        if ordinal < 1 or generation < 1 or not integer(column) or column < 1:
            raise _integrity("Optimization population failure position is inconsistent.")
    else:
        raise _integrity("Optimization failure candidate origin is unknown.")
    owner = value.get("owner")
    valid_owner = (
        isinstance(owner, Mapping)
        and (
            set(owner) == {"kind"} and owner.get("kind") in {"candidate", "dependency"}
            or set(owner) == {"kind", "leaf"} and owner.get("kind") == "leaf"
            or set(owner) == {"kind", "objective_id"} and owner.get("kind") == "objective" and isinstance(owner.get("objective_id"), str)
        )
    )
    if not valid_owner:
        raise _integrity("Optimization failure owner is malformed.")
    affected = value.get("affected_leaves")
    locators = [owner.get("leaf")] if owner.get("kind") == "leaf" else []
    if not isinstance(affected, list) or any(not isinstance(item, Mapping) for item in [*affected, *locators]):
        raise _integrity("Optimization failure leaf locators are malformed.")
    for locator in [*affected, *locators]:
        if set(locator) != {"objective_id", "term_ordinal"} or not isinstance(locator.get("objective_id"), str) or not integer(locator.get("term_ordinal")) or locator["term_ordinal"] < 1:
            raise _integrity("Optimization failure leaf locator is malformed.")
    if len({_canonical_bytes(item) for item in affected}) != len(affected):
        raise _integrity("Optimization affected leaves repeat a locator.")
    dependency = value.get("dependency")
    if dependency is not None and (
        not isinstance(dependency, Mapping)
        or set(dependency) != {"kind", "view_sha256", "dependency_sha256"}
        or dependency.get("kind") not in {"view", "quantity"}
        or _SHA256.fullmatch(str(dependency.get("view_sha256", ""))) is None
        or _SHA256.fullmatch(str(dependency.get("dependency_sha256", ""))) is None
    ):
        raise _integrity("Optimization failure dependency is malformed.")
    if owner.get("kind") == "dependency" and dependency is None:
        raise _integrity("Optimization dependency failure has no dependency identity.")
    witness = value.get("aggregation_witness")
    needs_witness = origin == "baseline" and value.get("phase") in {
        "objective_aggregation",
        "total_aggregation",
    }
    if needs_witness and witness is None:
        raise _integrity("Optimization baseline aggregation failure lacks its witness.")
    if not needs_witness and witness is not None:
        raise _integrity("Optimization failure carries an inapplicable aggregation witness.")
    if witness is not None:
        if not isinstance(witness, Mapping):
            raise _integrity("Optimization aggregation witness is malformed.")
        if value.get("phase") == "objective_aggregation":
            if (
                set(witness) != {"kind", "objective_id", "terms"}
                or witness.get("kind") != "objective"
                or not isinstance(witness.get("objective_id"), str)
                or not isinstance(witness.get("terms"), list)
            ):
                raise _integrity("Optimization objective aggregation witness is malformed.")
        elif value.get("phase") == "total_aggregation":
            if (
                set(witness) != {"kind", "objective_components"}
                or witness.get("kind") != "total"
                or not isinstance(witness.get("objective_components"), list)
            ):
                raise _integrity("Optimization total aggregation witness is malformed.")
        else:
            raise _integrity("Optimization non-aggregation failure carries an aggregation witness.")
    return value

def _is_projection_only_optimization_failure(value: object) -> bool:
    """Classify the closed selector failures that own no shared dependency."""

    if not isinstance(value, Mapping):
        return False
    evidence = value.get("evidence")
    return (
        value.get("kind") == "invalid_optimization_spec"
        and value.get("stage") in {"selector", "quantity_sum"}
        and isinstance(evidence, Mapping)
        and evidence.get("operation") == "optimize_direct"
        and evidence.get("context_kind") == "optimization_candidate"
    )

def _is_shared_element_root_failure(value: object, selector: Mapping[str, object]) -> bool:
    """Distinguish a shared numerical root from a leaf-local passive policy."""

    if selector.get("type") not in {"diagonal_root_projection", "operator_element_root_projection"} or not isinstance(value, Mapping):
        return False
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("operation") != "optimize_direct":
        return False
    return (
        evidence.get("context_kind") in {"direct_quantity", "direct_response"}
        and value.get("kind") in {
            "eliminated_block_solve_failure", "root_slope_unresolved",
            "numerical_resolution_unresolved", "direct_response_formation",
        }
        or evidence.get("context_kind") == "optimization_candidate"
        and value.get("kind") == "numerical_resolution_unresolved"
        and value.get("stage") == "series_rl"
    )

def _is_leaf_local_passive_root_failure(value: object, selector: Mapping[str, object]) -> bool:
    if not isinstance(value, Mapping) or value.get("kind") != "numerical_resolution_unresolved" or value.get("stage") != "newton_certificate":
        return False
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("operation") != "optimize_direct":
        return False
    return (
        selector.get("type") == "diagonal_root_projection" and evidence.get("context_kind") == "optimization_candidate"
        or selector.get("type") == "residue_coupling_projection" and evidence.get("context_kind") == "direct_quantity"
    )

def _optimization_failure_requires_context(value: object, operation: object) -> bool:
    """Return whether a sealed optimization failure is execution-owned."""

    if operation != "optimize_direct" or not isinstance(value, Mapping):
        return False
    evidence = value.get("evidence")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("operation") != "optimize_direct"
        or evidence.get("context_kind")
        not in {"optimization_candidate", "direct_quantity", "direct_response"}
    ):
        return False
    kind = value.get("kind")
    return (
        kind
        in {
            "direct_response_formation",
            "invalid_candidate_physical_parameter",
            "eliminated_block_solve_failure",
            "root_slope_unresolved",
            "numerical_resolution_unresolved",
            "unsupported_singular_capacitance_for_diagonal_root_v1",
            "port_realizability",
        }
        or kind == "compiler_invariant"
        and value.get("stage") not in {"optimization", "optimization_replay"}
        or _is_projection_only_optimization_failure(value)
    )

def _verify_failure_document(value: object, operation: object) -> None:
    if not isinstance(value, dict) or set(value) != {"category", "kind", "stage", "message", "evidence"}:
        raise _integrity("Failure envelope is open or malformed.")
    evidence = value.get("evidence")
    allowed = {
        "type", "operation", "context_kind", "plan_sha256", "request_sha256",
        "attempt_sha256", "workspace_instance_id", "component_path", "parameter",
        "coordinate_id", "port_id", "case_id", "candidate_ordinal", "artifact_id",
        "artifact_path", "expected_sha256", "actual_sha256", "backend_exit_code",
        "evidence_sha256", "optimization_context",
    }
    categories = {
        "plan_sealed": "state",
        "workspace_plan_replaced": "state",
        "workspace_commit_indeterminate": "state",
        "workspace_versioning_downgrade_forbidden": "state",
        "unsupported_runtime_platform": "capability",
        "unsupported_singular_capacitance_for_diagonal_root_v1": "capability",
        "scaffold_unavailable": "capability",
        "port_realizability": "validation",
        "invalid_diagonal_root_hint": "validation",
        "invalid_optimization_spec": "validation",
        "direct_response_formation": "execution",
        "invalid_candidate_physical_parameter": "execution",
        "compiler_invariant": "execution",
        "eliminated_block_solve_failure": "execution",
        "root_slope_unresolved": "execution",
        "numerical_resolution_unresolved": "execution",
        "runtime_preparation": "execution",
        "backend_protocol": "execution",
        "optimization_progress_callback": "execution",
        "result_unavailable": "evidence",
        "evidence_integrity": "evidence",
    }
    kind = value.get("kind")
    contexts = {
        "authoring", "workspace", "runtime", "compile", "direct_response",
        "direct_quantity", "optimization_candidate", "hb_case", "protocol",
        "artifact", "resolution", "scaffold",
    }
    if (
        not isinstance(evidence, dict)
        or not {"type", "operation", "context_kind"}.issubset(evidence)
        or not set(evidence).issubset(allowed)
        or evidence.get("type") != "failure_evidence"
        or evidence.get("operation") not in {operation, "backend_protocol"}
        or evidence.get("context_kind") not in contexts
        or kind not in categories
        or value.get("category") != categories.get(kind)
        or not isinstance(value.get("stage"), str)
        or not value["stage"]
        or not isinstance(value.get("message"), str)
        or not value["message"]
    ):
        raise _integrity("Failure discriminator or evidence is malformed.")
    context = evidence.get("optimization_context")
    if _optimization_failure_requires_context(value, operation) and context is None:
        raise _integrity("Optimization execution failure lacks its phase context.")
    if context is not None:
        if operation != "optimize_direct" or evidence.get("operation") != "optimize_direct":
            raise _integrity("Non-optimization failure carries optimization context.")
        _verify_optimization_context_shape(context)

def _verify_selector(value: object, plan: Mapping[str, object]) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise _integrity("Optimization selector is malformed.")
    kind = value.get("type")
    if kind == "quantity_sum":
        if set(value) != {"type", "terms"} or not isinstance(value.get("terms"), list) or not value["terms"]:
            raise _integrity("QuantitySum is malformed.")
        roles = [_verify_selector(item, plan) for item in value["terms"]]
        if any(role[1] != roles[0][1] for role in roles[1:]):
            raise _integrity("QuantitySum terms have incompatible physical roles.")
        return roles[0]
    if kind == "quantity_difference":
        if set(value) != {"type", "left", "right"}:
            raise _integrity("QuantityDifference is malformed.")
        roles = [_verify_selector(value[side], plan) for side in ("left", "right")]
        if roles[0][1] != roles[1][1]:
            raise _integrity("QuantityDifference operands have incompatible physical roles.")
        return roles[0]
    if kind == "quantity_absolute":
        if set(value) != {"type", "operand"}:
            raise _integrity("QuantityAbsolute is malformed.")
        return _verify_selector(value["operand"], plan)
    fields = {"type", "spec", "projection", "view"}
    kind = value.get("type")
    expected = {
        "diagonal_root_projection": ("diagonal_root", {"frequency", "linewidth"}, ("hertz", "inverse_time")),
        "operator_element_root_projection": ("operator_element_root", {"frequency"}, ("hertz", "inverse_time")),
        "hybridized_pole_projection": ("hybridized_pole", {"frequency", "linewidth"}, ("hertz", "inverse_time")),
        "transfer_zero_projection": ("transfer_zero", {"frequency"}, ("hertz", "inverse_time")),
        "residue_coupling_projection": ("residue_normalized_coupling", {"real", "imag", "magnitude"}, ("radian / second", "inverse_time")),
        "response_element_projection": ("response_element", {"magnitude", "real", "imag"}, None),
    }.get(kind)
    if set(value) != fields or expected is None or value.get("projection") not in expected[1] or not isinstance(value.get("spec"), dict) or value["spec"].get("type") != expected[0]:
        raise _integrity("Optimization selector is outside the Direct catalog.")
    terminal, port_realizable = _verify_view_declaration(value.get("view"), plan)
    _verify_v1_evaluation_spec(value["spec"], terminal, port_realizable)
    if expected[2] is not None:
        return expected[2]
    family = value["spec"].get("family")
    return {"S": ("dimensionless", "dimensionless"), "Y": ("siemens", "conductance"), "Z": ("ohm", "resistance")}[family]

def _plan_coordinates(plan: Mapping[str, object]) -> tuple[list[str], set[str]]:
    """Return the snapshot-owned compiler basis and public subset."""

    connectivity = plan.get("connectivity")
    nodes = connectivity.get("node_coordinates") if isinstance(connectivity, Mapping) else None
    if not isinstance(nodes, list) or not nodes:
        raise _integrity("Sealed Plan coordinate inventory is malformed.")
    order: list[str] = []
    public: set[str] = set()
    for node in nodes:
        if (
            not isinstance(node, Mapping)
            or set(node) != {"final_net", "compiler_node_id", "visibility", "public_aliases"}
            or not isinstance(node.get("compiler_node_id"), str)
            or not node["compiler_node_id"]
            or node.get("compiler_node_id") != node.get("final_net")
            or node.get("visibility") not in {"public", "internal"}
            or not isinstance(node.get("public_aliases"), list)
        ):
            raise _integrity("Sealed Plan node inventory is malformed.")
        compiler_id = node["compiler_node_id"]
        if compiler_id in order:
            raise _integrity("Sealed Plan compiler coordinates are not unique.")
        order.append(compiler_id)
        if node["visibility"] == "public":
            if not node["public_aliases"]:
                raise _integrity("Public compiler coordinate has no public alias.")
            public.add(compiler_id)
        elif node["public_aliases"]:
            raise _integrity("Internal compiler coordinate exposes a public alias.")
    return order, public

def _lineage_matrix(label: str, values: list[list[float]], applicability: str) -> dict[str, object]:
    rows = len(values)
    columns = len(values[0]) if rows else 0
    bits = [struct.pack(">d", value).hex() for row in values for value in row]
    digest = _sha256(_canonical_bytes({
        "schema": "scnsim.lineage_matrix",
        "schema_version": 1,
        "label": label,
        "applicability": applicability,
        "shape": [rows, columns],
        "row_major_f64": bits,
    }))
    return {"rows": rows, "columns": columns, "sha256": digest}

def _selector_terms(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        raise _integrity("Optimization objective quantity is malformed.")
    if value.get("type") == "quantity_sum":
        terms = value.get("terms")
        if not isinstance(terms, list) or not terms or any(not isinstance(term, Mapping) for term in terms):
            raise _integrity("Optimization QuantitySum terms are malformed.")
        return [leaf for term in terms for leaf in _selector_terms(term)]
    if value.get("type") == "quantity_difference":
        if set(value) != {"type", "left", "right"}:
            raise _integrity("Optimization QuantityDifference is malformed.")
        return _selector_terms(value["left"]) + _selector_terms(value["right"])
    if value.get("type") == "quantity_absolute":
        if set(value) != {"type", "operand"}:
            raise _integrity("Optimization QuantityAbsolute is malformed.")
        return _selector_terms(value["operand"])
    return [value]

def _verify_selector_lineage(
    selector: Mapping[str, object],
    lineage: object,
    plan: Mapping[str, object],
) -> None:
    declaration = selector.get("view")
    declared_terminal, declared_port_realizable = _verify_view_declaration(declaration, plan)
    terminal, port_realizable = _verify_v1_lineage(lineage, plan)
    if terminal != declared_terminal or port_realizable != declared_port_realizable:
        raise _integrity("Optimization term lineage disagrees with its declared View.")
    if not isinstance(declaration, Mapping) or not isinstance(lineage, Mapping):
        raise _integrity("Optimization term View evidence is malformed.")
    declared_ptc = declaration.get("ptc")
    actual_ptc = lineage.get("ptc")
    if (None if actual_ptc is None else {"selected_ports": actual_ptc.get("selected_ports")}) != declared_ptc:
        raise _integrity("Optimization term PTC lineage disagrees with its declaration.")
    declared_transforms = declaration.get("transforms")
    actual_transforms = lineage.get("transforms")
    if not isinstance(declared_transforms, list) or not isinstance(actual_transforms, list) or len(actual_transforms) != len(declared_transforms):
        raise _integrity("Optimization term transform lineage count is inconsistent.")
    for declared, actual in zip(declared_transforms, actual_transforms):
        if (
            not isinstance(declared, Mapping)
            or not isinstance(actual, Mapping)
            or actual.get("input_coordinates") != declared.get("input_coordinates")
            or [actual.get("common_id"), actual.get("differential_id")] != declared.get("output_coordinates")
        ):
            raise _integrity("Optimization term transform lineage disagrees with its declaration.")
    declared_retain = declaration.get("retain")
    actual_retain = lineage.get("retain")
    if (None if actual_retain is None else {"retained_coordinates": actual_retain.get("retained_coordinates")}) != declared_retain:
        raise _integrity("Optimization term retain lineage disagrees with its declaration.")

def _optimization_failure_context(failure: object) -> Mapping[str, object]:
    if not isinstance(failure, Mapping) or not isinstance(failure.get("evidence"), Mapping):
        raise _integrity("Optimization failure lacks evidence.")
    context = failure["evidence"].get("optimization_context")
    return _verify_optimization_context_shape(context)
