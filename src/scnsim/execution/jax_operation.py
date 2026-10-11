"""Same-process JAX operation coordinator over the verified task journal.

The numerical host owns lowering, candidate order, continuation and CMA state.
This module owns invocation/task binding, durable evidence before callbacks and
pure result projection. It never creates a Julia request, process or receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4
from contextlib import nullcontext
import sys

from ..workspace import evidence as storage
from ..workspace.operation_lease import request_execution_lease
from ..numeric_encoding import record_bytes, record_document
from ..canonical import canonical_json_bytes, float64_from_hex
from ..errors import EvidenceIntegrityError, OptimizationProgressCallbackError, ResultUnavailableError, RuntimePreparationError
from ..results.decode_jax import decode_jax_operation
from ..diagnostics.identity import (
    environment_identity,
    environment_snapshot,
    operation_task_identifier,
)
from .config import get_runtime_configuration, runtime_resource_identity


_POSITIVE_INFINITY_CANDIDATE_COST = "7ff0000000000000"


def _candidate_cost_from_hex(value: str) -> float:
    """Decode the optimizer's existing failed-candidate cost sentinel.

    Candidate failure rows use positive infinity as a numeric cost while
    retaining their typed failure record. Other cost tokens keep the canonical
    finite-Float64 decoder's validation contract.
    """
    if value == _POSITIVE_INFINITY_CANDIDATE_COST:
        return float("inf")
    return float64_from_hex(value)


def _decode(success, *, decoder, prepared_analysis, bound_spec, request=None):
    request = (prepared_analysis.request() if request is None else request)
    fixed_reader = success.get("fixed_reader")
    request_sha256 = success.get("request_sha256")
    if request_sha256 is None and prepared_analysis is not None:
        request_sha256 = prepared_analysis.request_sha256
    return decode_jax_operation(
        decoder, projection=success["projection"], request=request,
        plan_sha256=request["plan_sha256"],
        request_sha256=request_sha256,
        attempt_sha256=success["attempt_sha256"], result_sha256=success["result_sha256"],
        bound_spec=bound_spec, fixed_reader=fixed_reader,
    )


def resolve_jax_operation(*, binding, prepared_analysis=None, decoder, bound_spec,
                          attempt_id=None, expected_spec=None, expected_backend=None,
                          expected_precision=None):
    """Read one fixed success or explicitly named terminal without JAX startup."""
    if prepared_analysis is None and (not isinstance(attempt_id, str) or expected_spec is None):
        raise TypeError("exact Sensitivity resolution requires attempt_id and expected_spec")

    expected_request_sha256 = (None if prepared_analysis is None
                               else prepared_analysis.request_sha256)

    def decode_selected(success):
        if success is None:
            return None
        request = success.get("request")
        if not isinstance(request, dict):
            raise EvidenceIntegrityError("Selected JAX attempt has no verified request body",
                                         stage="resolve")
        if expected_request_sha256 is not None and success.get("request_sha256") != expected_request_sha256:
            raise ResultUnavailableError(
                "The selected terminal attempt belongs to another exact request",
                stage="resolve", evidence={"attempt_id": attempt_id,
                                            "request_sha256": success.get("request_sha256"),
                                            "expected_request_sha256": expected_request_sha256},
            )
        if prepared_analysis is None:
            runtime = request.get("runtime_semantic", {})
            if (request.get("operation") != "sensitivity"
                    or canonical_json_bytes(request.get("spec")) != canonical_json_bytes(expected_spec)
                    or (expected_backend is not None and runtime.get("backend") != expected_backend)
                    or (expected_precision is not None and runtime.get("precision") != expected_precision)):
                raise ResultUnavailableError(
                    "The named terminal Sensitivity attempt does not match its declaration",
                    stage="resolve", evidence={"attempt_id": attempt_id},
                )
        return _decode(
            success, decoder=decoder, prepared_analysis=prepared_analysis,
            bound_spec=bound_spec, request=request,
        )

    with binding.reader():
        if attempt_id is not None:
            result = storage.read_operation_attempt(
                binding, attempt_id, projection_consumer=decode_selected,
            )
        elif prepared_analysis is not None:
            result = storage.find_operation_success(
                binding, prepared_analysis.request_sha256,
                projection_consumer=decode_selected,
            )
        else:
            result = None
    if result is None:
        evidence = {"attempt_id": attempt_id}
        if expected_request_sha256 is not None:
            evidence["request_sha256"] = expected_request_sha256
        raise ResultUnavailableError("No verified JAX result exists for this exact request or attempt",
                                     stage="resolve", evidence=evidence)
    return result


def _sensitivity_reason(aux, *, observable_index=None):
    """Detach the checked derivative-unavailable facts for one saved block."""
    import numpy as np
    from ..numerics.evidence import evidence_bytes

    def host(value):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()

    certificates = []
    for certificate in aux.certificates:
        fields = getattr(certificate, "_fields", None)
        if fields is None:
            certificates.append(host(certificate))
        else:
            certificates.append({name: host(getattr(certificate, name)) for name in fields})
    observable_reasons = ()
    if observable_index is not None:
        observable_reasons = tuple(aux.observable_reasons[observable_index])
    return record_document(evidence_bytes({
        "status": host(aux.status),
        "failed_site": host(aux.failed_site),
        "derivative_reasons": tuple(aux.derivative_reasons),
        "observable_reasons": observable_reasons,
        "certificates": tuple(certificates),
    }))


def _sensitivity_blocks(quantity_program, prepared_analysis, request, derivative_outcome):
    """Project one checked derivative into ordered immutable primal/result blocks."""
    import numpy as np
    from ..numeric_encoding import array_record

    spec = request["spec"]
    derivative_spec = spec["derivative"]
    derivative_kind = derivative_spec["type"]
    quantity_metadata = quantity_program.quantity_metadata()
    output_rows = quantity_program.output_layout.rows
    rows_by_quantity = {
        row["key"]: tuple(item for item in output_rows if item.quantity_key == row["key"])
        for row in quantity_metadata
    }
    quantity_layout = [
        {"id": row["key"], "label": row["key"], "unit": row["si_unit"], "axes": row["axes"]}
        for row in quantity_metadata
    ]
    derivative_layout = {"kind": derivative_kind, "parameters": list(spec["wrt"])}
    if derivative_kind == "vjp":
        derivative_layout["functional_unit"] = spec["functional_unit"]

    primal = derivative_outcome.primal
    aux = derivative_outcome.aux
    derivative_status = int(np.asarray(aux.status))
    derivative_available = bool(np.asarray(aux.derivative_available))
    blocks = []
    primal_ids = {}
    for ordinal, row in enumerate(quantity_metadata):
        key = row["key"]
        block_id = f"primal:{ordinal}:{key}"
        primal_ids[key] = block_id
        blocks.append({
            "block_id": block_id,
            "kind": "primal",
            "quantity_id": key,
            "output_slice": {"start": 0, "stop": len(rows_by_quantity[key])},
            "parameter_slice": None,
            "summary_json": {"value_unit": row["si_unit"], "axes": row["axes"]},
            "primal_blocks": [],
            "primal_status": "complete",
            "payload": {"values": array_record(primal.values[row["observable_index"]])},
        })

    def append_derivative(*, block_id, row, values, unavailable, parameter_slice,
                          primal_links, parameters=()):
        key = row["key"]
        axes = row["axes"]
        summary = {"axes": axes, "parameters": list(parameters)}
        if key is not None:
            summary["value_unit"] = row["si_unit"]
        if unavailable is None:
            payload = {"values": array_record(values)}
            status = "complete"
        else:
            payload = {"values": None, "reason": unavailable}
            status = "unavailable"
        blocks.append({
            "block_id": block_id,
            "kind": "derivative",
            "quantity_id": key,
            "output_slice": (None if key is None else
                             {"start": 0, "stop": len(rows_by_quantity[key])}),
            "parameter_slice": parameter_slice,
            "summary_json": summary,
            "primal_blocks": list(primal_links),
            "primal_status": "complete",
            "derivative_status": status,
            "payload": payload,
        })

    if derivative_kind == "jacobian":
        payload = derivative_outcome.derivative
        row_positions = {row_index: index for index, row_index in enumerate(payload.row_indices)}
        matrix = np.asarray(payload.values)
        for row in quantity_metadata:
            key = row["key"]
            expected_rows = rows_by_quantity[key]
            selected_positions = [row_positions.get(item.row) for item in expected_rows]
            reason = None
            if (not bool(np.asarray(aux.observable_available[row["observable_index"]]))
                    or any(position is None for position in selected_positions)):
                reason = _sensitivity_reason(aux, observable_index=row["observable_index"])
            values = None
            if reason is None:
                values = matrix[selected_positions, :]
            append_derivative(
                block_id=f"derivative:{row['key']}", row=row, values=values,
                unavailable=reason,
                parameter_slice={"start": 0, "stop": len(spec["wrt"])},
                primal_links=[primal_ids[key]], parameters=spec["wrt"],
            )
    elif derivative_kind == "jvp":
        for row in quantity_metadata:
            key = row["key"]
            value = derivative_outcome.derivative[row["observable_index"]]
            reason = None
            if (not bool(np.asarray(aux.observable_available[row["observable_index"]]))
                    or value is None):
                reason = _sensitivity_reason(aux, observable_index=row["observable_index"])
            append_derivative(
                block_id=f"derivative:{key}", row=row,
                values=value if reason is None else None, unavailable=reason,
                parameter_slice=None, primal_links=[primal_ids[key]],
            )
    elif derivative_kind == "vjp":
        value = derivative_outcome.derivative
        reason = None
        if derivative_status != 0 or value is None:
            reason = _sensitivity_reason(aux)
        if reason is None:
            import jax.numpy as jnp
            indices = quantity_program.program.model_program.fields.layout.input_indices(
                prepared_analysis.active_keys
            )
            values = np.asarray(value.theta[jnp.asarray(indices, dtype=jnp.int32)])
        else:
            values = None
        append_derivative(
            block_id="derivative:vjp", row={"key": None, "axes": ()},
            values=values, unavailable=reason,
            parameter_slice={"start": 0, "stop": len(spec["wrt"])},
            primal_links=[primal_ids[row["key"]] for row in quantity_metadata],
            parameters=spec["wrt"],
        )
    else:
        raise TypeError(f"unsupported Sensitivity derivative {derivative_kind!r}")

    outcome = "success" if all(
        block.get("derivative_status") == "complete"
        for block in blocks if block["kind"] == "derivative"
    ) else "partial"
    return {
        "manifest": {
            "program_sha256": quantity_program.program_sha256,
            "quantity_layout": quantity_layout,
            "derivative_layout": derivative_layout,
        },
        "blocks": blocks,
        "outcome": outcome,
        "summary": {"program_sha256": quantity_program.program_sha256,
                    "quantity_count": len(quantity_layout),
                    "derivative_kind": derivative_kind,
                    "derivative_status": derivative_status,
                    "derivative_available": derivative_available},
    }


def execute_sensitivity_operation(*, binding, plan_document, prepared_analysis, decoder,
                                   bound_spec, trace, precision, backend="jax",
                                   program_owner=None):
    """Execute one checked Sensitivity program and atomically seal its result blocks."""
    request = prepared_analysis.request()
    request_sha = prepared_analysis.request_sha256
    if backend != "jax" or precision != request["runtime_semantic"]["precision"]:
        raise RuntimePreparationError(
            "Sensitivity execution does not match its prepared JAX runtime",
            stage="runtime_prepare",
        )

    with request_execution_lease(binding, request_sha):
        with trace.span("result_lookup"):
            with binding.reader():
                selected = storage.find_operation_success(
                    binding, request_sha,
                    projection_consumer=lambda success: (
                        success,
                        _decode(success, decoder=decoder, prepared_analysis=prepared_analysis,
                                bound_spec=bound_spec),
                    ),
                )
        if selected is not None:
            success, result = selected
            trace.bind(operation="sensitivity", request_sha256=request_sha,
                       task_id=success["task_id"], environment_sha256=None,
                       attempt_id=None, details={"cache_hit": True, "checkpoint_policy": "none"})
            trace.add_numerical_ref(success["result_ref"])
            return result

        from .jax_backend import get_jax_backend
        from .resources import operation_resources

        resources = request["runtime_semantic"]["resources"]
        if runtime_resource_identity() != resources:
            raise RuntimePreparationError(
                "Prepared JAX resources changed before Sensitivity execution",
                stage="runtime_prepare",
            )
        runtime_configuration = get_runtime_configuration()
        with operation_resources(cpu_threads=runtime_configuration.cpu_threads) as operation_resource:
            with trace.span("backend_prepare"):
                numerical_backend = get_jax_backend(
                    precision=precision, resources=runtime_configuration, trace=trace,
                    operation_resources=operation_resource, diagnostics=True,
                )
            primary = None
            try:
                return _execute_sensitivity_with_backend(
                    binding=binding, plan_document=plan_document,
                    prepared_analysis=prepared_analysis, decoder=decoder,
                    bound_spec=bound_spec, trace=trace, request=request,
                    request_sha=request_sha, precision=precision, resources=resources,
                    backend=numerical_backend, operation_resource=operation_resource,
                    program_owner=program_owner,
                )
            except BaseException as error:
                primary = error
                raise
            finally:
                try:
                    numerical_backend.close()
                except BaseException as cleanup_error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Sensitivity backend cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )


def _execute_sensitivity_with_backend(*, binding, plan_document, prepared_analysis, decoder,
                                     bound_spec, trace, request, request_sha, precision,
                                     resources, backend, operation_resource, program_owner=None):
    from ..compilation.compiler import parameter_key
    from ..compilation.quantity_program import compile_quantity_program

    operation = request["operation"]
    workspace = storage.operation_workspace(binding)
    attempt_id = str(uuid4())
    task_id = None
    environment_sha = None
    registration_attempted = False
    registration_committed = False
    terminal_attempted = False
    completed = False
    task_details = {"cache_hit": False, "checkpoint_policy": "none",
                    "commit_every_generations": None}
    try:
        arm = f"{trace.method}/jax/{precision}/none"
        environment = environment_snapshot(
            arm=arm, device="cpu", cpu_threads=resources["cpu_threads"],
            backend=backend.identity(),
        )
        environment["runtime_semantic"] = request["runtime_semantic"]
        environment["resources"] = resources
        environment_sha = environment_identity(environment)
        environment["environment_sha256"] = environment_sha
        task_id = operation_task_identifier(
            plan_sha256=request["plan_sha256"], request_sha256=request_sha,
            method=trace.method, backend="jax", precision=precision,
            resources=resources, algorithm_id=request["runtime_semantic"]["algorithm_id"],
            environment_sha256=environment_sha, checkpoint_policy="none",
            commit_every_generations=1,
        )
        trace.bind(operation=operation, request_sha256=request_sha, task_id=task_id,
                   environment_sha256=environment_sha, attempt_id=None,
                   details=task_details)
        trace.set_attempt(attempt_id)
        registration_attempted = True
        with binding.writer():
            registration = storage.register_operation_execution(
                binding,
                row=trace.root_row(),
                task={"task_id": task_id, "request_sha256": request_sha, "arm": arm,
                      "sample": 0, "attempts": [], "events": [], "measurements": [],
                      "environment": environment, "artifacts": []},
                request={"request_sha256": request_sha,
                         "request_bytes": prepared_analysis.request_bytes},
                attempt_id=attempt_id,
            )
            if registration["ack"].get("committed") is not True:
                raise EvidenceIntegrityError(
                    "Sensitivity execution registration was not durably acknowledged.",
                    stage="operation_store", evidence={"operation_id": trace.operation_id},
                )
            registration_committed = True
            task_details["request"] = registration["request_ref"]
            trace.bind(operation=operation, request_sha256=request_sha, task_id=task_id,
                       environment_sha256=environment_sha, attempt_id=attempt_id,
                       details=task_details)

        with trace.span("quantity_program_compile"):
            point = prepared_analysis.effective_point()
            spec = request["spec"]
            declarations = {row["id"]: row["quantity"] for row in spec["quantities"]}
            authorized = frozenset(
                parameter_key(reference) for reference in point.get("allow_extrapolation", ())
            )
            from .warmup import program_owner_for_device
            scoped_program_owner = program_owner_for_device(
                program_owner, backend.device, admit=False,
            )
            quantity_program = compile_quantity_program(
                plan_document, declarations, point,
                mesh=spec.get("discretization"), precision=precision,
                authorized=authorized, program_owner=scoped_program_owner,
                admit_program=False,
            )
        inputs = quantity_program.inputs(point)
        derivative_spec = spec["derivative"]
        derivative_kind = derivative_spec["type"]
        input_indices = quantity_program.program.model_program.fields.layout.input_indices(
            prepared_analysis.active_keys
        )
        with trace.span("sensitivity_evaluation", details={"derivative": derivative_kind}):
            if derivative_kind == "jacobian":
                derivative_outcome = backend.execute(
                    quantity_program.program, "jacobian", inputs,
                    input_indices=input_indices, sparse_runner=trace.sparse_runner,
                )
            elif derivative_kind == "jvp":
                from json import loads
                direction = quantity_program.direction_inputs(
                    inputs, loads(prepared_analysis.direction_bytes),
                )
                derivative_outcome = backend.execute(
                    quantity_program.program, "jvp", inputs, direction,
                    sparse_runner=trace.sparse_runner,
                )
            elif derivative_kind == "vjp":
                from json import loads
                weights = quantity_program.weight_vector(loads(prepared_analysis.weights_bytes))
                derivative_outcome = backend.execute(
                    quantity_program.program, "vjp", inputs, weights,
                    sparse_runner=trace.sparse_runner,
                )
            else:
                raise TypeError(f"unsupported Sensitivity derivative {derivative_kind!r}")
            quantity_program.require_primal(derivative_outcome)

        publication = _sensitivity_blocks(
            quantity_program, prepared_analysis, request, derivative_outcome,
        )
        with trace.span("result_publish"):
            with binding.writer():
                terminal_attempted = True
                published = storage.complete_sensitivity_operation(
                    binding, operation_id=trace.operation_id, task_id=task_id,
                    attempt_id=attempt_id, manifest=publication["manifest"],
                    result_summary=publication["summary"], blocks=publication["blocks"],
                    outcome=publication["outcome"],
                )
            ack = published["ack"]
            if ack.get("committed") is not True or ack.get("outcome") != publication["outcome"]:
                raise EvidenceIntegrityError(
                    "Sensitivity terminal publication was not durably acknowledged.",
                    stage="operation_store", evidence={"operation_id": trace.operation_id,
                                                         "attempt_id": attempt_id},
                )
            completed = True
        trace.add_numerical_ref(published["result_ref"])
        trace.publish_diagnostics()
        with trace.span("result_decode"):
            with binding.reader():
                return storage.read_sensitivity_result(
                    binding, task_id, attempt_id=attempt_id,
                    projection_consumer=lambda success: _decode(
                        success, decoder=decoder, prepared_analysis=prepared_analysis,
                        bound_spec=bound_spec, request=request,
                    ),
                )
    except BaseException as error:
        if attempt_id is not None and not completed:
            if registration_attempted and not registration_committed:
                outcome = getattr(error, "operation_registration_outcome", None)
                if outcome is None:
                    outcome = getattr(error, "operation_transaction_outcome", None)
                registration_committed = (
                    isinstance(outcome, dict) and outcome.get("status") == "committed"
                )
                if registration_committed:
                    committed_registration = getattr(error, "operation_registration_result", None)
                    if isinstance(committed_registration, dict):
                        task_id = committed_registration["attempt_binding"]["task_id"]
                        attempt_id = committed_registration["attempt_binding"]["attempt_id"]
                        task_details["request"] = committed_registration["request_ref"]
                        trace.bind(operation=operation, request_sha256=request_sha,
                                   task_id=task_id, environment_sha256=environment_sha,
                                   attempt_id=attempt_id, details=task_details)
                elif isinstance(outcome, dict) and outcome.get("status") == "unknown":
                    error.add_note(
                        "Sensitivity registration outcome is unknown; no attempt status was retried."
                    )
            publication_outcome = getattr(error, "operation_transaction_outcome", None)
            terminal_uncertain = terminal_attempted and isinstance(publication_outcome, dict) and (
                publication_outcome.get("status") in {"committed", "unknown"}
            )
            if registration_committed and task_id is not None and not terminal_uncertain:
                try:
                    failure = storage.error_document(error)
                    interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                    with binding.writer():
                        storage.update_attempt(
                            workspace, binding=binding, operation_id=trace.operation_id,
                            task_id=task_id, attempt_id=attempt_id,
                            status="interrupted" if interrupted else "failure",
                            interruption=failure if interrupted else None,
                            failure=None if interrupted else failure,
                        )
                except BaseException as recording_error:
                    error.add_note(
                        "Sensitivity failure recording also failed: "
                        f"{type(recording_error).__name__}: {recording_error}"
                    )
            elif terminal_uncertain:
                error.add_note(
                    "Sensitivity terminal publication was uncertain; no attempt status was overwritten."
                )
            if registration_attempted and not registration_committed:
                trace.bind(operation=operation, request_sha256=request_sha, task_id=None,
                           environment_sha256=None, attempt_id=None, details=task_details)
        raise


def _physical_optimization_controls(request, stage_id):
    optimizer = request["spec"]["optimizer"]
    if optimizer.get("type") == "hybrid":
        declaration = next(row for row in optimizer["stages"] if row["id"] == stage_id)
        return declaration["optimizer"]
    return optimizer


def _physical_progress(on_progress, request, values, *, phase, reused=False, trace=None):
    if on_progress is None:
        return
    from ..specs import OptimizationProgress

    stage_id = values["stage_id"]
    controls = _physical_optimization_controls(request, stage_id)
    strategy = controls["type"]
    cma = strategy == "cma_es"
    generations = values.get("completed_generations") if cma else None
    iterations = values.get("completed_iterations") if not cma else None
    progress = OptimizationProgress(
        phase=phase, reused=reused,
        completed_generations=generations,
        total_generations=controls.get("complete_generations") if cma else None,
        evaluated_count=values.get("evaluated_count", 0),
        requested_budget=controls["max_evaluations"],
        achievable_evaluations=(
            1 + controls["complete_generations"] * controls["resolved_population_size"]
            if cma else None
        ),
        best_cost=values.get("best_cost"),
        stage_id=stage_id,
        strategy="cmaes" if cma else "lbfgsb",
        committed_generations=values.get("committed_generations") if cma else None,
        completed_iterations=iterations,
        total_iterations=controls.get("max_iterations") if not cma else None,
        committed_iterations=values.get("committed_iterations") if not cma else None,
    )
    try:
        with (nullcontext() if trace is None else trace.span("progress_callback", details={"phase": phase})):
            on_progress(progress)
    except Exception as error:
        raise OptimizationProgressCallbackError(
            "optimization progress callback failed", stage="progress_callback",
            evidence={"error_type": type(error).__name__, "phase": phase,
                      "stage_id": stage_id},
        ) from error


def _optimization_candidate_summary(candidate):
    objectives = candidate.get("objectives", ())
    components = []
    for objective in objectives:
        component = {"objective_id": objective["id"], "status": objective["status"]}
        if objective["status"] == "success":
            component["weighted_cost_f64"] = objective["cost_f64"]
            component["normalized_residual_f64"] = objective["normalized_residual_f64"]
        component["terms"] = objective["terms"]
        components.append(component)
    return {
        "evaluation_ordinal": candidate["evaluation_ordinal"],
        "parameters": candidate["parameters"],
        "cost_f64": candidate["cost_f64"],
        "status": candidate["status"],
        "components": components,
    }


def _bind_resume_candidate_sources(value, *, task_id, attempt_id):
    """Attach the exact selected checkpoint owner to its candidate locators."""
    if isinstance(value, Mapping):
        result = {
            key: _bind_resume_candidate_sources(item, task_id=task_id, attempt_id=attempt_id)
            for key, item in value.items()
        }
        kind = result.get("kind", "evaluation")
        has_occurrence = (
            kind == "baseline"
            or kind == "evaluation" and "stage_id" in result and "evaluation_ordinal" in result
        )
        if has_occurrence and isinstance(result.get("candidate_ref"), Mapping):
            has_source_task = "source_task_id" in result
            has_source_attempt = "source_attempt_id" in result
            if has_source_task != has_source_attempt:
                raise EvidenceIntegrityError(
                    "Optimization checkpoint locator has incomplete source identity",
                    stage="resume_prepare",
                )
            if not has_source_task:
                result["source_task_id"] = task_id
                result["source_attempt_id"] = attempt_id
        return result
    if isinstance(value, list):
        return [_bind_resume_candidate_sources(item, task_id=task_id, attempt_id=attempt_id)
                for item in value]
    if isinstance(value, tuple):
        return tuple(_bind_resume_candidate_sources(item, task_id=task_id, attempt_id=attempt_id)
                     for item in value)
    return value


def _execute_physical_optimization(*, binding, plan_document, prepared_analysis, decoder,
                                   bound_spec, on_progress, progress_observer,
                                   checkpoint_policy, resume_from, restart_from,
                                   commit_every_generations, program_owner, trace, request,
                                   request_sha, precision, resources, runtime_configuration,
                                   backend, operation_resource):
    """Register, run, and seal the one current JAX physical optimizer."""
    from .operation_spool import OperationSpool
    from .physical_optimization import run_physical_optimization

    workspace = storage.operation_workspace(binding)
    arm = f"{trace.method}/jax/{precision}/{checkpoint_policy}"
    environment = environment_snapshot(
        arm=arm, device="cpu", cpu_threads=resources["cpu_threads"],
        backend=backend.identity(),
    )
    environment["runtime_semantic"] = request["runtime_semantic"]
    environment["resources"] = resources
    environment_sha = environment_identity(environment)
    environment["environment_sha256"] = environment_sha
    task_id = operation_task_identifier(
        plan_sha256=request["plan_sha256"], request_sha256=request_sha,
        method=trace.method, backend="jax", precision=precision, resources=resources,
        algorithm_id=request["runtime_semantic"]["algorithm_id"],
        environment_sha256=environment_sha, checkpoint_policy=checkpoint_policy,
        commit_every_generations=commit_every_generations,
    )

    exact_checkpoint = None
    accepted_source = None
    resume_selection = None
    latest_checkpoint_ref = None
    latest_checkpoint_stage = None
    if resume_from is not None:
        source_task_id = resume_from.get("task_id")
        source_attempt_id = resume_from.get("attempt_id")
        source_ref = resume_from.get("checkpoint_ref")
        if (not isinstance(source_task_id, str) or not isinstance(source_attempt_id, str)
                or not isinstance(source_ref, Mapping)):
            raise RuntimePreparationError(
                "Exact Optimization resume requires its task, attempt, and checkpoint reference",
                stage="resume_prepare",
            )
        with binding.reader():
            exact_checkpoint = storage.read_physical_optimization_checkpoint(
                binding, task_id=source_task_id, attempt_id=source_attempt_id,
                expected_request_sha256=request_sha, reference=source_ref,
                expected_stage_id=resume_from.get("stage_id"),
            )
        if exact_checkpoint is None:
            raise ResultUnavailableError(
                "The named Optimization attempt has no matching committed checkpoint",
                stage="resume_prepare",
                evidence={"task_id": source_task_id, "attempt_id": source_attempt_id},
            )
        if source_task_id != task_id:
            raise ResultUnavailableError(
                "The named checkpoint was created under a different execution environment",
                stage="resume_prepare",
                evidence={"checkpoint_task_id": source_task_id, "current_task_id": task_id},
            )
        resume_selection = exact_checkpoint["resume_from"]
        latest_checkpoint_ref = exact_checkpoint["checkpoint_ref"]
        latest_checkpoint_stage = exact_checkpoint["checkpoint"]["stage_id"]
    if restart_from is not None:
        if not isinstance(restart_from, Mapping):
            raise TypeError("restart_from must be an exact accepted-iterate selection")
        with binding.reader():
            accepted_source = storage.read_physical_accepted_iterate(
                binding, attempt_id=restart_from["attempt_id"],
                stage_id=restart_from["stage_id"],
                accepted_iterate_id=restart_from["accepted_iterate_id"],
            )
        if accepted_source is None or accepted_source["request_sha256"] != request_sha:
            raise ResultUnavailableError(
                "The selected accepted iterate is absent or belongs to another request",
                stage="restart_prepare", evidence=dict(restart_from),
            )

    attempt_id = str(uuid4())
    registration_attempted = False
    registration_committed = False
    terminal_attempted = False
    terminal_acknowledged = False
    latest_observation = {
        "stage_id": "main", "evaluated_count": 0, "best_cost": None,
        "completed_generations": None, "committed_generations": None,
        "completed_iterations": None, "committed_iterations": None,
    }
    last_notified = {}
    observer_enabled = progress_observer is not None

    def observe_progress(**values):
        nonlocal observer_enabled
        latest_observation.update(values)
        if not observer_enabled:
            return
        strategy = values["strategy"]
        display_strategy = "cmaes" if strategy == "cma_es" else "lbfgsb"
        try:
            progress_observer.update(
                phase=values["phase"], stage_id=values["stage_id"],
                strategy=display_strategy,
                completed_generations=values.get("completed_generations"),
                total_generations=values.get("total_generations"),
                completed_iterations=values.get("completed_iterations"),
                total_iterations=values.get("total_iterations"),
                evaluated_count=values.get("evaluated_count", 0),
                best_cost=values.get("best_cost"),
                committed_generations=values.get("committed_generations"),
                committed_iterations=values.get("committed_iterations"),
                reused=values.get("reused", False),
            )
        except Exception as display_error:
            observer_enabled = False
            import warnings
            warnings.warn(
                f"Optimization progress display disabled after {type(display_error).__name__}",
                RuntimeWarning, stacklevel=2,
            )

    if observer_enabled:
        optimizer = request["spec"]["optimizer"]
        if optimizer.get("type") == "hybrid":
            initial_stage = optimizer["stages"][0]
            initial_stage_id = initial_stage["id"]
            initial_strategy = initial_stage["optimizer"]["type"]
        else:
            initial_stage_id = "main"
            initial_strategy = optimizer["type"]
        observe_progress(
            phase="preparing", stage_id=initial_stage_id, strategy=initial_strategy,
            requested_budget=request["spec"]["optimizer"].get("max_evaluations", 0),
            achievable_evaluations=None, completed_generations=0,
            total_generations=None, committed_generations=0,
            completed_iterations=None, total_iterations=None,
            committed_iterations=None, evaluated_count=0, best_cost=None,
        )

    def publish_batch(batch):
        nonlocal latest_checkpoint_ref, latest_checkpoint_stage
        with trace.span("physical_stage_commit", details={"stage_id": batch["stage_id"]}):
            with binding.writer():
                published = storage.commit_physical_stage_batch(
                    binding, operation_id=trace.operation_id, task_id=task_id,
                    attempt_id=attempt_id, stage_id=batch["stage_id"],
                    model_identity=batch["model_identity"], rows=batch["rows"],
                    stage_state=batch["stage_state"], checkpoint=batch["checkpoint"],
                    transitions=batch["transitions"],
                )
        ack = published["ack"]
        if ack.get("committed") is not True:
            raise EvidenceIntegrityError(
                "Physical Optimization stage batch was not durably acknowledged",
                stage="operation_store", evidence={"operation_id": trace.operation_id,
                                                      "stage_id": batch["stage_id"]},
            )
        trace.add_numerical_ref(published["batch_ref"])
        latest_checkpoint_ref = published.get("checkpoint_ref")
        latest_checkpoint_stage = (batch["stage_id"]
                                   if latest_checkpoint_ref is not None else None)
        if latest_checkpoint_ref is not None:
            trace.add_numerical_ref(latest_checkpoint_ref)
        stage_state = batch["stage_state"]
        strategy = stage_state["strategy"]
        completed = stage_state["iteration_count"]
        if strategy == "cma_es" and isinstance(completed, int) and completed > 0:
            trace.complete_generation(completed)
        trace.publish_diagnostics()
        return ack

    def on_acknowledged(stage_state, ack):
        strategy = stage_state["strategy"]
        stage_id = stage_state["stage_id"]
        count = stage_state["iteration_count"]
        previous = last_notified.get(stage_id, 0)
        if not isinstance(count, int) or count <= previous or count <= 0:
            return
        last_notified[stage_id] = count
        phase = "generation" if strategy == "cma_es" else "iteration"
        current = dict(latest_observation)
        current["stage_id"] = stage_id
        if strategy == "cma_es":
            current["completed_generations"] = count
            current["committed_generations"] = count
        else:
            current["completed_iterations"] = count
            current["committed_iterations"] = count
        _physical_progress(on_progress, request, current, phase=phase, trace=trace)

    def cache_records(source_state):
        for key, locator in source_state.get("cache", {}).items():
            source_task_id = locator.get("source_task_id", source_state["source_task_id"])
            source_attempt_id = locator.get("source_attempt_id", source_state["source_attempt_id"])
            loaded = storage.read_physical_optimization_candidate(
                binding, task_id=source_task_id, attempt_id=source_attempt_id,
                locator=locator, expected_request_sha256=request_sha,
                resume_from=source_state["resume_from"],
            )
            candidate = loaded["candidate"]
            yield key, {"candidate": candidate, "kind": candidate["kind"],
                        "summary_json": _optimization_candidate_summary(candidate),
                        "source_locator": dict(locator),
                        "source_task_id": source_task_id,
                        "source_attempt_id": source_attempt_id}

    resume_state = None
    if exact_checkpoint is not None:
        checkpoint = exact_checkpoint["checkpoint"]
        resume_state = _bind_resume_candidate_sources(
            checkpoint["state"], task_id=exact_checkpoint["task_id"],
            attempt_id=exact_checkpoint["attempt_id"],
        )
        resume_state["attempt_id"] = exact_checkpoint["attempt_id"]
        resume_state["source_task_id"] = exact_checkpoint["task_id"]
        resume_state["source_attempt_id"] = exact_checkpoint["attempt_id"]
        resume_state["resume_from"] = exact_checkpoint["resume_from"]
        baseline_locator = resume_state.get("baseline_locator")
        if not isinstance(baseline_locator, Mapping):
            raise EvidenceIntegrityError(
                "Optimization checkpoint lacks its exact baseline candidate locator",
                stage="resume_prepare",
                evidence={"attempt_id": exact_checkpoint["attempt_id"]},
            )
        baseline = storage.read_physical_optimization_candidate(
            binding, task_id=baseline_locator.get("source_task_id", exact_checkpoint["task_id"]),
            attempt_id=baseline_locator.get("source_attempt_id", exact_checkpoint["attempt_id"]),
            locator=baseline_locator, expected_request_sha256=request_sha,
            resume_from=exact_checkpoint["resume_from"],
        )
        resume_state["baseline_body"] = baseline["candidate"]
        best_locator = resume_state.get("best")
        resume_state["best_candidate"] = None
        if best_locator is not None:
            if not isinstance(best_locator, Mapping):
                raise EvidenceIntegrityError(
                    "Optimization checkpoint best candidate locator is malformed",
                    stage="resume_prepare",
                    evidence={"attempt_id": exact_checkpoint["attempt_id"]},
                )
            best = storage.read_physical_optimization_candidate(
                binding, task_id=best_locator.get("source_task_id", exact_checkpoint["task_id"]),
                attempt_id=best_locator.get("source_attempt_id", exact_checkpoint["attempt_id"]),
                locator=best_locator, expected_request_sha256=request_sha,
                resume_from=exact_checkpoint["resume_from"],
            )
            resume_state["best_candidate"] = best["candidate"]
        resume_state["cache_records"] = cache_records(resume_state)
    elif accepted_source is not None:
        resume_state = {
            "restart_point": accepted_source["candidate"]["parameters"],
            "restart_ancestry": {
                "task_id": accepted_source["task_id"],
                "attempt_id": accepted_source["attempt_id"],
                "stage_id": accepted_source["stage_id"],
                "evaluation_ordinal": accepted_source["evaluation_ordinal"],
                "accepted_iterate_id": accepted_source["accepted_iterate_id"],
                "candidate_ref": accepted_source["candidate_ref"],
            },
        }

    operation_details = {
        "cache_hit": False, "checkpoint_policy": checkpoint_policy,
        "requested_commit_every_generations": commit_every_generations,
        "commit_every_generations": commit_every_generations,
    }
    if resume_selection is not None:
        operation_details["resume_from"] = resume_selection
    if accepted_source is not None:
        operation_details["restart_from"] = resume_state["restart_ancestry"]
    trace.bind(operation="optimize_direct", request_sha256=request_sha,
               task_id=task_id, environment_sha256=environment_sha,
               attempt_id=None, details=operation_details)
    trace.set_attempt(attempt_id)

    spool = OperationSpool(workspace, trace.operation_id, lease_root=binding.root)
    try:
        registration_attempted = True
        with binding.writer():
            registration = storage.register_operation_execution(
                binding, row=trace.root_row(),
                task={"task_id": task_id, "request_sha256": request_sha,
                      "arm": arm, "sample": 0, "attempts": [], "events": [],
                      "measurements": [], "environment": environment, "artifacts": []},
                request={"request_sha256": request_sha,
                         "request_bytes": prepared_analysis.request_bytes},
                attempt_id=attempt_id, resume_from=resume_selection,
            )
            if registration["ack"].get("committed") is not True:
                raise EvidenceIntegrityError(
                    "Optimization execution registration was not durably acknowledged",
                    stage="operation_store", evidence={"operation_id": trace.operation_id},
                )
        registration_committed = True
        operation_details["request"] = registration["request_ref"]
        trace.bind(operation="optimize_direct", request_sha256=request_sha,
                   task_id=task_id, environment_sha256=environment_sha,
                   attempt_id=attempt_id, details=operation_details)

        with spool.activate():
            computed = run_physical_optimization(
                plan_document=plan_document, request=request,
                effective_point=prepared_analysis.effective_point(), precision=precision,
                backend=backend, operation_resource=operation_resource, trace=trace,
                program_owner=program_owner, checkpoint_policy=checkpoint_policy,
                resume_state=resume_state,
                commit_every_generations=commit_every_generations,
                publish_batch=publish_batch, observe_progress=observe_progress,
                on_acknowledged=on_acknowledged, spool=spool,
            )

        with trace.span("physical_optimization_terminal"):
            with binding.writer():
                terminal_attempted = True
                publication = storage.complete_optimization_operation(
                    binding, operation_id=trace.operation_id, task_id=task_id,
                    attempt_id=attempt_id, result=computed["result"],
                    manifest=computed["manifest"], stages=computed["stages"],
                    transitions=computed["transitions"], outcome=computed["outcome"],
                )
        if publication["ack"].get("committed") is not True:
            raise EvidenceIntegrityError(
                "Optimization terminal was not durably acknowledged",
                stage="operation_store", evidence={"operation_id": trace.operation_id,
                                                      "attempt_id": attempt_id},
            )
        terminal_acknowledged = True
        for name in ("result_ref", "manifest_ref", "comparison_ref", "index_root_ref",
                     "completion_ref", "attempt_ref"):
            reference = publication.get(name)
            if reference is not None:
                trace.add_numerical_ref(reference)

        with trace.span("result_decode"):
            with binding.reader():
                result = storage.read_operation_attempt(
                    binding, attempt_id,
                    projection_consumer=lambda success: _decode(
                        success, decoder=decoder, prepared_analysis=prepared_analysis,
                        bound_spec=bound_spec,
                    ),
                )
        if result is None:
            raise EvidenceIntegrityError(
                "The sealed Optimization attempt could not be read back",
                stage="result_decode", evidence={"attempt_id": attempt_id},
            )
        final_phase = "complete" if computed["outcome"] == "success" else "partial"
        _physical_progress(on_progress, request, latest_observation,
                           phase=final_phase, trace=trace)
        if observer_enabled:
            values = dict(latest_observation)
            values["phase"] = final_phase
            observe_progress(**values)
        return result
    except BaseException as error:
        if not registration_committed and registration_attempted:
            outcome = getattr(error, "operation_registration_outcome", None)
            if outcome is None:
                outcome = getattr(error, "operation_transaction_outcome", None)
            registration_committed = isinstance(outcome, Mapping) and outcome.get("status") == "committed"
            if registration_committed:
                committed_registration = getattr(error, "operation_registration_result", None)
                if isinstance(committed_registration, Mapping):
                    binding_info = committed_registration.get("attempt_binding", {})
                    task_id = binding_info.get("task_id", task_id)
                    attempt_id = binding_info.get("attempt_id", attempt_id)
                    operation_details["request"] = committed_registration.get("request_ref")
                    trace.bind(operation="optimize_direct", request_sha256=request_sha,
                               task_id=task_id, environment_sha256=environment_sha,
                               attempt_id=attempt_id, details=operation_details)
            elif isinstance(outcome, Mapping) and outcome.get("status") == "unknown":
                error.add_note("Optimization registration outcome is unknown; no task status was retried.")

        transaction_outcome = getattr(error, "operation_transaction_outcome", None)
        transaction_status = (transaction_outcome.get("status")
                              if isinstance(transaction_outcome, Mapping) else None)
        transaction_uncertain = transaction_status in {"committed", "unknown"}
        uncertain_terminal = terminal_attempted and not terminal_acknowledged and transaction_uncertain
        preserve_attempt_state = terminal_acknowledged or transaction_uncertain
        if registration_committed and not preserve_attempt_state:
            try:
                failure = storage.error_document(error)
                interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                checkpoint_state = (None if latest_checkpoint_ref is None else {
                    "reference": latest_checkpoint_ref,
                    "stage_id": latest_checkpoint_stage,
                })
                with binding.writer():
                    storage.update_attempt(
                        workspace, binding=binding, operation_id=trace.operation_id,
                        task_id=task_id, attempt_id=attempt_id,
                        status="interrupted" if interrupted else "failure",
                        interruption=failure if interrupted else None,
                        failure=None if interrupted else failure,
                        checkpoint=checkpoint_state,
                    )
            except BaseException as recording_error:
                error.add_note(
                    f"Optimization failure recording also failed: "
                    f"{type(recording_error).__name__}: {recording_error}"
                )
        elif terminal_acknowledged:
            error.add_note(
                "Optimization terminal publication was acknowledged; a later result readback or "
                "notification failed, and the committed attempt status was preserved."
            )
        elif uncertain_terminal:
            error.add_note("Optimization terminal publication was uncertain; no attempt status was overwritten.")
        elif transaction_uncertain:
            if transaction_status == "unknown":
                error.add_note(
                    "An earlier operation transaction outcome was unknown; "
                    "no attempt status was overwritten."
                )
            else:
                error.add_note(
                    "An earlier operation transaction committed before the later failure; "
                    "no attempt status was overwritten."
                )
        raise
    finally:
        try:
            spool.close()
        except BaseException as cleanup_error:
            active = sys.exception()
            if active is not None:
                active.add_note(
                    f"Optimization spool cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            else:
                raise


def execute_jax_operation(*, binding, plan_document, prepared_analysis, decoder, bound_spec,
                          on_progress=None, progress_observer=None,
                          checkpoint_policy="generation", resume_from=None,
                          restart_from=None, commit_every_generations=1,
                          program_owner=None, trace):
    """Execute or reuse one request while serializing its exact result authority."""
    with request_execution_lease(binding, prepared_analysis.request_sha256):
        return _execute_jax_operation(
            binding=binding,
            plan_document=plan_document,
            prepared_analysis=prepared_analysis,
            decoder=decoder,
            bound_spec=bound_spec,
            on_progress=on_progress,
            progress_observer=progress_observer,
            checkpoint_policy=checkpoint_policy,
            resume_from=resume_from,
            restart_from=restart_from,
            commit_every_generations=commit_every_generations,
            program_owner=program_owner,
            trace=trace,
        )


def _execute_jax_operation(*, binding, plan_document, prepared_analysis, decoder, bound_spec,
                           on_progress=None, progress_observer=None,
                           checkpoint_policy="generation", resume_from=None,
                           restart_from=None, commit_every_generations=1,
                           program_owner=None, trace):
    """Execute or reuse one request with callbacks outside Workspace locks."""
    request = prepared_analysis.request()
    request_sha = prepared_analysis.request_sha256
    optimization = request["operation"] == "optimize_direct"
    if resume_from is not None and (not optimization or checkpoint_policy != "generation"):
        raise RuntimePreparationError("Only generation-checkpointed JAX Optimization can resume",
                                      stage="resume_prepare")
    if resume_from is not None and restart_from is not None:
        raise RuntimePreparationError("Exact resume and accepted-iterate restart are distinct operations",
                                      stage="resume_prepare")

    # A resume is an explicit request to continue its exact state, never a cache
    # lookup that silently disregards the supplied checkpoint.
    if resume_from is None and restart_from is None:
        with trace.span("result_lookup"):
            with binding.reader():
                def decode_selected(success):
                    with trace.span("result_decode"):
                        result = _decode(success, decoder=decoder, prepared_analysis=prepared_analysis,
                                         bound_spec=bound_spec)
                    return success, result

                selected = storage.find_operation_success(
                    binding, request_sha, projection_consumer=decode_selected,
                )
        if selected is None:
            success = None
        else:
            success, result = selected
        if success is not None:
            trace.bind(operation=request["operation"], request_sha256=request_sha,
                       task_id=success["task_id"], environment_sha256=success["environment_sha256"],
                       attempt_id=None, details={"cache_hit": True, "checkpoint_policy": checkpoint_policy,
                                                 "checkpoint_available": success["checkpoint_available"],
                                                 "requested_commit_every_generations": (
                                                     commit_every_generations if optimization else None
                                                 ),
                                                 "commit_every_generations": None})
            trace.add_numerical_ref(success["result_ref"])
            # Cache-hit trace rows are global operation facts; no fictitious
            # numerical attempt is allocated to store these observations.
            if optimization:
                stages = success["fixed_reader"].project("stages")
                if not stages:
                    raise EvidenceIntegrityError(
                        "A completed Optimization result has no entered stage",
                        stage="result_projection", evidence={"attempt_id": success["attempt_id"]},
                    )
                stage = stages[-1]
                strategy = stage["strategy"]
                cma = strategy == "cma_es"
                count = stage["iteration_count"]
                best = stage.get("observed_best")
                best_cost = (None if best is None else
                             _candidate_cost_from_hex(best["cost_f64"]))
                values = {
                    "stage_id": stage["stage_id"],
                    "evaluated_count": stage["evaluation_count"],
                    "best_cost": best_cost,
                    "completed_generations": count if cma else None,
                    "committed_generations": count if cma else None,
                    "completed_iterations": count if not cma else None,
                    "committed_iterations": count if not cma else None,
                }
                if progress_observer is not None:
                    progress_observer.update(
                        phase="result_reuse", stage_id=stage["stage_id"],
                        strategy="cmaes" if cma else "lbfgsb",
                        completed_generations=values["completed_generations"],
                        total_generations=(_physical_optimization_controls(
                            request, stage["stage_id"]
                        ).get("complete_generations") if cma else None),
                        completed_iterations=values["completed_iterations"],
                        total_iterations=(_physical_optimization_controls(
                            request, stage["stage_id"]
                        ).get("max_iterations") if not cma else None),
                        evaluated_count=values["evaluated_count"], best_cost=best_cost,
                        committed_generations=values["committed_generations"],
                        committed_iterations=values["committed_iterations"], reused=True,
                    )
                _physical_progress(on_progress, request, values,
                                   phase="result_reuse", reused=True, trace=trace)
                if progress_observer is not None:
                    progress_observer.update(phase="complete", reused=True, **{
                        key: progress_observer_value for key, progress_observer_value in (
                            ("stage_id", stage["stage_id"]),
                            ("strategy", "cmaes" if cma else "lbfgsb"),
                            ("completed_generations", values["completed_generations"]),
                            ("total_generations", _physical_optimization_controls(
                                request, stage["stage_id"]
                            ).get("complete_generations") if cma else None),
                            ("completed_iterations", values["completed_iterations"]),
                            ("total_iterations", _physical_optimization_controls(
                                request, stage["stage_id"]
                            ).get("max_iterations") if not cma else None),
                            ("evaluated_count", values["evaluated_count"]),
                            ("best_cost", best_cost),
                            ("committed_generations", values["committed_generations"]),
                            ("committed_iterations", values["committed_iterations"]),
                        )
                    })
                _physical_progress(on_progress, request, values,
                                   phase="complete", reused=True, trace=trace)
            return result

    from .jax_backend import get_jax_backend
    from .resources import operation_resources

    resources = request["runtime_semantic"]["resources"]
    precision = request["runtime_semantic"]["precision"]
    if runtime_resource_identity() != resources:
        raise RuntimePreparationError("Prepared JAX resources changed before execution", stage="runtime_prepare")
    runtime_configuration = get_runtime_configuration()
    with operation_resources(cpu_threads=runtime_configuration.cpu_threads) as operation_resource:
        with trace.span("backend_prepare"):
            backend = get_jax_backend(
                precision=precision, resources=runtime_configuration,
                trace=None if optimization else trace,
                operation_resources=operation_resource, diagnostics=not optimization,
            )

        if optimization:
            primary = None
            try:
                return _execute_physical_optimization(
                    binding=binding, plan_document=plan_document,
                    prepared_analysis=prepared_analysis, decoder=decoder,
                    bound_spec=bound_spec, on_progress=on_progress,
                    progress_observer=progress_observer,
                    checkpoint_policy=checkpoint_policy, resume_from=resume_from,
                    restart_from=restart_from,
                    commit_every_generations=commit_every_generations,
                    program_owner=program_owner, trace=trace, request=request,
                    request_sha=request_sha, precision=precision, resources=resources,
                    runtime_configuration=runtime_configuration, backend=backend,
                    operation_resource=operation_resource,
                )
            except BaseException as error:
                primary = error
                raise
            finally:
                try:
                    backend.close()
                except BaseException as cleanup_error:
                    if primary is None:
                        raise
                    primary.add_note(
                        f"Optimization backend cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )

        return _execute_with_backend(
            binding=binding, plan_document=plan_document, prepared_analysis=prepared_analysis,
            decoder=decoder, bound_spec=bound_spec, checkpoint_policy=checkpoint_policy,
            commit_every_generations=commit_every_generations, trace=trace,
            request=request, request_sha=request_sha, precision=precision,
            resources=resources, backend=backend, backend_identity=backend.identity(),
            operation_resource=operation_resource,
            program_owner=program_owner,
        )


def _execute_with_backend(*, binding, plan_document, prepared_analysis, decoder, bound_spec,
                          checkpoint_policy, commit_every_generations, trace, request,
                          request_sha, precision, resources, backend, backend_identity,
                          operation_resource, program_owner=None):
    """Run one ordinary JAX analysis through its task and result lifecycle."""
    from .python_host import execute_analysis

    writer = None
    attempt_id = None
    registration_attempted = False
    registration_committed = False
    completed = False
    workspace = storage.operation_workspace(binding)
    details = {
        "cache_hit": False,
        "checkpoint_policy": checkpoint_policy,
        "requested_commit_every_generations": None,
        "commit_every_generations": None,
    }
    try:
        arm = f"{trace.method}/jax/{precision}/{checkpoint_policy}"
        environment = environment_snapshot(
            arm=arm, device="cpu", cpu_threads=resources["cpu_threads"],
            backend=backend_identity,
        )
        environment["runtime_semantic"] = request["runtime_semantic"]
        environment["resources"] = resources
        environment_sha = environment_identity(environment)
        environment["environment_sha256"] = environment_sha
        task_id = operation_task_identifier(
            plan_sha256=request["plan_sha256"], request_sha256=request_sha,
            method=trace.method, backend="jax", precision=precision,
            resources=resources,
            algorithm_id=request["runtime_semantic"]["algorithm_id"],
            environment_sha256=environment_sha,
            checkpoint_policy=checkpoint_policy,
            commit_every_generations=commit_every_generations,
        )
        attempt_id = str(uuid4())
        trace.bind(operation=request["operation"], request_sha256=request_sha,
                   task_id=task_id, environment_sha256=environment_sha,
                   attempt_id=None, details=details)
        trace.set_attempt(attempt_id)
        registration_attempted = True
        with binding.writer():
            registration = storage.register_operation_execution(
                binding,
                row=trace.root_row(),
                task={"task_id": task_id, "request_sha256": request_sha, "arm": arm,
                      "sample": 0, "attempts": [], "events": [], "measurements": [],
                      "environment": environment, "artifacts": []},
                request={"request_sha256": request_sha,
                         "request_bytes": prepared_analysis.request_bytes},
                attempt_id=attempt_id,
            )
            if registration["ack"].get("committed") is not True:
                raise EvidenceIntegrityError(
                    "Operation execution registration was not durably acknowledged.",
                    stage="operation_store", evidence={"operation_id": trace.operation_id},
                )
            registration_committed = True
            details["request"] = registration["request_ref"]
            trace.bind(operation=request["operation"], request_sha256=request_sha,
                       task_id=task_id, environment_sha256=environment_sha,
                       attempt_id=attempt_id, details=details)
            writer = storage.begin_task_writer(
                workspace, binding=binding, operation_id=trace.operation_id,
                task_id=task_id, attempt_id=attempt_id, diagnostics="boundary",
            )

        def emit(kind, payload):
            value = {**payload, "attempt_id": attempt_id}
            if kind == "timing":
                trace.measure(
                    payload["stage"], start_tick_ns=payload["start_tick_ns"],
                    end_tick_ns=payload["end_tick_ns"],
                    details={"stage": payload["stage"], "counts": payload.get("counts", {})},
                )
                return None
            if kind == "baseline_ready":
                _, ack = storage.commit_barrier(
                    workspace, writer=writer, kind=kind, payload=value,
                )
                if ack.get("checkpoint") is not None:
                    trace.add_numerical_ref(ack["checkpoint"])
                if ack.get("committed") is True:
                    trace.publish_diagnostics()
                return ack
            if storage.is_diagnostic_event(kind):
                return storage.append_event(
                    workspace, task_id=task_id, kind=kind, payload=value, writer=writer,
                )
            with binding.writer():
                return storage.append_event(
                    workspace, task_id=task_id, kind=kind, payload=value, writer=writer,
                )

        terminal = execute_analysis(
            plan_document, prepared_analysis, backend=backend, emit=emit, trace=trace,
            checkpoint=None, checkpoint_policy=checkpoint_policy,
            operation_resources=operation_resource, program_owner=program_owner,
        )
        with trace.span("result_publish"):
            with binding.writer():
                result_ref, _ = storage.complete_operation(
                    binding, writer=writer, terminal_bytes=record_bytes(terminal),
                )
                completed = True
        trace.add_numerical_ref(result_ref)
        with trace.span("result_decode"):
            with binding.reader():
                return storage.read_operation_success(
                    binding, task_id, attempt_id=attempt_id,
                    projection_consumer=lambda success: _decode(
                        success, decoder=decoder, prepared_analysis=prepared_analysis,
                        bound_spec=bound_spec,
                    ),
                )
    except BaseException as error:
        if attempt_id is not None and not completed:
            try:
                if not registration_committed and registration_attempted:
                    outcome = getattr(error, "operation_registration_outcome", None)
                    if outcome is None:
                        outcome = getattr(error, "operation_transaction_outcome", None)
                    registration_committed = (
                        isinstance(outcome, dict) and outcome.get("status") == "committed"
                    )
                    if (not registration_committed and isinstance(outcome, dict)
                            and outcome.get("status") == "unknown"):
                        error.add_note(
                            "Execution registration outcome is unknown; no task failure update was retried."
                        )
                    if not registration_committed:
                        trace.bind(operation=request["operation"], request_sha256=request_sha,
                                   task_id=None, environment_sha256=None, attempt_id=None,
                                   details=details)
                    else:
                        committed_registration = getattr(error, "operation_registration_result", None)
                        if isinstance(committed_registration, dict):
                            details["request"] = committed_registration["request_ref"]
                            attempt_binding = committed_registration["attempt_binding"]
                            trace.bind(operation=request["operation"], request_sha256=request_sha,
                                       task_id=attempt_binding["task_id"],
                                       environment_sha256=attempt_binding["environment_sha256"],
                                       attempt_id=attempt_binding["attempt_id"], details=details)
                if registration_committed and writer is None:
                    try:
                        with binding.writer():
                            writer = storage.begin_task_writer(
                                workspace, binding=binding, operation_id=trace.operation_id,
                                task_id=task_id, attempt_id=attempt_id, diagnostics="boundary",
                            )
                    except BaseException as writer_setup_error:
                        error.add_note(
                            "Task writer initialization also failed after execution registration: "
                            f"{type(writer_setup_error).__name__}: {writer_setup_error}"
                        )
                if registration_committed:
                    failure = storage.error_document(error)
                    interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                    if writer is not None:
                        with binding.writer():
                            storage.append_event(
                                workspace, task_id=task_id,
                                kind="interrupted" if interrupted else "failed",
                                payload={"attempt_id": attempt_id,
                                         "interruption" if interrupted else "failure": failure},
                                writer=writer, force=True,
                            )
                            storage.update_attempt(
                                workspace, binding=binding, operation_id=trace.operation_id,
                                task_id=task_id, attempt_id=attempt_id,
                                status="interrupted" if interrupted else "failure",
                                interruption=failure if interrupted else None,
                                failure=None if interrupted else failure,
                            )
                    else:
                        with binding.writer():
                            storage.update_attempt(
                                workspace, binding=binding, operation_id=trace.operation_id,
                                task_id=task_id, attempt_id=attempt_id,
                                status="interrupted" if interrupted else "failure",
                                interruption=failure if interrupted else None,
                                failure=None if interrupted else failure,
                            )
            except BaseException as recording_error:
                error.add_note(
                    f"Operation failure recording also failed: "
                    f"{type(recording_error).__name__}: {recording_error}"
                )
        raise
    finally:
        primary = sys.exception()
        if backend is not None:
            try:
                backend.close()
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    f"Operation finalization also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

