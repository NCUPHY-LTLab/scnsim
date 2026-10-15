"""Shared checked analysis payloads and ephemeral execution ownership.

PhysicalAnalysis and native_ad remain the sole numerical/derivative producers.
This module owns only operation-local RAM sinks and synchronized host projection;
Results and Runtime own public models/lifecycle. It has no Workspace or disk sink.
"""
from __future__ import annotations
from collections.abc import Mapping
from types import MappingProxyType
import numpy as np

def project_sensitivity_host_blocks(quantity_program, prepared_analysis, request, derivative_outcome):
    """Project one checked derivative into ordered immutable primal/result blocks."""
    import numpy as np
    from .jax_operation import _sensitivity_reason, _sensitivity_equation_evidence

    spec = request["spec"]
    derivative_spec = spec["derivative"]
    derivative_kind = derivative_spec["type"]
    quantity_metadata = quantity_program.quantity_metadata()
    # Keep the compiler's factorized output blocks. Expanding every scalar
    # OutputRow here duplicates O(F·P²) Python metadata during readback.
    output_blocks = {
        block.quantity_key: block
        for block in quantity_program.output_layout.blocks
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
            "output_slice": {"start": 0, "stop": output_blocks[key].row_count},
            "parameter_slice": None,
            "summary_json": {"value_unit": row["si_unit"], "axes": row["axes"]},
            "primal_blocks": [],
            "primal_status": "complete",
            "payload": {"values": primal.values[row["observable_index"]]},
        })

    def append_derivative(*, block_id, row, values, unavailable, parameter_slice,
                          primal_links, parameters=()):
        key = row["key"]
        axes = row["axes"]
        summary = {"axes": axes, "parameters": list(parameters)}
        if key is not None:
            summary["value_unit"] = row["si_unit"]
        if unavailable is None:
            payload = {"values": values}
            status = "complete"
        else:
            payload = {"values": None, "reason": unavailable}
            status = "unavailable"
        blocks.append({
            "block_id": block_id,
            "kind": "derivative",
            "quantity_id": key,
            "output_slice": (None if key is None else
                             {"start": 0, "stop": output_blocks[key].row_count}),
            "parameter_slice": parameter_slice,
            "summary_json": summary,
            "primal_blocks": list(primal_links),
            "primal_status": "complete",
            "derivative_status": status,
            "payload": payload,
        })

    if derivative_kind == "jacobian":
        payload = derivative_outcome.derivative
        if payload is None:
            row_positions = {}
            matrix = None
        else:
            row_positions = {row_index: index for index, row_index in enumerate(payload.row_indices)}
            matrix = np.asarray(payload.values)
        for row in quantity_metadata:
            key = row["key"]
            block = output_blocks[key]
            selected_positions = [row_positions.get(index)
                                  for index in range(block.offset, block.offset + block.row_count)]
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
            value = (None if derivative_outcome.derivative is None else
                     derivative_outcome.derivative[row["observable_index"]])
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
            indices = quantity_program.program.model_program.fields.layout.input_indices(
                prepared_analysis.active_keys
            )
            values = np.asarray(value)[np.asarray(indices, dtype=np.intp)]
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
        "equation_evidence": _sensitivity_equation_evidence(
            quantity_program, request, derivative_outcome,
        ),
    }


def _freeze_host(value, memo=None):
    """Detach synchronized typed host data once, preserving shared references."""
    from dataclasses import fields, is_dataclass, replace
    memo = {} if memo is None else memo
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, Mapping):
        detached = MappingProxyType({key: _freeze_host(item, memo) for key, item in value.items()})
    elif hasattr(value, '_fields'):
        detached = type(value)(*(_freeze_host(item, memo) for item in value))
    elif isinstance(value, (tuple, list)):
        detached = tuple(_freeze_host(item, memo) for item in value)
    elif is_dataclass(value):
        detached = replace(value, **{field.name: _freeze_host(getattr(value, field.name), memo)
                                     for field in fields(value)})
    elif hasattr(value, 'shape') and hasattr(value, 'dtype'):
        array = np.asarray(value)
        detached = np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)
    else:
        detached = value
    memo[identity] = detached
    return detached


class InMemoryAnalysisSink:
    """Same ordered Sweep producer with typed public RAM occurrences.

    Work arrays exist only while one candidate is projected. Requested point
    outputs/history remain owned by the returned Result; no scratch spill occurs.
    """
    domain = 'sweep'
    in_memory_analysis = True

    def __init__(self, request):
        # Static C has no frequency work or RF Sweep sink. It uses the same
        # checked whole-point producer and this existing detached record owner.
        if request['spec']['type'] in ('static_capacitance', 'static_capacitance_element'):
            if request['parameter_source']['kind'] != 'point':
                raise TypeError('Static capacitance memory analysis requires one point')
            self.domain = 'point'
        self.request = request
        self.metadata = None
        self.baseline = None
        self.points = []
        self._cache = []
        self._work = {}
        self._continuations = {}

    def set_metadata(self, metadata):
        self.metadata = dict(metadata)

    def stage_native_work(self, namespace, work_index, record):
        key = namespace, work_index
        if key in self._work:
            raise RuntimeError('native frequency work was staged twice in one candidate')
        self._work[key] = record

    def native_work(self, namespace, work_index):
        return self._work[namespace, work_index]

    def release_native_work(self, namespace):
        for key in tuple(self._work):
            if key[0] == namespace:
                del self._work[key]

    def cache_sweep_point(self, point):
        index = len(self._cache)
        self._cache.append(point)
        return index

    def sweep_point(self, reference):
        return self._cache[reference]

    def emit(self, kind, row):
        if kind == 'baseline_ready':
            return None
        if kind != 'evaluation':
            raise TypeError(f'unsupported ephemeral analysis event {kind!r}')
        if row.get('origin') == 'root_continuation':
            self._continuations.setdefault(row['source_index'], []).append(row)
            return None
        if row.get('origin') == 'baseline':
            self.baseline = row
            return self.cache_sweep_point(row)
        copied = dict(row)
        copied['continuations'] = tuple(self._continuations.pop(row['source_index'], ()))
        self.points.append(copied)
        return self.cache_sweep_point(copied)

    def finish(self, terminal, execution_id):
        def point(source):
            if source is None:
                return None
            # IDs here are operation-local numerical occurrence labels, never
            # persistent body/request/attempt/result identities.
            return dict(source, origin='baseline' if source.get('origin') == 'baseline' else 'point')
        payload = _freeze_host({'execution_id':execution_id,'persistence':'memory',
            'operation':self.request['operation'],'result_kind':self.request['spec']['type'],
            'spec':self.request['spec'],'parameter_source':self.request['parameter_source'],
            'metadata':self.metadata,'baseline':point(self.baseline),
            'points':tuple(point(row) for row in self.points),'summary':terminal})
        self.clear()
        return payload

    def clear(self):
        self._work.clear()
        self._cache.clear()
        self._continuations.clear()
        self.points.clear()
        self.baseline = None


def _require_prepared_jax(prepared_analysis):
    from ..errors import RuntimePreparationError
    from .config import get_runtime_configuration, runtime_resource_identity
    request = prepared_analysis.request()
    if request['runtime_semantic']['backend'] != 'jax':
        raise RuntimePreparationError('Memory analysis requires the prepared JAX backend',stage='memory_analysis_prepare')
    if runtime_resource_identity() != request['runtime_semantic']['resources']:
        raise RuntimePreparationError('Prepared JAX resources changed before execution',stage='runtime_prepare')
    return request, get_runtime_configuration()


def _close_backend(backend, primary):
    if backend is not None:
        try:
            backend.close()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f'Memory analysis backend cleanup also failed: {type(error).__name__}: {error}')


def execute_memory_analysis(*, plan_document, prepared_analysis, trace,
                            program_owner, execution_policy):
    """Run the shared ordered numerical producer into ephemeral typed payloads."""
    request, configuration = _require_prepared_jax(prepared_analysis)
    if request['operation'] == 'solve_direct' and request['parameter_source']['kind'] == 'point':
        # Existing bounded Direct sink is retained as the single Direct adapter.
        from .memory_direct import execute_memory_direct
        return execute_memory_direct(plan_document=plan_document,prepared_analysis=prepared_analysis,
            trace=trace,program_owner=program_owner,execution_policy=execution_policy)
    if request['operation'] not in ('solve_direct','evaluate_direct'):
        raise TypeError(f'unsupported memory analysis operation {request["operation"]!r}')
    from .jax_backend import get_jax_backend
    from .python_host import execute_analysis
    from .resources import operation_resources
    sink = InMemoryAnalysisSink(request)
    backend = primary = None
    try:
        with operation_resources(cpu_threads=configuration.cpu_threads) as operation_resource:
            with trace.span('backend_prepare'):
                backend = get_jax_backend(precision=request['runtime_semantic']['precision'],
                    resources=configuration,trace=trace,operation_resources=operation_resource,diagnostics=True)
            terminal = execute_analysis(plan_document,prepared_analysis,backend=backend,
                emit=sink.emit,trace=trace,operation_resources=operation_resource,
                program_owner=program_owner,execution_policy=execution_policy,result_sink=sink)
            return sink.finish(terminal,trace.execution_id)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            _close_backend(backend,primary)
        finally:
            sink.clear()


def execute_memory_sensitivity(*, plan_document, prepared_analysis, trace,
                               program_owner, execution_policy):
    """Run the same bounded checked native AD and project shared host blocks."""
    request, configuration = _require_prepared_jax(prepared_analysis)
    if request['operation'] != 'sensitivity':
        raise TypeError('memory Sensitivity requires a prepared sensitivity request')
    from .jax_backend import get_jax_backend
    from .resources import operation_resources
    from .compile_owner import CompileOwner
    from .jax_operation import _native_sensitivity_transform
    from ..compilation.quantity_program import compile_quantity_program
    from ..compilation.compiler import parameter_key
    frame_owner = CompileOwner(policy=execution_policy)
    backend = primary = None
    try:
        with operation_resources(cpu_threads=configuration.cpu_threads) as operation_resource:
            backend = get_jax_backend(precision=request['runtime_semantic']['precision'],
                resources=configuration,trace=trace,operation_resources=operation_resource,diagnostics=True)
            spec = request['spec']
            point = prepared_analysis.effective_point()
            program = compile_quantity_program(plan_document,
                {row['id']:row['quantity'] for row in spec['quantities']},point,
                mesh=spec.get('discretization'),precision=backend.precision,
                authorized=frozenset(parameter_key(ref) for ref in point.get('allow_extrapolation',())),
                program_owner=program_owner,admit_program=False)
            inputs = program.inputs(point)
            derivative_kind = spec['derivative']['type']
            with trace.span('sensitivity_evaluation',details={'derivative':derivative_kind}):
                outcome = _native_sensitivity_transform(program,inputs,prepared_analysis,
                    derivative_kind=derivative_kind,backend=backend,program_owner=program_owner,
                    execution_policy=execution_policy,frame_owner=frame_owner,compact_evidence=True)
            projected = project_sensitivity_host_blocks(program,prepared_analysis,request,outcome)
            primal_blocks = {row['quantity_id']:row for row in projected['blocks'] if row['kind']=='primal'}
            derivatives = {row['quantity_id']:row for row in projected['blocks'] if row['kind']=='derivative'}
            blocks = []
            for row in program.quantity_metadata():
                actual = primal_blocks[row['key']]
                derivative = derivatives.get(row['key'])
                available = (bool(np.asarray(outcome.aux.observable_available[row['observable_index']]))
                    if derivative is None else derivative['derivative_status']=='complete')
                blocks.append({'quantity_key':row['key'],'observable_index':row['observable_index'],
                    'axes':row['axes'],'si_unit':row['si_unit'],'primal':actual['payload']['values'],
                    'derivative':None if derivative is None else derivative['payload']['values'],
                    'available':available,'reason':None if derivative is None else derivative['payload'].get('reason'),
                    'output_slice':actual['output_slice'],'parameter_keys':tuple(spec['wrt'])})
            aggregate = derivatives.get(None)
            functional = (None if aggregate is None else {'derivative':aggregate['payload']['values'],
                'available':aggregate['derivative_status']=='complete','reason':aggregate['payload'].get('reason'),
                'parameter_keys':tuple(spec['wrt']),'unit':spec['functional_unit']})
            layout = program.program.model_program.fields.layout
            indices = layout.input_indices(prepared_analysis.active_keys)
            parameter_layout = tuple({'parameter':{'definitions_id':layout.axes[index].key[0],
                'parameter_id':layout.axes[index].key[1]}, 'label':layout.axes[index].name,
                'si_unit':layout.axes[index].si_unit,'input_index':index,
                'matrix_field':layout.axes[index].matrix_field,'matrix_entry':layout.axes[index].matrix_entry}
                for index in indices)
            return _freeze_host({'execution_id':trace.execution_id,'persistence':'memory',
                'operation':'sensitivity','spec':spec,'parameters':point,
                'quantity_metadata':program.quantity_metadata(),'observable_metadata':program.observable_metadata(),'context':program.point_metadata(inputs),
                'output_layout':program.output_layout.compact_record(),
                'parameter_layout':parameter_layout,'derivative_kind':derivative_kind,
                'primal':outcome.primal,'derivative':outcome.derivative,'aux':outcome.aux,
                'blocks':tuple(blocks),'functional':functional,'summary':projected['summary'],
                'manifest':projected['manifest'],'outcome':projected['outcome'],
                'equation_evidence':dict(projected['equation_evidence'],certificates=outcome.aux.certificates)})
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            _close_backend(backend,primary)
        finally:
            frame_owner.clear()
