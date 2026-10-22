"""Maintained M0 raw-byte observations and deterministic capture comparison.

This harness owns benchmark evidence, never physical equations or acceptance.
Capture every public value and staged Direct record; private owners/readers are
not serialized. Numeric leaves use actual dtype/shape and SHA-bound raw bytes.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import struct
import threading
import traceback
import uuid


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _semantic_projection(document, directory, *, operation=None):
    """Separate Human-excluded provenance; preserve every other captured field.

    Hashes that bind numeric blobs stay numeric evidence. Named identity hashes,
    execution UUIDs, recorded timings and formatted tracebacks are reported in
    the adjacent provenance document, with a typed marker at their old position.
    """
    excluded = []
    # These are producer identity fields in the captured Result/context/View
    # and canonical evidence records. Never classify by a suffix or hash value
    # alone: a blob's `sha256`, scientific IDs and unknown fields stay semantic.
    identity_fields = {'plan_sha256', 'program_sha256', 'request_sha256',
                       'attempt_sha256', 'result_sha256', 'python_source_sha256',
                       'environment_sha256'}
    execution_uuid_fields = {'execution_id', 'operation_id', 'terminal_id'}
    timing_fields = {'elapsed_ns', 'duration_ns', 'start_ns', 'end_ns',
                     'elapsed_ns_sum', 'duration_seconds'}

    backend_paths = (('outcome','result','fields','metadata','backend_identity'),
                     ('outcome','result','fields','metadata','terminal','backend'))
    direct_optimizer_inputs = {'optimization_parameter_set',
                               'optimization_initial_parameters', 'optimization_point_policy'}

    def category(key, value, schema_path):
        if schema_path[:-1] in backend_paths:
            if key == 'initialization_ns' and isinstance(value,int) and not isinstance(value,bool):
                return 'timing'
            if (key == 'cpu_affinity' and isinstance(value,dict)
                    and set(value) == {'sequence','type'} and value['type'] == 'tuple'
                    and isinstance(value['sequence'],list)
                    and all(isinstance(item,int) and not isinstance(item,bool) for item in value['sequence'])):
                return 'environment_provenance'
        if (key in identity_fields and isinstance(value,str) and len(value) == 64
                and all(character in '0123456789abcdef' for character in value)):
            return 'identity_hash'
        if (key == 'formatted_traceback' and isinstance(value,str)
                and any(name in ('exception','cause','context','failure') for name in schema_path)):
            return 'formatted_traceback'
        if key in timing_fields and any(name in ('timings','timing_batches') for name in schema_path) and (isinstance(value,(int,float)) and not isinstance(value,bool)
                or isinstance(value,dict) and set(value) == {'float64_bits'}):
            return 'timing'
        if key in execution_uuid_fields and isinstance(value, str):
            try:
                parsed = uuid.UUID(value)
            except ValueError:
                pass
            else:
                if str(parsed) == value:
                    return 'execution_uuid'
        return None

    # Only declared public Optimization Result carriers establish locator
    # context. Similar-looking mappings in physical evidence remain untouched.
    locator_nodes = set()
    result = document.get('outcome', {}).get('result', {})
    if (isinstance(result, dict) and result.get('class') == 'scnsim.results.OptimizationResult'
            and isinstance(result.get('fields'), dict)):
        result_fields = result['fields']
        best = result_fields.get('best')
        if (isinstance(best, dict) and best.get('class') == 'scnsim.results.OptimizationBest'
                and isinstance(best.get('fields'), dict)):
            locator_nodes.add(id(best['fields'].get('locator')))
        stages = result_fields.get('stages')
        if (isinstance(stages, dict) and set(stages) == {'sequence','type'}
                and stages['type'] == 'tuple' and isinstance(stages['sequence'], list)):
            for stage in stages['sequence']:
                if (not isinstance(stage, dict)
                        or stage.get('class') != 'scnsim.results.optimization.OptimizationStage'
                        or not isinstance(stage.get('fields'), dict)):
                    continue
                stage_fields = stage['fields']
                observed = stage_fields.get('observed_best')
                if (isinstance(observed, dict)
                        and observed.get('class') == 'scnsim.results.OptimizationBest'
                        and isinstance(observed.get('fields'), dict)):
                    locator_nodes.add(id(observed['fields'].get('locator')))
                history = stage_fields.get('history')
                if (isinstance(history, dict) and set(history) == {'sequence','type'}
                        and history['type'] in ('tuple','LazyOptimizationStageHistory')
                        and isinstance(history['sequence'], list)):
                    for row in history['sequence']:
                        if isinstance(row, dict) and set(row) == {'mapping'}:
                            pairs = row['mapping']
                            if (isinstance(pairs, list)
                                    and all(isinstance(pair, list) and len(pair) == 2
                                            and isinstance(pair[0], str) for pair in pairs)
                                    and len({pair[0] for pair in pairs}) == len(pairs)):
                                locator_nodes.add(id(dict(pairs).get('locator')))

    locator_identity_paths = {}

    def closed_mapping(value):
        if not isinstance(value, dict) or set(value) != {'mapping'}:
            return None
        pairs = value['mapping']
        if (not isinstance(pairs, list)
                or not all(isinstance(pair, list) and len(pair) == 2
                           and isinstance(pair[0], str) for pair in pairs)
                or len({pair[0] for pair in pairs}) != len(pairs)):
            return None
        return dict(pairs)

    def digest(value):
        return (isinstance(value, str) and len(value) == 64
                and all(character in '0123456789abcdef' for character in value))

    def register_locator(value, path):
        fields = closed_mapping(value)
        required = {'kind','stage_id','evaluation_ordinal','candidate_ref',
                    'source_task_id','source_attempt_id','source_batch_ref'}
        if (fields is None or not required.issubset(fields)
                or not set(fields).issubset(required | {'cost_f64'})
                or fields['kind'] not in ('baseline','evaluation')
                or not isinstance(fields['stage_id'], str)
                or not isinstance(fields['evaluation_ordinal'], int)
                or isinstance(fields['evaluation_ordinal'], bool)
                or not digest(fields['source_task_id'])
                or not isinstance(fields['source_attempt_id'], str)):
            return
        try:
            if str(uuid.UUID(fields['source_attempt_id'])) != fields['source_attempt_id']:
                return
        except ValueError:
            return
        reference_keys = {'byte_length','database','role','schema_version','sha256','storage'}
        for name, role in (('candidate_ref','optimization_candidate'),
                           ('source_batch_ref','optimization_stage_batch')):
            reference = closed_mapping(fields[name])
            if (reference is None or set(reference) != reference_keys
                    or reference['database'] != 'operations.sqlite3'
                    or reference['storage'] != 'sqlite' or reference['schema_version'] != 7
                    or reference['role'] != role or not digest(reference['sha256'])
                    or not isinstance(reference['byte_length'], int)
                    or isinstance(reference['byte_length'], bool)):
                return
        # These exact object addresses bind storage envelopes containing
        # request/execution identity. Numeric blob digests stay semantic.
        for index, (name, item) in enumerate(value['mapping']):
            field_path = path+'/mapping/'+str(index)+'/value'
            if name == 'source_task_id':
                locator_identity_paths[field_path] = 'source_task_identity'
            elif name == 'source_attempt_id':
                locator_identity_paths[field_path] = 'execution_uuid'
            elif name in ('candidate_ref','source_batch_ref'):
                for ref_index, (ref_key, _) in enumerate(item['mapping']):
                    if ref_key == 'sha256':
                        locator_identity_paths[field_path+'/mapping/'+str(ref_index)+'/value'] = 'typed_optimization_object_address'

    def visit(value, path, key='', schema_path=()):
        if id(value) in locator_nodes:
            register_locator(value, path)
        kind = locator_identity_paths.get(path) or category(key, value, schema_path)
        if kind is not None:
            excluded.append({'path': path, 'category': kind, 'value': value})
            return {'excluded_provenance': kind, 'captured_type': type(value).__name__}
        if isinstance(value, dict):
            if key == 'evidence_bytes' and set(value) == {'bytes'}:
                blob = value['bytes']
                data = (directory/blob['file']).read_bytes()
                if len(data) != blob['bytes'] or hashlib.sha256(data).hexdigest() != blob['sha256']:
                    raise ValueError('captured evidence blob integrity mismatch: '+blob['file'])
                # Evidence is a closed producer JSON record. Its actual scalar
                # encodings and arrays remain intact; source hashes inside it
                # follow the same provenance rule as unencoded evidence.
                decoded = json.loads(data)
                return {'encoded_json': visit(decoded, path+'/encoded_json',schema_path=schema_path)}
            if set(value) == {'mapping'}:
                projected = []
                for k,v in value['mapping']:
                    # Only Direct's unused optimizer inputs/registry descriptions
                    # are tools provenance. Keep named physical operands and the
                    # relative order of every remaining captured mapping entry.
                    unused_optimizer = operation == 'direct' and (
                        schema_path == ('case_metadata',) and k in direct_optimizer_inputs
                        or schema_path == ('case_metadata','operations') and k in ('cma','lbfgsb'))
                    if unused_optimizer:
                        excluded.append({'path':path+'/mapping/key:'+json.dumps(k)+'/value',
                                         'category':'unused_optimizer_tool_input', 'value':v})
                        continue
                    i = len(projected)
                    projected.append([visit(k,path+'/mapping/'+str(i)+'/key'),
                                      visit(v,path+'/mapping/'+str(i)+'/value',k if isinstance(k,str) else '',
                                            schema_path+(k,) if isinstance(k,str) else schema_path)])
                return {'mapping':projected}
            return {k:visit(v,path+'/'+k,k,schema_path+(k,)) for k,v in value.items()}
        if isinstance(value, list):
            return [visit(item,path+'/'+str(i),schema_path=schema_path) for i,item in enumerate(value)]
        return value

    return visit(document, ''), excluded


class Capture:
    def __init__(self, directory):
        self.directory = Path(directory)
        (self.directory / "blobs").mkdir(parents=True, exist_ok=True)

    def blob(self, data):
        digest = hashlib.sha256(data).hexdigest()
        path = self.directory / "blobs" / (digest + ".bin")
        if not path.exists():
            path.write_bytes(data)
        return {"sha256": digest, "bytes": len(data), "file": "blobs/" + digest + ".bin"}

    def project(self, value):
        import numpy as np
        from pint import Quantity, Unit
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, np.generic):
            return self.project(np.asarray(value))
        if isinstance(value, float):
            return {"float64_bits": struct.pack("<d", value).hex()}
        if isinstance(value, complex):
            return {"complex128_bits": struct.pack("<dd", value.real, value.imag).hex()}
        if isinstance(value, bytes):
            return {"bytes": self.blob(value)}
        if isinstance(value, Path):
            return {"path": str(value)}
        if isinstance(value, Unit):
            return {"unit": str(value)}
        if isinstance(value, Quantity):
            return {"quantity": self.project(value.magnitude), "unit": str(value.units)}
        if isinstance(value, BaseException):
            return self.exception(value)
        if isinstance(value, np.ndarray) or type(value).__module__.startswith(("jax.", "jaxlib.")):
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError("golden capture does not serialize object arrays")
            return {"array": self.blob(array.tobytes(order="C")), "dtype": array.dtype.str,
                    "shape": list(array.shape), "order": "C"}
        if isinstance(value, Enum):
            return {"enum": type(value).__module__ + "." + type(value).__qualname__, "value": self.project(value.value)}
        if isinstance(value, Mapping):
            return {"mapping": [[self.project(key), self.project(item)] for key, item in value.items()]}
        if is_dataclass(value):
            return {"class": type(value).__module__ + "." + type(value).__qualname__,
                    "fields": {field.name: self.project(getattr(value, field.name))
                               for field in fields(value) if not field.name.startswith("_")}}
        if isinstance(value, (set, frozenset)):
            items = [self.project(item) for item in value]
            return {"set": sorted(items, key=_json), "type": type(value).__name__}
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return {"class": type(value).__module__ + "." + type(value).__qualname__,
                    "fields": {name: self.project(getattr(value, name)) for name in value._fields}}
        if isinstance(value, Sequence):
            return {"sequence": [self.project(item) for item in value], "type": type(value).__name__}
        # These authoring values have one existing canonical record authority.
        if type(value).__module__ == "scnsim.authoring.parameters" and type(value).__name__ == "ParameterSet":
            return {"parameter_set": self.project(value._record())}
        raise TypeError("unassigned golden capture type " + type(value).__module__ + "." + type(value).__qualname__)

    def checked_error(self, error):
        """Read every predicate/code/payload through the pinned public PyTree protocol."""
        (predicates, codes, payloads), _metadata = error.tree_flatten()
        effects = []
        for effect in sorted(payloads):
            kind = effect.error_type
            metadata = {name: self.project(getattr(effect, name))
                        for name in ("site", "role", "stage", "derivative_order", "dependency", "reason")
                        if hasattr(effect, name)}
            effects.append({"error_type": kind.__module__ + "." + kind.__qualname__,
                            "metadata": metadata, "predicate": self.project(predicates[effect]),
                            "code": self.project(codes[effect]),
                            "payload": self.project(payloads[effect])})
        return effects

    def materialize(self, records):
        """Materialize retained raw chunk observations after the public call."""
        from scnsim.numerics.native_sparse import sparse_certificates
        result = []
        for record in records:
            if "_raw_chunk" not in record:
                result.append(record); continue
            rows, outcome = record["_raw_chunk"]
            result.append({"role": "native_chunk_before_projection",
                           "source_rows": self.project(rows),
                           "check_status": self.project(outcome.check_status),
                           "certificate_status": self.project(outcome.certificate_status),
                           "checked_error": self.checked_error(outcome.errors),
                           "certificates": self.project(sparse_certificates(outcome.errors)),
                           "outcomes": self.project(outcome.outcomes)})
        return result

    def exception(self, error):
        from scnsim.errors import SCNSimError
        from scnsim.numerics.native_sparse import SparseEquationCheck
        args = error.args
        omitted = None
        if isinstance(error, SparseEquationCheck):
            sparse_payload, sparse_metadata = error.tree_flatten()
            traceback_owner, site, role, stage, derivative_order, dependency = sparse_metadata
            # Pinned checkify stores a live traceback owner as the first
            # constructor argument; semantic operands remain exact below.
            if args and args[0] is traceback_owner:
                args = args[1:]
                omitted = {"field": "args[0]/traceback_info",
                    "class": type(traceback_owner).__module__ + "." + type(traceback_owner).__qualname__,
                    "reason": "opaque JAX traceback ownership object",
                    "retained_reference": "formatted_traceback"}
        row = {"class": type(error).__module__ + "." + type(error).__qualname__,
               "message": str(error), "formatted_traceback": ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
               "args": self.project(args), "cause": self.project(error.__cause__),
               "context": None if error.__suppress_context__ else self.project(error.__context__)}
        if isinstance(error, SparseEquationCheck):
            row["sparse_equation"] = {name: self.project(value) for name, value in (
                ("site", site), ("role", role), ("stage", stage),
                ("derivative_order", derivative_order), ("dependency", dependency),
                ("payload", sparse_payload))}
            if omitted is not None:
                row["omitted_traceback_owner"] = omitted
        if isinstance(error, SCNSimError):
            row.update(kind=error.kind, category=error.category, stage=error.stage,
                       evidence=self.project(error.evidence))
        return row

    def save(self, filename, value):
        data = _json(value)
        (self.directory / filename).write_bytes(data)
        return hashlib.sha256(data).hexdigest()


@contextmanager
def observe_direct(capture):
    """R1/R2/R4 Golden-only authorized read-only same-call internal seams.

    Wrappers forward original arguments, returns and failures. No equation is
    reevaluated. Ordinary tests and timing workers do not activate these hooks.
    """
    from scnsim.execution.memory_direct import InMemoryDirectSink
    from scnsim.execution.physical_analysis import ChunkedResultSink
    records = []
    def wrap(original):
        def stage(sink, work_index, record):
            records.append({"work_index": capture.project(work_index), "record": capture.project(record)})
            return original(sink, work_index, record)
        return stage
    from scnsim.execution import physical_analysis, jax_operation, physical_optimization, memory_optimization
    originals = []
    def returned(owner, name, role):
        original = getattr(owner, name)
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            records.append({"role": role, "record": capture.project(result)})
            return result
        originals.append((owner, name, original))
        setattr(owner, name, call)
    try:
        for cls in (InMemoryDirectSink, ChunkedResultSink):
            originals.append((cls, "stage_direct_work", cls.stage_direct_work))
            cls.stage_direct_work = wrap(cls.stage_direct_work)
        from scnsim.execution.jax_backend import JaxBackend
        source_context = threading.local()
        original_chunk = physical_analysis.PhysicalAnalysis._execute_native_chunk
        def chunk(analysis, native, candidate, rows, omega_source, **kwargs):
            previous = getattr(source_context, "rows", None)
            source_context.rows = tuple(rows)
            try:
                return original_chunk(analysis, native, candidate, rows, omega_source, **kwargs)
            finally:
                if previous is None:
                    del source_context.rows
                else:
                    source_context.rows = previous
        originals.append((physical_analysis.PhysicalAnalysis, "_execute_native_chunk", original_chunk))
        physical_analysis.PhysicalAnalysis._execute_native_chunk = chunk
        original_execute = JaxBackend.execute_native
        def execute(backend, *args, **kwargs):
            outcome = original_execute(backend, *args, **kwargs)
            if kwargs.get("phase") == "native_frequency_chunk":
                # Same synchronized outcome, before the caller may throw its
                # checked error. Completion order and actual source rows both
                # remain visible; parallel lanes never share row context.
                records.append({"_raw_chunk": (source_context.rows, outcome)})
            return outcome
        originals.append((JaxBackend, "execute_native", original_execute))
        JaxBackend.execute_native = execute
        returned(physical_analysis, "program_evidence", "primal_equation_evidence")
        returned(jax_operation, "_sensitivity_equation_evidence", "derivative_equation_evidence")
        for cls in (physical_optimization._SavedCandidateRecords, memory_optimization.MemoryCandidateRecords):
            returned(cls, "success", "candidate_success")
            returned(cls, "failure", "candidate_failure")
        yield records
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)




def _observation_inputs(directory):
    """Require a retained raw or source-occurrence observation, never an empty baseline."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    observations = sorted(directory.glob('*.json'))
    if not observations:
        raise ValueError('golden input requires observation JSON')
    for path in observations:
        document = json.loads(path.read_bytes())
        if not isinstance(document, dict):
            raise ValueError('golden observation must be a document: '+str(path))
        raw_keys = {'case_metadata','outcome','direct_source_order'}
        projected_keys = {'case_metadata','outcome','source_occurrences',
                          'staged_source_records','other_evidence_records'}
        if set(document) == raw_keys:
            container_names = ('direct_source_order',)
        elif set(document) == projected_keys:
            container_names = ('source_occurrences','staged_source_records','other_evidence_records')
        else:
            raise ValueError('unsupported golden observation structure: '+str(path))
        metadata = document['case_metadata']
        if (not isinstance(metadata, dict) or set(metadata) != {'mapping'}
                or not isinstance(metadata['mapping'], list) or not metadata['mapping']
                or any(not isinstance(pair,list) or len(pair)!=2
                       or not isinstance(pair[0],str) or not pair[0] for pair in metadata['mapping'])):
            raise ValueError('golden observation requires captured metadata: '+str(path))
        if any(not isinstance(document[name],list)
               or any(not isinstance(row,dict) or not row for row in document[name])
               for name in container_names):
            raise ValueError('golden observation requires typed source containers: '+str(path))
        outcome = document['outcome']
        if not isinstance(outcome,dict) or set(outcome) not in ({'result'},{'exception'}):
            raise ValueError('golden observation requires one captured outcome: '+str(path))
        record = next(iter(outcome.values()))
        if (not isinstance(record,dict) or not isinstance(record.get('class'),str)
                or not record['class']):
            raise ValueError('golden observation requires a typed outcome record: '+str(path))
        if 'result' in outcome:
            if not isinstance(record.get('fields'),dict) or not record['fields']:
                raise ValueError('golden Result record requires public fields: '+str(path))
        else:
            args = record.get('args')
            if (not isinstance(args,dict) or set(args) != {'sequence','type'}
                    or args['type'] != 'tuple' or not isinstance(args['sequence'],list)
                    or not {'cause','context'}.issubset(record)):
                raise ValueError('golden exception record requires actual arguments and cause/context: '+str(path))
    return observations

def _projection_operation(source, explicit):
    """Use declared worker input or explicit CLI input, never infer from values."""
    worker = source.parent/'worker.json'
    declared = None
    record = {'explicit_operation':explicit, 'worker_input':None}
    if worker.is_file():
        data = worker.read_bytes()
        document = json.loads(data)
        declared = document.get('input',{}).get('operation')
        if declared is not None and (not isinstance(declared,str) or not declared):
            raise ValueError('worker input operation must be a nonempty string')
        record['worker_input'] = {'path':str(worker.resolve()),
                                  'sha256':hashlib.sha256(data).hexdigest(), 'operation':declared}
    if explicit is not None and declared is not None and explicit != declared:
        raise ValueError('explicit projection operation differs from captured worker input')
    selected = explicit if explicit is not None else declared
    record['selected_operation'] = selected
    return selected, record


def project_occurrences(source, destination, *, operation=None):
    """Project retained Direct bytes by source occurrence, without execution.

    Raw chunk boundaries, completion order and intermediate reducers remain in
    the unchanged source archive. Their hashes live outside the comparison tree.
    No lane, status, performed/present flag or structured failure is discarded.
    Human-excluded identity/UUID/timing/traceback values remain in the sidecar.
    """
    import numpy as np
    source, destination = Path(source), Path(destination)
    observations = _observation_inputs(source)
    operation, operation_input = _projection_operation(source, operation)
    destination.mkdir(parents=True, exist_ok=False)
    capture = Capture(destination)

    def clone(value):
        if isinstance(value, dict):
            if 'array' in value or 'bytes' in value:
                key = 'array' if 'array' in value else 'bytes'
                blob = value[key]
                data = (source/blob['file']).read_bytes()
                if hashlib.sha256(data).hexdigest() != blob['sha256'] or len(data) != blob['bytes']:
                    raise ValueError('captured blob integrity mismatch: '+blob['file'])
                return dict(value, **{key: capture.blob(data)})
            return {key: clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone(item) for item in value]
        return value

    def lane(value, index, width):
        if isinstance(value, dict):
            if 'array' in value:
                shape = value['shape']
                if not shape or shape[0] != width:
                    raise ValueError('lane-bearing field differs from actual source-row count')
                data = (source/value['array']['file']).read_bytes()
                dtype = np.dtype(value['dtype'])
                size = dtype.itemsize
                for dimension in shape[1:]:
                    size *= dimension
                if len(data) != width*size or hashlib.sha256(data).hexdigest() != value['array']['sha256']:
                    raise ValueError('captured lane array integrity mismatch')
                return dict(value, shape=shape[1:], array=capture.blob(data[index*size:(index+1)*size]))
            return {key: lane(item,index,width) for key,item in value.items()}
        if isinstance(value, list):
            return [lane(item,index,width) for item in value]
        return value

    def mapping(value):
        return dict(value['mapping'])

    provenance = []
    for path in observations:
        document = json.loads(path.read_bytes())
        if set(document) != {'case_metadata','direct_source_order','outcome'}:
            raise ValueError('source occurrence projection requires a captured observation document')
        records = document['direct_source_order']
        chunks = [row for row in records if row.get('role') == 'native_chunk_before_projection']
        occurrences = []
        if chunks:
            metadata = mapping(document['case_metadata'])
            direct = mapping(mapping(metadata['specs'])['direct'])
            frequencies = direct['frequencies']
            frequency_array = frequencies['quantity']
            frequency_count = frequency_array['shape'][0]
            for chunk in chunks:
                source_rows = chunk['source_rows']['sequence']
                width = len(source_rows)
                for index, source_row in enumerate(source_rows):
                    identity = mapping(source_row)
                    certificates = []
                    for certificate in chunk['certificates']['sequence']:
                        fields = certificate['fields']
                        lane_fields = {'status','eta','residual_numerator','residual_denominator','performed','present'}
                        certificates.append(dict(certificate, fields={
                            key: lane(value,index,width) if key in lane_fields else clone(value)
                            for key,value in fields.items()}))
                    checked = [dict(effect, metadata=clone(effect['metadata']),
                                    predicate=lane(effect['predicate'],index,width),
                                    code=lane(effect['code'],index,width),
                                    payload=lane(effect['payload'],index,width))
                               for effect in chunk['checked_error']]
                    occurrences.append(dict(source=clone(source_row),
                        frequency=dict(frequencies, quantity=lane(frequency_array,identity['omega_index'],frequency_count)),
                        outcomes=lane(chunk['outcomes'],index,width),
                        certificates=certificates, checked_error=checked))
            # Keep every occurrence, including repeated executions at one source.
            # Sorting removes only concurrent chunk completion order, not source order.
            occurrences.sort(key=lambda row: (
                mapping(row['source'])['source_index'], mapping(row['source'])['work_index'],
                mapping(row['source'])['view_index'], mapping(row['source'])['omega_index']))
        retained = [row for row in records if row.get('role') != 'native_chunk_before_projection']
        staged = [row for row in retained if 'work_index' in row]
        other = [row for row in retained if 'work_index' not in row]
        staged.sort(key=lambda row: row['work_index'])
        projected = dict(case_metadata=clone(document['case_metadata']),
                         outcome=clone(document['outcome']), source_occurrences=occurrences,
                         staged_source_records=clone(staged), other_evidence_records=clone(other))
        semantic, excluded = _semantic_projection(projected, destination, operation=operation)
        capture.save(path.name,semantic)
        provenance.append(dict(observation=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               raw_chunk_count=len(chunks),source_occurrence_count=len(occurrences),
                               excluded_values=excluded))
    # Excluded encoded records may have copied blobs before projection. Keep
    # only semantic blob references in this new tree; originals stay archived.
    referenced = set()
    def references(value):
        if isinstance(value, dict):
            for key,item in value.items():
                if key in ('array','bytes') and isinstance(item,dict) and 'file' in item:
                    referenced.add(item['file'])
                references(item)
        elif isinstance(value,list):
            for item in value:
                references(item)
    for path in destination.glob('*.json'):
        references(json.loads(path.read_bytes()))
    for path in (destination/'blobs').iterdir():
        if str(path.relative_to(destination)) not in referenced:
            path.unlink()
    # Provenance and raw diagnostic differences are reported separately from
    # the semantic comparison. The original capture tree remains untouched.
    record = dict(projection_contract='R1-Human-supplement-2026-10-21',
                  source=str(source.resolve()), observations=provenance, operation_input=operation_input,
                  raw_files=[dict(path=str(path.relative_to(source)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                             for path in sorted(source.rglob('*')) if path.is_file()],
                  archived_diagnostics='original raw chunks/completion order/check and certificate reducers retained in source; no discarded or rewritten files')
    destination.with_name(destination.name+'.provenance.json').write_bytes(_json(record))
    return {'observations':len(provenance),'source_occurrences':sum(row['source_occurrence_count'] for row in provenance),
            'excluded_provenance_values':sum(len(row['excluded_values']) for row in provenance),
            'provenance_path':str(destination.with_name(destination.name+'.provenance.json'))}


def _provenance_differences(left, right):
    """Report excluded values separately; never add them to semantic diff count."""
    projected = any('source_occurrences' in json.loads(path.read_bytes())
                    for directory in (left,right) for path in directory.glob('*.json'))
    paths = [path.with_name(path.name+'.provenance.json') for path in (left,right)]
    documents = [json.loads(path.read_bytes()) if path.exists() else None for path in paths]
    contract = 'R1-Human-supplement-2026-10-21'
    if not projected and not any(doc is not None and doc.get('projection_contract') == contract for doc in documents):
        return None
    if any(doc is None or doc.get('projection_contract') != contract for doc in documents):
        raise ValueError('semantic comparison requires both current projection provenance documents')
    def entries(document):
        return {(row['observation'],item['path']):item
                for row in document['observations'] for item in row['excluded_values']}
    a,b = map(entries,documents)
    differences = []
    for key in sorted(a.keys() | b.keys()):
        if a.get(key) != b.get(key):
            differences.append({'observation':key[0], 'path':key[1], 'left':a.get(key), 'right':b.get(key)})
    return {'difference_count':len(differences),'differences':differences,
            'left_record':str(paths[0]),'right_record':str(paths[1])}


def compare(left, right):
    """Compare all files, reporting exact tree/bytes and numeric leaf differences."""
    import numpy as np
    left, right = Path(left), Path(right)
    _observation_inputs(left)
    _observation_inputs(right)
    lfiles = {p.relative_to(left).as_posix() for p in left.rglob("*") if p.is_file()}
    rfiles = {p.relative_to(right).as_posix() for p in right.rglob("*") if p.is_file()}
    differences = []
    def walk(a, b, path):
        if type(a) is not type(b):
            differences.append({"path": path, "reason": "type"}); return
        if isinstance(a, dict):
            if set(a) != set(b):
                differences.append({"path": path, "reason": "keys", "left_only": sorted(set(a)-set(b)), "right_only": sorted(set(b)-set(a))})
            if "array" in a and "array" in b and a != b and a.get("dtype") == b.get("dtype") and a.get("shape") == b.get("shape"):
                x = np.frombuffer((left/a["array"]["file"]).read_bytes(), dtype=a["dtype"])
                y = np.frombuffer((right/b["array"]["file"]).read_bytes(), dtype=b["dtype"])
                changed = np.flatnonzero(np.any(x.view(np.uint8).reshape(len(x), -1) != y.view(np.uint8).reshape(len(y), -1), axis=1)) if x.size else np.array([], dtype=int)
                entry = {"path": path, "reason": "array_bits", "flat_indices": changed.tolist()}
                if x.dtype.kind in "fciub" and x.size:
                    with np.errstate(all="ignore"):
                        absolute = np.abs(x.astype(np.complex128)-y.astype(np.complex128))
                        denominator = np.abs(x.astype(np.complex128))
                        relative = absolute / denominator
                    entry["max_absolute_hex"] = float(np.max(absolute)).hex()
                    entry["max_relative_hex"] = float(np.max(relative)).hex()
                differences.append(entry)
            for key in sorted(set(a)&set(b)): walk(a[key], b[key], path+"/"+key)
        elif isinstance(a, list):
            if len(a)!=len(b): differences.append({"path":path,"reason":"length"})
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+"/"+str(i))
        elif a != b: differences.append({"path":path,"reason":"value","left":a,"right":b})
    for name in sorted(lfiles | rfiles):
        if name not in lfiles or name not in rfiles:
            differences.append({"path":name,"reason":"missing_file","side":"left" if name not in lfiles else "right"});continue
        a,b=(left/name).read_bytes(),(right/name).read_bytes()
        if a==b:continue
        if name.endswith(".json"):
            positions = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            first_positions = (positions + list(range(min(len(a), len(b)), min(max(len(a), len(b)), min(len(a), len(b))+16))))[:16]
            differences.append({"path": name, "reason": "json_bytes",
                                "differing_byte_count": len(positions) + abs(len(a)-len(b)),
                                "first_byte_positions": first_positions,
                                "left_bytes": len(a), "right_bytes": len(b),
                                "left_sha256": hashlib.sha256(a).hexdigest(),
                                "right_sha256": hashlib.sha256(b).hexdigest()})
            walk(json.loads(a),json.loads(b),name)
        else:
            differences.append({"path":name,"reason":"bytes","left_sha256":hashlib.sha256(a).hexdigest(),"right_sha256":hashlib.sha256(b).hexdigest()})
    return {"differences":differences,"difference_count":len(differences),"left_files":len(lfiles),"right_files":len(rfiles),
            "provenance_differences":_provenance_differences(left,right)}


def main(argv=None):
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "project":
        parser = argparse.ArgumentParser(description="Offline source-occurrence comparison projection")
        parser.add_argument("command", choices=("project",))
        parser.add_argument("source");parser.add_argument("destination")
        parser.add_argument("--operation", help="captured operation; must match worker.input.operation when present")
        args = parser.parse_args(arguments)
        print(json.dumps(project_occurrences(args.source,args.destination,operation=args.operation),indent=2))
        return 0
    if arguments and arguments[0] == "capture":
        parser = argparse.ArgumentParser(description="Capture one declared public operation")
        parser.add_argument("command", choices=("capture",))
        if "--" not in arguments:
            parser.error("separate capture from run.py worker options with literal --")
        boundary = arguments.index("--")
        parser.parse_args(arguments[:boundary])
        from run import main as run_main
        return run_main(arguments[boundary+1:])
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compare",))
    parser.add_argument("left");parser.add_argument("right")
    args=parser.parse_args(arguments)
    result=compare(args.left,args.right);print(json.dumps(result,indent=2));return int(bool(result["difference_count"]))


if __name__ == "__main__":
    raise SystemExit(main())
