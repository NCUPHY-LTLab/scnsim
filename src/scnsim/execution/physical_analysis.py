"""Ordinary physical-program execution and detached numerical body projection.

The caller owns request lookup, operation leases, persistence and final decoding.
This adapter owns requested-point occurrences and the existing baseline-anchored
continuation policy. Every value/branch uses one compiled QuantityProgram; it
never constructs a second physical model or runs an optimizer.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from ..canonical import canonical_json_bytes
from ..compilation.compiler import parameter_key, parameter_values
from ..compilation.mesh import quantity
from ..compilation.quantity_declarations import quantity_declarations
from ..compilation.quantity_program import compile_point_quantity_program
from ..errors import SCNSimError
from ..numeric_encoding import array_record, bits, record_bytes, record_document
from ..numerics.evidence import evidence_bytes, numerical_error
from ..numerics.models import EvaluationResult, NumericalFailure
from .quantities import (
    EvaluationFailure, continue_root, quantity_body_id, quantity_record,
)
from .quantity_policy import diagonal_passive_policy
from .warmup import program_owner_for_device

_ROOT_TYPES = frozenset(('diagonal_root', 'operator_element_root', 'hybridized_pole',
                         'transfer_zero', 'residue_normalized_coupling'))
_NUMERICAL_FAILURES = frozenset(('direct_response_formation', 'port_realizability',
                                'eliminated_block_solve_failure', 'root_slope_unresolved',
                                'numerical_resolution_unresolved',
                                'unsupported_singular_capacitance_for_diagonal_root_v1'))


def point_identity(point):
    return canonical_json_bytes({
        'bindings': sorted(point['bindings'], key=lambda row: parameter_key(row['parameter'])),
        'allow_extrapolation': sorted(point['allow_extrapolation'], key=parameter_key),
    })


def _point_at(template, values):
    return dict(template, bindings=[
        dict(row, value=values[parameter_key(row['parameter'])]
             if isinstance(values[parameter_key(row['parameter'])], dict)
             else dict(row['value'], si_value_f64=bits(values[parameter_key(row['parameter'])])))
        for row in template['bindings']
    ])


def program_evidence(outcome, context):
    """Detach the actual same-frame primal certificates, never derivative claims."""
    return evidence_bytes({
        'certificates': tuple({key: np.asarray(value) for key, value in zip(row._fields, row)}
                              for row in outcome.aux.certificates),
        'realized_capabilities': outcome.aux.realized_capabilities,
        'terminal_ids':context['terminal_ids'],
        'axes':{key:context[key] for key in ('probe_loads','reference_coordinate_ids',
                                           'reference_impedances_ohm','load_mask')},
    })


def point_contexts(program, inputs):
    """Detach actual binary64 View lineage and realized sections per observable."""
    metadata = program.point_metadata(inputs)
    views = {row['view_index']: row for row in metadata['views']}
    return {row['observable_index']: {
        'lineage': views[row['view_index']]['lineage'],
        **{key:views[row['view_index']][key] for key in ('terminal_ids','probe_loads',
              'reference_coordinate_ids','reference_impedances_ohm','load_mask')},
        'discretization': metadata['discretization'],
    } for row in metadata['observables']}


def project_primal_bodies(program, outcome, inputs):
    """Numerical body encoding authority shared by ordinary and optimizer sinks.

    Locators refer exclusively to the supplied synchronized graph outcome. No
    selector evaluation, sparse solve, objective reduction or View lowering runs.
    """
    metadata = program.observable_metadata()
    contexts = point_contexts(program, inputs)
    branches, quantities = dict(outcome.aux.branch_results), dict(outcome.aux.quantity_results)
    bodies = {}
    for row in metadata:
        locator, index = row['primal_locator'], row['observable_index']
        identity = f"{row['quantity_key']}:observable:{index}"
        if locator['kind'] == 'branch_results':
            result = replace(branches[locator['start_index']], id=identity)
        elif locator['kind'] == 'quantity_results':
            result = replace(quantities[locator['observable_index']], id=identity)
        else:
            raw = np.asarray(outcome.aux.observable_raw_values[locator['observable_index']])
            field = (row['kind'] if row['kind'] in ('S', 'Y', 'Z') else
                     'operator_values' if row['kind'] == 'operator' else 'response_value')
            result = EvaluationResult(identity, **{field: raw if raw.shape else raw[()]},
                                      evidence_bytes=program_evidence(outcome, contexts[index]))
        bodies[index] = result
    for row in metadata:
        if row['kind'] != 'residue_coupling':
            continue
        dependencies = []
        for role, spec in (('a', row['spec']['branch_a']), ('b', row['spec']['branch_b'])):
            branch = next(item for item in metadata if item['start_index'] is not None
                          and canonical_json_bytes(item['spec']) == canonical_json_bytes(spec)
                          and canonical_json_bytes(item['view']) == canonical_json_bytes(row['view']))
            dependencies.append({'role': role, 'binding': {'kind': spec['type'], 'spec': dict(spec), 'view': dict(row['view'])},
                                 'body_id': quantity_body_id(bodies[branch['observable_index']])[0]})
        result = bodies[row['observable_index']]
        evidence = record_document(result.evidence_bytes)
        evidence['branches'] = dependencies
        bodies[row['observable_index']] = replace(result, evidence_bytes=record_bytes(evidence))
    return bodies


class PhysicalAnalysis:
    """One request-local static program; points and outcomes are never retained in Run."""

    def __init__(self, plan, analysis, backend, trace, emit, points, *, program_owner=None):
        self.analysis, self.backend, self.trace, self.emit = analysis, backend, trace, emit
        self.spec = analysis['spec']
        self.baseline = {
            'type': 'parameter_set_v2', 'allow_extrapolation': [],
            'bindings': [{'parameter': {'definitions_id': row['definitions_id'],
                                       'parameter_id': row['parameter_id']},
                          'value': row['baseline']}
                         for row in plan['parameter_closure']['definitions']],
        }
        self.baseline_values = parameter_values(self.baseline)
        self.anchors = {}
        self._plan = plan
        self._program_owner = program_owner_for_device(program_owner, backend.device, admit=False)
        self._program_variants = {}
        self._point_programs = {}
        self.program = self.program_at(self.baseline if self.spec['type'] in _ROOT_TYPES else points[0])
        self.metadata = self.program.observable_metadata()
        self.roots = tuple(row for row in self.metadata if row['start_index'] is not None)

    def program_at(self, point):
        key = point_identity(point)
        if key not in self._point_programs:
            with self.trace.span('physical_program_preparation'):
                self._point_programs[key] = compile_point_quantity_program(
                    self._plan, quantity_declarations(self.spec, self.analysis['view']), point,
                    default_view=self.analysis['view'], mesh=self.spec.get('discretization'),
                    precision=self.backend.precision,
                    authorized=frozenset(parameter_key(ref) for ref in point['allow_extrapolation']),
                    program_variants=self._program_variants, program_owner=self._program_owner,
                )
        return self._point_programs[key]

    def context(self, point):
        program = self.program_at(point)
        return point_contexts(program, program.inputs(point))[0]

    @staticmethod
    def _dependency_key(row):
        return canonical_json_bytes({'spec': row['spec'], 'view': row['view'],
                                     'type': row['spec']['type']}).decode()

    def _branch(self, point, row, omega, identity):
        program = self.program_at(point)
        starts = list(program.branch_starts_rad_s)
        starts[row['start_index']] = omega
        inputs = program.inputs(point, branch_starts_rad_s=starts)
        result = self.backend.execute(
            program.program, 'evaluate_branch', inputs, row['start_index'],
            sparse_runner=self.trace.sparse_runner,
        )
        return replace(result, id=identity)

    def _root_results(self, point, identity, *, baseline):
        result_by_index = {}
        values = parameter_values(point)
        for row in self.roots:
            index = row['start_index']
            if index in result_by_index:
                continue
            key = self._dependency_key(row)
            omega = (self.program.branch_starts_rad_s[index] if baseline
                     else self.anchors[key].root_omega_rad_s)
            result = self._branch(point, row, omega, identity)
            if not baseline and result.failure is not None:
                def make_job(job_id, actual, start):
                    return SimpleNamespace(id=job_id, point=actual, omega=start)

                def evaluate(jobs):
                    return tuple(self._branch(job.point, row, job.omega, job.id) for job in jobs)

                def observe(values_at, actual, attempt, t):
                    record = quantity_record(attempt)
                    record.update(parameters=actual, origin='root_continuation',
                                  source_index=int(identity), continuation_t_f64=bits(t),
                                  **self.context(actual))
                    self.emit('evaluation', record)

                result = continue_root(
                    baseline=self.baseline_values, values=values,
                    baseline_root=self.anchors[key].root_omega_rad_s,
                    endpoint_view=point, initial_result=result,
                    candidate_view=lambda values_at: _point_at(point, values_at),
                    make_job=make_job, evaluate=evaluate, identity=identity, observe=observe,
                )
            # The baseline diagonal anchor intentionally precedes the existing
            # passive-imaginary policy; requested/candidate bodies still enforce it.
            if row['kind'] == 'diagonal_root' and not (baseline and self.spec['type'] == 'diagonal_root'):
                result = diagonal_passive_policy(result)
            result_by_index[index] = result
            if result.failure is not None:
                raise EvaluationFailure(result.failure, result=result,
                                        dependencies=self._dependencies(result_by_index))
            if baseline:
                self.anchors[key] = result
        return result_by_index

    @staticmethod
    def _dependencies(results):
        return dict(quantity_body_id(result) for result in results.values())

    def evaluate(self, point, identity, *, baseline=False):
        branch_results = {}
        try:
            branch_results = self._root_results(point, identity, baseline=baseline)
            program = self.program_at(point)
            starts = list(program.branch_starts_rad_s)
            for index, result in branch_results.items():
                starts[index] = result.root_omega_rad_s
            inputs = program.inputs(point, branch_starts_rad_s=starts)
            outcome = self.backend.execute(program.program, 'value', inputs,
                                           sparse_runner=self.trace.sparse_runner)
            program.require_primal(outcome)
            bodies = project_primal_bodies(program, outcome, inputs)
            branches = dict(outcome.aux.branch_results)
            for row in self.roots:
                index = row['start_index']
                actual = branches[index]
                if row['kind'] == 'diagonal_root' and not (baseline and self.spec['type'] == 'diagonal_root'):
                    actual = diagonal_passive_policy(actual)
                branches[index] = actual
                if actual.failure is not None:
                    raise EvaluationFailure(actual.failure, result=actual,
                                            dependencies=self._dependencies(branches))
            visible = [row for row in program.observable_metadata() if not row['hidden']]
            if self.spec['type'] == 'direct_solve':
                values = {row['kind']: getattr(bodies[row['observable_index']], row['kind']) for row in visible}
                result = EvaluationResult(identity, **values, evidence_bytes=program_evidence(
                    outcome, point_contexts(program, inputs)[visible[0]['observable_index']]))
            else:
                result = replace(bodies[visible[0]['observable_index']], id=identity)
            dependencies = {quantity_body_id(bodies[row['observable_index']])[0]:
                            quantity_body_id(bodies[row['observable_index']])[1]
                            for row in program.observable_metadata() if row['hidden']}
            if baseline:
                for row in self.roots:
                    anchor = bodies[row['observable_index']]
                    if not row['hidden']:
                        anchor = result
                    self.anchors[self._dependency_key(row)] = anchor
            return result, dependencies
        except EvaluationFailure as error:
            result = error.result if error.result is not None else EvaluationResult(identity, failure=error.failure)
            result = replace(result, id=identity)
            dependencies = dict(error.dependencies)
            dependencies.pop(quantity_body_id(result)[0], None)
            return result, dependencies
        except SCNSimError as error:
            if error.kind not in _NUMERICAL_FAILURES:
                raise
            failure = NumericalFailure(error.kind, error.stage, str(error), record_bytes(error.evidence))
            return EvaluationResult(identity, failure=failure), self._dependencies(branch_results)


    def record(self, point, result, dependencies, **occurrence):
        record = quantity_record(result)
        record.update(parameters=point, dependencies=dependencies, **self.context(point), **occurrence)
        if self.spec['type'] in ('direct_solve', 'operator'):
            record['frequencies_hz'] = array_record(np.asarray(
                [quantity(value) for value in self.spec['frequencies']], dtype=np.float64))
        elif self.spec['type'] == 'response_element':
            record['frequencies_hz'] = array_record(np.asarray([quantity(self.spec['frequency'])], dtype=np.float64))
        return record

    def run(self, points):
        cache, seen = {}, set()
        baseline_needed = self.spec['type'] in _ROOT_TYPES
        if baseline_needed:
            result, dependencies = self.evaluate(self.baseline, 'baseline', baseline=True)
            record = self.record(self.baseline, result, dependencies, origin='baseline', cache_hit=False)
            self.emit('evaluation', record)
            if result.failure is not None:
                raise numerical_error(result.failure)
            self.emit('baseline_ready', {
                'schema': 'scnsim.benchmark_root_anchor', 'schema_version': 1,
                'baseline': record,
                'anchors': {key: quantity_body_id(value)[0] for key, value in self.anchors.items()},
                'resume_state': None,
            })
            if self.spec['type'] == 'diagonal_root':
                result = diagonal_passive_policy(result)
            key = point_identity(self.baseline)
            cache[key], seen = (result, dependencies), {key}
        first_failure = None
        for ordinal, point in enumerate(points):
            key = point_identity(point)
            if key not in cache:
                cache[key] = self.evaluate(point, str(ordinal))
            result, dependencies = cache[key]
            self.emit('evaluation', self.record(
                point, result, dependencies, source_index=ordinal, origin='requested_point',
                cache_hit=key in seen, numerical_source_id=result.id,
            ))
            seen.add(key)
            if first_failure is None:
                first_failure = result.failure
        if first_failure is not None and self.analysis['parameter_source']['kind'] == 'point':
            raise numerical_error(first_failure)
        terminal = {'type': self.spec['type'], 'point_count': len(points), 'backend': self.backend.identity()}
        if baseline_needed:
            terminal['baseline_anchor_schema'] = 'scnsim.benchmark_root_anchor'
        return terminal
