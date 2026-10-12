"""Detached array-only providers for one finite physical route partition.

Static providers contain declarations, patterns and stable functions only.
Original binary64 physical classification chooses the partition at acquisition;
runtime points still rebuild coefficients and report domain/route mismatch.
Operation scheduling and persistence do not live in this numerical module.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import partial
from types import MappingProxyType
from typing import Any, NamedTuple
from threading import RLock
import hashlib
import json
import numpy as np

from ..canonical import canonical_json_bytes
from ..compilation.model_program import PhysicalChecks
from .native_frequency import NativeFrequencyProgram, NativeCandidate
from .observable_projection import _flatten, _projection


class NativeCandidateState(NamedTuple):
    model: Any
    views: tuple
    operands: tuple
    physical_checks: Any
    realized_capabilities: tuple
    partition_status: Any


def _candidate(model, views, assemblies, capabilities, precision, theta):
    import jax
    import jax.numpy as jnp
    # Physical validation/rank is intentionally original binary64, not the
    # selected32 cast. Numerical assembly begins AFTER its selected cast.
    with jax.enable_x64(True):
        original = jnp.asarray(theta, dtype=jnp.float64)
        checks = model.physical_checks(original)
        physical = model(original)
        actual = tuple(view.capability_array(physical, original) for view in views)
        mismatch = jnp.any(jnp.stack(tuple(value != expected for value, expected in
                                          zip(actual, capabilities, strict=True)))) if views else jnp.bool_(False)
        rd = jnp.float64 if precision == 'float64' else jnp.float32
        selected = original.astype(rd)
        numerical = physical if precision == 'float64' else model(selected)
        routed = tuple(view.apply_model(numerical, selected, port_realizable=capability)
                       for view, capability in zip(views, capabilities, strict=True))
        operands = tuple(assembly.operands(view) for assembly, view in zip(assemblies, routed, strict=True))
        return NativeCandidateState(numerical, routed, operands, checks, actual,
                                    jnp.where(mismatch, jnp.int32(1), jnp.int32(0)))


def _frequency(program, view_index, demand, capability, candidate, omega,
               prior_check_status=0, prior_certificate_status=0):
    return program(candidate.operands[view_index], candidate.views[view_index], omega,
                   families=demand, capability=capability,
                   prior_check_status=prior_check_status,
                   prior_certificate_status=prior_certificate_status,
                   dependency=('responses', view_index, tuple(sorted(demand))))


class NativeProgramAux(NamedTuple):
    physical_checks: Any
    partition_status: Any
    realized_capabilities: tuple
    check_status: Any
    certificate_status: Any
    frequency_outcomes: tuple
    derivative_available: tuple
    branch_packets: tuple = ()
    quantity_packets: tuple = ()
    observable_values: tuple = ()
    observable_raw_values: tuple = ()
    output_values: tuple = ()


class NativePublicProjection(NamedTuple):
    observable_values: tuple
    observable_raw_values: tuple
    output_values: tuple
    derivative_available: tuple


class NativeChunkOutcome(NamedTuple):
    outcomes: Any
    errors: Any
    check_status: Any
    certificate_status: Any


def _compile_chunk(function, bucket):
    # Capture only the admitted immutable point equation and static bucket.
    # Individually functionalized frame errors become scan OUTPUTS, not one
    # overwritten checkify effect payload for the whole frequency scan.
    from .native_sparse import checked_native
    checked = checked_native(function)
    def chunk(candidate, omega, prior_check_status=0, prior_certificate_status=0):
        import jax
        import jax.numpy as jnp
        from .jax_core import first
        if omega.shape != (bucket,):
            raise ValueError('native chunk operand shape differs from its declared bucket')
        def body(carry, frequency):
            error, outcome = checked(candidate, frequency, *carry)
            checks = first(carry[0], outcome.check_status != 0, outcome.check_status)
            certificates = first(carry[1], outcome.certificate_status != 0, outcome.certificate_status)
            return (checks, certificates), (outcome, error)
        initial = (jnp.asarray(prior_check_status, dtype=jnp.int32),
                   jnp.asarray(prior_certificate_status, dtype=jnp.int32))
        final, (outcomes, errors) = jax.lax.scan(body, initial, omega)
        return NativeChunkOutcome(outcomes, errors, *final)
    return chunk


class NativeBranchOutcome(NamedTuple):
    values: tuple                 # parallel branch_observable_indices, raw scalar leaves
    derivative_available: tuple
    branch_packets: tuple         # ordered (start_index, actual RootPacket)
    quantity_packets: tuple       # ordered (observable_index, actual coupling packet)
    check_status: Any
    certificate_status: Any


class BranchADFrame(NamedTuple):
    """Completed primal branches and operation-owned JAX linear residuals."""
    outcome: NativeBranchOutcome
    linearized_branch: Any
    input_templates: tuple


def _branch_ad(branch_function, candidate, omega, seeds, prior_check_status=0,
               prior_certificate_status=0):
    import jax
    import jax.numpy as jnp
    with jax.enable_x64(True):
        def values(point, frequencies, starts):
            outcome = branch_function(point, frequencies, starts,
                                      prior_check_status, prior_certificate_status)
            return outcome.values, outcome
        _, linearized, outcome = jax.linearize(values, candidate, omega, seeds, has_aux=True)
        # Linearized inputs are tangent trees: discrete candidate leaves
        # carry float0, never their primal bool/integer dtype.
        templates = jax.tree.map(
            lambda value: jnp.zeros_like(value) if np.issubdtype(value.dtype, np.inexact)
            else np.zeros(value.shape, dtype=jax.dtypes.float0),
            (candidate, omega, seeds))
        return BranchADFrame(outcome, linearized, templates)


def _branch_tangent(frame, candidate_dot, omega_dot, seeds_dot):
    """One direction; check this executed linear function before mapping."""
    import jax
    with jax.enable_x64(True):
        return frame.linearized_branch(candidate_dot, omega_dot, seeds_dot)


def _branch_pullback(frame, covectors):
    """One cotangent tuple; no branch primal is evaluated again."""
    import jax
    with jax.enable_x64(True):
        transpose = jax.linear_transpose(frame.linearized_branch, *frame.input_templates)
        return transpose(covectors)


class CandidateADFrame(NamedTuple):
    """Operation-owned candidate and JAX linearization residual PyTree.

    JAX Partial's function/jaxpr is structural; its arguments contain actual
    residual arrays. Neither this frame nor its arrays enter a source registry.
    """
    candidate: Any
    linearized_candidate: Any
    theta_template: Any


class ResponseADChunk(NamedTuple):
    public_values: tuple             # family arrays [frequency, ...]
    primal_outcomes: Any             # actual compact NativeFrequencyOutcome
    primal_errors: Any               # checked Error [frequency]
    derivatives: tuple               # JVP family arrays [frequency, direction, ...]
    candidate_cotangents: Any         # VJP state PyTree [direction, ...], summed over frequency
    derivative_errors: Any           # checked Error [frequency, direction]
    check_status: Any
    certificate_status: Any


def _candidate_ad(candidate_function, theta):
    import jax
    import jax.numpy as jnp
    with jax.enable_x64(True):
        candidate, linearized = jax.linearize(candidate_function, theta)
        return CandidateADFrame(candidate, linearized, jnp.zeros_like(theta))


def _candidate_tangents(frame, directions):
    import jax
    with jax.enable_x64(True):
        return jax.lax.map(frame.linearized_candidate, directions, batch_size=8)


def _candidate_pullback(frame, covectors):
    import jax
    with jax.enable_x64(True):
        transpose = jax.linear_transpose(frame.linearized_candidate, frame.theta_template)
        return jax.lax.map(lambda cotangent: transpose(cotangent)[0], covectors, batch_size=8)


def _zero_candidate_covectors(candidate, count):
    import jax
    import jax.numpy as jnp
    def zero(value):
        shape = (count,) + value.shape
        return (jnp.zeros(shape, dtype=value.dtype) if np.issubdtype(value.dtype, np.inexact)
                else np.zeros(shape, dtype=jax.dtypes.float0))
    return jax.tree.map(zero, candidate)


def _add_candidate_covectors(left, right):
    import jax
    return jax.tree.map(lambda a, b: a if a.dtype == jax.dtypes.float0 else a + b, left, right)


def _compile_response_ad_chunk(function, demand, bucket, mode):
    """Check each point and transformed direction BEFORE frequency mapping.

    One point linearization is consumed by all its supplied direction banks.
    Only public values, equation evidence and (reverse) summed candidate
    cotangents cross the point boundary. No frequency-stacked solve tape leaves
    this function. Reverse calls explicitly replay the primal point equations.
    """
    from .native_sparse import checked_native
    families = tuple(sorted(demand))

    def point(frame, frequency, directional_operands, prior_checks, prior_certificates):
        import jax
        with jax.enable_x64(True):
            def primal(candidate):
                outcome = function(candidate, frequency, prior_checks, prior_certificates)
                return tuple(getattr(outcome, family) for family in families), outcome
            if mode == 'jvp':
                values, pushforward, outcome = jax.linearize(primal, frame.candidate, has_aux=True)
                checked_direction = checked_native(pushforward)
                def direction(theta_dot):
                    candidate_dot = frame.linearized_candidate(theta_dot)
                    return checked_direction(candidate_dot)
                errors, derivatives = jax.lax.map(direction, directional_operands, batch_size=8)
                return values, outcome, derivatives, (), errors
            values, pullback, outcome = jax.vjp(primal, frame.candidate, has_aux=True)
            checked_direction = checked_native(lambda cotangent: pullback(cotangent)[0])
            errors, cotangents = jax.lax.map(checked_direction, directional_operands, batch_size=8)
            return values, outcome, (), cotangents, errors

    checked_point = checked_native(point)

    def chunk(frame, omega, directional_operands, prior_check_status=0, prior_certificate_status=0):
        import jax
        import jax.numpy as jnp
        from .jax_core import first
        with jax.enable_x64(True):
            if omega.shape != (bucket,):
                raise ValueError('response AD omega shape differs from its static bucket')
            checks = jnp.asarray(prior_check_status, dtype=jnp.int32)
            certificates = jnp.asarray(prior_certificate_status, dtype=jnp.int32)
            if mode == 'jvp':
                if directional_operands.ndim != 2 or directional_operands.shape[1:] != frame.theta_template.shape:
                    raise ValueError('response JVP directions differ from the candidate parameter layout')
                def body(carry, frequency):
                    error, (values, outcome, derivatives, _, direction_errors) = checked_point(
                        frame, frequency, directional_operands, *carry)
                    final = (first(carry[0], outcome.check_status != 0, outcome.check_status),
                             first(carry[1], outcome.certificate_status != 0, outcome.certificate_status))
                    return final, (values, outcome, error, derivatives, direction_errors)
                final, (values, outcomes, errors, derivatives, derivative_errors) = jax.lax.scan(
                    body, (checks, certificates), omega)
                return ResponseADChunk(values, outcomes, errors, derivatives, (), derivative_errors, *final)
            if not isinstance(directional_operands, tuple) or len(directional_operands) != len(families):
                raise ValueError('response VJP covectors require the ordered demanded family tuple')
            count = directional_operands[0].shape[1]
            if any(value.shape[:2] != (bucket, count) for value in directional_operands):
                raise ValueError('response VJP covectors require frequency and direction leading axes')
            initial = (checks, certificates, _zero_candidate_covectors(frame.candidate, count))
            def body(carry, item):
                frequency, covectors = item
                error, (values, outcome, _, cotangents, direction_errors) = checked_point(
                    frame, frequency, covectors, carry[0], carry[1])
                final = (first(carry[0], outcome.check_status != 0, outcome.check_status),
                         first(carry[1], outcome.certificate_status != 0, outcome.certificate_status),
                         _add_candidate_covectors(carry[2], cotangents))
                return final, (values, outcome, error, direction_errors)
            final, (values, outcomes, errors, derivative_errors) = jax.lax.scan(
                body, initial, (omega, directional_operands))
            return ResponseADChunk(values, outcomes, errors, (), final[2], derivative_errors, *final[:2])
    return chunk


@dataclass(frozen=True, slots=True, weakref_slot=True)
class NativePhysicalCore:
    """Grid-independent physical kernel source; owner contains no point arrays."""
    candidate_function: Any
    frequency_programs: tuple
    capabilities: tuple
    precision: str
    structural_identity: str
    owner_key: Any
    _functions: dict = field(default_factory=dict, repr=False, compare=False)
    _lock: Any = field(default_factory=RLock, repr=False, compare=False)

    @property
    def candidate_ad_function(self):
        with self._lock:
            function = self._functions.get(('candidate_ad',))
            if function is None:
                function = partial(_candidate_ad, self.candidate_function)
                self._functions[('candidate_ad',)] = function
            return function

    candidate_tangents_function = staticmethod(_candidate_tangents)
    candidate_pullback_function = staticmethod(_candidate_pullback)
    add_candidate_cotangents = staticmethod(_add_candidate_covectors)

    def response_ad_chunk(self, view_index, demand, bucket, mode='jvp'):
        if isinstance(bucket, (bool, np.bool_)) or not isinstance(bucket, (int, np.integer)) or bucket <= 0:
            raise ValueError('response AD bucket requires a positive static integer')
        if mode not in ('jvp', 'vjp'):
            raise ValueError('response AD mode must be jvp or vjp')
        demand, bucket = tuple(sorted(demand)), int(bucket)
        key = ('response_ad', view_index, demand, bucket, mode)
        with self._lock:
            function = self._functions.get(key)
            if function is None:
                function = _compile_response_ad_chunk(self.frequency_function(view_index, demand), demand, bucket, mode)
                self._functions[key] = function
            return function

    def frequency_function(self, view_index, demand):
        demand = tuple(sorted(demand))
        if not set(demand) <= {'S', 'Y', 'Z', 'operator'} or not demand:
            raise ValueError('native frequency demand is not a current declared family')
        key = ('frequency', view_index, demand)
        with self._lock:
            function = self._functions.get(key)
            if function is None:
                function = partial(_frequency, self.frequency_programs[view_index], view_index,
                                   frozenset(demand), self.capabilities[view_index])
                self._functions[key] = function
            return function

    def chunk_function(self, view_index, demand, bucket, transform='value'):
        if isinstance(bucket, (bool, np.bool_)) or not isinstance(bucket, (int, np.integer)) or bucket <= 0:
            raise ValueError('native chunk bucket requires a positive static integer')
        bucket = int(bucket)
        if transform != 'value':
            raise TypeError('native chunk_function supports value; use response_ad_chunk for response derivatives')
        key = ('chunk', view_index, tuple(sorted(demand)), bucket, transform)
        with self._lock:
            function = self._functions.get(key)
            if function is None:
                function = _compile_chunk(self.frequency_function(view_index, demand), bucket)
                self._functions[key] = function
            return function

    def root_function(self, view_index, declaration):
        """Admit one declared root over the SAME candidate/selected-state core."""
        from .native_frequency import selected_jet_function
        from .native_root import compile_native_root
        # Hint/seed ordinal and user labels are aggregate metadata, not a
        # physical equation signature. Independent branches share this stable
        # function while still supplying independent runtime seed operands.
        key = ('root', view_index, declaration.kind, declaration.coordinate_index,
               declaration.row_index, declaration.column_index, declaration.family,
               declaration.output_index, declaration.input_index)
        availability_dependency = ('root_occurrence', view_index, declaration.start_index, key[2:])
        key = (*key, availability_dependency)
        with self._lock:
            function = self._functions.get(key)
            if function is None:
                frequency = self.frequency_programs[view_index]
                view = frequency.view_program
                if declaration.kind == 'transfer_zero':
                    from .native_transfer import compile_native_transfer
                    function = compile_native_transfer(declaration, frequency,
                        precision=self.precision, view_index=view_index,
                        capability=self.capabilities[view_index], availability_dependency=availability_dependency)
                else:
                    function = compile_native_root(declaration,
                        selected_jet_function(frequency, view_index, dependency=availability_dependency), precision=self.precision,
                        node_count=len(view.node_ids), selected=view.selected,
                        eliminated=view.eliminated, pattern=frequency.assembly.pattern,
                        availability_dependency=availability_dependency)
                self._functions[key] = function
            return function

    def coupling_function(self, view_index, branch_declarations, *, midpoint, availability_dependency):
        from .native_frequency import selected_jet_function
        from .native_coupling import compile_native_coupling
        signature = tuple((item.kind, item.coordinate_index) for item in branch_declarations)
        key = ('coupling', view_index, signature, bool(midpoint), availability_dependency)
        with self._lock:
            function = self._functions.get(key)
            if function is None:
                frequency = self.frequency_programs[view_index]
                function = compile_native_coupling(frequency,
                    selected_jet_function(frequency, view_index, dependency=availability_dependency), branch_declarations,
                    precision=self.precision, midpoint=midpoint, availability_dependency=availability_dependency)
                self._functions[key] = function
            return function

    @staticmethod
    def operand_signature(operands):
        """Shared warm/execution signature of typed array/abstract operands."""
        import jax
        return tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in jax.tree.leaves(operands))

    def variant_key(self, kind, *, device, operands, view_index=None, demand=(),
                    bucket=None, transform='value', equation_signature=None):
        if kind not in ('candidate', 'candidate_ad', 'candidate_jvp', 'candidate_vjp', 'response_ad', 'chunk', 'root', 'coupling'):
            raise ValueError('unknown native physical variant kind')
        return ('native_physical_core', self.structural_identity, kind, view_index,
                tuple(sorted(demand)), bucket, transform, equation_signature, self.precision,
                (device.platform, device.id, device.process_index), self.operand_signature(operands))


def acquire_native_core(quantity_program, capabilities, *, owner=None, admit=False, existing=None):
    program = quantity_program.program
    model = program.model_program
    record = {'plan_sha256': quantity_program.identity['plan_sha256'],
        'sections': [[list(path), count] for path, count in model.mesh.sections],
        'operand_schema': [asdict(axis) for axis in model.fields.layout.axes],
        'authorized': model.fields.authorized,
        'material_authorized': [[term.authorized for term in material.terms]
                                for material in model.fields.structured],
        'line_domains': [{'sections': line.sections, 'declared_lossless': line.declared_lossless,
                          'electrical_resolution': line.electrical_resolution,
                          'resolution_policy': None if line.resolution_policy is None else
                          [line.resolution_policy[0], float(line.resolution_policy[1]).hex()]}
                         for line in model.lines],
        'views': [json.loads(view.declaration_bytes) for view in program.views],
        'precision': program.precision, 'capabilities': capabilities,
        'implementation': 'scnsim.jax.native_physical_core.v1'}
    identity = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    key = ('native_physical_core', identity)
    if owner is not None:
        admitted = owner.get(key)
        if admitted is not None:
            return admitted
    if existing is not None and existing.structural_identity == identity:
        result = existing
    else:
        programs = tuple(NativeFrequencyProgram.create(view) for view in program.views)
        candidate = partial(_candidate, model, program.views, tuple(item.assembly for item in programs),
                            capabilities, program.precision)
        result = NativePhysicalCore(candidate, programs, capabilities, program.precision, identity, key)
    if owner is not None and admit:
        owner.admit(key, result, kind='program', known_bytes=None)
        return owner.get(key, result)
    return result


@dataclass(frozen=True, slots=True, weakref_slot=True)
class NativePhysicalFunction:
    numeric_function: Any
    candidate_function: Any
    completed_primal_function: Any
    _frequency_functions: Any
    _root_functions: Any
    _root_declarations: Any
    _coupling_functions: Any
    input_layout: Any
    output_layout: Any
    aux_layout: Any
    structural_identity: str
    work_layout: tuple
    occurrence_layout: tuple
    root_layout: tuple
    coupling_layout: tuple
    project_frequency_outcomes: Any
    project_matrices: Any  # Ordered requested public matrices; no equation frames.
    project_public_values: Any
    core: NativePhysicalCore
    aggregate_identity: str
    output_dependencies: tuple
    output_covector_rows: Any
    branch_values_function: Any
    branch_ad_function: Any
    compact_projection_function: Any
    response_leaf_function: Any
    compact_leaf_projection_function: Any
    response_observable_indices: tuple
    branch_observable_indices: tuple

    branch_tangent_function = staticmethod(_branch_tangent)
    branch_pullback_function = staticmethod(_branch_pullback)

    @property
    def candidate_ad_function(self): return self.core.candidate_ad_function

    @property
    def candidate_tangents_function(self): return self.core.candidate_tangents_function

    @property
    def candidate_pullback_function(self): return self.core.candidate_pullback_function

    def response_ad_chunk(self, view_index, demand, bucket, mode='jvp'):
        return self.core.response_ad_chunk(view_index, demand, bucket, mode)

    def frequency_function(self, view_index, demand):
        """Stable exact-demand function; no coefficient/point admission."""
        return self.core.frequency_function(view_index, demand)

    def chunk_function(self, view_index, demand, bucket, transform='value'):
        return self.core.chunk_function(view_index, demand, bucket, transform)

    def variant_key(self, kind, **options):
        if kind in ('branch_jvp', 'branch_vjp'):
            raise TypeError('branch derivative executables are operation-owned by their exact BranchADFrame; no Run variant key')
        if kind in ('value', 'branch_ad'):
            device, operands = options.pop('device'), options.pop('operands')
            transform = options.pop('transform', 'value')
            if options:
                raise TypeError('aggregate variant accepts device, operands and transform')
            return ('native_physical_aggregate', self.core.structural_identity,
                self.aggregate_identity, transform if kind == 'value' else (kind, transform), self.core.precision,
                (device.platform, device.id, device.process_index),
                self.core.operand_signature(operands))
        return self.core.variant_key(kind, **options)

    def root_function(self, view_index, declaration):
        return self.core.root_function(view_index, declaration)

    def branch_function(self, start_index):
        """Declared seed ordinal to the SAME detached compiled root callable."""
        return self._root_functions[start_index]

    def coupling_function(self, observable_index):
        return self._coupling_functions[observable_index]

    def coupling_variant_key(self, observable_index, *, device, operands, transform='value'):
        row = next(item for item in self.coupling_layout if item['observable_index'] == observable_index)
        return self.core.variant_key('coupling', device=device, operands=operands,
            view_index=row['view_index'], transform=transform, equation_signature=row['equation_signature'])

    def branch_variant_key(self, start_index, *, device, operands, transform='value'):
        declaration = self._root_declarations[start_index]
        row = next(item for item in self.root_layout if item['start_index'] == start_index)
        signature = (declaration.kind, declaration.coordinate_index,
            declaration.row_index, declaration.column_index, declaration.family,
            declaration.output_index, declaration.input_index,
            ('root_occurrence', row['view_index'], start_index,
                (declaration.kind, declaration.coordinate_index, declaration.row_index,
                 declaration.column_index, declaration.family, declaration.output_index, declaration.input_index)))
        return self.core.variant_key('root', device=device, operands=operands,
            view_index=row['view_index'], transform=transform, equation_signature=signature)


def compile_native_program(quantity_program, inputs, *, capabilities=None, core=None):
    """Acquire one detached native partition from the ACTUAL original point.

    Responses, scalar roots and residue coupling consume one candidate and
    exact selected-state authority. Unsupported descriptor kinds fail before
    execution rather than acquiring a second numerical evaluator.
    """
    program = quantity_program.program
    capabilities = program._capabilities(inputs) if capabilities is None else tuple(capabilities)
    root_kinds = frozenset(('diagonal_root', 'operator_element_root', 'hybridized_pole', 'transfer_zero'))
    supported = {'S', 'Y', 'Z', 'operator', 'response_element', 'residue_coupling'} | root_kinds
    if any(descriptor.kind not in supported for descriptor in program.observables):
        raise TypeError('unsupported native observable kind: ' + repr(tuple(
            descriptor.kind for descriptor in program.observables if descriptor.kind not in supported)))
    core = acquire_native_core(quantity_program, capabilities) if core is None else core
    frequency_programs = core.frequency_programs
    candidate = core.candidate_function
    # Fixed declaration aliases only. Independent root/coupling coordinates
    # never enter this response namespace. Actual omega values stay runtime.
    works, by_key, gathers = [], {}, []
    source_indices = []
    root_entries, root_by_start, root_declarations = [], {}, {}
    coupling_entries = []
    from .program_models import RootDeclaration
    for ordinal, descriptor in enumerate(program.observables):
        if descriptor.kind == 'residue_coupling':
            midpoint = descriptor.frequency_mode == 'complex_root_midpoint'
            evaluation_index = None
            if not midpoint:
                evaluation_index = len(source_indices)
                source_indices.append(descriptor.frequency_slice[0])
            coupling_entries.append((ordinal, descriptor, midpoint, evaluation_index))
            gathers.append(())
            continue
        if descriptor.kind in root_kinds:
            start = descriptor.branch_indices[0]
            arguments = {}
            if descriptor.kind == 'diagonal_root':
                arguments['coordinate_index'] = descriptor.indices[0]
            elif descriptor.kind == 'operator_element_root':
                arguments['row_index'], arguments['column_index'] = descriptor.indices
            elif descriptor.kind == 'transfer_zero':
                arguments['family'] = ('S', 'Y', 'Z')[descriptor.indices[0]]
                arguments['output_index'], arguments['input_index'] = descriptor.indices[1:]
            declaration = RootDeclaration(descriptor.kind, start, descriptor.quantity_key, **arguments)
            structure = (descriptor.view_index, descriptor.kind, descriptor.indices)
            if start in root_by_start:
                if root_by_start[start][0] != structure:
                    raise ValueError('one declared root seed refers to inconsistent physical equations')
            else:
                function = core.root_function(descriptor.view_index, declaration)
                root_by_start[start] = (structure, function)
                root_declarations[start] = declaration
                root_entries.append((start, function))
            gathers.append(())
            continue
        family = ('S', 'Y', 'Z')[descriptor.indices[0]] if descriptor.kind == 'response_element' else descriptor.kind
        selected = []
        for occurrence in range(*descriptor.frequency_slice):
            bits = np.asarray(quantity_program.frequencies_rad_s[occurrence], dtype=np.complex128).tobytes()
            key = descriptor.view_index, bits
            if key not in by_key:
                by_key[key] = len(works)
                works.append([descriptor.view_index, set(), len(source_indices)])
                source_indices.append(occurrence)
            index = by_key[key]
            works[index][1].add(family)
            selected.append(index)
        gathers.append(tuple(selected))
    coupling_functions, coupling_rows = {}, []
    for ordinal, descriptor, midpoint, evaluation_index in coupling_entries:
        declarations = tuple(root_declarations[program.observables[index].branch_indices[0]]
                             for index in descriptor.dependency_indices)
        branch_occurrences = tuple(('root_occurrence', descriptor.view_index, item.start_index,
            (item.kind, item.coordinate_index, item.row_index, item.column_index,
             item.family, item.output_index, item.input_index)) for item in declarations)
        availability_dependency = ('coupling_occurrence', descriptor.view_index, ordinal,
                                   branch_occurrences, midpoint)
        function = core.coupling_function(descriptor.view_index, declarations, midpoint=midpoint,
                                         availability_dependency=availability_dependency)
        coupling_functions[ordinal] = function
        signature = (tuple((item.kind, item.coordinate_index) for item in declarations), midpoint,
                     availability_dependency)
        coupling_rows.append(MappingProxyType({'observable_index': ordinal,
            'view_index': descriptor.view_index, 'branch_indices': tuple(item.start_index for item in declarations),
            'midpoint': midpoint, 'evaluation_index': evaluation_index, 'equation_signature': signature}))
    functions = {}
    for view_index, demand, _ in works:
        key = view_index, tuple(sorted(demand))
        if key not in functions:
            functions[key] = core.frequency_function(view_index, demand)
    # Adjacent same-structure occurrences share a scan body. No all-frequency
    # Python graph or frequency-specific jitted function is created.
    segments = []
    for index, (view_index, demand, source) in enumerate(works):
        key = view_index, tuple(sorted(demand))
        if segments and segments[-1][0] == key:
            segments[-1][1].append((index, source))
        else:
            segments.append((key, [(index, source)]))
    frozen_segments = tuple((functions[key], tuple(indices)) for key, indices in segments)
    locations = {work: (segment, local) for segment, (_, indices) in enumerate(frozen_segments)
                 for local, (work, _) in enumerate(indices)}
    selection_runs = []
    for selected in gathers:
        runs = []
        for work in selected:
            segment, local = locations[work]
            if runs and runs[-1][0] == segment:
                runs[-1][1].append(local)
            else:
                runs.append((segment, [local]))
        selection_runs.append(tuple((segment, tuple(indices)) for segment, indices in runs))
    selection_runs = tuple(selection_runs)
    descriptors, expressions = program.observables, program.output_expressions
    precision = program.precision
    def root_dependency(item):
        return ('root_occurrence', root_by_start[item.start_index][0][0], item.start_index,
            (item.kind, item.coordinate_index, item.row_index, item.column_index,
             item.family, item.output_index, item.input_index))
    leaf_dependencies = []
    for ordinal, descriptor in enumerate(descriptors):
        roots = tuple(root_dependency(root_declarations[start])
                      for start in descriptor.branch_indices if start in root_declarations)
        if descriptor.kind == 'residue_coupling':
            roots = tuple(root_dependency(root_declarations[
                descriptors[index].branch_indices[0]]) for index in descriptor.dependency_indices)
            leaf_dependencies.append((*roots, ('coupling_occurrence', descriptor.view_index, ordinal,
                roots, descriptor.frequency_mode == 'complex_root_midpoint')))
        else:
            response_family = (('S', 'Y', 'Z')[descriptor.indices[0]]
                if descriptor.kind == 'response_element' else descriptor.kind)
            leaf_dependencies.append(roots if roots else (('response', descriptor.view_index, response_family),))
    output_dependencies = (tuple(tuple(dict.fromkeys(dependency
        for index in expression.indices for dependency in leaf_dependencies[index]))
        for expression in expressions) if expressions else tuple(leaf_dependencies))

    def response_leaf_function(observable_index, family_array):
        """Select one declared raw response leaf, retaining its frequency axis.

        This exact callable also defines tangent selection and its JAX
        transpose. It contains no numerical solve or expression projection.
        """
        descriptor = descriptors[observable_index]
        if descriptor.kind == 'response_element':
            return family_array[:, descriptor.indices[1], descriptor.indices[2]]
        if descriptor.kind not in ('S', 'Y', 'Z', 'operator'):
            raise TypeError('response leaf selection requires a response observable')
        return family_array

    def project_selected_values(raw_leaves, *, availability=None):
        """One scalar normalization, projection and expression authority."""
        import jax.numpy as jnp
        raw_values, values, available = [], [], []
        for ordinal, (descriptor, selection, value) in enumerate(zip(descriptors, gathers, raw_leaves, strict=True)):
            if descriptor.kind == 'response_element' and len(selection) == 1:
                value = value[0]
            raw_values.append(value)
            unavailable = jnp.bool_(False)
            if descriptor.projection in ('absolute', 'magnitude', 'phase'):
                unavailable = value == 0
                if descriptor.projection == 'phase':
                    unavailable = unavailable | ((value.imag == 0) & (value.real < 0))
            eligible = jnp.bool_(True) if availability is None else availability[ordinal]
            available.append(eligible & ~jnp.any(unavailable))
            if descriptor.kind in root_kinds:
                projected = (value.real / (2 * jnp.pi) if descriptor.projection == 'frequency'
                             else -value.imag / jnp.pi if descriptor.projection == 'linewidth'
                             else value if descriptor.projection is None
                             else None)
                if projected is None:
                    raise ValueError('unknown declared root projection')
            else:
                projected = _projection(value, descriptor.projection)
            values.append(projected)
        observable_values = tuple(values)
        if expressions:
            available = tuple((~expression.nondifferentiable(*values)) &
                              jnp.all(jnp.stack(tuple(available[index] for index in expression.indices)))
                              for expression in expressions)
            values = tuple(expression.apply(*values) for expression in expressions)
        return NativePublicProjection(observable_values, tuple(raw_values), tuple(values), tuple(available))

    def project_public_values(matrices, *, availability=None):
        selected = tuple(matrix if descriptor.kind in root_kinds or descriptor.kind == 'residue_coupling'
                         else response_leaf_function(index, matrix)
                         for index, (descriptor, matrix) in enumerate(zip(descriptors, matrices, strict=True)))
        return project_selected_values(selected, availability=availability)

    def project_matrices(matrices):
        projection = project_public_values(matrices)
        return _flatten(projection.output_values), projection.derivative_available

    def project_frequency_outcomes(outcomes, *, branch_packets=(), quantity_packets=()):
        """Project ordered unique-work outcomes with original occurrence gathers.

        Supplied arrays remain operation-owned. Both raw and scheduled paths
        use the same static selectors and expression lowering.
        """
        import jax.numpy as jnp
        matrices = []
        packet_by_start = dict(branch_packets)
        packet_by_observable = dict(quantity_packets)
        for ordinal, (descriptor, selection) in enumerate(zip(descriptors, gathers, strict=True)):
            if descriptor.kind == 'residue_coupling':
                matrices.append(packet_by_observable[ordinal].coupling_rad_s)
                continue
            if descriptor.kind in root_kinds:
                matrices.append(packet_by_start[descriptor.branch_indices[0]].omega)
                continue
            family = ('S', 'Y', 'Z')[descriptor.indices[0]] if descriptor.kind == 'response_element' else descriptor.kind
            matrices.append(jnp.stack(tuple(getattr(outcomes[work], family) for work in selection)))
        return project_matrices(tuple(matrices))

    branch_observable_indices = tuple(index for index, descriptor in enumerate(descriptors)
        if descriptor.kind in root_kinds or descriptor.kind == 'residue_coupling')
    response_observable_indices = tuple(index for index in range(len(descriptors))
        if index not in branch_observable_indices)

    def branch_values_function(point, omega, seeds, prior_check_status=0,
                               prior_certificate_status=0, completed_branches=None):
        """SAME differentiable root/coupling subtree, with original stage order.

        Raw aggregate and bounded response-AD composition share this callable.
        Completed packets are an internal primal-only continuation option;
        differentiation uses the ordinary runtime candidate and implicit roots.
        """
        import jax
        import jax.numpy as jnp
        from .jax_core import first
        with jax.enable_x64(True):
            cd = jnp.complex128 if precision == 'float64' else jnp.complex64
            omega = jnp.asarray(omega, dtype=cd)
            check_status = first(point.physical_checks.status, point.partition_status != 0, point.partition_status)
            check_status = first(check_status, prior_check_status != 0, prior_check_status)
            certificate_status = jnp.asarray(prior_certificate_status, dtype=jnp.int32)
            branch_packets = []
            packet_by_start = {}
            supplied = {} if completed_branches is None else dict(completed_branches)
            if completed_branches is not None and tuple(start for start, _ in completed_branches) != tuple(start for start, _ in root_entries):
                raise ValueError('completed branch packets differ from compiled ordered root starts')
            for start, function in root_entries:
                if completed_branches is not None:
                    packet = supplied[start]
                    check_status = first(check_status, packet.check_status != 0, packet.check_status)
                    certificate_status = first(certificate_status, packet.certificate_status != 0, packet.certificate_status)
                    branch_packets.append((start, packet))
                    packet_by_start[start] = packet
                    continue
                seed = jnp.asarray(seeds[start], dtype=cd)
                shape = jax.eval_shape(function, point, seed)
                empty = jax.tree.map(lambda item: jnp.zeros(item.shape, dtype=item.dtype), shape)
                prior = first(check_status, certificate_status != 0, certificate_status)
                skipped = empty._replace(omega=seed, status=prior, check_status=check_status,
                                         certificate_status=certificate_status)
                packet = jax.lax.cond(prior == 0, lambda _: function(point, seed),
                                      lambda _: skipped, operand=None)
                check_status = first(check_status, packet.check_status != 0, packet.check_status)
                certificate_status = first(certificate_status, packet.certificate_status != 0,
                                           packet.certificate_status)
                branch_packets.append((start, packet))
                packet_by_start[start] = packet
            quantity_packets, packet_by_observable = [], {}
            for row in coupling_rows:
                branch_packets_for_coupling = tuple(packet_by_start[start] for start in row['branch_indices'])
                evaluation = (jnp.asarray(0, dtype=cd) if row['midpoint'] else omega[row['evaluation_index']])
                function = coupling_functions[row['observable_index']]
                shape = jax.eval_shape(function, point, branch_packets_for_coupling, evaluation)
                empty = jax.tree.map(lambda item: jnp.zeros(item.shape, dtype=item.dtype), shape)
                prior = first(check_status, certificate_status != 0, certificate_status)
                skipped = empty._replace(status=prior, check_status=check_status,
                                         certificate_status=certificate_status)
                packet = jax.lax.cond(prior == 0,
                    lambda _: function(point, branch_packets_for_coupling, evaluation),
                    lambda _: skipped, operand=None)
                check_status = first(check_status, packet.check_status != 0, packet.check_status)
                certificate_status = first(certificate_status, packet.certificate_status != 0, packet.certificate_status)
                quantity_packets.append((row['observable_index'], packet))
                packet_by_observable[row['observable_index']] = packet
            values = tuple(packet_by_observable[index].coupling_rad_s
                if descriptors[index].kind == 'residue_coupling' else
                packet_by_start[descriptors[index].branch_indices[0]].omega
                for index in branch_observable_indices)
            available = tuple(packet_by_observable[index].derivative_available
                if descriptors[index].kind == 'residue_coupling' else jnp.bool_(True)
                for index in branch_observable_indices)
            return NativeBranchOutcome(values, available, tuple(branch_packets),
                tuple(quantity_packets), check_status, certificate_status)

    def compact_leaf_projection_function(response_values, branch_values=(), branch_available=None):
        """Sole expression/projection authority over compact public leaves.

        Ordered response arrays retain their original occurrence grids. Root
        leaves may be differentiated from the shared subtree; they are never
        installed here as surrogate constant roots by the numerical authority.
        """
        import jax.numpy as jnp
        if len(response_values) != len(response_observable_indices) or len(branch_values) != len(branch_observable_indices):
            raise ValueError('compact public operands differ from the declared response/branch leaf layout')
        values = [None] * len(descriptors)
        available = [jnp.bool_(True)] * len(descriptors)
        for index, value in zip(response_observable_indices, response_values, strict=True):
            values[index] = value
        for ordinal, (index, value) in enumerate(zip(branch_observable_indices, branch_values, strict=True)):
            values[index] = value
            if branch_available is not None:
                available[index] = branch_available[ordinal]
        projection = project_selected_values(tuple(values), availability=tuple(available))
        return _flatten(projection.output_values), projection

    def compact_projection_function(response_values, branch_values=(), branch_available=None):
        """Full-family callers delegate to the same selected-leaf authority."""
        selected = tuple(response_leaf_function(index, value)
                         for index, value in zip(response_observable_indices, response_values, strict=True))
        return compact_leaf_projection_function(selected, branch_values, branch_available)

    def consume_point(point, omega, seeds, completed_branches=None):
        import jax
        import jax.numpy as jnp
        from .jax_core import first
        with jax.enable_x64(True):
            cd = jnp.complex128 if precision == 'float64' else jnp.complex64
            omega = jnp.asarray(omega, dtype=cd)
            check_status = first(point.physical_checks.status, point.partition_status != 0, point.partition_status)
            certificate_status = jnp.int32(0)
            outcomes = []
            for function, indices in frozen_segments:
                positions = jnp.asarray(tuple(source for _, source in indices), dtype=jnp.int32)
                def body(carry, frequency):
                    result = function(point, frequency, *carry)
                    checks = first(carry[0], result.check_status != 0, result.check_status)
                    certs = first(carry[1], result.certificate_status != 0, result.certificate_status)
                    return (checks, certs), result
                (check_status, certificate_status), result = jax.lax.scan(
                    body, (check_status, certificate_status), omega[positions])
                outcomes.append(result)
            branch = branch_values_function(point, omega, seeds, check_status, certificate_status, completed_branches)
            check_status, certificate_status = branch.check_status, branch.certificate_status
            branch_packets, quantity_packets = branch.branch_packets, branch.quantity_packets
            packet_by_start, packet_by_observable = dict(branch_packets), dict(quantity_packets)
            matrices = []
            for ordinal, (descriptor, runs) in enumerate(zip(descriptors, selection_runs, strict=True)):
                if descriptor.kind == 'residue_coupling':
                    matrices.append(packet_by_observable[ordinal].coupling_rad_s)
                    continue

                if descriptor.kind in root_kinds:
                    matrices.append(packet_by_start[descriptor.branch_indices[0]].omega)
                    continue
                family = ('S', 'Y', 'Z')[descriptor.indices[0]] if descriptor.kind == 'response_element' else descriptor.kind
                pieces = tuple(getattr(outcomes[segment], family)[jnp.asarray(indices, dtype=jnp.int32)]
                               for segment, indices in runs)
                matrices.append(jnp.concatenate(pieces, axis=0))
            local_available = tuple(packet_by_observable[index].derivative_available
                if descriptor.kind == 'residue_coupling' else jnp.bool_(True)
                for index, descriptor in enumerate(descriptors))
            projection = project_public_values(tuple(matrices), availability=local_available)
            flat, available = _flatten(projection.output_values), projection.derivative_available
            return flat, NativeProgramAux(point.physical_checks, point.partition_status,
                point.realized_capabilities, check_status, certificate_status, tuple(outcomes), available, tuple(branch_packets), tuple(quantity_packets),
                projection.observable_values, projection.observable_raw_values, projection.output_values)
    def numeric_function(theta, omega, seeds):
        import jax
        with jax.enable_x64(True):
            return consume_point(core.candidate_function(theta), omega, seeds)

    output_blocks = program.output_layout.blocks
    def output_covector_rows(cotangents):
        """Pack JAX bilinear cotangents into declared real functional rows."""
        import jax.numpy as jnp
        pieces = []
        for block, value in zip(output_blocks, cotangents, strict=True):
            if tuple(value.shape) != block.shape:
                raise ValueError('output cotangent shape differs from its declared block')
            pieces.append((jnp.stack((value.real, -value.imag), axis=-1)
                if len(block.components) == 2 else value.real).reshape(-1))
        return jnp.concatenate(tuple(pieces)) if pieces else jnp.zeros((0,), dtype=
            jnp.float64 if precision == 'float64' else jnp.float32)

    seed_count = len(quantity_program.branch_starts_rad_s)
    def completed_primal_function(candidate_state, omega, completed_branch_packets):
        """Consume actual continuation packets, without restarting any root.

        The operation owns association with the same candidate/core/View/mesh.
        This primal-only seam deliberately does not substitute for the joint
        theta/root derivative authority exposed by numeric_function.
        """
        import jax
        import jax.numpy as jnp
        with jax.enable_x64(True):
            return consume_point(candidate_state, omega,
                jnp.zeros((seed_count,), dtype=omega.dtype),
                completed_branch_packets)

    rd = 'float64'  # Original physical operands; selected numerical dtype explicit below.
    cd = 'complex128' if precision == 'float64' else 'complex64'
    layout = MappingProxyType({'theta': {'shape': (len(program.model_program.fields.layout.axes),), 'dtype': rd},
        'omega': {'shape': (len(source_indices),), 'dtype': cd, 'source_indices': tuple(source_indices)},
        'seeds': {'shape': (len(quantity_program.branch_starts_rad_s),), 'dtype': cd},
        'selected_precision': precision, 'occurrence_gathers': tuple(gathers)})
    aux = MappingProxyType({'candidate': ('physical_checks', 'partition_status', 'realized_capabilities'),
        'reducers': ('check_status', 'certificate_status'),
        'root_start_indices': tuple(start for start, _ in root_entries),
        'segments': tuple({'work_indices': tuple(work for work, _ in indices),
                           'certificate_layout': frequency_programs[works[indices[0][0]][0]].certificate_layout(
                               families=works[indices[0][0]][1], capability=capabilities[works[indices[0][0]][0]])}
                          for _, indices in frozen_segments)})
    identity = hashlib.sha256(canonical_json_bytes({'program': quantity_program.program_sha256,
        'capabilities': capabilities, 'source_indices': source_indices, 'gathers': gathers,
        'implementation': 'scnsim.jax.native_physical.v1'})).hexdigest()
    work_layout = tuple(MappingProxyType({'work_index': index, 'view_index': view_index,
        'demand': tuple(sorted(demand)), 'omega_index': omega_index,
        'source_index': source_indices[omega_index]})
        for index, (view_index, demand, omega_index) in enumerate(works))
    occurrence_layout = tuple(MappingProxyType({'observable_index': index,
        'quantity_key': descriptor.quantity_key, 'view_index': descriptor.view_index,
        'family': ('S', 'Y', 'Z')[descriptor.indices[0]] if descriptor.kind == 'response_element' else descriptor.kind,
        'work_indices': gather, 'selector_indices': descriptor.indices,
        'projection': descriptor.projection, 'frequency_slice': descriptor.frequency_slice,
        'branch_indices': descriptor.branch_indices})
        for index, (descriptor, gather) in enumerate(zip(descriptors, gathers, strict=True)))
    root_layout = tuple(MappingProxyType({'start_index': start,
        'view_index': root_by_start[start][0][0], 'kind': root_by_start[start][0][1],
        'indices': root_by_start[start][0][2], 'declaration': root_declarations[start]}) for start, _ in root_entries)
    def static_record(value):
        if isinstance(value, float):
            return {'binary64': value.hex()}
        if isinstance(value, (tuple, list)):
            return [static_record(item) for item in value]
        if isinstance(value, dict):
            return {key: static_record(item) for key, item in value.items()}
        return value
    aggregate_identity = hashlib.sha256(canonical_json_bytes(static_record({
        'core': core.structural_identity,
        'descriptors': [{key: value for key, value in asdict(item).items() if key != 'axes'}
                        for item in descriptors],
        'expressions': [{key: value for key, value in asdict(item).items() if key != 'coordinates'}
                        for item in expressions],
        'source_indices': source_indices, 'gathers': gathers}))).hexdigest()
    return NativePhysicalFunction(numeric_function, candidate, completed_primal_function, MappingProxyType(functions),
                                  MappingProxyType(dict(root_entries)), MappingProxyType(root_declarations),
                                  MappingProxyType(coupling_functions), layout,
                                  program.output_layout.compact_record(), aux, identity,
                                  work_layout, occurrence_layout, root_layout, tuple(coupling_rows), project_frequency_outcomes, project_matrices, project_public_values, core, aggregate_identity, output_dependencies, output_covector_rows,
                                  branch_values_function, partial(_branch_ad, branch_values_function), compact_projection_function,
                                  response_leaf_function, compact_leaf_projection_function,
                                  response_observable_indices, branch_observable_indices)
