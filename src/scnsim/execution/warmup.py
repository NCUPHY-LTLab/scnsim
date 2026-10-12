"""Compile declared detached native signatures without numerical execution.

Preparation chooses the same physical program used by ordinary execution and
Sensitivity. This module lowers/compiles only: no SuperLU, Newton, objective
baseline evaluation, attempt, Result or recovery publication. Active cache
borrows cover compilation handoff; terminal AOT handles never replace the
transformable callable used to build derivative variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter_ns

from .compile_owner import CompileOwnerStats


@dataclass(frozen=True, slots=True)
class WarmupKernel:
    signature_sha256: str
    kernel: str
    argument_shapes: tuple[tuple[tuple[int, ...], str], ...]
    reused: bool
    lookup_and_compile_ns: int
    compile_ns: int | None
    admitted: bool
    known_bytes: int | None
    generated_code_bytes: int | None


@dataclass(frozen=True, slots=True)
class WarmupSummary:
    backend: str
    precision: str
    request_sha256: str | None
    parameter_points: int
    kernels: tuple[WarmupKernel, ...]
    elapsed_ns: int
    preparation_ns: int
    cache_stats: CompileOwnerStats
    prepared_function_sha256: str | None = None
    unknown_shapes_lazy: bool = True
    lazy_work: tuple[str, ...] = ('unrequested_transforms', 'undeclared_batch_shapes',
                                 'future_stage_or_parameter_shapes')

    @property
    def compiled_count(self):
        return sum(not kernel.reused for kernel in self.kernels)

    @property
    def reused_count(self):
        return sum(kernel.reused for kernel in self.kernels)

    @property
    def compile_ns(self):
        return sum(kernel.compile_ns for kernel in self.kernels if kernel.compile_ns is not None)


def program_owner_for_device(owner, device, *, admit=False):
    """Return the one bounded owner; compiler keys bind the actual device.

    No nested registry or program payload is retained under a device key.
    Admission and pinning use the explicit CompileOwner APIs at each consumer.
    """
    return owner





def prepared_variant_key(prepared, transform, batch_shape, device, argument_shapes):
    """Shared static lookup key; no runtime parameter or factor values."""
    return ('prepared_function', prepared.structural_identity, transform,
            tuple(batch_shape), prepared.precision,
            (device.platform, device.id, device.process_index), argument_shapes)


def _compile_variants(prepared, *, executable_owner, device, transforms, batch_shapes, resource):
    import jax
    from ..canonical import canonical_json_bytes

    # The native acquisition layout is the sole shape/dtype authority.
    descriptors = tuple(jax.ShapeDtypeStruct(tuple(prepared.input_layout[name]['shape']),
                                             prepared.input_layout[name]['dtype'])
                        for name in ('theta', 'omega', 'seeds'))
    kernels = []
    for batch_shape in ((), *tuple(tuple(shape) for shape in batch_shapes)):
        arguments = descriptors
        raw = prepared.numeric_function
        if batch_shape:
            for _ in batch_shape:
                raw = jax.vmap(raw, in_axes=(0, None, None))
            arguments = (jax.ShapeDtypeStruct(batch_shape + descriptors[0].shape,
                                              descriptors[0].dtype), *descriptors[1:])
        for transform in transforms:
            if transform == 'value':
                function, signature = raw, arguments
            elif transform == 'jvp':
                def function(theta, omega, seeds, tangent, raw=raw):
                    return jax.jvp(lambda value: raw(value, omega, seeds)[0],
                                   (theta,), (tangent,))
                signature = (*arguments, arguments[0])
            elif transform == 'vjp':
                def function(theta, omega, seeds, cotangent, raw=raw):
                    value, pullback = jax.vjp(lambda value: raw(value, omega, seeds)[0], theta)
                    return value, pullback(cotangent)[0]
                with jax.enable_x64(True):
                    packed = jax.eval_shape(raw, *arguments)[0]
                signature = (*arguments, packed)
            elif transform in ('jacfwd', 'jacrev'):
                derivative = jax.jacfwd if transform == 'jacfwd' else jax.jacrev
                function = derivative(lambda theta, omega, seeds, raw=raw:
                                      raw(theta, omega, seeds)[0], argnums=0)
                signature = arguments
            else:
                raise ValueError(f'unsupported warmup transform {transform!r}')
            shapes = tuple((tuple(arg.shape), str(arg.dtype)) for arg in signature)
            key = prepared_variant_key(prepared, transform, batch_shape, device, shapes)
            digest = sha256(canonical_json_bytes(key)).hexdigest()
            started_ns = perf_counter_ns()
            compile_ns = None
            def build():
                nonlocal compile_ns
                with resource.exclusive_assembly(), jax.default_device(device), jax.enable_x64(True):
                    compile_start = perf_counter_ns()
                    executable = jax.jit(function).lower(*signature).compile()
                    compile_ns = perf_counter_ns() - compile_start
                    return executable

            def code_bytes(executable):
                # This is known generated-code size, not full host residency or
                # temporary device memory. Unexposed sizes remain unknown.
                size = getattr(executable, 'size_of_generated_code_in_bytes', None)
                return None if size is None else size()

            with executable_owner.acquire(key, build, kind='variant', known_bytes=None) as executable:
                admitted = executable_owner.get(key, kind='variant') is executable
                kernels.append(WarmupKernel(digest, transform, shapes, compile_ns is None,
                    perf_counter_ns() - started_ns, compile_ns, admitted, None, code_bytes(executable)))
            executable = function = None
    return tuple(kernels)



def _compile_execution_variants(native, *, executable_owner, device, resource,
                                checked_derivatives=False):
    """Lower the SAME candidate and checked value chunks used by execution.

    eval_shape establishes only the candidate PyTree signature. It does not
    execute physical assembly, sparse callbacks or Newton. Work declarations
    choose exact View/family namespaces; numerical omega values are not keys.
    """
    import jax
    from ..canonical import canonical_json_bytes
    from ..numerics.native_sparse import checked_native

    theta = jax.ShapeDtypeStruct(native.input_layout['theta']['shape'], 'float64')
    status = jax.ShapeDtypeStruct((), 'int32')
    omega_dtype = native.input_layout['omega']['dtype']
    kernels = []

    def compile_one(kind, function, arguments, *, key=None, **options):
        if key is None:
            key = native.variant_key(kind, device=device, operands=arguments, **options)
        shapes = native.core.operand_signature(arguments)
        digest = sha256(canonical_json_bytes(key)).hexdigest()
        started_ns = perf_counter_ns()
        compile_ns = None

        def build():
            nonlocal compile_ns
            with resource.exclusive_assembly(), jax.default_device(device), jax.enable_x64(True):
                compile_start = perf_counter_ns()
                executable = jax.jit(function).lower(*arguments).compile()
                compile_ns = perf_counter_ns() - compile_start
                return executable

        with executable_owner.acquire(key, build, kind='variant', known_bytes=None) as executable:
            admitted = executable_owner.get(key, kind='variant') is executable
            size = getattr(executable, 'size_of_generated_code_in_bytes', None)
            kernels.append(WarmupKernel(digest, kind, shapes, compile_ns is None,
                perf_counter_ns() - started_ns, compile_ns, admitted, None,
                None if size is None else size()))

    with executable_owner.hold_exact_core(native.core):
        compile_one('candidate', native.candidate_function, (theta,))
        with resource.exclusive_assembly(), jax.default_device(device), jax.enable_x64(True):
            candidate = jax.eval_shape(native.candidate_function, theta)
        if checked_derivatives:
            # The producer is the SAME cached checked linearization as execution.
            # Do not construct consumers from an independently traced Partial:
            # identical leaf shapes do not establish identical static jaxpr.
            compile_one('candidate_ad', checked_native(native.candidate_ad_function), (theta,),
                transform='checked_candidate_linearization')
            if native.root_layout or native.coupling_layout:
                arguments = (candidate,
                    jax.ShapeDtypeStruct(native.input_layout['omega']['shape'], omega_dtype),
                    jax.ShapeDtypeStruct(native.input_layout['seeds']['shape'],
                                         native.input_layout['seeds']['dtype']), status, status)
                compile_one('branch_ad', checked_native(native.branch_ad_function), arguments,
                    transform='checked_joint_branch_linearization')
        namespaces = tuple(dict.fromkeys((row['view_index'], tuple(row['demand']))
                                        for row in native.work_layout))
        for view_index, demand in namespaces:
            for bucket in dict.fromkeys((executable_owner.policy.frequency_chunk_size, 1)):
                arguments = (candidate, jax.ShapeDtypeStruct((bucket,), omega_dtype), status, status)
                compile_one('chunk', native.chunk_function(view_index, demand, bucket), arguments,
                            view_index=view_index, demand=demand, bucket=bucket, transform='value')
        seed = jax.ShapeDtypeStruct((), native.input_layout['seeds']['dtype'])
        root_packets = {}
        for row in native.root_layout:
            start = row['start_index']
            branch = native.branch_function(start)
            arguments = (candidate, seed)
            compile_one('root', checked_native(branch), arguments,
                key=native.branch_variant_key(start, device=device, operands=arguments,
                                               transform='checked_value'))
            with resource.exclusive_assembly(), jax.default_device(device), jax.enable_x64(True):
                root_packets[start] = jax.eval_shape(branch, *arguments)
        for row in native.coupling_layout:
            ordinal = row['observable_index']
            packets = tuple(root_packets[start] for start in row['branch_indices'])
            arguments = (candidate, packets, jax.ShapeDtypeStruct((), omega_dtype))
            compile_one('coupling', checked_native(native.coupling_function(ordinal)), arguments,
                key=native.coupling_variant_key(ordinal, device=device, operands=arguments,
                                                 transform='checked_value'))
    return tuple(kernels)


def warm_prepared(prepared, *, executable_owner, device=None,
                  transforms=('value',), batch_shapes=()):
    """Compile explicitly requested native transforms and point-batch shapes."""
    from .prepared_function import PreparedFunction
    from .config import ensure_jax_configuration, mark_jax_initialized
    from .resources import operation_resources
    if not isinstance(prepared, PreparedFunction):
        raise TypeError('prepared must be PreparedFunction')
    started_ns = perf_counter_ns()
    configuration = ensure_jax_configuration()
    import jax
    if device is None:
        device = jax.devices('cpu')[0]
    mark_jax_initialized(configuration)
    preparation_ns = perf_counter_ns() - started_ns
    with executable_owner.hold_exact_core(prepared._compile_core), \
            operation_resources(cpu_threads=configuration.cpu_threads) as resource:
        kernels = _compile_variants(prepared, executable_owner=executable_owner,
            device=device, transforms=tuple(transforms), batch_shapes=batch_shapes, resource=resource)
    return WarmupSummary('jax', prepared.precision, None, 1, kernels,
        perf_counter_ns() - started_ns, preparation_ns, executable_owner.stats(),
        prepared_function_sha256=prepared.structural_identity)


def prepare_warmup(*, plan_document, analysis, precision,
                   executable_owner, baseline_parameters,
                   transforms=('value',), batch_shapes=()):
    """Prepare the existing view/spec form through the same native acquisition."""
    from ..compilation.compiler import parameter_key
    from ..compilation.mesh import quantity
    from ..compilation.quantity_program import compile_point_quantity_program, compile_quantity_program
    from ..compilation.quantity_declarations import quantity_declarations
    from ..compilation.objective_program import compile_objective_quantities
    from ..compilation.prepared_function import prepared_from_quantity_program
    from .python_host import iter_resolved_points
    from .config import ensure_jax_configuration, mark_jax_initialized
    from .resources import operation_resources

    started_ns = perf_counter_ns()
    # Establish the declared runtime configuration before native acquisition
    # can initialize JAX while tracing its static callable.
    configuration = ensure_jax_configuration()
    request = analysis.request()
    spec = request['spec']
    points = iter_resolved_points(request['parameter_source'])
    if request['operation'] == 'evaluate_direct' and spec['type'] in (
            'diagonal_root', 'operator_element_root', 'hybridized_pole',
            'transfer_zero', 'residue_normalized_coupling'):
        # Root anchor preparation is compile-only. Cache admission deduplicates
        # repeated baseline signatures without retaining a sweep-wide point set.
        from itertools import chain
        points = chain((baseline_parameters,), points)
    mesh = spec.get('discretization')
    if request['operation'] == 'optimize_direct' and spec['optimizer']['type'] == 'hybrid':
        first = spec['optimizer']['stages'][0]
        mesh = first['discretization']
        if first['seed'] is not None:
            points = [first['seed']]
    kernels, preparation_ns = [], 0
    point_count = 0
    for point in points:
        point_count += 1
        preparation_start = perf_counter_ns()
        authorized = frozenset(parameter_key(ref) for ref in point.get('allow_extrapolation', ()))
        authorized |= frozenset(parameter_key(ref) for ref in spec.get('allow_extrapolation', ()))
        import jax
        with jax.enable_x64(True):
            if request['operation'] == 'sensitivity':
                program = compile_quantity_program(plan_document,
                    {row['id']: row['quantity'] for row in spec['quantities']}, point,
                    mesh=mesh, precision=precision, authorized=authorized,
                    program_owner=executable_owner, admit_program=True)
            elif request['operation'] == 'optimize_direct':
                bounds = {parameter_key(row['parameter']): (quantity(row['lower']), quantity(row['upper']))
                          for row in spec['variables'] if row.get('domain') is None}
                program, _ = compile_objective_quantities(plan_document, spec, point,
                    default_view=request['view'], mesh=mesh, bounds=bounds,
                    precision=precision, authorized=authorized,
                    program_owner=executable_owner, admit_program=True)
            else:
                program = compile_point_quantity_program(plan_document,
                    quantity_declarations(spec, request['view']), point,
                    default_view=request['view'], mesh=mesh, precision=precision,
                    authorized=authorized, program_variants={},
                    program_owner=executable_owner, admit_program=True)
            inputs = program.inputs(point)
            native = program.native_program(inputs, program_owner=executable_owner, admit_core=True)
        device = jax.devices('cpu')[0]
        mark_jax_initialized(configuration)
        preparation_ns += perf_counter_ns() - preparation_start
        with operation_resources(cpu_threads=configuration.cpu_threads) as resource:
            kernels.extend(_compile_execution_variants(native, executable_owner=executable_owner,
                                                       device=device, resource=resource,
                checked_derivatives=(request['operation'] == 'sensitivity' or
                    any(transform != 'value' for transform in transforms))))
        # Explicit raw derivatives/batches are different callable signatures and
        # reported separately; they never imply a checked chunk cache hit.
        raw_transforms = tuple(transform for transform in transforms if transform != 'value')
        if raw_transforms or batch_shapes:
            prepared = prepared_from_quantity_program(program, point,
                wrt=(tuple(parameter_key(ref) for ref in spec['wrt'])
                     if request['operation'] == 'sensitivity' else None),
                program_owner=executable_owner)
            summary = warm_prepared(prepared, executable_owner=executable_owner, device=device,
                transforms=transforms if batch_shapes else raw_transforms, batch_shapes=batch_shapes)
            preparation_ns += summary.preparation_ns
            kernels.extend(summary.kernels)
            del prepared, summary
        del native, inputs, program
    return WarmupSummary('jax', precision, analysis.request_sha256, point_count,
        tuple(kernels), perf_counter_ns() - started_ns, preparation_ns,
        executable_owner.stats(), lazy_work=(
            ('checked_sensitivity_' + spec['derivative']['type'] + '_frame_consumers',)
            if request['operation'] == 'sensitivity' else
            ('checked_frame_consumers',) if any(transform != 'value' for transform in transforms) else ()) +
            ('unrequested_transforms', 'undeclared_batch_shapes', 'future_stage_or_parameter_shapes'))
