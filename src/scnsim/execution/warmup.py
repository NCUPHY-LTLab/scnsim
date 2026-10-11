"""Explicit Run-owned executable preparation, never an analysis attempt.

The existing request/compiler/View authorities select real declared shapes.
Only lowering and compilation run: no assembly execution, SuperLU, Newton,
CMA, Workspace publication, checkpoint or numerical result cache participates.
Strong ownership is the explicit device-scoped Run warm holder. Ordinary
operations may borrow its exact static programs but never auto-admit candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prepared import PreparedAnalysis
from hashlib import sha256
from time import perf_counter_ns


@dataclass(frozen=True, slots=True)
class WarmupKernel:
    signature_sha256: str
    kernel: str
    argument_shapes: tuple[tuple[tuple[int, ...], str], ...]
    reused: bool
    lookup_and_compile_ns: int
    compile_ns: int | None


@dataclass(frozen=True, slots=True)
class WarmupSummary:
    backend: str
    precision: str
    request_sha256: str
    parameter_points: int
    kernels: tuple[WarmupKernel, ...]
    elapsed_ns: int
    preparation_ns: int
    unknown_shapes_lazy: bool = True
    lazy_work: tuple[str, ...] = ('sparse_solve_and_factor_callbacks',
                                 'implicit_derivative_blocks',
                                 'Newton_and_continuation_execution',
                                 'future_stage_or_parameter_shapes',
                                 'future_candidate_ready_batch_widths')

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
    """Scope a detached explicit warm holder to the actual device.

    Ordinary operations only look up an existing scope; unmatched programs
    remain operation-owned. No backend discovery occurs in this helper.
    """
    if owner is None:
        return None
    key = ('physical_program_device', device.platform, device.id, device.process_index)
    return owner.setdefault(key, {}) if admit else owner.get(key)


def prepare_warmup(*, plan_document, analysis: PreparedAnalysis, precision,
                   executable_owner, baseline_parameters):
    """Lower/compile known pure assembly signatures without evaluating them.

    Only the explicit Run holder keeps admitted static programs and compiled
    executable handles alive; actual numerical calls borrow matching entries.
    Unknown sparse/implicit AD blocks and future stage shapes remain lazy;
    warmup is not a solved result or a promise of complete subsequent warmness.
    """
    started_ns = perf_counter_ns()
    from ..compilation.compiler import parameter_key
    from ..compilation.mesh import quantity
    from ..compilation.quantity_program import compile_point_quantity_program
    from ..compilation.quantity_declarations import quantity_declarations
    from ..compilation.objective_program import compile_objective_quantities
    from ..canonical import canonical_json_bytes
    from .config import ensure_jax_configuration, mark_jax_initialized
    from .python_host import resolved_points
    from .resources import operation_resources

    preparation_start = perf_counter_ns()
    request = analysis.request()
    operation, spec = request['operation'], request['spec']
    points = list(resolved_points(request['parameter_source']))
    if operation == 'evaluate_direct' and spec['type'] in (
            'diagonal_root', 'operator_element_root', 'hybridized_pole',
            'transfer_zero', 'residue_normalized_coupling'):
        if canonical_json_bytes(baseline_parameters) not in {
                canonical_json_bytes(point) for point in points}:
            points.insert(0, baseline_parameters)
    mesh = spec.get('discretization')
    if operation == 'optimize_direct' and spec['optimizer']['type'] == 'hybrid':
        first = spec['optimizer']['stages'][0]
        mesh = first['discretization']
        if first['seed'] is not None:
            points = [first['seed']]
    configuration = ensure_jax_configuration()
    import jax
    device = jax.devices('cpu')[0]
    mark_jax_initialized(configuration)
    device_identity = (device.platform, device.id, device.process_index)
    scoped_owner = program_owner_for_device(executable_owner, device, admit=True)
    preparation_ns = perf_counter_ns() - preparation_start
    kernels = []
    point_variants = {}
    with operation_resources(cpu_threads=configuration.cpu_threads) as resource:
        for point in points:
            preparation_start = perf_counter_ns()
            authorized = frozenset(
                parameter_key(item) for item in point.get('allow_extrapolation', ())
            ) | frozenset(parameter_key(item) for item in spec.get('allow_extrapolation', ()))
            if operation == 'optimize_direct':
                bounds = {parameter_key(variable['parameter']):
                          (quantity(variable['lower']), quantity(variable['upper']))
                          for variable in spec['variables'] if variable.get('domain') is None}
                program, _ = compile_objective_quantities(
                    plan_document, spec, point, default_view=request['view'],
                    mesh=mesh, bounds=bounds, precision=precision, authorized=authorized,
                    program_owner=scoped_owner, admit_program=True,
                )
            else:
                program = compile_point_quantity_program(
                    plan_document, quantity_declarations(spec, request['view']), point,
                    default_view=request['view'], mesh=mesh, precision=precision, authorized=authorized,
                    program_variants=point_variants, program_owner=scoped_owner,
                )
            program = scoped_owner.setdefault(('quantity_program', program.program_sha256), program)
            inputs = program.inputs(point)
            signatures = program.program.preparation_signatures(inputs)
            preparation_ns += perf_counter_ns() - preparation_start
            for structural_key, function, descriptors in signatures:
                shapes = tuple((tuple(argument.shape), str(argument.dtype))
                               for argument in descriptors)
                signature = (program.program_sha256, structural_key, shapes,
                             precision, device_identity)
                digest = sha256(canonical_json_bytes(signature)).hexdigest()
                lookup_start = perf_counter_ns()
                reused = signature in scoped_owner
                compile_ns = None
                if not reused:
                    with resource.exclusive_assembly(), jax.default_device(device), jax.enable_x64(precision == 'float64'):
                        compile_start = perf_counter_ns()
                        executable = function.lower(*descriptors).compile()
                        compile_ns = perf_counter_ns() - compile_start
                    # Publish each successful handle immediately so a later
                    # compilation failure does not erase earlier owned work.
                    scoped_owner[signature] = executable
                program.program.install_prepared_signature(structural_key, scoped_owner[signature])
                kernels.append(WarmupKernel(
                    digest, structural_key[1], shapes, reused,
                    perf_counter_ns() - lookup_start, compile_ns,
                ))
            # Only the explicitly admitted static transformable program and
            # compiled handles survive; point arrays/frames remain call-local.
            inputs = signatures = program = None
    return WarmupSummary(
        'jax', precision, analysis.request_sha256, len(points), tuple(kernels),
        perf_counter_ns() - started_ns, preparation_ns,
    )
