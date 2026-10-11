"""One parameter/field/View authority with explicit checked numerical jets.

Static programs retain declarations and stable arithmetic callables only.
Point arrays and checked equation frames belong to one call. Sparse callbacks
own their actual factors locally; no Run, Workspace, history or native handle
is captured. Real/imag flattening follows physical C-order, component fastest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
import numpy as np
from ..compilation.program import OutputAxis, SparsePattern, _indices
from .assembly_program import compile_assembly
from .sparse_solve import SparseEquation
from .checked_graph import CheckedGraph, zero_tangent

_ROOT_KINDS = frozenset(('diagonal_root', 'operator_element_root', 'hybridized_pole', 'transfer_zero'))


class ProgramInputs(NamedTuple):
    theta: Any
    omega_rad_s: Any
    branch_starts_rad_s: Any


class PreparedPoint(NamedTuple):
    """Operation-owned sparse bootstrap arrays, never retained by a program."""
    program_token: Any
    input_binding: tuple
    capabilities: tuple
    assemblies: tuple
    root_bootstraps: tuple
    failure: Any = None


@dataclass(frozen=True, slots=True)
class ObservableDescriptor:
    kind: str
    view_index: int
    frequency_slice: tuple[int, int]
    indices: tuple[int, ...] = ()
    branch_indices: tuple[int, ...] = ()
    projection: str | None = None
    quantity_key: str = ''
    si_unit: str = ''
    axes: tuple[tuple[str, tuple[str, ...]], ...] = ()
    dependency_indices: tuple[int, ...] = ()
    frequency_mode: str | None = None


@dataclass(frozen=True, slots=True)
class OutputRow:
    row: int
    observable_index: int
    quantity_key: str
    index: tuple[int, ...]
    component: str
    si_unit: str


@dataclass(frozen=True, slots=True)
class OutputLayout:
    axes: tuple[OutputAxis, ...]
    rows: tuple[OutputRow, ...]
    flatten_order: str = 'physical_C_order_then_component_real_imag'


class ProgramAux(NamedTuple):
    status: Any
    failed_site: Any
    certificates: tuple
    derivative_available: Any
    derivative_reasons: tuple[str, ...]
    realized_capabilities: tuple[bool, ...]
    observable_available: tuple
    observable_reasons: tuple[tuple[str, ...], ...]
    branch_results: tuple = ()
    quantity_results: tuple = ()
    observable_values: tuple = ()
    observable_raw_values: tuple = ()


class ProgramOutcome(NamedTuple):
    values: tuple
    aux: ProgramAux


class JacobianPayload(NamedTuple):
    values: Any
    row_indices: tuple[int, ...]
    shape: tuple[int, int]


class DerivativeOutcome(NamedTuple):
    primal: ProgramOutcome
    derivative: Any
    aux: ProgramAux


def _dense(pattern, values):
    import jax.numpy as jnp
    return jnp.zeros(pattern.shape, dtype=values.dtype).at[
        jnp.asarray(pattern.rows), jnp.asarray(pattern.cols)].add(values)


def _block(pattern, rows, columns):
    rmap, cmap = {value: index for index, value in enumerate(rows)}, {value: index for index, value in enumerate(columns)}
    sources, rr, cc = [], [], []
    for index, (r, c) in enumerate(zip(pattern.rows, pattern.cols, strict=True)):
        if int(r) in rmap and int(c) in cmap:
            sources.append(index); rr.append(rmap[int(r)]); cc.append(cmap[int(c)])
    result, slots = SparsePattern.from_coo((len(rows), len(columns)), rr, cc)
    # Parent CSC traversal gives submatrix CSC traversal with preserved row order.
    order = np.argsort(slots)
    return result, _indices(np.asarray(sources)[order])


def _full_pattern(size):
    return SparsePattern.from_coo((size, size), np.tile(np.arange(size), size),
                                  np.repeat(np.arange(size), size))[0]


def _flatten(values):
    import jax.numpy as jnp
    parts = []
    for value in values:
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            value = jnp.stack((value.real, value.imag), axis=-1)
        parts.append(value.ravel())
    return jnp.concatenate(parts) if parts else jnp.zeros((0,))


def _first(prior, current):
    import jax.numpy as jnp
    return jnp.where(prior != 0, prior, current).astype(jnp.int32)


def _projection(value, name):
    import jax.numpy as jnp
    if name is None:
        return value
    if name == 'real':
        return value.real
    if name == 'imag':
        return value.imag
    if name in ('magnitude', 'absolute'):
        return jnp.abs(value)
    if name == 'phase':
        return jnp.angle(value)
    raise ValueError(f'unknown observable projection {name}')


def _selected_precision(method):
    from functools import wraps
    @wraps(method)
    def checked(self, *args, **kwargs):
        import jax
        with jax.enable_x64(self.precision == 'float64'):
            return method(self, *args, **kwargs)
    return checked


@dataclass(frozen=True, slots=True, eq=False, init=False)
class PhysicalProgram:
    model_program: Any
    views: tuple
    observables: tuple[ObservableDescriptor, ...]
    precision: str
    assemblies: tuple
    output_layout: OutputLayout
    output_expressions: tuple
    _functions: Any
    _equations: Any
    _arithmetic_functions: Any
    _function_lock: Any
    _prepared_executables: Any
    _prepared_token: Any

    def __init__(self, model_program, views, observables, *, precision='float64', output_expressions=()):
        if precision not in ('float32', 'float64'):
            raise ValueError('precision must be float32 or float64')
        views, observables = tuple(views), tuple(observables)
        if any(view.model_program is not model_program for view in views):
            raise ValueError('all Views must bind this same ModelProgram')
        supported = {'S', 'Y', 'Z', 'operator', 'response_element', 'diagonal_root',
                     'operator_element_root', 'hybridized_pole', 'transfer_zero', 'residue_coupling'}
        for descriptor in observables:
            if descriptor.kind not in supported:
                raise TypeError(f'PhysicalProgram numerical route is not yet integrated: {descriptor.kind}')
            if descriptor.kind == 'response_element' and len(descriptor.indices) != 3:
                raise ValueError('response_element indices are (family index S/Y/Z, output, input)')
        assemblies = tuple(compile_assembly(view) for view in views)
        from threading import RLock
        functions, equations = {}, {}
        for index, (view, assembly) in enumerate(zip(views, assemblies, strict=True)):
            self._prepare_view(index, view, assembly, functions, equations)
        axes, rows = [], []
        for ordinal, descriptor in enumerate(observables):
            view = views[descriptor.view_index]
            count = descriptor.frequency_slice[1] - descriptor.frequency_slice[0]
            scalar = descriptor.kind in ('response_element', 'diagonal_root', 'operator_element_root', 'hybridized_pole', 'transfer_zero', 'residue_coupling')
            shape = () if scalar and (count == 1 or descriptor.branch_indices) else (count,) if scalar else (count, len(view.terminal_ids), len(view.terminal_ids))
            key = descriptor.quantity_key or str(ordinal)
            components = ('real',) if descriptor.projection is not None else ('real', 'imag')
            for physical_index in np.ndindex(shape):
                for component in components:
                    rows.append(OutputRow(len(rows), ordinal, key, physical_index, component, descriptor.si_unit))
            for component in components:
                axes.append(OutputAxis(key, component, descriptor.si_unit,
                                       tuple(view.terminal_ids), shape))
        output_expressions = tuple(output_expressions)
        if output_expressions:
            axes, rows = [], []
            for ordinal, expression in enumerate(output_expressions):
                components = ('real', 'imag') if expression.complex_output else ('real',)
                for index in np.ndindex(expression.shape):
                    for component in components:
                        rows.append(OutputRow(len(rows), ordinal, expression.quantity_key, index, component, expression.si_unit))
                for component in components:
                    axes.append(OutputAxis(expression.quantity_key, component, expression.si_unit, expression.coordinates, expression.shape))
        for name, value in (('model_program', model_program), ('views', views), ('observables', observables),
                            ('precision', precision), ('output_expressions', output_expressions), ('assemblies', assemblies), ('output_layout', OutputLayout(tuple(axes), tuple(rows))),
                            ('_functions', functions), ('_equations', equations), ('_arithmetic_functions', {}), ('_function_lock', RLock()), ('_prepared_executables', {}), ('_prepared_token', object())):
            object.__setattr__(self, name, value)

    @staticmethod
    def _prepare_view(index, view, assembly, functions, equations):
        import functools
        q, n = len(view.terminal_ids), len(view.node_ids)
        selected, eliminated = view.selected, view.eliminated
        rr, rr_map = _block(assembly.pattern, selected, selected)
        re, re_map = _block(assembly.pattern, selected, eliminated)
        er, er_map = _block(assembly.pattern, eliminated, selected)
        ee, ee_map = _block(assembly.pattern, eliminated, eliminated)
        maps = tuple((p, sources) for p, sources in ((rr, rr_map), (re, re_map), (er, er_map), (ee, ee_map)))
        def extract(values, which):
            import jax.numpy as jnp
            return values[jnp.asarray(maps[which][1])]
        for which in range(4):
            functions[index, 'block', which] = functools.partial(extract, which=which)
        functions[index, 'rr_dense'] = functools.partial(_dense, rr)
        functions[index, 'er_dense'] = functools.partial(_dense, er)
        functions[index, 'B_dense'] = functools.partial(_dense, view.patterns[3])
        equations[index, 'ee'] = SparseEquation(ee, 'MMD_AT_PLUS_A')
        equations[index, 're'] = SparseEquation(re)
        equations[index, 'W'] = SparseEquation(assembly.pattern)
        equations[index, 'small'] = SparseEquation(_full_pattern(q))
        functions[index, 'column_major'] = lambda matrix: matrix.T.ravel()
        functions[index, 'identity'] = lambda reference: __import__('jax').numpy.eye(q, dtype=reference.dtype)
        def schur(rr_values, re_values, x):
            return _dense(rr, rr_values) - equations[index, 're'].matmul(re_values, x)
        functions[index, 'schur'] = schur
        def schur_pair(rr_values, re_values, x):
            import jax
            import jax.numpy as jnp
            from . import compensated as cp
            kw = dict(xp=jnp, barrier=jax.lax.optimization_barrier)
            initial = tuple(_dense(rr, part) for part in rr_values)
            rows, cols = jnp.asarray(re.rows), jnp.asarray(re.cols)
            # Ordered entries, compensated products; selected output is small.
            def add_at(position, result):
                coefficient = tuple(part[position] for part in re_values)
                term = cp.multiply(coefficient, tuple(part[cols[position]] for part in x), **kw)
                at = rows[position]
                update = cp.add(tuple(part[at] for part in result), cp.negate(term), **kw)
                return tuple(part.at[at].set(value) for part, value in zip(result, update, strict=True))
            result = jax.lax.fori_loop(0, len(re.rows), add_at, initial)
            return cp.project(result, **kw)
        functions[index, 'schur_pair'] = schur_pair
        B = view.patterns[3]
        load_map = assembly.load_slots
        def stamp(B_values, matrix):
            import jax.numpy as jnp
            cols = jnp.asarray(B.cols)
            values = B_values[:, None] * matrix[cols[:, None], cols[None, :]] * B_values[None, :]
            return jnp.zeros((len(assembly.pattern.rows),), dtype=matrix.dtype).at[jnp.asarray(load_map).ravel()].add(values.ravel())
        functions[index, 'stamp'] = stamp

    def _cast(self, inputs):
        import jax.numpy as jnp
        rd, cd = (jnp.float32, jnp.complex64) if self.precision == 'float32' else (jnp.float64, jnp.complex128)
        return ProgramInputs(jnp.asarray(inputs.theta, dtype=rd), jnp.asarray(inputs.omega_rad_s, dtype=cd),
                             jnp.asarray(inputs.branch_starts_rad_s, dtype=cd))

    def _assembly_function(self, key):
        with self._function_lock:
            if key not in self._functions:
                index, _, loaded, compensated, capability = key
                assembly = self.assemblies[index]
                self._functions[key] = lambda theta, omega: assembly(theta, omega, loaded=loaded,
                                                                      compensated=compensated, port_realizable=capability)
            return self._functions[key]

    @staticmethod
    def _input_binding(inputs):
        return tuple((array.dtype.str, array.shape, array.tobytes(order='C'))
                     for array in (np.asarray(value) for value in inputs))

    def _prepared_capabilities(self, inputs, prepared):
        if prepared is None:
            return self._capabilities(inputs)
        if prepared.program_token is not self._prepared_token or prepared.input_binding != (self._input_binding(inputs), self._input_binding(self._cast(inputs))):
            from ..errors import CompilerInvariantError
            raise CompilerInvariantError('prepared assembly packet does not bind this program/input', stage='prepared_assembly')
        if prepared.failure is not None:
            from .evidence import numerical_error
            raise numerical_error(prepared.failure)
        return prepared.capabilities

    @_selected_precision
    def prepare_batch(self, inputs_sequence, *, sparse_runner=None):
        """Batch only known pure sparse assembly operands of one ready wave.

        Physical validation precedes selected casts. Each row binds its original physical and selected
        numerical operands; no factor, root, callback or AD frame executes here.
        Batch callables capture only assembly and immutable control values.
        """
        import jax
        import jax.numpy as jnp
        inputs_sequence = tuple(inputs_sequence)
        from ..errors import InvalidCandidatePhysicalParameter
        from .models import NumericalFailure
        from .evidence import evidence_bytes
        capabilities, failures = [], []
        for point in inputs_sequence:
            try:
                capabilities.append(self._capabilities(point))
                failures.append(None)
            except InvalidCandidatePhysicalParameter as error:
                capabilities.append(())
                failures.append(NumericalFailure(error.kind, error.stage, str(error), evidence_bytes(dict(error.evidence))))
        points = tuple(self._cast(point) for point in inputs_sequence)
        assemblies, roots, groups = [dict() for _ in points], [dict() for _ in points], {}
        for ordinal, (point, capability) in enumerate(zip(points, capabilities, strict=True)):
            if failures[ordinal] is not None:
                continue
            requests = {}
            for descriptor in self.observables:
                index = descriptor.view_index
                actual = capability[index]
                if descriptor.kind in _ROOT_KINDS:
                    start = descriptor.branch_indices[0]
                    loaded = descriptor.kind != 'transfer_zero' or not actual
                    # RootDeclaration owns the exact family interpretation.
                    root = self._root_program(descriptor)
                    compensated = root.declaration.kind == 'transfer_zero' and root.declaration.family == 'Z' and not actual
                    requests['root', start] = ((index, 'assembly', loaded, compensated, actual), point.branch_starts_rad_s[start])
                elif descriptor.kind != 'residue_coupling':
                    family = ('S', 'Y', 'Z')[descriptor.indices[0]] if descriptor.kind == 'response_element' else descriptor.kind
                    loaded = family == 'operator' or not actual
                    for frequency in range(*descriptor.frequency_slice):
                        key = (index, frequency, loaded, False, actual)
                        requests['assembly', key] = ((index, 'assembly', loaded, False, actual), point.omega_rad_s[frequency])
            for destination, (control, omega) in requests.items():
                groups.setdefault(control, []).append((ordinal, destination, point.theta, omega))
        for control, entries in groups.items():
            function = self._assembly_function(control)
            key = ('batch_assembly', control, len(entries))
            with self._function_lock:
                batched = self._functions.get(key)
                if batched is None:
                    batched = jax.jit(jax.vmap(function))
                    self._functions[key] = batched
            outputs = batched(jnp.stack([row[2] for row in entries]), jnp.stack([row[3] for row in entries]))
            for row, (ordinal, (kind, target), _, _) in enumerate(entries):
                packet = jax.tree.map(lambda value: value[row], outputs)
                (roots if kind == 'root' else assemblies)[ordinal][target] = packet
        return tuple(PreparedPoint(self._prepared_token, (self._input_binding(original), self._input_binding(point)), capability,
                                    tuple(assembly.items()), tuple(root.items()), failure)
                     for original, point, capability, assembly, root, failure in zip(inputs_sequence, points, capabilities, assemblies, roots, failures, strict=True))

    def _assemble_node(self, graph, view_index, frequency, loaded, compensated):
        # This callable is immutable structural metadata; no point value is
        # captured. The same instance is reused by all checked transformations.
        capability = graph.realized_capabilities[view_index]
        key = (view_index, 'assembly', loaded, compensated, capability)
        function = self._assembly_function(key)
        theta_key, omega_key = ('theta_operand',), ('omega_operand', frequency)
        if theta_key not in self._functions:
            self._functions[theta_key] = lambda point: point.theta
        if omega_key not in self._functions:
            self._functions[omega_key] = lambda point: point.omega_rad_s[frequency]
        theta = graph.pure(self._functions[theta_key], 0)
        omega = graph.pure(self._functions[omega_key], 0)
        packet_key = (view_index, frequency, loaded, compensated, capability)
        packet = graph.prepared_assemblies.get(packet_key)
        executor = self._prepared_executables.get(key)
        if packet is not None:
            executor = lambda *operands: packet
        return graph.pure(function, theta, omega, primal_executor=executor)

    @staticmethod
    def _status(graph, statuses):
        import jax.numpy as jnp
        status = jnp.int32(0)
        for value in statuses:
            status = _first(status, value)
        for certificate in graph.certificates:
            status = _first(status, certificate.status)
        return status

    def _small_solve(self, graph, view_index, matrix, rhs, stage, statuses):
        values = graph.pure(self._functions[view_index, 'column_major'], matrix)
        return graph.solve(self._equations[view_index, 'small'], values, rhs, stage=stage,
                           prior_status=self._status(graph, statuses))

    def _selected(self, graph, view_index, frequency, compensated, statuses):
        import jax.numpy as jnp
        assembled = self._assemble_node(graph, view_index, frequency, True, compensated)
        statuses.append(graph.values[assembled][0].status)
        high = graph.pure(lambda pair: pair[0].values, assembled)
        if compensated:
            pair = graph.pure(lambda value: (value[0].values, value[0].low_values), assembled)
            blocks = [graph.pure(lambda values, fn=self._functions[view_index, 'block', index]: tuple(fn(part) for part in values), pair)
                      for index in range(4)]
            rhs = graph.pure(lambda value: tuple(self._functions[view_index, 'er_dense'](part) for part in value), blocks[2])
            x = graph.solve_pair(self._equations[view_index, 'ee'], blocks[3], rhs, stage=3,
                                 prior_status=self._status(graph, statuses))
            F = graph.pure(self._functions[view_index, 'schur_pair'], blocks[0], blocks[1], x)
        else:
            blocks = [graph.pure(self._functions[view_index, 'block', index], high) for index in range(4)]
            rhs = graph.pure(self._functions[view_index, 'er_dense'], blocks[2])
            x = graph.solve(self._equations[view_index, 'ee'], blocks[3], rhs, stage=3,
                            prior_status=self._status(graph, statuses))
            F = graph.pure(self._functions[view_index, 'schur'], blocks[0], blocks[1], x)
        return F, assembled

    def _selected_frequency_jet(self, graph, view_index, frequency, statuses):
        """Schur value and analytic frequency derivative with one checked jet."""
        assembled = self._assemble_node(graph, view_index, frequency, True, False)
        statuses.append(graph.values[assembled][0].status)
        high = graph.pure(lambda pair: pair[0].values, assembled)
        prime = graph.pure(lambda pair: pair[0].frequency_values, assembled)
        blocks = [graph.pure(self._functions[view_index, 'block', i], high) for i in range(4)]
        primes = [graph.pure(self._functions[view_index, 'block', i], prime) for i in range(4)]
        rhs = graph.pure(self._functions[view_index, 'er_dense'], blocks[2])
        rhs_prime = graph.pure(self._functions[view_index, 'er_dense'], primes[2])
        solved = graph.frequency_solve(self._equations[view_index, 'ee'], blocks[3], primes[3],
                                       rhs, rhs_prime, prior_status=self._status(graph, statuses))
        x = graph.pure(lambda pair: pair[0], solved)
        xp = graph.pure(lambda pair: pair[1], solved)
        F = graph.pure(self._functions[view_index, 'schur'], blocks[0], blocks[1], x)
        def derivative(rrp, rep, re, x, xp):
            return self._functions[view_index, 'rr_dense'](rrp) - self._equations[view_index, 're'].matmul(rep, x) - self._equations[view_index, 're'].matmul(re, xp)
        Fp = graph.pure(derivative, primes[0], primes[1], blocks[1], x, xp)
        return F, Fp

    def _coupling_frame(self, inputs, roots, descriptor, branch_results, capabilities, *, sparse_runner):
        import jax.numpy as jnp
        from .null_vector import NullVectorEquation
        from .coupling_program import CouplingFrame
        from ..numeric_encoding import record_document, array_from_record
        midpoint = descriptor.frequency_mode == 'complex_root_midpoint'
        evaluation = (roots[0] + roots[1]) / 2 if midpoint else inputs.omega_rad_s[descriptor.frequency_slice[0]]
        point = ProgramInputs(inputs.theta, jnp.asarray([*roots, evaluation], dtype=inputs.omega_rad_s.dtype),
                              inputs.branch_starts_rad_s)
        registry = self._arithmetic_functions.setdefault(('coupling_equation', descriptor, capabilities), {})
        frame = CheckedGraph(point, arithmetic_functions=registry, sparse_runner=sparse_runner)
        frame.realized_capabilities = capabilities
        index, statuses, vectors, slopes = descriptor.view_index, [], [], []
        for at, (dependency, result) in enumerate(zip(descriptor.dependency_indices, branch_results, strict=True)):
            branch = self.observables[dependency]
            F, Fp = self._selected_frequency_jet(frame, index, at, statuses)
            if branch.kind == 'diagonal_root':
                coordinate = branch.indices[0]
                vector = frame.pure(lambda point, coordinate=coordinate: jnp.zeros((len(self.views[index].terminal_ids),), dtype=point.omega_rad_s.dtype).at[coordinate].set(1), 0)
            else:
                original = jnp.asarray(result.null_vector, dtype=point.omega_rad_s.dtype)
                original = original / jnp.linalg.norm(original)
                left = array_from_record(record_document(result.evidence_bytes)['left_null_vector'])
                factory = lambda F=frame.values[F], vector=original, left=left: NullVectorEquation(F, vector, left)
                vector = frame.implicit_equation(factory, original, F)
            vectors.append(vector)
            slopes.append(frame.pure(lambda v, fp: -v.T @ fp @ v, vector, Fp))
        F, _ = self._selected(frame, index, 2, False, statuses)
        output = frame.pure(lambda v0, matrix, v1, s0, s1: (v0.T @ matrix @ v1) / jnp.sqrt(s0 * s1),
                             vectors[0], F, vectors[1], *slopes)
        return CouplingFrame(frame, output, inputs, frequency_index=descriptor.frequency_slice[0], midpoint=midpoint)

    def _coupling_node(self, graph, descriptor, observable_index):
        import jax.numpy as jnp
        from .coupling_program import coupling_value
        from ..numeric_encoding import record_document
        dependencies = tuple(self.observables[i] for i in descriptor.dependency_indices)
        root_nodes = tuple(self._root_node(graph, branch) for branch in dependencies)
        programs = tuple(self._root_program(branch) for branch in dependencies)
        results = tuple(graph.nodes[node].primal for node in root_nodes)
        capability = graph.realized_capabilities[descriptor.view_index]
        primal = coupling_value(programs[0], graph.inputs, tuple(zip(programs, results, strict=True)),
                                capability=capability, frequency_index=descriptor.frequency_slice[0],
                                midpoint=descriptor.frequency_mode == 'complex_root_midpoint')
        roots = tuple(graph.values[node] for node in root_nodes)
        inputs, capabilities, runner = graph.inputs, graph.realized_capabilities, graph.sparse_runner
        factory = lambda: self._coupling_frame(inputs, roots, descriptor, results, capabilities,
                                               sparse_runner=runner)
        node = graph.implicit_equation(factory, jnp.asarray(primal.coupling_rad_s, dtype=graph.inputs.omega_rad_s.dtype), 0, *root_nodes)
        available, reasons = True, []
        for branch, result in zip(dependencies, results, strict=True):
            if branch.kind == 'hybridized_pole':
                magnitude = np.abs(result.null_vector)
                if np.count_nonzero(magnitude == magnitude.max()) != 1:
                    available = False
                    reasons.append('null-vector phase gauge has an exact largest-coordinate tie')
        slopes = record_document(primal.evidence_bytes)['branch_slopes']
        cd = np.complex64 if self.precision == 'float32' else np.complex128
        product = cd(cd(complex(slopes[0]['real'], slopes[0]['imag'])) * cd(complex(slopes[1]['real'], slopes[1]['imag'])))
        if product.imag == 0 and product.real < 0:
            available = False
            reasons.append('principal-square-root residue denominator lies on its branch cut')
        graph.quantity_results.append((observable_index, primal))
        return node, jnp.asarray(available), tuple(reasons)

    def _network(self, graph, view_index, frequency, families, statuses):
        import jax.numpy as jnp
        assembled = self._assemble_node(graph, view_index, frequency, False, False)
        statuses.append(graph.values[assembled][0].status)
        view = graph.pure(lambda value: value[1], assembled)
        Rk = graph.pure(lambda value: value.Rk.astype(value.model.C.dtype), view)
        eye = graph.pure(self._functions[view_index, 'identity'], Rk)
        Rinv = self._small_solve(graph, view_index, Rk, eye, 1, statuses)
        def source_rhs(value):
            return self._functions[view_index, 'B_dense'](value.model.B) @ value.selected_map.T
        Bk = graph.pure(source_rhs, view)
        def w_values(state, inverse, point):
            current, value = state
            B = self._functions[view_index, 'B_dense'](value.model.B)
            effective = value.selected_map.T @ inverse @ value.selected_map + value.Go
            load = self._functions[view_index, 'stamp'](value.model.B, effective)
            return current.values / (-jnp.asarray(1j, dtype=point.omega_rad_s.dtype) * point.omega_rad_s[frequency]) + load
        W = graph.pure(w_values, assembled, Rinv, 0)
        X = graph.solve(self._equations[view_index, 'W'], W, Bk, stage=5, prior_status=self._status(graph, statuses))
        Zsrc = graph.pure(lambda b, x: b.T @ x, Bk, X)
        Ysrc = self._small_solve(graph, view_index, Zsrc, eye, 6, statuses)
        Y = graph.pure(lambda source, inverse: source - inverse, Ysrc, Rinv)
        result = {'Y': Y}
        if 'Z' in families:
            result['Z'] = self._small_solve(graph, view_index, Y, eye, 7, statuses)
        if 'S' in families:
            D = graph.pure(lambda value: value.Dk, view)
            normalized = graph.pure(lambda d, y: d @ y @ d, D, Y)
            P = graph.pure(lambda identity, value: identity + value, eye, normalized)
            N = graph.pure(lambda identity, value: identity - value, eye, normalized)
            S = self._small_solve(graph, view_index, P, N, 8, statuses)
            Dinv = self._small_solve(graph, view_index, D, eye, 1, statuses)
            source = graph.pure(lambda b, d: jnp.asarray(2, dtype=b.dtype) * b @ d, Bk, Dinv)
            voltage = graph.solve(self._equations[view_index, 'W'], W, source, stage=9,
                                  prior_status=self._status(graph, statuses))
            source_response = graph.pure(lambda b, v: b.T @ v, Bk, voltage)
            source_s = self._small_solve(graph, view_index, D, source_response, 9, statuses)
            def certificate(src, scattering, identity):
                src = src - identity
                num = jnp.max(jnp.abs(src - scattering), initial=0)
                den = 1 + jnp.max(jnp.abs(src), initial=0) + jnp.max(jnp.abs(scattering), initial=0)
                eta = num / den
                threshold = jnp.asarray(256 * (len(self.views[view_index].node_ids) + 1), dtype=eta.dtype) * jnp.finfo(eta.dtype).eps
                return jnp.where(jnp.isfinite(eta) & (eta <= threshold), jnp.int32(0), jnp.int32(9))
            check = graph.pure(certificate, source_s, S, eye)
            statuses.append(graph.values[check])
            result['S'] = S
        return result

    def _root_program_unlocked(self, descriptor):
        from .root_program import RootProgram, RootDeclaration
        index = descriptor.view_index
        key = (index, 'root_program', descriptor.kind, descriptor.indices, descriptor.branch_indices)
        if key not in self._functions:
            arguments = {}
            if descriptor.kind == 'diagonal_root':
                arguments['coordinate_index'] = descriptor.indices[0]
            elif descriptor.kind == 'operator_element_root':
                arguments['row_index'], arguments['column_index'] = descriptor.indices
            elif descriptor.kind == 'transfer_zero':
                arguments['family'] = ('S', 'Y', 'Z')[descriptor.indices[0]]
                arguments['output_index'], arguments['input_index'] = descriptor.indices[1:]
            declaration = RootDeclaration(descriptor.kind, descriptor.branch_indices[0],
                                          descriptor.quantity_key, **arguments)
            self._functions[key] = RootProgram(self.assemblies[index], declaration, self.precision)
        return self._functions[key]

    def _root_program(self, descriptor):
        # Admission is structural only: no point/factor/frame is retained.
        with self._function_lock:
            return self._root_program_unlocked(descriptor)

    def _root_node(self, graph, descriptor):
        program = self._root_program(descriptor)
        capabilities, runner = graph.realized_capabilities, graph.sparse_runner
        factory = lambda inputs, omega: self._root_frame(inputs, omega, descriptor, capabilities,
                                                         sparse_runner=runner)
        return graph.root(program, factory, capability=graph.realized_capabilities[descriptor.view_index])

    @_selected_precision
    def evaluate_branch(self, inputs, start_index, *, sparse_runner=None):
        """ONE declared branch; dyadic continuation remains caller-owned."""
        descriptor = next(item for item in self.observables
                          if item.kind in _ROOT_KINDS and item.branch_indices[0] == start_index)
        capability = self._capabilities(inputs)[descriptor.view_index]
        return self._root_program(descriptor).evaluate(self._cast(inputs), capability=capability, sparse_runner=sparse_runner)

    def install_prepared_signature(self, key, executable):
        """Bind an explicitly warmed binary to its same transformable program."""
        with self._function_lock:
            if key[1] == 'root_assembly':
                index, _, loaded, compensated, capability = key
                for descriptor in self.observables:
                    if descriptor.view_index == index and descriptor.kind in _ROOT_KINDS:
                        self._root_program(descriptor).install_prepared((loaded, compensated, capability), executable)
            else:
                self._prepared_executables[key] = executable

    @_selected_precision
    def preparation_signatures(self, inputs):
        """Exact pure assembly signatures; no equation or assembly execution.

        Field and View arithmetic is fused into each stable assembly callable.
        Sparse/implicit derivative partitions remain lazy. Returned descriptors
        retain shapes and selected dtypes, never the effective point arrays.
        """
        import jax
        capabilities = self._capabilities(inputs)
        selected = self._cast(inputs)
        arguments = (jax.ShapeDtypeStruct(selected.theta.shape, selected.theta.dtype),
                     jax.ShapeDtypeStruct((), selected.omega_rad_s.dtype))
        result, seen = [], set()
        for descriptor in self.observables:
            index, capability = descriptor.view_index, capabilities[descriptor.view_index]
            root = descriptor.kind in _ROOT_KINDS
            # Include both actual source-network and loaded selected variants
            # where the declared observable consumes them; runtime omega values
            # do not change these signatures.
            if root:
                controls = ((descriptor.kind != 'transfer_zero' or not capability,
                             descriptor.kind == 'transfer_zero' and descriptor.indices[0] == 2 and not capability),)
            elif descriptor.kind in ('operator', 'residue_coupling') or not capability:
                controls = ((True, False),)
            else:
                controls = ((False, False),)
            for loaded, compensated in controls:
                key = (index, 'assembly', loaded, compensated, capability)
                if key not in seen:
                    seen.add(key)
                    with self._function_lock:
                        function = self._functions.get(key)
                        if function is None:
                            assembly = self.assemblies[index]
                            function = lambda theta, omega, assembly=assembly, loaded=loaded, compensated=compensated, capability=capability: assembly(theta, omega, loaded=loaded, compensated=compensated, port_realizable=capability)
                            self._functions[key] = function
                        compiled_key = ('compiled', function)
                        registry = self._arithmetic_functions.setdefault(capabilities, {})
                        compiled = registry.get(compiled_key)
                        if compiled is None:
                            compiled = jax.jit(function)
                            registry[compiled_key] = compiled
                    result.append((key, compiled, arguments))
                if root:
                    program = self._root_program(descriptor)
                    key = (index, 'root_assembly', loaded, compensated, capability)
                    if key not in seen:
                        seen.add(key)
                        result.append((key, program.assembly_function(loaded=loaded, compensated=compensated, capability=capability), arguments))
        return tuple(result)

    def _root_frame(self, inputs, omega, descriptor, capabilities, *, sparse_runner=None):
        import jax.numpy as jnp
        from .root_implicit import ImplicitRootFrame
        from .determinant_ad import determinant
        point = ProgramInputs(inputs.theta, jnp.asarray([omega], dtype=inputs.omega_rad_s.dtype),
                              inputs.branch_starts_rad_s)
        registry = self._arithmetic_functions.setdefault(('root_equation', descriptor, capabilities), {})
        frame = CheckedGraph(point, arithmetic_functions=registry, sparse_runner=sparse_runner)
        frame.realized_capabilities = capabilities
        index, statuses = descriptor.view_index, []
        if descriptor.kind == 'transfer_zero':
            family = ('S', 'Y', 'Z')[descriptor.indices[0]]
            output, input_ = descriptor.indices[1:]
            if capabilities[index]:
                values = self._network(frame, index, 0, {'Y'}, statuses)
                Y = values['Y']
            else:
                F, _ = self._selected(frame, index, 0, family == 'Z', statuses)
                Y = frame.pure(lambda f, current: f / (-1j * current.omega_rad_s[0]), F, 0)
            if family == 'Z':
                q = len(self.views[index].terminal_ids)
                rows = tuple(i for i in range(q) if i != input_)
                columns = tuple(i for i in range(q) if i != output)
                def numerator(y):
                    minor = y[jnp.asarray(rows, dtype=jnp.int32)[:, None],
                              jnp.asarray(columns, dtype=jnp.int32)[None, :]]
                    return (-1 if (input_ + output) % 2 else 1) * determinant(minor)
                node = frame.pure(numerator, Y)
            elif family == 'S':
                state = frame.pure(lambda current: self.views[index](current.theta, port_realizable=True), 0)
                def numerator(y, current):
                    eye = jnp.eye(y.shape[0], dtype=y.dtype)
                    P, N = eye + current.Dk @ y @ current.Dk, eye - current.Dk @ y @ current.Dk
                    border = jnp.zeros((y.shape[0] + 1,) * 2, dtype=y.dtype)
                    border = border.at[:-1, :-1].set(P).at[:-1, -1].set(N[:, input_]).at[-1, output].set(-1)
                    return determinant(border)
                node = frame.pure(numerator, Y, state)
            else:
                node = frame.pure(lambda y: y[output, input_], Y)
        else:
            F, _ = self._selected(frame, index, 0, False, statuses)
            if descriptor.kind == 'hybridized_pole':
                node = frame.pure(determinant, F)
            else:
                row = descriptor.indices[0]
                column = row if descriptor.kind == 'diagonal_root' else descriptor.indices[1]
                node = frame.pure(lambda f: f[row, column], F)
        return ImplicitRootFrame(frame, node, 0, inputs)

    def _capabilities(self, inputs):
        # Host physical decisions precede the selected numerical cast. Only a
        # finite declaration-derived route mask keys the structural partitions.
        fields = self.model_program.physical_fields(inputs.theta)
        return tuple(view.capability(fields) for view in self.views)

    def _graph(self, inputs, capabilities, *, sparse_runner=None, prepared=None):
        import jax.numpy as jnp
        registry = self._arithmetic_functions.setdefault(capabilities, {})
        graph, statuses, output = CheckedGraph(inputs, arithmetic_functions=registry, sparse_runner=sparse_runner), [], []
        graph.realized_capabilities = capabilities
        graph.quantity_results = []
        graph.prepared_assemblies = {} if prepared is None else dict(prepared.assemblies)
        graph.prepared_roots = {} if prepared is None else dict(prepared.root_bootstraps)
        requests = {}
        for descriptor in self.observables:
            if descriptor.kind in _ROOT_KINDS or descriptor.kind == 'residue_coupling':
                continue
            if descriptor.kind == 'response_element':
                family = ('S', 'Y', 'Z')[descriptor.indices[0]]
            else:
                family = descriptor.kind
            for frequency in range(*descriptor.frequency_slice):
                requests.setdefault((descriptor.view_index, frequency), set()).add(family)
        frames = {}
        for (view_index, frequency), families in requests.items():
            view = self.views[view_index]
            response_families = families - {'operator'}
            current = {}
            if response_families and capabilities[view_index]:
                current.update(self._network(graph, view_index, frequency, response_families, statuses))
            elif response_families:
                from .determinants import QuantityError
                if any(descriptor.view_index == view_index and descriptor.kind in ('S', 'Y', 'Z')
                       for descriptor in self.observables) or 'S' in response_families:
                    raise QuantityError('port_realizability', 'selected_network', 'Direct matrix response requires a Port-realizable View')
                # Compensation remains the existing retained TransferZero.Z
                # scope; an ordinary response does not activate that scope.
                F, _ = self._selected(graph, view_index, frequency, False, statuses)
                Y = graph.pure(lambda f, point, at=frequency: f / (-jnp.asarray(1j, dtype=point.omega_rad_s.dtype) * point.omega_rad_s[at]), F, 0)
                current['Y'] = Y
                if 'Z' in response_families:
                    identity = graph.pure(self._functions[view_index, 'identity'], Y)
                    current['Z'] = self._small_solve(graph, view_index, Y, identity, 7, statuses)
            if 'operator' in families:
                F, _ = self._selected(graph, view_index, frequency, False, statuses)
                current['operator'] = F
            frames[view_index, frequency] = current
        derivative_available, reasons, block_available, block_reasons = jnp.asarray(True), [], [], []
        raw_values = []
        for observable_index, descriptor in enumerate(self.observables):
            available, local_reasons = jnp.asarray(True), []
            if descriptor.kind == 'residue_coupling':
                raw, available, local_reasons = self._coupling_node(graph, descriptor, observable_index)
                value = graph.values[raw]
                if descriptor.projection in ('magnitude', 'absolute') and bool(value == 0):
                    available = jnp.asarray(False)
                    local_reasons = tuple(local_reasons) + ('coupling magnitude zero derivative unavailable',)
                node = graph.pure(lambda value, projection=descriptor.projection: _projection(value, projection), raw)
                output.append(node)
                raw_values.append(graph.values[raw])
                block_available.append(available)
                block_reasons.append(tuple(local_reasons))
                continue
            if descriptor.kind in _ROOT_KINDS:
                root = self._root_node(graph, descriptor)
                if descriptor.projection == 'frequency':
                    node = graph.pure(lambda omega: omega.real / (2 * jnp.pi), root)
                elif descriptor.projection == 'linewidth':
                    node = graph.pure(lambda omega: -omega.imag / jnp.pi, root)
                elif descriptor.projection is None:
                    node = root
                else:
                    raise ValueError('unknown root projection')
                output.append(node)
                raw_values.append(graph.values[root])
                block_available.append(available)
                block_reasons.append(())
                continue
            family = ('S', 'Y', 'Z')[descriptor.indices[0]] if descriptor.kind == 'response_element' else descriptor.kind
            nodes = tuple(frames[descriptor.view_index, frequency][family] for frequency in range(*descriptor.frequency_slice))
            def collect(*values, descriptor=descriptor):
                array = jnp.stack(values)
                if descriptor.kind == 'response_element':
                    array = array[:, descriptor.indices[1], descriptor.indices[2]]
                    if array.shape[0] == 1:
                        array = array[0]
                return array
            if descriptor.projection in ('magnitude', 'absolute', 'phase'):
                raw = tuple(graph.values[node] for node in nodes)
                for value in raw:
                    if descriptor.kind == 'response_element':
                        value = value[descriptor.indices[1], descriptor.indices[2]]
                    unavailable = (value == 0)
                    if descriptor.projection == 'phase':
                        unavailable = unavailable | ((value.imag == 0) & (value.real < 0))
                    available = available & ~jnp.any(unavailable)
                local_reasons.append(f'{descriptor.quantity_key}: {descriptor.projection} zero/branch-cut derivative unavailable')
            derivative_available = derivative_available & available
            block_available.append(available)
            block_reasons.append(tuple(local_reasons))
            reasons.extend(local_reasons)
            raw = graph.pure(collect, *nodes)
            raw_values.append(graph.values[raw])
            output.append(graph.pure(lambda value, projection=descriptor.projection: _projection(value, projection), raw))
        observable_values = tuple(graph.values[index] for index in output)
        if self.output_expressions:
            public_output, public_available, public_reasons = [], [], []
            for expression in self.output_expressions:
                nodes = tuple(output[index] for index in expression.indices)
                values = tuple(graph.values[node] for node in nodes)
                available = ~expression.nondifferentiable(*values)
                local_reasons = []
                for index in expression.indices:
                    available = available & block_available[index]
                    local_reasons.extend(block_reasons[index])
                if expression.has_absolute:
                    local_reasons.append(f'{expression.quantity_key}: exact absolute-value zero derivative unavailable')
                public_output.append(graph.pure(expression.apply, *nodes))
                public_available.append(available)
                public_reasons.append(tuple(local_reasons))
            output, block_available, block_reasons = public_output, public_available, public_reasons
            derivative_available = jnp.all(jnp.asarray(block_available))
            reasons = [reason for current in block_reasons for reason in current]
        aux = ProgramAux(self._status(graph, statuses), jnp.int32(-1), tuple(graph.certificates),
                         derivative_available, tuple(reasons), capabilities, tuple(block_available), tuple(block_reasons),
                         tuple((node.root_program.declaration.start_index, node.primal)
                               for node in graph.nodes[1:] if hasattr(node, 'root_program')),
                         tuple(graph.quantity_results), observable_values, tuple(raw_values))
        return graph, tuple(output), aux

    @_selected_precision
    def value(self, inputs, *, sparse_runner=None, prepared=None):
        graph, output, aux = self._graph(self._cast(inputs), self._prepared_capabilities(inputs, prepared), sparse_runner=sparse_runner, prepared=prepared)
        return ProgramOutcome(tuple(graph.values[index] for index in output), aux)

    @_selected_precision
    def jvp(self, inputs, direction, *, sparse_runner=None):
        graph, output, aux = self._graph(self._cast(inputs), self._capabilities(inputs), sparse_runner=sparse_runner)
        jets, certificates = graph.jvp(self._cast(direction))
        derivative_aux = self._output_derivative_aux(aux, certificates, graph, output)
        return DerivativeOutcome(ProgramOutcome(tuple(graph.values[index] for index in output), aux),
                                 tuple(jets[index] if bool(derivative_aux.observable_available[ordinal]) else None
                                       for ordinal, index in enumerate(output)), derivative_aux)

    @_selected_precision
    def vjp(self, inputs, weights, *, sparse_runner=None):
        import jax
        graph, output, aux = self._graph(self._cast(inputs), self._capabilities(inputs), sparse_runner=sparse_runner)
        # Public weights are real flattened output rows. JAX handles the explicit
        # real/imag last-axis coordinate conversion; sparse nodes convert its
        # bilinear cotangent encoding at their adjoint boundary.
        values = tuple(graph.values[index] for index in output)
        _, pullback = jax.vjp(_flatten, values)
        unavailable = tuple(row.row for row in self.output_layout.rows
                            if not bool(aux.observable_available[row.observable_index]))
        if unavailable and np.any(np.asarray(weights)[list(unavailable)] != 0):
            return DerivativeOutcome(ProgramOutcome(values, aux), None, aux)
        parts = pullback(weights)[0]
        seeds = tuple((node, part) for ordinal, (node, part) in enumerate(zip(output, parts, strict=True))
                      if bool(aux.observable_available[ordinal]))
        derivative, certificates = graph.vjp(seeds)
        checked_aux = self._output_derivative_aux(aux, certificates, graph, output)
        # Zero-weight unavailable leaves did not contribute to this functional.
        checked_aux = checked_aux._replace(derivative_available=checked_aux.status == 0)
        return DerivativeOutcome(ProgramOutcome(values, aux), derivative if bool(checked_aux.derivative_available) else None, checked_aux)

    @_selected_precision
    def jacobian(self, inputs, *, input_indices=None, output_indices=None, sparse_runner=None):
        import jax.numpy as jnp
        point = self._cast(inputs)
        graph, output, aux = self._graph(point, self._capabilities(inputs), sparse_runner=sparse_runner)
        indices = tuple(range(point.theta.shape[0])) if input_indices is None else tuple(input_indices)
        selected_rows = tuple(range(len(self.output_layout.rows))) if output_indices is None else tuple(output_indices)
        available_blocks = tuple(bool(value) for value in aux.observable_available)
        columns, certificates = [], []
        for index in indices:
            direction = ProgramInputs(jnp.zeros_like(point.theta).at[index].set(1),
                                      jnp.zeros_like(point.omega_rad_s), jnp.zeros_like(point.branch_starts_rad_s))
            jets, current = graph.jvp(direction)
            current_available = graph.available_outputs(output)
            available_blocks = tuple(left and right for left, right in zip(available_blocks, current_available, strict=True))
            columns.append(_flatten(tuple(jets[node] for node in output)))
            certificates.extend(current)
        values = tuple(graph.values[index] for index in output)
        available_rows = tuple(row for row in selected_rows if available_blocks[self.output_layout.rows[row].observable_index])
        derivative = jnp.stack(columns, axis=1)[jnp.asarray(available_rows, dtype=jnp.int32)] if columns else jnp.zeros((len(available_rows), 0), dtype=point.theta.dtype)
        payload = JacobianPayload(derivative, available_rows, (len(selected_rows), len(indices)))
        checked_aux = self._derivative_aux(aux, tuple(certificates))
        checked_aux = checked_aux._replace(derivative_available=(checked_aux.status == 0) &
                                           (len(available_rows) == len(selected_rows)),
                                           observable_available=available_blocks,
                                           observable_reasons=tuple(reasons + (() if available else ('checked derivative equation unavailable',))
                                                                    for available, reasons in zip(available_blocks, aux.observable_reasons, strict=True)))
        return DerivativeOutcome(ProgramOutcome(values, aux), payload, checked_aux)

    @_selected_precision
    def frequency_parameter_jet(self, inputs, direction, *, sparse_runner=None):
        import jax.numpy as jnp
        point = self._cast(inputs)
        graph, output, aux = self._graph(point, self._capabilities(inputs), sparse_runner=sparse_runner)
        frequency = ProgramInputs(jnp.zeros_like(point.theta), jnp.ones_like(point.omega_rad_s),
                                  jnp.zeros_like(point.branch_starts_rad_s))
        _, _, mixed, certificates = graph.mixed(frequency, self._cast(direction))
        checked_aux = self._output_derivative_aux(aux, certificates, graph, output)
        return DerivativeOutcome(ProgramOutcome(tuple(graph.values[index] for index in output), aux),
                                 tuple(mixed[index] if bool(checked_aux.observable_available[ordinal]) else None
                                       for ordinal, index in enumerate(output)), checked_aux)

    @classmethod
    def _output_derivative_aux(cls, primal_aux, certificates, graph, output):
        aux = cls._derivative_aux(primal_aux, certificates)
        available = tuple(bool(a) and b for a, b in zip(primal_aux.observable_available, graph.available_outputs(output), strict=True))
        reasons = tuple(current + (() if good else ('checked derivative equation unavailable',))
                        for good, current in zip(available, primal_aux.observable_reasons, strict=True))
        return aux._replace(observable_available=available, observable_reasons=reasons,
                            derivative_available=(aux.status == 0) and all(available))

    @staticmethod
    def _derivative_aux(primal_aux, certificates):
        status = primal_aux.status
        for certificate in certificates:
            status = _first(status, certificate.status)
        return primal_aux._replace(status=status, certificates=certificates,
                                   derivative_available=primal_aux.derivative_available & (status == 0))
