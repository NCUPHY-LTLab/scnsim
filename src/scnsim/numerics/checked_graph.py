"""Call-local checked equation graph, shared by value and explicit derivatives.

Pure nodes describe physical array equations; sparse nodes own checked solves.
A graph is an ephemeral linearization of ONE point, never a global tape or
native-handle registry. Integer certificates are auxiliaries, not observables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PureNode:
    function: Any
    inputs: tuple[int, ...]
    value: Any


@dataclass(frozen=True, slots=True)
class SolveNode:
    equation: Any
    inputs: tuple[int, int]
    value: Any
    aux: Any
    stage: int
    derivative_stage: int
    paired: bool = False



@dataclass(slots=True)
class RootNode:
    root_program: Any
    frame_factory: Any
    inputs: tuple[int, ...]
    value: Any
    primal: Any
    frame: Any = None

    def derivative_frame(self, inputs):
        if self.frame is None:
            self.frame = self.frame_factory(inputs, self.value)
        return self.frame


@dataclass(slots=True)
class EquationFrameNode:
    frame: Any
    inputs: tuple[int, ...]
    value: Any
    factory: Any = None

    def derivative_frame(self):
        if self.frame is None:
            self.frame = self.factory()
        return self.frame


def zero_tangent(value):
    import jax
    import jax.numpy as jnp
    def zero(array):
        dtype = array.dtype if jnp.issubdtype(array.dtype, jnp.inexact) else jax.dtypes.float0
        return jnp.zeros(array.shape, dtype=dtype)
    return jax.tree.map(zero, value)


def _sum(a, b):
    import jax
    return jax.tree.map(lambda x, y: x if x.dtype == jax.dtypes.float0 else x + y, a, b)


class CheckedGraph:
    def __init__(self, inputs, *, arithmetic_functions=None, sparse_runner=None):
        self.inputs = inputs
        self.sparse_runner = sparse_runner
        self.arithmetic_functions = {} if arithmetic_functions is None else arithmetic_functions
        self.nodes = [None]
        self.values = [inputs]
        self.certificates = []
        self.derivative_certificate_map = {}
        self.root_nodes = {}
        self.prepared_assemblies = {}
        self.prepared_roots = {}

    def pure(self, function, *inputs, primal_executor=None):
        # The compiled declaration owns one function per structural node. A
        # runtime frame supplies only operands; no point-dependent closure is
        # retained. setdefault installs one identical structural callable when
        # an eagerly used program first reaches this node.
        import jax
        # Structural callable identity, not call-frame ordinal, selects the
        # executable. Root dedup and output selection can change node ordinals.
        # Stable declaration frames admit their dynamic Python wrappers once.
        # Code+ordinal separates operators if output selection changes a frame;
        # the admitted function contains declaration constants only.
        structural = ('pure', len(self.nodes), getattr(function, '__code__', None))
        function = self.arithmetic_functions.setdefault(structural, function)
        key = ('compiled', function)
        compiled = self.arithmetic_functions.get(key)
        if compiled is None:
            compiled = self.arithmetic_functions.setdefault(key, jax.jit(function))
        function = compiled
        executor = function if primal_executor is None else primal_executor
        value = executor(*(self.values[index] for index in inputs))
        self.nodes.append(PureNode(function, tuple(inputs), value))
        self.values.append(value)
        return len(self.nodes) - 1

    def solve(self, equation, values, rhs, *, stage, derivative_stage=4, prior_status=0):
        result = equation.solve(self.values[values], self.values[rhs], stage=stage, prior_status=prior_status)
        self.nodes.append(SolveNode(equation, (values, rhs), result.value, result.aux, stage, derivative_stage))
        self.values.append(result.value)
        self.certificates.append(result.aux)
        return len(self.nodes) - 1

    def solve_pair(self, equation, values, rhs, *, stage, derivative_stage=4, prior_status=0):
        result = equation.solve_pair(self.values[values], self.values[rhs], stage=stage, prior_status=prior_status, runner=self.sparse_runner)
        self.nodes.append(SolveNode(equation, (values, rhs), result.value, result.aux, stage, derivative_stage, True))
        self.values.append(result.value)
        self.certificates.append(result.aux)
        return len(self.nodes) - 1

    def implicit_equation(self, factory, value, *inputs):
        """Lazy call-local frame of an already certified physical observable."""
        self.nodes.append(EquationFrameNode(None, tuple(inputs), value, factory))
        self.values.append(value)
        return len(self.nodes) - 1

    def frequency_solve(self, equation, a, aw, b, bw, *, prior_status=0, stage=3, derivative_stage=4):
        from .solve_jet import FrequencySolveFrame
        indices = (a, aw, b, bw)
        frame = FrequencySolveFrame(equation, tuple(self.values[i] for i in indices),
                                    prior=prior_status, stage=stage, derivative_stage=derivative_stage)
        self.nodes.append(EquationFrameNode(frame, indices, frame.value))
        self.values.append(frame.value)
        self.certificates.extend(frame.certificates)
        return len(self.nodes) - 1

    def root(self, root_program, frame_factory, *, capability):
        import jax.numpy as jnp
        if root_program in self.root_nodes:
            return self.root_nodes[root_program]
        primal = root_program.value(self.inputs, capability=capability, sparse_runner=self.sparse_runner,
                                   prepared_bootstrap=self.prepared_roots.get(root_program.declaration.start_index))
        value = jnp.asarray(primal.root_omega_rad_s, dtype=self.inputs.omega_rad_s.dtype)
        self.nodes.append(RootNode(root_program, frame_factory, (0,), value, primal))
        self.values.append(value)
        index = len(self.nodes) - 1
        self.root_nodes[root_program] = index
        return index

    def jvp(self, direction):
        jets, certificates, by_node = self._jvp(direction)
        self.derivative_certificate_map = by_node
        return jets, certificates

    def _jvp(self, direction):
        import jax
        jets, certificates, by_node = [direction], [], {}
        for ordinal, node in enumerate(self.nodes[1:], start=1):
            arguments = tuple(self.values[index] for index in node.inputs)
            directions = tuple(jets[index] for index in node.inputs)
            if isinstance(node, PureNode):
                _, derivative = jax.jvp(node.function, arguments, directions)
            elif isinstance(node, EquationFrameNode):
                derivative, checked = node.derivative_frame().jvp(directions)
                certificates.extend(checked)
                by_node[ordinal] = checked
            elif isinstance(node, RootNode):
                derivative, checked = node.derivative_frame(self.inputs).jvp(directions[0])
                certificates.extend(checked)
                by_node[ordinal] = checked
            else:
                A, B = arguments
                Ad, Bd = directions
                if node.paired:
                    checked = node.equation.jvp_pair(A, B, Ad, Bd, primal_stage=node.stage,
                                                     tangent_stage=node.derivative_stage, prior_status=node.aux.status, runner=self.sparse_runner).tangent
                else:
                    checked = node.equation.jvp(A, B, Ad, Bd, primal_stage=node.stage,
                                                tangent_stage=node.derivative_stage, prior_status=node.aux.status).tangent
                derivative = checked.value
                certificates.append(checked.aux)
                by_node[ordinal] = (checked.aux,)
            jets.append(derivative)
        return jets, tuple(certificates), by_node

    def available_outputs(self, output_nodes):
        """A failed equation affects only outputs which consume that node."""
        import numpy as np
        available = [True]
        for ordinal, node in enumerate(self.nodes[1:], start=1):
            current = all(available[index] for index in node.inputs)
            current = current and all(int(np.asarray(aux.status)) == 0
                                      for aux in self.derivative_certificate_map.get(ordinal, ()))
            available.append(current)
        return tuple(available[index] for index in output_nodes)

    def vjp(self, output_weights):
        """Reverse traversal; weights use JAX's bilinear cotangent encoding.

        Conversion to/from the real-loss conjugate convention is explicit ONLY
        at sparse adjoints. Physical transpose in all primal equations is intact.
        """
        import jax
        import jax.numpy as jnp
        weights = [zero_tangent(value) for value in self.values]
        for index, value in output_weights:
            weights[index] = _sum(weights[index], value)
        certificates, by_node = [], {}
        for index in range(len(self.nodes) - 1, 0, -1):
            # An exactly zero requested covector has no derivative dependency;
            # avoid evaluating an unrelated undefined local pullback as 0/0.
            import numpy as np
            if all(leaf.dtype == jax.dtypes.float0 or not np.any(np.asarray(leaf) != 0)
                   for leaf in jax.tree.leaves(weights[index])):
                continue
            node = self.nodes[index]
            arguments = tuple(self.values[key] for key in node.inputs)
            if isinstance(node, PureNode):
                _, pullback = jax.vjp(node.function, *arguments)
                parts = pullback(weights[index])
            elif isinstance(node, EquationFrameNode):
                parts, checked = node.derivative_frame().vjp(weights[index])
                certificates.extend(checked)
                by_node[index] = checked
            elif isinstance(node, RootNode):
                derivative, checked = node.derivative_frame(self.inputs).vjp(weights[index])
                parts = (derivative,)
                certificates.extend(checked)
                by_node[index] = checked
            else:
                A, B = arguments
                if node.paired:
                    from . import compensated as cp
                    # Pair parts represent ONE physical solution. The logical
                    # cotangent is taken in its high coordinate; low is an EFT
                    # residue, not a second independent physical variable.
                    covector = jnp.conj(weights[index][0])
                    checked = node.equation.solve_pair(A, cp.from_value(covector, xp=jnp),
                                                       stage=node.derivative_stage, prior_status=node.aux.status, mode='H', runner=self.sparse_runner)
                    adjoint = cp.project(checked.value, xp=jnp)
                    value = cp.project(node.value, xp=jnp)
                else:
                    checked = node.equation.solve(A, jnp.conj(weights[index]), stage=node.derivative_stage,
                                                  prior_status=node.aux.status, mode='H')
                    adjoint, value = checked.value, node.value
                certificates.append(checked.aux)
                by_node[index] = (checked.aux,)
                rows = jnp.asarray(node.equation.pattern.rows)
                cols = jnp.asarray(node.equation.pattern.cols)
                bar = -adjoint[rows] * jnp.conj(value[cols])
                if bar.ndim > 1:
                    bar = jnp.sum(bar, axis=tuple(range(1, bar.ndim)))
                dtype = A[0].dtype if node.paired else A.dtype
                Abar = jnp.conj(bar) if jnp.issubdtype(dtype, jnp.complexfloating) else jnp.real(bar)
                rhs_dtype = B[0].dtype if node.paired else B.dtype
                Bbar = jnp.conj(adjoint) if jnp.issubdtype(rhs_dtype, jnp.complexfloating) else jnp.real(adjoint)
                # Each cotangent has its ORIGINAL physical operand dtype;
                # promoting A X=B for the solve does not complexify real B.
                Abar = jnp.asarray(Abar, dtype=dtype)
                Bbar = jnp.asarray(Bbar, dtype=rhs_dtype)
                parts = ((Abar, Abar), (Bbar, Bbar)) if node.paired else (Abar, Bbar)
            for key, part in zip(node.inputs, parts, strict=True):
                weights[key] = _sum(weights[key], part)
        self.derivative_certificate_map = by_node
        return weights[0], tuple(certificates)

    def mixed(self, first_direction, second_direction):
        import jax
        import jax.numpy as jnp
        first, first_certificates, first_by_node = self._jvp(first_direction)
        second, second_certificates, second_by_node = self._jvp(second_direction)
        mixed, certificates, mixed_by_node = [zero_tangent(self.inputs)], [], {}
        for node in self.nodes[1:]:
            args = tuple(self.values[key] for key in node.inputs)
            d1 = tuple(first[key] for key in node.inputs)
            d2 = tuple(second[key] for key in node.inputs)
            d12 = tuple(mixed[key] for key in node.inputs)
            if isinstance(node, PureNode):
                def first_at(*arguments):
                    return jax.jvp(node.function, arguments, d1)[1]
                _, hessian = jax.jvp(first_at, args, d2)
                _, propagated = jax.jvp(node.function, args, d12)
                value = _sum(hessian, propagated)
            elif isinstance(node, EquationFrameNode):
                value, checked = node.derivative_frame().mixed(d1, d2, d12)
                certificates.extend(checked)
                mixed_by_node[len(mixed)] = checked
            elif isinstance(node, RootNode):
                frame = node.derivative_frame(self.inputs)
                value, checked = frame.mixed(d1[0], d2[0])
                propagated, extra = frame.jvp(d12[0])
                value = value + propagated
                certificates.extend(checked + extra)
                mixed_by_node[len(mixed)] = checked + extra
            else:
                A, B = args
                A1, _ = d1
                A2, _ = d2
                A12, B12 = d12
                X1, X2 = first[len(mixed)], second[len(mixed)]
                prior = node.aux.status
                # The derivative equation keeps its own status. A failed first
                # or second solve is never published as a valid mixed buffer.
                for aux in first_by_node[len(mixed)] + second_by_node[len(mixed)]:
                    prior = jnp.where(prior != 0, prior, aux.status)
                if node.paired:
                    result = node.equation.mixed_pair(A, B, A1, d1[1], A2, d2[1], A12, B12,
                                                      primal_stage=node.stage, derivative_stage=node.derivative_stage,
                                                      prior_status=prior, runner=self.sparse_runner).mixed
                else:
                    result = node.equation.mixed(A, B, A1, d1[1], A2, d2[1], A12, B12,
                                                 primal_stage=node.stage, derivative_stage=node.derivative_stage,
                                                 prior_status=prior).mixed
                value = result.value
                certificates.append(result.aux)
                mixed_by_node[len(mixed)] = (result.aux,)
            mixed.append(value)
        self.derivative_certificate_map = {index: first_by_node.get(index, ()) + second_by_node.get(index, ()) + mixed_by_node.get(index, ())
                                           for index in set(first_by_node) | set(second_by_node) | set(mixed_by_node)}
        return first, second, mixed, first_certificates + second_certificates + tuple(certificates)
