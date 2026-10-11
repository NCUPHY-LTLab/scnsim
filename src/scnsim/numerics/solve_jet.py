"""Checked value/frequency-jet equations for residue slopes.

A node represents A X=B AND A Xw=Bw-Aw X, not two independent
physical solves. Each callback owns one local factor and certifies every
requested primal/directional/adjoint equation with the existing denominator.
The eight fixed slots below are the value, frequency and two parameter
mixed directions of this equation; they are not a general tensor AD engine.
"""
from __future__ import annotations
import numpy as np
from .sparse_solve import SolveOutcome, SolveAux, _host_solve


def _host_jet(pattern, coefficients, right_sides, prior, *, ordering, stage, derivative_stage):
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
    dtype = np.asarray(right_sides[0]).dtype
    rd = np.empty((), dtype=dtype).real.dtype.type
    def failure(code):
        return SolveOutcome(np.zeros_like(right_sides[0]), SolveAux(np.int32(code), rd(np.inf), rd(np.inf), rd(np.inf)))
    if prior:
        return tuple(failure(prior) for _ in coefficients)
    if not (np.all(np.isfinite(coefficients[0])) and np.all(np.isfinite(right_sides[0]))):
        return tuple(failure(stage) for _ in coefficients)
    matrices = tuple(csc_matrix((a, pattern.rows, pattern.indptr), shape=pattern.shape) for a in coefficients)
    factor = None
    if pattern.shape[0]:
        try:
            factor = splu(matrices[0]) if ordering is None else splu(matrices[0], permc_spec=ordering)
        except RuntimeError as error:
            if str(error) != 'Factor is exactly singular':
                raise
            return tuple(failure(stage) for _ in coefficients)
    results = []
    for mask, source in enumerate(right_sides):
        rhs = np.asarray(source, dtype=dtype).copy()
        code = np.int32(0)
        # Ordered distinct-direction Leibniz terms; each submask occurs once.
        for subset in range(1, mask + 1):
            if subset & mask == subset:
                dependency = results[mask ^ subset]
                if code == 0:
                    code = dependency.aux.status
                rhs -= matrices[subset] @ dependency.value
        result = _host_solve(pattern, coefficients[0], rhs, code,
                             stage=stage if mask == 0 else derivative_stage,
                             mode='N', ordering=ordering, factor=factor)
        results.append(result)
    return tuple(results)


def solve_jet(equation, coefficients, right_sides, *, prior=0, stage=3, derivative_stage=4):
    """2/4/8 fixed equation slots in frequency-first binary mask order."""
    import jax
    import jax.numpy as jnp
    count = len(coefficients)
    if count not in (2, 4, 8) or len(right_sides) != count:
        raise ValueError('a frequency jet requires two, four or eight equation slots')
    dtype = jnp.result_type(*coefficients, *right_sides)
    coefficients = tuple(jnp.asarray(a, dtype=dtype) for a in coefficients)
    right_sides = tuple(jnp.asarray(b, dtype=dtype) for b in right_sides)
    b = right_sides[0]
    scalar = lambda dtype: jax.ShapeDtypeStruct((), dtype)
    shape = SolveOutcome(jax.ShapeDtypeStruct(b.shape, b.dtype),
                         SolveAux(scalar(jnp.int32), scalar(b.real.dtype), scalar(b.real.dtype), scalar(b.real.dtype)))
    return equation.call_callback(('frequency_jet', count, stage, derivative_stage), _host_jet,
                                  (shape,) * count, coefficients, right_sides,
                                  jnp.asarray(prior, dtype=jnp.int32),
                                  stage=stage, derivative_stage=derivative_stage)


def _host_adjoint(pattern, a, aw, first_rhs, frequency_rhs, prior, *, ordering, stage):
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
    dtype = np.asarray(first_rhs).dtype
    rd = np.empty((), dtype=dtype).real.dtype.type
    def failure(code):
        return SolveOutcome(np.zeros_like(first_rhs), SolveAux(np.int32(code), rd(np.inf), rd(np.inf), rd(np.inf)))
    if prior:
        return failure(prior), failure(prior)
    A = csc_matrix((a, pattern.rows, pattern.indptr), shape=pattern.shape)
    Aw = csc_matrix((aw, pattern.rows, pattern.indptr), shape=pattern.shape)
    factor = None
    if pattern.shape[0]:
        try:
            factor = splu(A) if ordering is None else splu(A, permc_spec=ordering)
        except RuntimeError as error:
            if str(error) != 'Factor is exactly singular':
                raise
            return failure(stage), failure(stage)
    frequency = _host_solve(pattern, a, frequency_rhs, 0, stage=stage, mode='H', ordering=ordering, factor=factor)
    rhs = np.asarray(first_rhs, dtype=dtype) - Aw.conjugate().T @ frequency.value
    first = _host_solve(pattern, a, rhs, frequency.aux.status, stage=stage, mode='H', ordering=ordering, factor=factor)
    return first, frequency


def adjoint_jet(equation, a, aw, weights, *, prior=0, stage=4):
    """Triangular Hermitian adjoint of the TWO primal frequency equations."""
    import jax
    import jax.numpy as jnp
    dtype = jnp.result_type(a, aw, *weights)
    a, aw, first, frequency = (jnp.asarray(v, dtype=dtype) for v in (a, aw, *weights))
    scalar = lambda dtype: jax.ShapeDtypeStruct((), dtype)
    shape = SolveOutcome(jax.ShapeDtypeStruct(first.shape, first.dtype),
                         SolveAux(scalar(jnp.int32), scalar(first.real.dtype), scalar(first.real.dtype), scalar(first.real.dtype)))
    return equation.call_callback(('frequency_adjoint', stage), _host_adjoint,
                                  (shape, shape), a, aw, first, frequency,
                                  jnp.asarray(prior, dtype=jnp.int32), stage=stage)


class FrequencySolveFrame:
    """Call-local derivative frame; inputs are A, Aw, B, Bw."""
    def __init__(self, equation, inputs, *, prior=0, stage=3, derivative_stage=4):
        self.equation, self.inputs = equation, tuple(inputs)
        self.prior, self.stage, self.derivative_stage = prior, stage, derivative_stage
        a, aw, b, bw = self.inputs
        self.results = solve_jet(equation, (a, aw), (b, bw), prior=prior,
                                 stage=stage, derivative_stage=derivative_stage)
        self.value = tuple(result.value for result in self.results)
        self.certificates = tuple(result.aux for result in self.results)

    def jvp(self, directions):
        a, aw, b, bw = self.inputs
        ad, awd, bd, bwd = directions
        result = solve_jet(self.equation, (a, aw, ad, awd), (b, bw, bd, bwd),
                           prior=self.prior, stage=self.stage, derivative_stage=self.derivative_stage)
        return (result[2].value, result[3].value), tuple(item.aux for item in result)

    def mixed(self, first, second, mixed):
        a, aw, b, bw = self.inputs
        a1, aw1, b1, bw1 = first
        a2, aw2, b2, bw2 = second
        a12, aw12, b12, bw12 = mixed
        result = solve_jet(self.equation, (a, aw, a1, aw1, a2, aw2, a12, aw12),
                           (b, bw, b1, bw1, b2, bw2, b12, bw12), prior=self.prior,
                           stage=self.stage, derivative_stage=self.derivative_stage)
        return (result[6].value, result[7].value), tuple(item.aux for item in result)

    def vjp(self, weights):
        import jax.numpy as jnp
        a, aw, b, bw = self.inputs
        prior = self.prior
        for certificate in self.certificates:
            prior = jnp.where(prior != 0, prior, certificate.status)
        first, frequency = adjoint_jet(self.equation, a, aw, tuple(jnp.conj(w) for w in weights),
                                        prior=prior, stage=self.derivative_stage)
        rows, cols = jnp.asarray(self.equation.pattern.rows), jnp.asarray(self.equation.pattern.cols)
        x, xw = self.value
        def product(adjoint, value):
            result = -jnp.conj(adjoint[rows]) * value[cols]
            return jnp.sum(result, axis=tuple(range(1, result.ndim))) if result.ndim > 1 else result
        def project(value, operand):
            if not jnp.issubdtype(operand.dtype, jnp.complexfloating):
                value = jnp.real(value)
            return jnp.asarray(value, dtype=operand.dtype)
        parts = (project(product(first.value, x) + product(frequency.value, xw), a),
                 project(product(frequency.value, x), aw),
                 project(jnp.conj(first.value), b), project(jnp.conj(frequency.value), bw))
        return parts, (first.aux, frequency.aux)
