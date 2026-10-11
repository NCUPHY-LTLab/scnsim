"""Checked local chart derivatives of the accepted simple-null-vector gauge.

Primal vectors are the existing SVD certificate's vectors, with its exact
largest-coordinate positive-real phase. A nonsingular retained minor supplies
implicit chart derivatives. No SVD-vector derivative or new rank threshold is
introduced; physical normalization uses the original Hermitian norm only.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..compilation.program import SparsePattern
from .sparse_solve import SparseEquation


@dataclass(slots=True)
class NullVectorFrame:
    matrix: object
    vector: object
    left_vector: object
    _rows: tuple = ()
    _columns: tuple = ()
    _pivot: int = 0
    _equation: object = None

    def __post_init__(self):
        magnitude = np.abs(np.asarray(self.vector))
        self._pivot = int(np.argmax(magnitude))
        omitted_row = int(np.argmax(np.abs(np.asarray(self.left_vector))))
        self._rows = tuple(i for i in range(self.matrix.shape[0]) if i != omitted_row)
        self._columns = tuple(i for i in range(self.matrix.shape[1]) if i != self._pivot)
        q = len(self._rows)
        pattern = SparsePattern.from_coo((q, q), np.tile(np.arange(q), q), np.repeat(np.arange(q), q))[0]
        self._equation = SparseEquation(pattern)

    @property
    def available(self):
        magnitude = np.abs(np.asarray(self.vector))
        return np.count_nonzero(magnitude == magnitude.max()) == 1

    def _chart(self):
        return self.vector / self.vector[self._pivot]

    def _minor(self):
        import jax.numpy as jnp
        return self.matrix[jnp.asarray(self._rows, dtype=jnp.int32)[:, None],
                           jnp.asarray(self._columns, dtype=jnp.int32)[None, :]]

    @staticmethod
    def _normalize(vector):
        import jax.numpy as jnp
        return vector / jnp.linalg.norm(vector)

    def _chart_jvp(self, direction):
        import jax.numpy as jnp
        chart = self._chart()
        rhs = -(direction @ chart)[jnp.asarray(self._rows, dtype=jnp.int32)]
        solution = self._equation.solve(self._minor().T.ravel(), rhs, stage=13)
        result = jnp.zeros_like(chart).at[jnp.asarray(self._columns, dtype=jnp.int32)].set(solution.value)
        return result, (solution.aux,)

    def jvp(self, direction):
        import jax
        chart_dot, certificates = self._chart_jvp(direction)
        _, value = jax.jvp(self._normalize, (self._chart(),), (chart_dot,))
        return value, certificates

    def vjp(self, weight):
        import jax
        import jax.numpy as jnp
        _, pullback = jax.vjp(self._normalize, self._chart())
        chart_bar = pullback(weight)[0]
        rhs = jnp.conj(chart_bar[jnp.asarray(self._columns, dtype=jnp.int32)])
        solution = self._equation.solve(self._minor().T.ravel(), rhs, stage=13, mode='H')
        products = -jnp.conj(solution.value)[:, None] * self._chart()[None, :]
        result = jnp.zeros_like(self.matrix).at[jnp.asarray(self._rows, dtype=jnp.int32)].set(products)
        if not jnp.issubdtype(self.matrix.dtype, jnp.complexfloating):
            result = jnp.real(result)
        return jnp.asarray(result, dtype=self.matrix.dtype), (solution.aux,)

    def mixed(self, first, second, matrix_mixed):
        import jax
        import jax.numpy as jnp
        a, ca = self._chart_jvp(first)
        b, cb = self._chart_jvp(second)
        rows = jnp.asarray(self._rows, dtype=jnp.int32)
        rhs = -(matrix_mixed @ self._chart() + first @ b + second @ a)[rows]
        prior = jnp.int32(0)
        for certificate in ca + cb:
            prior = jnp.where(prior != 0, prior, certificate.status)
        solution = self._equation.solve(self._minor().T.ravel(), rhs, stage=13, prior_status=prior)
        chart_mixed = jnp.zeros_like(a).at[jnp.asarray(self._columns, dtype=jnp.int32)].set(solution.value)
        chart = self._chart()
        def first_at(value):
            return jax.jvp(self._normalize, (value,), (a,))[1]
        _, hessian = jax.jvp(first_at, (chart,), (b,))
        _, propagated = jax.jvp(self._normalize, (chart,), (chart_mixed,))
        return hessian + propagated, ca + cb + (solution.aux,)


class NullVectorEquation:
    """Single matrix operand adapter to the call-local checked graph."""
    def __init__(self, matrix, vector, left_vector):
        self.frame = NullVectorFrame(matrix, vector, left_vector)

    def jvp(self, directions):
        return self.frame.jvp(directions[0])

    def vjp(self, weight):
        value, certificates = self.frame.vjp(weight)
        return (value,), certificates

    def mixed(self, first, second, mixed):
        return self.frame.mixed(first[0], second[0], mixed[0])
