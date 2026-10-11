"""Polynomial determinant derivatives for checked retained root equations.

The primal determinant retains selected-dtype arithmetic. Its JAX derivative
uses signed minors, so a singular numerator does not invoke an inverse or a
singular-value-vector derivative. Nested derivatives recurse through smaller
minors; the physical transpose and distinct asymmetric entries are preserved.
"""
from __future__ import annotations

_DETERMINANT = None


def determinant(matrix):
    global _DETERMINANT
    if _DETERMINANT is None:
        import jax
        import jax.numpy as jnp
        @jax.custom_jvp
        def polynomial(value):
            n = value.shape[0]
            if n == 0:
                return jnp.asarray(1, dtype=value.dtype)
            if n == 1:
                return value[0, 0]
            if n == 2:
                return value[0, 0] * value[1, 1] - value[0, 1] * value[1, 0]
            return jnp.linalg.det(value)
        @polynomial.defjvp
        def directional(primals, tangents):
            (value,), (direction,) = primals, tangents
            n = value.shape[0]
            derivative = jnp.asarray(0, dtype=value.dtype)
            for row in range(n):
                rows = tuple(index for index in range(n) if index != row)
                for column in range(n):
                    columns = tuple(index for index in range(n) if index != column)
                    minor = value[jnp.asarray(rows, dtype=jnp.int32)[:, None],
                                  jnp.asarray(columns, dtype=jnp.int32)[None, :]]
                    sign = -1 if (row + column) % 2 else 1
                    derivative = derivative + sign * polynomial(minor) * direction[row, column]
            return polynomial(value), derivative
        _DETERMINANT = polynomial
    return _DETERMINANT(matrix)
