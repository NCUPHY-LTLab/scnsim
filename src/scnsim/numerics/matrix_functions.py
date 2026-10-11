"""Small physical matrix functions with equation-defined directional rules.

Principal SPD roots retain the existing reconstruction certificate. Their
Sylvester derivative remains meaningful at repeated positive eigenvalues;
individual eigenvector derivatives are never used as physical authority.
"""
from __future__ import annotations


def _principal_root(matrix):
    import jax.numpy as jnp
    values, vectors = jnp.linalg.eigh(matrix)
    return (vectors * jnp.sqrt(values)) @ vectors.T


def spd_root_checked(matrix):
    import jax.numpy as jnp
    root = spd_root(matrix)
    chol = jnp.linalg.cholesky(matrix)
    num = jnp.max(jnp.abs(root @ root - matrix), initial=0)
    den = jnp.max(jnp.abs(root) @ jnp.abs(root) + jnp.abs(matrix), initial=0)
    eta = jnp.where(jnp.isfinite(num) & jnp.isfinite(den),
                    jnp.where(den == 0, jnp.where(num == 0, 0, jnp.inf), num / den), jnp.inf)
    threshold = jnp.asarray(256 * (matrix.shape[0] + 1), dtype=matrix.dtype) * jnp.finfo(matrix.dtype).eps
    status = jnp.where(jnp.all(jnp.isfinite(chol)) & jnp.isfinite(eta) & (eta <= threshold), 0, 1).astype(jnp.int32)
    return root, status


# Lazy module-local definition keeps metadata compilation runtime-free. Only a
# stable numerical function is retained, never a numerical point or AD tape.
_ROOT = None


def spd_root(matrix):
    global _ROOT
    if _ROOT is None:
        import jax
        import jax.numpy as jnp
        root = jax.custom_jvp(_principal_root)
        @root.defjvp
        def directional(primals, tangents):
            (matrix,), (direction,) = primals, tangents
            value = root(matrix)
            # Solve the Sylvester equation directly on this small Port matrix.
            # Nested frequency/parameter jets therefore never differentiate
            # an arbitrary eigenbasis at repeated positive eigenvalues.
            eye = jnp.eye(value.shape[0], dtype=value.dtype)
            operator = jnp.kron(value, eye) + jnp.kron(eye, value.T)
            derivative = jnp.linalg.solve(operator, direction.ravel()).reshape(value.shape)
            return value, derivative
        _ROOT = root
    return _ROOT(matrix)
