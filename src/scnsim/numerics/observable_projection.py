"""Declared observable projections and real/imag row packing.

These pure JAX operations are the single native public-output arithmetic
boundary. Physical equations, execution ownership and persistence live elsewhere.
"""
from __future__ import annotations

def _flatten(values):
    import jax.numpy as jnp
    parts = []
    for value in values:
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            value = jnp.stack((value.real, value.imag), axis=-1)
        parts.append(value.ravel())
    return jnp.concatenate(parts) if parts else jnp.zeros((0,))


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
