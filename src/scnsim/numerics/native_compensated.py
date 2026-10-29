"""Finite native arithmetic for one compensated physical array operand.

The two channels represent one physical value, not two public variables. Exact
selected-dtype EFT formation owns the primal and directional representation;
JAX differentiates its physical sum/product equation, never splitting bits.
Only addition, multiplication and final projection are admitted here. Sparse
solve certificates and factors remain in native_pair, outside this arithmetic.
"""
from __future__ import annotations
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.interpreters import ad, batching, mlir
from . import compensated as cp

_pair_p = core.Primitive('scnsim_physical_pair_arithmetic')
_pair_p.multiple_results = True
_batch_p = core.Primitive('scnsim_physical_pair_arithmetic_batch')
_batch_p.multiple_results = True


def _impl(*parts, operation):
    kw = dict(xp=jnp, barrier=jax.lax.optimization_barrier)
    if operation == 'add':
        return cp.add(parts[:2], parts[2:], **kw)
    if operation == 'multiply':
        return cp.multiply(parts[:2], parts[2:], **kw)
    if operation == 'project':
        return (cp.project(parts, **kw),)
    raise ValueError('unknown finite compensated arithmetic operation')


def _abstract(*parts, operation):
    count = 2 if operation == 'project' else 4
    if operation not in ('add', 'multiply', 'project') or len(parts) != count:
        raise TypeError('finite compensated arithmetic operand schema differs')
    dtype = parts[0].dtype
    if any(item.dtype != dtype for item in parts):
        raise TypeError('compensated channels require one selected dtype')
    shape = np.broadcast_shapes(*(item.shape for item in parts))
    output = core.ShapedArray(shape, dtype)
    return (output,) if operation == 'project' else (output, output)


_pair_p.def_impl(partial(jax.jit, static_argnames=('operation',))(_impl))
_pair_p.def_abstract_eval(_abstract)
mlir.register_lowering(_pair_p, mlir.lower_fun(_impl, multiple_results=True))


def _batch_impl(*parts, operation):
    return jax.lax.map(lambda point: _impl(*point, operation=operation), parts)


def _batch_abstract(*parts, operation):
    size = parts[0].shape[0]
    shapes = _abstract(*(core.ShapedArray(item.shape[1:], item.dtype) for item in parts),
                       operation=operation)
    return tuple(core.ShapedArray((size,) + item.shape, item.dtype) for item in shapes)


_batch_p.def_impl(partial(jax.jit, static_argnames=('operation',))(_batch_impl))
_batch_p.def_abstract_eval(_batch_abstract)
mlir.register_lowering(_batch_p, mlir.lower_fun(_batch_impl, multiple_results=True))


def _jvp(primals, tangents, *, operation, primitive=_pair_p):
    value = primitive.bind(*primals, operation=operation)
    tangent = tuple(ad.instantiate_zeros(item) for item in tangents)
    if operation in ('add', 'project'):
        direction = primitive.bind(*tangent, operation=operation)
    else:
        left = primitive.bind(*tangent[:2], *primals[2:], operation='multiply')
        right = primitive.bind(*primals[:2], *tangent[2:], operation='multiply')
        direction = primitive.bind(*left, *right, operation='add')
    return value, direction


ad.primitive_jvps[_pair_p] = _jvp
ad.primitive_jvps[_batch_p] = partial(_jvp, primitive=_batch_p)


def _unbroadcast(value, shape):
    while value.ndim > len(shape):
        value = jnp.sum(value, axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = jnp.sum(value, axis=axis, keepdims=True)
    return value.reshape(shape)


def _bar(value, operand):
    if not np.issubdtype(operand.aval.dtype, np.complexfloating):
        value = jnp.real(value)
    return jnp.asarray(_unbroadcast(value, operand.aval.shape), dtype=operand.aval.dtype)


def _transpose(cotangents, *parts, operation, primitive=_pair_p):
    # Low is a rounding residue of the same logical output. Its separate
    # cotangent is not a second functional weight; projection supplies high.
    weight = cotangents[0]
    if isinstance(weight, ad.Zero):
        return tuple(ad.Zero(item.aval) if ad.is_undefined_primal(item) else None for item in parts)
    if operation == 'project':
        return (_bar(weight, parts[0]) if ad.is_undefined_primal(parts[0]) else None,
                _bar(weight, parts[1]) if ad.is_undefined_primal(parts[1]) else None)
    if operation == 'add':
        return tuple(_bar(weight, item) if ad.is_undefined_primal(item) else None for item in parts)
    left, right = parts[:2], parts[2:]
    unknown_left = any(ad.is_undefined_primal(item) for item in left)
    unknown_right = any(ad.is_undefined_primal(item) for item in right)
    if unknown_left and unknown_right:
        raise TypeError('linearized compensated product requires one retained physical operand')
    if unknown_left:
        value = weight * (right[0] + right[1])
        return tuple(_bar(value, item) if ad.is_undefined_primal(item) else None for item in left) + (None, None)
    if unknown_right:
        value = weight * (left[0] + left[1])
        return (None, None) + tuple(_bar(value, item) if ad.is_undefined_primal(item) else None for item in right)
    return (None,) * 4


ad.primitive_transposes[_pair_p] = _transpose
ad.primitive_transposes[_batch_p] = partial(_transpose, primitive=_batch_p)


def _batch(parts, axes, *, operation):
    size = next(item.shape[axis] for item, axis in zip(parts, axes, strict=True)
                if axis is not None)
    front = tuple(batching.bdim_at_front(item, axis, size) for item, axis in zip(parts, axes, strict=True))
    result = _batch_p.bind(*front, operation=operation)
    return result, (0,) * len(result)


def _nested_batch(parts, axes, *, operation):
    size = next(item.shape[axis] for item, axis in zip(parts, axes, strict=True)
                if axis is not None)
    front = tuple(batching.bdim_at_front(item, axis, size) for item, axis in zip(parts, axes, strict=True))
    old = front[0].shape[1]
    flat = tuple(item.reshape((size * old,) + item.shape[2:]) for item in front)
    result = _batch_p.bind(*flat, operation=operation)
    return tuple(item.reshape((size, old) + item.shape[1:]) for item in result), (0,) * len(result)


batching.primitive_batchers[_pair_p] = _batch
batching.primitive_batchers[_batch_p] = _nested_batch


def add(left, right):
    return tuple(_pair_p.bind(*left, *right, operation='add'))


def multiply(left, right):
    return tuple(_pair_p.bind(*left, *right, operation='multiply'))


def project(pair):
    return _pair_p.bind(*pair, operation='project')[0]


def from_value(value, *, xp):
    return cp.from_value(value, xp=xp)


def negate(pair):
    return cp.negate(pair)
