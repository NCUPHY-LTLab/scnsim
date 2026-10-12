"""Controlled native equations for one two-component physical operand.

Finite first jets share one callback-local high factor across primal and
first tangent; separate retained-solution equations own local factors. Each
equation uses the existing single fixed correction. Tangents consume the actual primal solution;
no cross-callback factor, bit-operation AD, Run, or Workspace is retained. EFT
parts are representation channels, not independently observable model inputs.
The existing native_sparse bridge owns all equation certificate collection.
"""
from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.interpreters import ad, batching, mlir, partial_eval as pe

from .native_sparse import NativeSparseAux, register_equation_certificate, equation_lowering
from .sparse_solve import (PairSolveOutcome, PairDirectionalOutcome, SolveAux,
                           _host_pair_equation, _host_pair_product, _host_pair_dispatch)


class NativePairOutcome(NamedTuple):
    value: tuple[Any, Any]
    aux: NativeSparseAux


def _host_equation(pattern, ah, al, bh, bl, prior, *products,
                   ordering, stage, mode):
    """B minus ordered C Y products, followed by the SAME pair equation."""
    from scipy.sparse import csc_matrix
    from . import compensated as cp
    rhs = bh, bl
    if np.asarray(prior).item() == 0:
        for offset in range(0, len(products), 4):
            ch, cl, yh, yl = products[offset:offset + 4]
            matrices = tuple(csc_matrix((part, pattern.rows, pattern.indptr),
                                       shape=pattern.shape) for part in (ch, cl))
            if mode != 'N':
                matrices = tuple(matrix.T if mode == 'T' else matrix.conjugate().T
                                 for matrix in matrices)
            product = cp.sparse_matmul(*(matrix.tocsr() for matrix in matrices),
                                      (yh, yl), execute=_host_pair_product)
            rhs = cp.add(rhs, cp.negate(product), xp=np)
    return _host_pair_equation(pattern, (ah, al), rhs, prior,
                               ordering=ordering, stage=stage, mode=mode)


_pair_p = core.Primitive('scnsim_checked_pair_equation')
_pair_p.multiple_results = True
_batch_pair_p = core.Primitive('scnsim_checked_pair_equation_batch')
_batch_pair_p.multiple_results = True
_residual_pair_p = core.Primitive('scnsim_checked_pair_residual_equation')
_residual_pair_p.multiple_results = True
_batch_residual_pair_p = core.Primitive('scnsim_checked_pair_residual_equation_batch')
_batch_residual_pair_p.multiple_results = True


def _callback(ah, al, bh, bl, prior, *products, equation, stage, mode, role,
              derivative_order=0, certificate_site=(), dependency=(), parameter_ad_depth=0, frequency_jet=False):
    shape = PairSolveOutcome(
        (jax.ShapeDtypeStruct(bh.shape, bh.dtype), jax.ShapeDtypeStruct(bl.shape, bl.dtype)),
        SolveAux(jax.ShapeDtypeStruct((), jnp.int32),
                 *(jax.ShapeDtypeStruct((), bh.real.dtype) for _ in range(3))))
    result = equation.call_callback(('native_pair', stage, mode, role, derivative_order, len(products)),
        _host_equation, shape, ah, al, bh, bl, prior, *products, stage=stage, mode=mode)
    return (*result.value, *result.aux, prior == 0)


_static = ('equation', 'stage', 'mode', 'role', 'derivative_order', 'certificate_site', 'dependency', 'parameter_ad_depth', 'frequency_jet')
_pair_p.def_impl(partial(jax.jit, static_argnames=_static)(_callback))
mlir.register_lowering(_pair_p, equation_lowering(_callback))


def _batch_callback(*operands, **params):
    return jax.lax.map(lambda items: _callback(*items, **params), operands)


_batch_pair_p.def_impl(partial(jax.jit, static_argnames=_static)(_batch_callback))
mlir.register_lowering(_batch_pair_p, equation_lowering(_batch_callback))


def _abstract(ah, al, bh, bl, prior, *products, equation, **params):
    del params
    shape = (len(equation.pattern.rows),)
    if ah.shape != shape or al.shape != shape:
        raise TypeError('pair coefficient shape disagrees with declared CSC pattern')
    if bh.ndim not in (1, 2) or bh.shape[0] != equation.pattern.shape[0] or bl.shape != bh.shape:
        raise TypeError('pair RHS requires the declared node axis and optional RHS axis')
    if any(value.dtype != bh.dtype for value in (ah, al, bl, *products)):
        raise TypeError('pair equation operands require one selected dtype')
    if prior.shape != () or prior.dtype != np.dtype('int32'):
        raise TypeError('pair equation status requires scalar int32')
    if len(products) % 4:
        raise TypeError('pair derivative products require matrix/vector pairs')
    for offset in range(0, len(products), 4):
        ch, cl, yh, yl = products[offset:offset + 4]
        if ch.shape != shape or cl.shape != shape or yh.shape != bh.shape or yl.shape != bh.shape:
            raise TypeError('pair derivative product shape disagrees with its equation')
    rd = np.empty((), dtype=bh.dtype).real.dtype
    return (bh, bl, core.ShapedArray((), np.dtype('int32')),
            *(core.ShapedArray((), rd) for _ in range(3)), core.ShapedArray((), np.dtype('bool')))


_pair_p.def_abstract_eval(_abstract)


def _batch_abstract(*operands, **params):
    size = operands[0].shape[0]
    avals = _abstract(*(core.ShapedArray(value.shape[1:], value.dtype) for value in operands), **params)
    return tuple(core.ShapedArray((size,) + value.shape, value.dtype) for value in avals)


_batch_pair_p.def_abstract_eval(_batch_abstract)


def _residual_callback(ah, al, bh, bl, prior, retained_prior, *products, **params):
    # The forward equation keeps its exact directional skip prerequisite.
    # The separate primal status is a reverse residual, not a substitute here.
    del retained_prior
    return _callback(ah, al, bh, bl, prior, *products, **params)


def _residual_abstract(ah, al, bh, bl, prior, retained_prior, *products, **params):
    if retained_prior.shape != prior.shape or retained_prior.dtype != prior.dtype:
        raise TypeError('pair residual requires the retained primal status shape/dtype')
    return _abstract(ah, al, bh, bl, prior, *products, **params)


def _batch_residual_callback(*operands, **params):
    return jax.lax.map(lambda items: _residual_callback(*items, **params), operands)


def _batch_residual_abstract(*operands, **params):
    size = operands[0].shape[0]
    result = _residual_abstract(*(core.ShapedArray(value.shape[1:], value.dtype)
                                  for value in operands), **params)
    return tuple(core.ShapedArray((size,) + value.shape, value.dtype) for value in result)


for primitive, callback, abstract in (
        (_residual_pair_p, _residual_callback, _residual_abstract),
        (_batch_residual_pair_p, _batch_residual_callback, _batch_residual_abstract)):
    primitive.def_impl(partial(jax.jit, static_argnames=_static)(callback))
    primitive.def_abstract_eval(abstract)
    mlir.register_lowering(primitive, equation_lowering(callback))


# A finite first jet keeps both actual solution certificates in one primitive.
# Its next JVP can consume BOTH first-direction statuses before the mixed solve.
_first_p = core.Primitive('scnsim_checked_pair_first_jet')
_first_p.multiple_results = True
_batch_first_p = core.Primitive('scnsim_checked_pair_first_jet_batch')
_batch_first_p.multiple_results = True
_linear_p = core.Primitive('scnsim_checked_pair_linear_tangent')
_linear_p.multiple_results = True
_batch_linear_p = core.Primitive('scnsim_checked_pair_linear_tangent_batch')
_batch_linear_p.multiple_results = True


def _next(params):
    order = params.get('derivative_order', 0) + 1
    return dict(params, stage=4, role='tangent' if order == 1 else 'mixed',
                derivative_order=order, frequency_jet=False)


def _first_callback(ah, al, bh, bl, prior, a1h, a1l, b1h, b1l, *, equation,
                    stage, mode, role, derivative_order=0, certificate_site=(), dependency=(), parameter_ad_depth=0, frequency_jet=False):
    # The existing host authority keeps ONE high factor through primal and
    # first tangent, then releases it before callback return. No JAX function
    # or opaque factor handle crosses that boundary.
    shape = PairSolveOutcome(
        (jax.ShapeDtypeStruct(bh.shape, bh.dtype), jax.ShapeDtypeStruct(bl.shape, bl.dtype)),
        SolveAux(jax.ShapeDtypeStruct((), jnp.int32),
                 *(jax.ShapeDtypeStruct((), bh.real.dtype) for _ in range(3))))
    result = equation.call_callback(
        ('native_pair_first', stage, mode, role, derivative_order),
        _host_pair_dispatch, PairDirectionalOutcome(shape, shape),
        ah, al, bh, bl, prior, a1h, a1l, b1h, b1l,
        stage=stage, mode=mode, derivative_stage=4, jet_order=1)
    primal, tangent = result
    return (*primal.value, *tangent.value, *primal.aux, prior == 0,
            *tangent.aux, (prior == 0) & (primal.aux.status == 0))


def _first_abstract(ah, al, bh, bl, prior, a1h, a1l, b1h, b1l, **params):
    primal = _abstract(ah, al, bh, bl, prior, **params)
    _abstract(a1h, a1l, b1h, b1l, prior, **params)
    return *primal[:2], *primal[:2], *primal[2:], *primal[2:]


def _linear_callback(ah, al, xh, xl, prior, a1h, a1l, b1h, b1l, **params):
    return _callback(ah, al, b1h, b1l, prior, a1h, a1l, xh, xl, **params)


def _linear_abstract(ah, al, xh, xl, prior, a1h, a1l, b1h, b1l, **params):
    return _abstract(ah, al, b1h, b1l, prior, a1h, a1l, xh, xl, **params)


def _jet_batch_callback(*operands, callback, **params):
    return jax.lax.map(lambda items: callback(*items, **params), operands)


def _jet_batch_abstract(*operands, abstract, **params):
    size = operands[0].shape[0]
    result = abstract(*(core.ShapedArray(value.shape[1:], value.dtype)
                        for value in operands), **params)
    return tuple(core.ShapedArray((size,) + value.shape, value.dtype) for value in result)


for primitive, callback, abstract in ((_first_p, _first_callback, _first_abstract),
                                      (_linear_p, _linear_callback, _linear_abstract)):
    primitive.def_impl(partial(jax.jit, static_argnames=_static)(callback))
    primitive.def_abstract_eval(abstract)
    mlir.register_lowering(primitive, equation_lowering(callback))
for primitive, callback, abstract in ((_batch_first_p, _first_callback, _first_abstract),
                                      (_batch_linear_p, _linear_callback, _linear_abstract)):
    function = partial(_jet_batch_callback, callback=callback)
    primitive.def_impl(partial(jax.jit, static_argnames=_static)(function))
    primitive.def_abstract_eval(partial(_jet_batch_abstract, abstract=abstract))
    mlir.register_lowering(primitive, equation_lowering(function))


def _jvp(primals, tangents, *, primitive=_pair_p, **params):
    if params.get('parameter_ad_depth', 0) >= 2:
        raise TypeError('finite pair jet supports two parameter directions, not third parameter derivatives')
    if len(primals) != 5:
        # A physical-frequency first jet leaves its first parameter direction
        # as AX = B - sum(CY). Its admitted second parameter direction must
        # differentiate that represented equation, not the EFT bit mechanics.
        if params.get('parameter_ad_depth', 0) != 1 or len(primals) not in (9, 17):
            raise TypeError('pair residual products require the admitted second parameter direction')
        ah, al, bh, bl, prior, *products = primals
        a2h, a2l, b2h, b2l, _, *product_dots = map(ad.instantiate_zeros, tangents)
        primal = primitive.bind(*primals, **params)
        derivative_products = [a2h, a2l, *primal[:2]]
        for offset in range(0, len(products), 4):
            ch, cl, yh, yl = products[offset:offset + 4]
            c2h, c2l, y2h, y2l = product_dots[offset:offset + 4]
            derivative_products.extend((c2h, c2l, yh, yl, ch, cl, y2h, y2l))
        derivative = primitive.bind(ah, al, b2h, b2l, primal[2],
            *derivative_products, **dict(_next(params), parameter_ad_depth=2))
        tangent = (*derivative[:2],
                   *(ad.Zero(core.typeof(value).to_tangent_aval()) for value in primal[2:]))
        return primal, tangent
    ah, al, bh, bl, prior = primals
    a1h, a1l, b1h, b1l, _ = map(ad.instantiate_zeros, tangents)
    jet = (_batch_first_p if primitive is _batch_pair_p else _first_p).bind(
        ah, al, bh, bl, prior, a1h, a1l, b1h, b1l,
        **dict(params, parameter_ad_depth=params.get('parameter_ad_depth', 0) + 1))
    primal = *jet[:2], *jet[4:9]
    tangent = *jet[2:4], *(ad.Zero(core.typeof(value).to_tangent_aval()) for value in primal[2:])
    return primal, tangent


def _jet_jvp(primals, tangents, *, primitive=_first_p, **params):
    # Frequency-order metadata belongs to the physical equation; it is not
    # the number of parameter directions introduced by this finite AD rule.
    if params.get('parameter_ad_depth', 0) != 1 and not (
            params.get('frequency_jet', False) and params.get('parameter_ad_depth', 0) == 0):
        raise TypeError('finite pair jet supports two parameter directions, not third parameter derivatives')
    ah, al, bh, bl, prior, a1h, a1l, b1h, b1l = primals
    a2h, a2l, b2h, b2l, _, a12h, a12l, b12h, b12l = map(ad.instantiate_zeros, tangents)
    primal = primitive.bind(*primals, **params)
    depth = params.get('parameter_ad_depth', 0) + 1
    solve = _batch_residual_pair_p if primitive is _batch_first_p else _residual_pair_p
    second = solve.bind(ah, al, b2h, b2l, primal[4],
                        primal[4], a2h, a2l, *primal[:2], **dict(_next(params), parameter_ad_depth=depth))
    # Original priority: primal, first tangent, second tangent. The callback
    # skips ALL residual products when a prerequisite equation is unavailable.
    mixed_prior = jnp.where(primal[4] != 0, primal[4],
                            jnp.where(primal[9] != 0, primal[9], second[2]))
    mixed = solve.bind(ah, al, b12h, b12l, mixed_prior, primal[4],
        a12h, a12l, *primal[:2], a1h, a1l, *second[:2], a2h, a2l, *primal[2:4],
        **dict(params, stage=4, role='mixed',
               derivative_order=params.get('derivative_order', 0) + 2, parameter_ad_depth=depth, frequency_jet=False))
    derivatives = *second[:2], *mixed[:2], *(ad.Zero(core.typeof(value).to_tangent_aval())
                                            for value in primal[4:])
    return primal, derivatives


def _partial_eval(trace, *tracers, primitive=_first_p, **params):
    all_known = tuple(tracer.pval.get_known() for tracer in tracers)
    if all(value is not None for value in all_known):
        return tuple(trace.to_jaxpr_tracer(value) for value in primitive.bind(*all_known, **params))
    known = all_known[:5]
    if any(value is None for value in known):
        return trace.default_process_primitive(primitive, tracers, params)
    solve = _batch_pair_p if primitive is _batch_first_p else _pair_p
    primal = solve.bind(*known, **params)
    linear = _batch_linear_p if primitive is _batch_first_p else _linear_p
    operands = (*(trace.to_jaxpr_tracer(value) for value in (*known[:2], *primal[:2], primal[2])),
                *tracers[5:])
    tangent = trace.default_process_primitive(linear, operands, _next(params))
    return (*(trace.to_jaxpr_tracer(value) for value in primal[:2]), *tangent[:2],
            *(trace.to_jaxpr_tracer(value) for value in primal[2:]), *tangent[2:])


ad.primitive_jvps[_pair_p] = _jvp
ad.primitive_jvps[_batch_pair_p] = partial(_jvp, primitive=_batch_pair_p)
for primitive in (_first_p, _batch_first_p):
    ad.primitive_jvps[primitive] = partial(_jet_jvp, primitive=primitive)
    pe.custom_partial_eval_rules[primitive] = partial(_partial_eval, primitive=primitive)
    register_equation_certificate(primitive, value_count=4,
        certificate_groups=((None, None, None), ('derivative', 4, 'next')))


def _physical(pair):
    return pair[0] + pair[1]


def _cast_cotangent(value, operand):
    dtype = operand.aval.dtype
    if not np.issubdtype(dtype, np.complexfloating):
        value = jnp.real(value)
    return jnp.asarray(value, dtype=dtype)


def _product_bars(equation, pair, vector, weight, mode, *, matrix_unknown):
    rows, columns = equation.pattern.rows, equation.pattern.cols
    if mode != 'N':
        rows, columns = columns, rows
    rows, columns = jnp.asarray(rows), jnp.asarray(columns)
    if matrix_unknown:
        products = -weight[rows] * _physical(vector)[columns]
        value = jnp.sum(products, axis=tuple(range(1, products.ndim))) if products.ndim > 1 else products
        return jnp.conj(value) if mode == 'H' else value
    coefficient = _physical(pair)
    coefficient = jnp.conj(coefficient) if mode == 'H' else coefficient
    weights = coefficient.reshape((len(rows),) + (1,) * (weight.ndim - 1))
    return jnp.zeros_like(weight).at[columns].add(-weights * weight[rows])


def _transpose(cotangents, ah, al, bh, bl, prior, *products,
               primitive=_pair_p, **params):
    operands = (ah, al, bh, bl, prior, *products)
    if any(ad.is_undefined_primal(value) for value in (ah, al, prior)):
        raise TypeError('pair transpose requires the primal matrix and checked status')
    # Existing pair graph convention: high is the logical output covector;
    # low is an EFT residue, never a second independently weighted observable.
    cotangent = cotangents[0]
    if isinstance(cotangent, ad.Zero):
        return tuple(ad.Zero(value.aval) if ad.is_undefined_primal(value) else None for value in operands)
    mode = params['mode']
    if mode == 'H':
        matrix, transpose_mode = (jnp.conj(ah), jnp.conj(al)), 'N'
    else:
        matrix, transpose_mode = (ah, al), 'T' if mode == 'N' else 'N'
    adjoint = primitive.bind(*matrix, cotangent, jnp.zeros_like(cotangent), prior,
        **dict(params, stage=4, mode=transpose_mode, role='adjoint',
               derivative_order=max(1, params.get('derivative_order', 0))))
    weight = _physical(adjoint[:2])
    result = [None, None,
              _cast_cotangent(weight, bh) if ad.is_undefined_primal(bh) else None,
              _cast_cotangent(weight, bl) if ad.is_undefined_primal(bl) else None, None]
    for offset in range(0, len(products), 4):
        ch, cl, yh, yl = products[offset:offset + 4]
        matrix_unknown = ad.is_undefined_primal(ch) or ad.is_undefined_primal(cl)
        vector_unknown = ad.is_undefined_primal(yh) or ad.is_undefined_primal(yl)
        if matrix_unknown and vector_unknown:
            raise TypeError('pair transpose requires one retained primal product operand')
        if matrix_unknown or vector_unknown:
            if matrix_unknown:
                def product(vector, weight):
                    return _product_bars(params['equation'], None, vector, weight, mode,
                                         matrix_unknown=True)
                known = yh, yl
            else:
                def product(matrix, weight):
                    return _product_bars(params['equation'], matrix, None, weight, mode,
                                         matrix_unknown=False)
                known = ch, cl
            if primitive is _batch_pair_p:
                product = jax.vmap(product)
            bar = product(known, weight)
            result.extend(_cast_cotangent(bar, value) if ad.is_undefined_primal(value) else None
                          for value in (ch, cl, yh, yl))
        else:
            result.extend((None,) * 4)
    return tuple(result)


ad.primitive_transposes[_pair_p] = _transpose
ad.primitive_transposes[_batch_pair_p] = partial(_transpose, primitive=_batch_pair_p)


def _residual_jvp(primals, tangents, *, primitive=_residual_pair_p, **params):
    if params.get('parameter_ad_depth', 0) != 1 or len(primals) not in (10, 18):
        raise TypeError('pair residual products require the admitted second parameter direction')
    ah, al, bh, bl, prior, retained_prior, *products = primals
    a2h, a2l, b2h, b2l, _, _, *product_dots = map(ad.instantiate_zeros, tangents)
    primal = primitive.bind(*primals, **params)
    derivative_products = [a2h, a2l, *primal[:2]]
    for offset in range(0, len(products), 4):
        ch, cl, yh, yl = products[offset:offset + 4]
        c2h, c2l, y2h, y2l = product_dots[offset:offset + 4]
        derivative_products.extend((c2h, c2l, yh, yl, ch, cl, y2h, y2l))
    derivative = primitive.bind(ah, al, b2h, b2l, primal[2], retained_prior,
        *derivative_products, **dict(_next(params), parameter_ad_depth=2))
    return primal, (*derivative[:2],
                   *(ad.Zero(core.typeof(value).to_tangent_aval()) for value in primal[2:]))


def _residual_transpose(cotangents, ah, al, bh, bl, prior, retained_prior, *products,
                        primitive=_residual_pair_p, **params):
    # A reverse adjoint consumes the actual retained primal prerequisite;
    # direction-dependent forward certificate outputs are not linear operands.
    solve = _batch_pair_p if primitive is _batch_residual_pair_p else _pair_p
    bars = _transpose(cotangents, ah, al, bh, bl, retained_prior, *products,
                      primitive=solve, **params)
    return *bars[:5], None, *bars[5:]


def _residual_batch(operands, axes, **params):
    size = next(value.shape[axis] for value, axis in zip(operands, axes) if axis is not None)
    values = tuple(batching.bdim_at_front(value, axis, size) for value, axis in zip(operands, axes))
    result = _batch_residual_pair_p.bind(*values, **params)
    return result, (0,) * len(result)


def _residual_nested_batch(operands, axes, **params):
    size = next(value.shape[axis] for value, axis in zip(operands, axes) if axis is not None)
    values = tuple(batching.bdim_at_front(value, axis, size) for value, axis in zip(operands, axes))
    inner = values[0].shape[1]
    flat = tuple(value.reshape((size * inner,) + value.shape[2:]) for value in values)
    result = _batch_residual_pair_p.bind(*flat, **params)
    return tuple(value.reshape((size, inner) + value.shape[1:]) for value in result), (0,) * len(result)


ad.primitive_jvps[_residual_pair_p] = _residual_jvp
ad.primitive_jvps[_batch_residual_pair_p] = partial(_residual_jvp, primitive=_batch_residual_pair_p)
ad.primitive_transposes[_residual_pair_p] = _residual_transpose
ad.primitive_transposes[_batch_residual_pair_p] = partial(_residual_transpose, primitive=_batch_residual_pair_p)
batching.primitive_batchers[_residual_pair_p] = _residual_batch
batching.primitive_batchers[_batch_residual_pair_p] = _residual_nested_batch
register_equation_certificate(_residual_pair_p, value_count=2)
register_equation_certificate(_batch_residual_pair_p, value_count=2)


def _linear_transpose(cotangents, ah, al, xh, xl, prior, a1h, a1l, b1h, b1l,
                      *, primitive=_linear_p, **params):
    if any(ad.is_undefined_primal(value) for value in (ah, al, xh, xl, prior)):
        raise TypeError('linear pair transpose requires retained matrix/solution/status')
    solve = _batch_pair_p if primitive is _batch_linear_p else _pair_p
    bars = _transpose(cotangents, ah, al, b1h, b1l, prior, a1h, a1l, xh, xl,
                      primitive=solve, **params)
    return None, None, None, None, None, bars[5], bars[6], bars[2], bars[3]


for primitive in (_linear_p, _batch_linear_p):
    ad.primitive_transposes[primitive] = partial(_linear_transpose, primitive=primitive)
    register_equation_certificate(primitive, value_count=2)


def _batch(operands, axes, **params):
    size = next(value.shape[axis] for value, axis in zip(operands, axes) if axis is not None)
    front = tuple(batching.bdim_at_front(value, axis, size) for value, axis in zip(operands, axes))
    result = _batch_pair_p.bind(*front, **params)
    return result, (0,) * len(result)


batching.primitive_batchers[_pair_p] = _batch


def _nested_batch(operands, axes, **params):
    size = next(value.shape[axis] for value, axis in zip(operands, axes) if axis is not None)
    front = tuple(batching.bdim_at_front(value, axis, size) for value, axis in zip(operands, axes))
    inner = front[0].shape[1]
    flat = tuple(value.reshape((size * inner,) + value.shape[2:]) for value in front)
    result = _batch_pair_p.bind(*flat, **params)
    return tuple(value.reshape((size, inner) + value.shape[1:]) for value in result), (0,) * len(result)


batching.primitive_batchers[_batch_pair_p] = _nested_batch
register_equation_certificate(_pair_p, value_count=2)
register_equation_certificate(_batch_pair_p, value_count=2)


def _jet_batch(operands, axes, *, primitive, **params):
    size = next(value.shape[axis] for value, axis in zip(operands, axes) if axis is not None)
    front = tuple(batching.bdim_at_front(value, axis, size) for value, axis in zip(operands, axes))
    target = _batch_first_p if primitive is _first_p else _batch_linear_p
    result = target.bind(*front, **params)
    return result, (0,) * len(result)


def _jet_nested_batch(operands, axes, *, primitive, **params):
    size = next(value.shape[axis] for value, axis in zip(operands, axes) if axis is not None)
    front = tuple(batching.bdim_at_front(value, axis, size) for value, axis in zip(operands, axes))
    inner = front[0].shape[1]
    flat = tuple(value.reshape((size * inner,) + value.shape[2:]) for value in front)
    result = primitive.bind(*flat, **params)
    return tuple(value.reshape((size, inner) + value.shape[1:]) for value in result), (0,) * len(result)


for primitive in (_first_p, _linear_p):
    batching.primitive_batchers[primitive] = partial(_jet_batch, primitive=primitive)
for primitive in (_batch_first_p, _batch_linear_p):
    batching.primitive_batchers[primitive] = partial(_jet_nested_batch, primitive=primitive)


def checked_pair_solve(equation, values_pair, rhs_pair, *, stage, prior_status=0,
                       mode='N', role='primal', derivative_order=0, dependency=()) -> NativePairOutcome:
    """Solve the represented pair equation; certificate remains one authority."""
    if mode not in ('N', 'T', 'H'):
        raise ValueError('pair solve mode must be N, T or H')
    dtype = jnp.result_type(*values_pair, *rhs_pair)
    values = tuple(jnp.asarray(part, dtype=dtype) for part in values_pair)
    rhs = tuple(jnp.asarray(part, dtype=dtype) for part in rhs_pair)
    result = _pair_p.bind(*values, *rhs, jnp.asarray(prior_status, dtype=jnp.int32),
                         equation=equation, stage=stage, mode=mode, role=role,
                         derivative_order=derivative_order, dependency=tuple(dependency))
    return NativePairOutcome(result[:2], NativeSparseAux(*result[2:]))


def checked_pair_frequency_jet(equation, values_pair, rhs_pair, frequency_values_pair,
                               frequency_rhs_pair, *, prior_status=0, dependency=()):
    """One local high-component factor, two represented pair equations.

    Primal and frequency tangent retain separate stage3/stage4 certificates.
    Each equation keeps its existing correction and postverification. The
    frequency channel does not consume the two supported parameter AD depths.
    """
    dtype = jnp.result_type(*values_pair, *rhs_pair, *frequency_values_pair, *frequency_rhs_pair)
    operands = tuple(jnp.asarray(value, dtype=dtype) for value in
        (*values_pair, *rhs_pair, *frequency_values_pair, *frequency_rhs_pair))
    result = _first_p.bind(*operands[:4], jnp.asarray(prior_status, dtype=jnp.int32),
        *operands[4:], equation=equation, stage=3, mode='N', role='primal',
        derivative_order=0, dependency=tuple(dependency), parameter_ad_depth=0, frequency_jet=True)
    return (NativePairOutcome(result[:2], NativeSparseAux(*result[4:9])),
            NativePairOutcome(result[2:4], NativeSparseAux(*result[9:14])))
