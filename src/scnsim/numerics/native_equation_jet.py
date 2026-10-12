"""Finite implicit equation jets; JAX owns differentiation and linearization.

A first jet retains its actual primal and tangent certificates together. Its
next JVP can therefore skip the mixed equation when EITHER first tangent failed.
Partial evaluation separates the known primal from the linear tangent equation;
reverse consumes retained arrays, never an opaque factor or Python AD tape.
"""
from __future__ import annotations

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.interpreters import ad, batching, mlir, partial_eval as pe
from . import native_sparse as ns

_first_p = core.Primitive('scnsim_checked_equation_first_jet')
_first_p.multiple_results = True
_batch_first_p = core.Primitive('scnsim_checked_equation_first_jet_batch')
_batch_first_p.multiple_results = True
_linear_p = core.Primitive('scnsim_checked_equation_linear_tangent')
_linear_p.multiple_results = True
_batch_linear_p = core.Primitive('scnsim_checked_equation_linear_tangent_batch')
_batch_linear_p.multiple_results = True


def _next(params):
    order = params.get('derivative_order', 0) + 1
    return dict(params, stage=4, role='tangent' if order == 1 else 'mixed',
                derivative_order=order)


def _first(prior, *statuses):
    for status in statuses:
        prior = jnp.where(prior != 0, prior, status)
    return prior


def _product(values, x, params, batched=False):
    function = partial(ns._matmul, params['equation'], mode=params['mode'])
    return (jax.vmap(function) if batched else function)(values, x)


def _callback(values, rhs, prior, values1, rhs1, **params):
    if params.get('backend', 'sparse') == 'sparse' and params['mode'] == 'N':
        from .sparse_solve import _high_callback
        # The existing host first-jet equation factors once and solves the
        # ready primal/tangent bundle in order. Wide transport is unchanged.
        result = _high_callback(params['equation'], values, rhs, prior,
            stage=params['stage'], a_dot=values1, b_dot=rhs1, derivative_stage=4)
        performed = prior == 0
        tangent_performed = performed & (result.primal.aux.status == 0)
        return (result.primal.value, result.tangent.value,
                *result.primal.aux, performed, *result.tangent.aux, tangent_performed)
    # Dense equations stay on device. T/H preserve their established mode
    # solve implementation rather than changing factor ordering by transpose.
    primal = ns._callback(values, rhs, prior, **params)
    residual = rhs1 - _product(values1, primal[0], params)
    tangent = ns._callback(values, residual, primal[1], **_next(params))
    return primal[0], tangent[0], *primal[1:], *tangent[1:]


def _linear_callback(values, x, prior, values1, rhs1, **params):
    residual = rhs1 - _product(values1, x, params)
    return ns._callback(values, residual, prior, **params)


def _batch_callback(*operands, callback, **params):
    return jax.lax.map(lambda items: callback(*items, **params), operands)


def _abstract(values, rhs, prior, values1, rhs1, **params):
    primal = ns._abstract(values, rhs, prior, **params)
    ns._abstract(values1, rhs1, prior, **params)
    return primal[0], primal[0], *primal[1:], *primal[1:]


def _linear_abstract(values, x, prior, values1, rhs1, **params):
    primal = ns._abstract(values, rhs1, prior, **params)
    if x.shape != rhs1.shape or x.dtype != rhs1.dtype:
        raise TypeError('linear equation tangent requires the retained primal solution shape/dtype')
    ns._abstract(values1, rhs1, prior, **params)
    return primal


def _batch_abstract(*operands, abstract, **params):
    size = operands[0].shape[0]
    result = abstract(*(core.ShapedArray(value.shape[1:], value.dtype)
                        for value in operands), **params)
    return tuple(core.ShapedArray((size,) + value.shape, value.dtype) for value in result)


for primitive, callback, abstract in ((_first_p, _callback, _abstract),
                                      (_linear_p, _linear_callback, _linear_abstract)):
    primitive.def_impl(partial(jax.jit, static_argnames=ns._static)(callback))
    primitive.def_abstract_eval(abstract)
    mlir.register_lowering(primitive, ns.equation_lowering(callback))
for primitive, callback, abstract in ((_batch_first_p, _callback, _abstract),
                                      (_batch_linear_p, _linear_callback, _linear_abstract)):
    function = partial(_batch_callback, callback=callback)
    primitive.def_impl(partial(jax.jit, static_argnames=ns._static)(function))
    primitive.def_abstract_eval(partial(_batch_abstract, abstract=abstract))
    mlir.register_lowering(primitive, ns.equation_lowering(function))


def _solve_jvp(primals, tangents, *, primitive=ns._solve_p, **params):
    if params.get('parameter_ad_depth', 0) >= 2:
        raise TypeError('finite equation jet supports two parameter directions, not third parameter derivatives')
    values, rhs, prior = primals
    values1, rhs1, _ = map(ad.instantiate_zeros, tangents)
    jet = (_batch_first_p if primitive is ns._batch_solve_p else _first_p).bind(
        values, rhs, prior, values1, rhs1,
        **dict(params, parameter_ad_depth=params.get('parameter_ad_depth', 0) + 1))
    primal = jet[0], *jet[2:7]
    tangent = jet[1], *(ad.Zero(core.typeof(value).to_tangent_aval()) for value in primal[1:])
    return primal, tangent


def _jet_jvp(primals, tangents, *, primitive=_first_p, **params):
    # Frequency order is physical equation metadata, independent of the two
    # admitted parameter directions in this finite AD rule.
    if params.get('parameter_ad_depth', 0) != 1:
        raise TypeError('finite equation jet supports two parameter directions, not third parameter derivatives')
    values, rhs, prior, values1, rhs1 = primals
    values2, rhs2, _, values12, rhs12 = map(ad.instantiate_zeros, tangents)
    primal = primitive.bind(*primals, **params)
    batched = primitive is _batch_first_p
    solve = ns._batch_solve_p if batched else ns._solve_p
    tangent_params = dict(_next(params), parameter_ad_depth=2)
    residual2 = rhs2 - _product(values2, primal[0], params, batched)
    second = solve.bind(values, residual2, primal[2], **tangent_params)
    # Both actual first equations, not a global Error predicate, own the mixed
    # prerequisites. No inactive conditional site enters this local status.
    mixed_prior = _first(primal[2], primal[7], second[1])
    residual12 = rhs12 - _product(values12, primal[0], params, batched)
    residual12 = residual12 - _product(values1, second[0], params, batched)
    residual12 = residual12 - _product(values2, primal[1], params, batched)
    mixed = solve.bind(values, residual12, mixed_prior,
        **dict(params, stage=4, role='mixed',
               derivative_order=params.get('derivative_order', 0) + 2, parameter_ad_depth=2))
    derivatives = second[0], mixed[0], *(ad.Zero(core.typeof(value).to_tangent_aval())
                                        for value in primal[2:])
    return primal, derivatives


def _partial_eval(trace, *tracers, primitive=_first_p, **params):
    known = tuple(tracer.pval.get_known() for tracer in tracers[:3])
    if any(value is None for value in known):
        return trace.default_process_primitive(primitive, tracers, params)
    solve = ns._batch_solve_p if primitive is _batch_first_p else ns._solve_p
    primal = solve.bind(*known, **params)
    linear = _batch_linear_p if primitive is _batch_first_p else _linear_p
    operands = (trace.to_jaxpr_tracer(known[0]), trace.to_jaxpr_tracer(primal[0]),
                trace.to_jaxpr_tracer(primal[1]), *tracers[3:])
    tangent = trace.default_process_primitive(linear, operands, _next(params))
    return (trace.to_jaxpr_tracer(primal[0]), tangent[0],
            *(trace.to_jaxpr_tracer(value) for value in primal[1:]), *tangent[1:])


def _project(value, operand):
    dtype = operand.aval.dtype
    if not np.issubdtype(dtype, np.complexfloating):
        value = jnp.real(value)
    return jnp.asarray(value, dtype=dtype)


def _linear_transpose(cotangents, values, x, prior, values1, rhs1,
                      *, primitive=_linear_p, **params):
    if any(ad.is_undefined_primal(value) for value in (values, x, prior)):
        raise TypeError('linear equation transpose requires retained matrix/solution/status')
    cotangent = cotangents[0]
    if isinstance(cotangent, ad.Zero):
        return (None, None, None,
                ad.Zero(values1.aval) if ad.is_undefined_primal(values1) else None,
                ad.Zero(rhs1.aval) if ad.is_undefined_primal(rhs1) else None)
    mode = params['mode']
    transposed_values = jnp.conj(values) if mode == 'H' else values
    transposed_mode = 'N' if mode in ('T', 'H') else 'T'
    solve = ns._batch_solve_p if primitive is _batch_linear_p else ns._solve_p
    adjoint = solve.bind(transposed_values, cotangent, prior,
        **dict(params, stage=4, mode=transposed_mode, role='adjoint'))
    def matrix_bar(weight, solution):
        rows, cols = params['equation'].pattern.rows, params['equation'].pattern.cols
        if mode != 'N':
            rows, cols = cols, rows
        result = -weight[jnp.asarray(rows)] * solution[jnp.asarray(cols)]
        if result.ndim > 1:
            result = jnp.sum(result, axis=tuple(range(1, result.ndim)))
        return jnp.conj(result) if mode == 'H' else result
    abar = None
    if ad.is_undefined_primal(values1):
        bar = (jax.vmap(matrix_bar) if primitive is _batch_linear_p else matrix_bar)(adjoint[0], x)
        abar = _project(bar, values1)
    bbar = _project(adjoint[0], rhs1) if ad.is_undefined_primal(rhs1) else None
    return None, None, None, abar, bbar


def _batch(operands, axes, *, primitive, **params):
    size = next(value.shape[axis] for value, axis in zip(operands, axes) if axis is not None)
    values = tuple(batching.bdim_at_front(value, axis, size) for value, axis in zip(operands, axes))
    target = _batch_first_p if primitive is _first_p else _batch_linear_p
    result = target.bind(*values, **params)
    if primitive is _first_p and all(axis is None for axis in axes[:3]):
        # A tangent-basis vmap does not map the primal equation. jacfwd
        # explicitly requires that solution unmapped. Retain every actually
        # executed primal/tangent certificate lane, despite their shared value.
        empty_primal = None if size else ns._solve_p.bind(*operands[:3], **params)
        primal = result[0][0] if size else empty_primal[0]
        # The bound batched primitive still owns every certificate lane;
        # returned primal status/residual values are basis-independent too.
        primal_aux = (tuple(value[0] for value in result[2:7]) if size else
                      tuple(empty_primal[1:]))
        return (primal, result[1], *primal_aux, *result[7:]), (None, 0, *(None for _ in primal_aux), *(0 for _ in result[7:]))
    return result, (0,) * len(result)


def _nested_batch(operands, axes, *, primitive, **params):
    size = next(value.shape[axis] for value, axis in zip(operands, axes) if axis is not None)
    values = tuple(batching.bdim_at_front(value, axis, size) for value, axis in zip(operands, axes))
    inner = values[0].shape[1]
    flat = tuple(value.reshape((size * inner,) + value.shape[2:]) for value in values)
    result = primitive.bind(*flat, **params)
    result = tuple(value.reshape((size, inner) + value.shape[1:]) for value in result)
    if primitive is _batch_first_p and all(axis is None for axis in axes[:3]):
        empty_primal = None if size else ns._batch_solve_p.bind(*operands[:3], **params)
        primal = result[0][0] if size else empty_primal[0]
        primal_aux = (tuple(value[0] for value in result[2:7]) if size else
                      tuple(empty_primal[1:]))
        return (primal, result[1], *primal_aux, *result[7:]), (None, 0, *(None for _ in primal_aux), *(0 for _ in result[7:]))
    return result, (0,) * len(result)


ad.primitive_jvps[ns._solve_p] = _solve_jvp
ad.primitive_jvps[ns._batch_solve_p] = partial(_solve_jvp, primitive=ns._batch_solve_p)
for primitive in (_first_p, _batch_first_p):
    ad.primitive_jvps[primitive] = partial(_jet_jvp, primitive=primitive)
    pe.custom_partial_eval_rules[primitive] = partial(_partial_eval, primitive=primitive)
    ns.register_equation_certificate(primitive, value_count=2,
        certificate_groups=((None, None, None), ('derivative', 4, 'next')))
for primitive in (_linear_p, _batch_linear_p):
    ad.primitive_transposes[primitive] = partial(_linear_transpose, primitive=primitive)
    ns.register_equation_certificate(primitive, value_count=1)
for primitive in (_first_p, _linear_p):
    batching.primitive_batchers[primitive] = partial(_batch, primitive=primitive)
for primitive in (_batch_first_p, _batch_linear_p):
    batching.primitive_batchers[primitive] = partial(_nested_batch, primitive=primitive)
