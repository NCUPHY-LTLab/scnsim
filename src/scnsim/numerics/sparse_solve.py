"""Checked sparse equation operations for shared physical-program derivatives.

Each callback factors its actual inputs locally and releases the factor before
return. Primal, tangent, adjoint and mixed equations expose independent existing
solve certificates. No native handle, SQL, progress side effect or callback
counter is retained. These explicit checked transformations do not advertise an
unchecked arbitrary grad-through-pure_callback contract.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from functools import partial
from threading import RLock
import numpy as np

from ..compilation.program import SparsePattern


class SolveAux(NamedTuple):
    status: Any
    residual: Any
    numerator: Any
    denominator: Any


class SolveOutcome(NamedTuple):
    value: Any
    aux: SolveAux


class DirectionalOutcome(NamedTuple):
    primal: SolveOutcome
    tangent: SolveOutcome


class AdjointOutcome(NamedTuple):
    primal: SolveOutcome
    adjoint: SolveOutcome
    values_cotangent: Any
    rhs_cotangent: Any


class MixedOutcome(NamedTuple):
    primal: SolveOutcome
    first: SolveOutcome
    second: SolveOutcome
    mixed: SolveOutcome


def _host_transport(*operands, callback, input_tree, specifications,
                    output_tree, output_specifications):
    """Decode exact selected operands before any host equation arithmetic.

    JAX's callback worker may use a different thread-local x64 setting. Only
    uint32 words cross that boundary for wider operands; no numeric conversion
    or JAX dispatch occurs here. The cached metadata contains no point arrays.
    """
    decoded = []
    for operand, (dtype, shape, words) in zip(operands, specifications, strict=True):
        value = np.asarray(operand)
        if words:
            # Complex words have separate component and word axes. Collapse
            # both before viewing the native dtype; retain their C-order bits.
            value = np.ascontiguousarray(value).reshape(-1).view(np.dtype(dtype)).reshape(shape)
        decoded.append(value)
    result = output_tree.flatten_up_to(callback(*input_tree.unflatten(decoded)))
    encoded = []
    for value, (dtype, shape, words) in zip(result, output_specifications, strict=True):
        value = np.asarray(value)
        if value.dtype != np.dtype(dtype) or value.shape != shape:
            raise TypeError(f'checked sparse callback output requires {dtype}{shape}, '
                            f'received {value.dtype}{value.shape}')
        if words:
            # Return only uint32 to the callback framework: its worker-local
            # device transfer must not canonicalize the selected wide dtype.
            value = np.ascontiguousarray(value).reshape(-1).view(np.uint32).reshape(
                shape + (np.dtype(dtype).itemsize // np.dtype(np.uint32).itemsize,))
        encoded.append(value)
    return output_tree.unflatten(encoded)


def _host_solve(pattern, values, rhs, prior_status, *, stage, mode, ordering, factor=None):
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
    try:
        dtype = np.asarray(rhs).dtype
        real_dtype = np.empty((), dtype=dtype).real.dtype.type
        status = np.int32(prior_status)
        unavailable = SolveOutcome(np.zeros_like(rhs), SolveAux(status, real_dtype(np.inf),
                                                              real_dtype(np.inf), real_dtype(np.inf)))
        if status:
            return unavailable
        if not (np.all(np.isfinite(values)) and np.all(np.isfinite(rhs))):
            return unavailable._replace(aux=unavailable.aux._replace(status=np.int32(stage)))
        if pattern.shape[0] == 0:
            zero = real_dtype(0)
            return SolveOutcome(np.zeros_like(rhs), SolveAux(np.int32(0), zero, zero, zero))
        matrix = csc_matrix((np.asarray(values, dtype=dtype), pattern.rows, pattern.indptr), shape=pattern.shape)
        try:
            if factor is None:
                factor = splu(matrix) if ordering is None else splu(matrix, permc_spec=ordering)
        except RuntimeError as error:
            if str(error) != 'Factor is exactly singular':
                raise
            return unavailable._replace(aux=unavailable.aux._replace(status=np.int32(stage)))
        value = np.asarray(factor.solve(np.asarray(rhs, dtype=dtype), trans=mode), dtype=dtype)
        actual = matrix if mode == 'N' else matrix.T if mode == 'T' else matrix.conjugate().T
        error = actual @ value - rhs
        scale = abs(actual) @ np.abs(value) + np.abs(rhs)
        num = real_dtype(np.max(np.abs(error), initial=0))
        den = real_dtype(np.max(np.abs(scale), initial=0))
        eta = real_dtype(np.inf) if not (np.isfinite(num) and np.isfinite(den)) else (
            real_dtype(0 if num == 0 else np.inf) if den == 0 else real_dtype(num / den))
        threshold = real_dtype(256) * real_dtype(pattern.shape[0] + 1) * real_dtype(np.finfo(real_dtype).eps)
        if not (np.all(np.isfinite(value)) and np.isfinite(eta) and eta <= threshold):
            status = np.int32(stage)
        return SolveOutcome(value, SolveAux(status, eta, num, den))
    finally:
        # Traceback frames must not extend ownership of a native factor.
        factor = None


def _host_high_equation(pattern, a, b, prior, *, stage, ordering, a_dot, b_dot,
                        derivative_stage, second=None):
    """One local factor for checked primal and directional/mixed equations."""
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
    factor = None
    try:
        dtype = np.asarray(b).dtype
        rd = np.empty((), dtype=dtype).real.dtype.type
        def failed(code):
            return SolveOutcome(np.zeros_like(b), SolveAux(np.int32(code), rd(np.inf), rd(np.inf), rd(np.inf)))
        def unavailable(primal):
            value = failed(primal.aux.status)
            return DirectionalOutcome(primal, value) if second is None else MixedOutcome(primal, value, value, value)
        if prior:
            return unavailable(failed(prior))
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return unavailable(failed(stage))
        matrix = csc_matrix((a, pattern.rows, pattern.indptr), shape=pattern.shape)
        factor = None
        if pattern.shape[0]:
            try:
                factor = splu(matrix) if ordering is None else splu(matrix, permc_spec=ordering)
            except RuntimeError as error:
                if str(error) != 'Factor is exactly singular':
                    raise
                return unavailable(failed(stage))
        def solve(rhs, code):
            return _host_solve(pattern, a, np.asarray(rhs, dtype=dtype), 0, stage=code,
                               mode='N', ordering=ordering, factor=factor)
        def product(values, x):
            return csc_matrix((values, pattern.rows, pattern.indptr), shape=pattern.shape) @ x
        primal = solve(b, stage)
        if primal.aux.status:
            return unavailable(primal)
        first = solve(b_dot - product(a_dot, primal.value), derivative_stage)
        if second is None:
            return DirectionalOutcome(primal, first)
        a2, b2, a12, b12 = second
        second_value = solve(b2 - product(a2, primal.value), derivative_stage)
        if first.aux.status or second_value.aux.status:
            mixed = failed(first.aux.status if first.aux.status else second_value.aux.status)
        else:
            rhs = b12 - product(a12, primal.value) - product(a_dot, second_value.value) - product(a2, first.value)
            mixed = solve(rhs, derivative_stage)
        return MixedOutcome(primal, first, second_value, mixed)
    finally:
        # Traceback frames must not extend ownership of a native factor.
        factor = None


def _host_high_dispatch(pattern, av, bv, code, ad, bd, *more,
                        ordering, stage, derivative_stage, mixed_mode):
    return _host_high_equation(pattern, av, bv, code, stage=stage, ordering=ordering,
                               a_dot=ad, b_dot=bd, derivative_stage=derivative_stage,
                               second=more if mixed_mode else None)


def _high_callback(equation, a, b, prior, *, stage, a_dot, b_dot, derivative_stage, second=None):
    import jax
    import jax.numpy as jnp
    dtype = jnp.result_type(a, b, a_dot, b_dot)
    a, b, a_dot, b_dot = (jnp.asarray(value, dtype=dtype) for value in (a, b, a_dot, b_dot))
    rd = b.real.dtype
    scalar = lambda dtype: jax.ShapeDtypeStruct((), dtype)
    shape = SolveOutcome(jax.ShapeDtypeStruct(b.shape, b.dtype),
                         SolveAux(scalar(jnp.int32), scalar(rd), scalar(rd), scalar(rd)))
    result_shape = DirectionalOutcome(shape, shape) if second is None else MixedOutcome(shape, shape, shape, shape)
    operands = () if second is None else tuple(jnp.asarray(value, dtype=dtype) for value in second)
    mixed_mode = second is not None
    return equation.call_callback(('high', stage, derivative_stage, mixed_mode), _host_high_dispatch,
                                  result_shape, a, b, jnp.asarray(prior, dtype=jnp.int32),
                                  a_dot, b_dot, *operands, stage=stage,
                                  derivative_stage=derivative_stage, mixed_mode=mixed_mode)


@dataclass(frozen=True, slots=True, eq=False)
class SparseEquation:
    pattern: SparsePattern
    ordering: str | None = None
    _callbacks: dict = field(default_factory=dict, repr=False)
    _callback_lock: Any = field(default_factory=RLock, repr=False)

    def call_callback(self, key, function, result_shape, *operands, **options):
        """One exact input/output transport for all checked sparse callbacks.

        Wider values are bitcast under the caller's selected precision scope,
        before pure_callback's worker-local device_put can canonicalize them.
        Complex values use contiguous real/imag channels, preserving signs and
        all represented bits. Float32 stays float32; this is not wider math.
        """
        import jax
        import jax.numpy as jnp
        leaves, input_tree = jax.tree.flatten(operands)
        encoded, specifications = [], []
        for operand in leaves:
            value = jnp.asarray(operand)
            dtype = np.dtype(value.dtype)
            words = dtype.itemsize > np.dtype(np.uint32).itemsize
            specifications.append((dtype.str, value.shape, words))
            if words:
                channels = jnp.stack((value.real, value.imag), axis=-1) if dtype.kind == 'c' else value
                value = jax.lax.bitcast_convert_type(channels, jnp.uint32)
            encoded.append(value)
        specifications = tuple(specifications)
        output_leaves, output_tree = jax.tree.flatten(result_shape)
        output_specifications = tuple((np.dtype(value.dtype).str, tuple(value.shape),
                                       np.dtype(value.dtype).itemsize > np.dtype(np.uint32).itemsize)
                                      for value in output_leaves)
        wire_shape = output_tree.unflatten([
            jax.ShapeDtypeStruct(shape + (np.dtype(dtype).itemsize // np.dtype(np.uint32).itemsize,),
                                 jnp.uint32) if words else value
            for value, (dtype, shape, words) in zip(output_leaves, output_specifications, strict=True)])
        transport_key = ('operand_transport', key, input_tree, specifications,
                         output_tree, output_specifications)
        with self._callback_lock:
            callback = self._callbacks.get(transport_key)
            if callback is None:
                host = partial(function, self.pattern, ordering=self.ordering, **options)
                callback = partial(_host_transport, callback=host,
                                   input_tree=input_tree, specifications=specifications,
                                   output_tree=output_tree, output_specifications=output_specifications)
                self._callbacks[transport_key] = callback
        wire = jax.pure_callback(callback, wire_shape, *encoded, vmap_method='sequential')
        restored = []
        for value, (dtype, shape, words) in zip(output_tree.flatten_up_to(wire),
                                               output_specifications, strict=True):
            if words:
                dtype = np.dtype(dtype)
                if dtype.kind == 'c':
                    component_dtype = np.dtype('float32' if dtype.itemsize == 8 else 'float64')
                    if component_dtype.itemsize > np.dtype(np.uint32).itemsize:
                        value = value.reshape(shape + (2, component_dtype.itemsize // 4))
                    channels = jax.lax.bitcast_convert_type(value, component_dtype)
                    value = jax.lax.complex(channels[..., 0], channels[..., 1])
                else:
                    value = jax.lax.bitcast_convert_type(value, dtype)
            restored.append(value)
        return output_tree.unflatten(restored)

    def solve(self, values, rhs, *, stage: int, prior_status=0, mode='N') -> SolveOutcome:
        """Stateless callback with fixed-shape checked output; no hidden factor reuse."""
        import jax
        import jax.numpy as jnp
        if mode not in ('N', 'T', 'H'):
            raise ValueError('sparse solve mode must be N, T or H')
        dtype = jnp.result_type(values, rhs)
        rhs = jnp.asarray(rhs, dtype=dtype)
        values = jnp.asarray(values, dtype=dtype)
        rd = rhs.real.dtype
        scalar = lambda dtype: jax.ShapeDtypeStruct((), dtype)
        result = SolveOutcome(jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
                              SolveAux(scalar(jnp.int32), scalar(rd), scalar(rd), scalar(rd)))
        return self.call_callback(('solve', stage, mode), _host_solve, result, values, rhs,
                                  jnp.asarray(prior_status, dtype=jnp.int32), stage=stage, mode=mode)

    def matmul(self, values, x):
        import jax.numpy as jnp
        rows, cols = jnp.asarray(self.pattern.rows), jnp.asarray(self.pattern.cols)
        weights = jnp.asarray(values).reshape((len(self.pattern.rows),) + (1,) * (x.ndim - 1))
        return jnp.zeros((self.pattern.shape[0],) + x.shape[1:], dtype=x.dtype).at[rows].add(weights * x[cols])

    def jvp(self, values, rhs, values_dot, rhs_dot, *, primal_stage=3, tangent_stage=4, prior_status=0):
        return _high_callback(self, values, rhs, prior_status, stage=primal_stage,
                              a_dot=values_dot, b_dot=rhs_dot, derivative_stage=tangent_stage)

    def vjp(self, values, rhs, value_cotangent, *, primal_stage=3, adjoint_stage=4):
        """Cotangent c satisfies dL=Re(conj(c) dot dx), unlike raw JAX encoding."""
        import jax.numpy as jnp
        primal = self.solve(values, rhs, stage=primal_stage)
        adjoint = self.solve(values, value_cotangent, stage=adjoint_stage,
                             prior_status=primal.aux.status, mode='H')
        rows, cols = jnp.asarray(self.pattern.rows), jnp.asarray(self.pattern.cols)
        products = -adjoint.value[rows] * jnp.conj(primal.value[cols])
        value_bar = jnp.sum(products, axis=tuple(range(1, products.ndim))) if products.ndim > 1 else products
        if not jnp.issubdtype(jnp.asarray(values).dtype, jnp.complexfloating):
            value_bar = jnp.real(value_bar)
        rhs_bar = adjoint.value if jnp.issubdtype(jnp.asarray(rhs).dtype, jnp.complexfloating) else jnp.real(adjoint.value)
        return AdjointOutcome(primal, adjoint, jnp.asarray(value_bar, dtype=jnp.asarray(values).dtype),
                              jnp.asarray(rhs_bar, dtype=jnp.asarray(rhs).dtype))

    def solve_pair(self, values, rhs, *, stage, prior_status=0, mode='N', runner=None):
        return _pair_callback(self, values, rhs, prior_status, stage=stage, mode=mode, runner=runner)

    def jvp_pair(self, values, rhs, values_dot, rhs_dot, *, primal_stage=3,
                 tangent_stage=4, prior_status=0, runner=None):
        return _pair_callback(self, values, rhs, prior_status, stage=primal_stage, mode='N',
                              a_dot=values_dot, b_dot=rhs_dot, derivative_stage=tangent_stage, runner=runner)

    def mixed_pair(self, values, rhs, values_first, rhs_first, values_second, rhs_second,
                   values_mixed, rhs_mixed, *, primal_stage=3, derivative_stage=4, prior_status=0, runner=None):
        return _pair_callback(self, values, rhs, prior_status, stage=primal_stage, mode='N',
                              a_dot=values_first, b_dot=rhs_first, derivative_stage=derivative_stage,
                              second=(values_second, rhs_second, values_mixed, rhs_mixed), runner=runner)

    def mixed(self, values, rhs, values_first, rhs_first, values_second, rhs_second,
              values_mixed, rhs_mixed, *, primal_stage=3, derivative_stage=4, prior_status=0):
        return _high_callback(self, values, rhs, prior_status, stage=primal_stage,
                              a_dot=values_first, b_dot=rhs_first, derivative_stage=derivative_stage,
                              second=(values_second, rhs_second, values_mixed, rhs_mixed))


class PairSolveOutcome(NamedTuple):
    value: tuple[Any, Any]
    aux: SolveAux


class PairDirectionalOutcome(NamedTuple):
    primal: PairSolveOutcome
    tangent: PairSolveOutcome


class PairMixedOutcome(NamedTuple):
    primal: PairSolveOutcome
    first: PairSolveOutcome
    second: PairSolveOutcome
    mixed: PairSolveOutcome


def _host_pair_product(payload):
    """Required compiled ordered CSR kernel; borrowed buffers stay call-local.

    No numerical Python-loop fallback or nested JAX dispatch is available.
    Installation/preparation owns a missing native extension's ImportError.
    """
    from .._native import _compensated_cpu
    hi_data, hi_indices, hi_indptr, lo_data, lo_indices, lo_indptr, xhi, xlo = payload
    operands = tuple(np.ascontiguousarray(value) for value in payload)
    shape = (len(hi_indptr) - 1,) + xhi.shape[1:]
    high, low = np.empty(shape, dtype=xhi.dtype), np.empty(shape, dtype=xhi.dtype)
    _compensated_cpu.csr_pair_product(*operands, high, low)
    return high, low


def _host_pair_equation(pattern, a, b, prior, *, stage, ordering, mode='N',
                        a_dot=None, b_dot=None, derivative_stage=4, second=None, observer=None):
    """One actual high factor, one fixed correction for each required equation."""
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
    from . import compensated as cp
    factor = None
    try:
        def phase(name, equation):
            return nullcontext() if observer is None else observer(name, equation=equation)
        dtype = np.asarray(b[0]).dtype
        rd = np.empty((), dtype=dtype).real.dtype.type
        status = np.int32(prior)
        def unavailable(code):
            return PairSolveOutcome((np.zeros_like(b[0]), np.zeros_like(b[0])),
                                    SolveAux(np.int32(code), rd(np.inf), rd(np.inf), rd(np.inf)))
        def matrices(pair):
            pair = tuple(csc_matrix((np.asarray(part, dtype=dtype), pattern.rows, pattern.indptr), shape=pattern.shape)
                         for part in pair)
            return pair if mode == 'N' else tuple(part.T if mode == 'T' else part.conjugate().T for part in pair)
        def failure(value):
            return (PairMixedOutcome(value, value, value, value) if second is not None else
                    value if a_dot is None else PairDirectionalOutcome(value, value))
        if status:
            return failure(unavailable(status))
        A = matrices(a)
        if not all(np.all(np.isfinite(part)) for pair in (a, b) for part in pair):
            value = unavailable(stage)
            return failure(value)
        if pattern.shape[0] == 0:
            value = PairSolveOutcome((np.zeros_like(b[0]), np.zeros_like(b[0])), SolveAux(np.int32(0), rd(0), rd(0), rd(0)))
            return failure(value)
        try:
            high = csc_matrix((np.asarray(a[0], dtype=dtype), pattern.rows, pattern.indptr), shape=pattern.shape)
            with phase('factor', 'shared'):
                factor = splu(high) if ordering is None else splu(high, permc_spec=ordering)
        except RuntimeError as error:
            if str(error) != 'Factor is exactly singular':
                raise
            value = unavailable(stage)
            return failure(value)
        csr = tuple(part.tocsr() for part in A)
        def equation(rhs, code, name):
            if not all(np.all(np.isfinite(part)) for part in rhs):
                return unavailable(code)
            with phase('seed_solve', name):
                seed = np.asarray(factor.solve(cp.project(rhs, xp=np), trans=mode), dtype=dtype)
            x = cp.from_value(seed, xp=np)
            with phase('correction_residual', name):
                remainder = cp.residual(*csr, x, rhs, execute=_host_pair_product)
            with phase('correction_solve', name):
                correction = np.asarray(factor.solve(cp.project(cp.negate(remainder), xp=np), trans=mode), dtype=dtype)
            with phase('pair_update', name):
                x = cp.add(x, cp.from_value(correction, xp=np), xp=np)
            with phase('postverification', name):
                eta, num, den = cp.solve_certificate(*csr, x, rhs, real_dtype=rd, execute=_host_pair_product)
            threshold = rd(256) * rd(pattern.shape[0] + 1) * rd(np.finfo(rd).eps)
            failed = not (all(np.all(np.isfinite(part)) for part in x) and np.isfinite(eta) and eta <= threshold)
            return PairSolveOutcome(x, SolveAux(np.int32(code if failed else 0), eta, num, den))
        primal = equation(b, stage, 'primal')
        if a_dot is None:
            return primal
        if primal.aux.status:
            value = unavailable(primal.aux.status)
            return PairMixedOutcome(primal, value, value, value) if second is not None else PairDirectionalOutcome(primal, value)
        Ad = tuple(part.tocsr() for part in matrices(a_dot))
        with phase('derivative_rhs', 'tangent_first'):
            rhs = cp.add(b_dot, cp.negate(cp.sparse_matmul(*Ad, primal.value, execute=_host_pair_product)), xp=np)
        first = equation(rhs, derivative_stage, 'tangent_first')
        if second is None:
            return PairDirectionalOutcome(primal, first)
        a_second, b_second, a_mixed, b_mixed = second
        def product(pair, x):
            return cp.sparse_matmul(*(part.tocsr() for part in matrices(pair)), x, execute=_host_pair_product)
        with phase('derivative_rhs', 'tangent_second'):
            second_rhs = cp.add(b_second, cp.negate(product(a_second, primal.value)), xp=np)
        second_value = equation(second_rhs, derivative_stage, 'tangent_second')
        if first.aux.status or second_value.aux.status:
            mixed = unavailable(first.aux.status if first.aux.status else second_value.aux.status)
        else:
            with phase('derivative_rhs', 'mixed'):
                mixed_rhs = b_mixed
                for pair, x in ((a_mixed, primal.value), (a_dot, second_value.value), (a_second, first.value)):
                    mixed_rhs = cp.add(mixed_rhs, cp.negate(product(pair, x)), xp=np)
            mixed = equation(mixed_rhs, derivative_stage, 'mixed')
        return PairMixedOutcome(primal, first, second_value, mixed)
    finally:
        # Traceback frames must not extend ownership of a native factor.
        factor = None


def _host_pair_dispatch(pattern, ah, al, bh, bl, code, *derivatives,
                        ordering, stage, mode, derivative_stage, jet_order):
    options = dict(stage=stage, mode=mode, ordering=ordering, derivative_stage=derivative_stage)
    if jet_order:
        adh, adl, bdh, bdl, *more = derivatives
        options.update(a_dot=(adh, adl), b_dot=(bdh, bdl))
        if jet_order == 2:
            a2h, a2l, b2h, b2l, amh, aml, bmh, bml = more
            options['second'] = ((a2h, a2l), (b2h, b2l), (amh, aml), (bmh, bml))
    return _host_pair_equation(pattern, (ah, al), (bh, bl), code, **options)


def _pair_callback(equation, a, b, prior, *, stage, mode, a_dot=None, b_dot=None, derivative_stage=4, second=None, runner=None):
    import jax
    import jax.numpy as jnp
    dtype = jnp.result_type(*a, *b)
    b = tuple(jnp.asarray(part, dtype=dtype) for part in b)
    a = tuple(jnp.asarray(part, dtype=dtype) for part in a)
    if runner is not None:
        # This checked-host route uses exactly the callback equation. The runner
        # owns optional observations; no observer enters a JAX closure or cache.
        host_pair = lambda pair: tuple(np.asarray(part, dtype=np.dtype(dtype)) for part in pair)
        options = dict(stage=stage, mode=mode, ordering=equation.ordering, derivative_stage=derivative_stage)
        if a_dot is not None:
            options.update(a_dot=host_pair(a_dot), b_dot=host_pair(b_dot))
        if second is not None:
            options['second'] = tuple(host_pair(pair) for pair in second)
        outcome = runner(_host_pair_equation, (equation.pattern, host_pair(a), host_pair(b), np.asarray(prior, dtype=np.int32)), options)
        return jax.tree.map(jnp.asarray, outcome)
    rd = b[0].real.dtype
    scalar = lambda dtype: jax.ShapeDtypeStruct((), dtype)
    shape = PairSolveOutcome(tuple(jax.ShapeDtypeStruct(part.shape, part.dtype) for part in b),
                             SolveAux(scalar(jnp.int32), scalar(rd), scalar(rd), scalar(rd)))
    status = jnp.asarray(prior, dtype=jnp.int32)
    jet_order = 0 if a_dot is None else 1 if second is None else 2
    def invoke(result, *operands):
        return equation.call_callback(('pair', stage, mode, derivative_stage, jet_order),
                                      _host_pair_dispatch, result, *operands, stage=stage,
                                      mode=mode, derivative_stage=derivative_stage, jet_order=jet_order)
    if a_dot is None:
        return invoke(shape, *a, *b, status)
    a_dot = tuple(jnp.asarray(part, dtype=b[0].dtype) for part in a_dot)
    b_dot = tuple(jnp.asarray(part, dtype=b[0].dtype) for part in b_dot)
    if second is not None:
        operands = tuple(tuple(jnp.asarray(part, dtype=b[0].dtype) for part in pair) for pair in second)
        return invoke(PairMixedOutcome(shape, shape, shape, shape),
                                 *a, *b, status, *a_dot, *b_dot,
                                 *(part for pair in operands for part in pair))
    return invoke(PairDirectionalOutcome(shape, shape), *a, *b, status, *a_dot, *b_dot)
