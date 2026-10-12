"""Pinned native JAX sparse equation and its finite checked transform rules.

Only this equation boundary uses JAX's private primitive/checkify interfaces.
Ordinary arithmetic remains JAX arithmetic. Native factors belong to one host
call; primitive parameters contain immutable structure, never points or tapes.
Checkify must wrap the complete transformed computation, including an executed
reverse pullback, so tangent and adjoint equation checks are functionalized too.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple, Any
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from jax._src import core, checkify
from jax._src.interpreters import ad, batching, mlir
from jax.experimental import checkify as public_checkify

from .sparse_solve import SolveAux, SolveOutcome, SparseEquation, _host_solve


class NativeSparseAux(NamedTuple):
    status: Any
    eta: Any
    residual_numerator: Any
    residual_denominator: Any
    performed: Any  # Equation attempted; false when prior checked failure skips it.


class NativeSparseOutcome(NamedTuple):
    value: Any
    aux: NativeSparseAux


def _owned_error_effect_key(effect):
    # Error dictionary flattening is the loop carry ABI. Distinct ordinary,
    # archived and directional payload schemas must never compare as tied:
    # checkify's probe admits effects from a set, the body in encounter order.
    return (str(effect.error_type),
            tuple((item.shape, str(item.dtype)) for item in effect.shape_dtypes),
            type(effect).__module__, type(effect).__qualname__,
            repr(effect.site), repr(effect.dependency),
            getattr(effect, 'role', ''), getattr(effect, 'stage', -1),
            getattr(effect, 'derivative_order', -1), getattr(effect, 'reason', ''))


@dataclass(frozen=True, slots=True)
class SparseCertificateEffect(checkify.ErrorEffect):
    """A separate fixed payload slot for each transformed equation site."""
    site: tuple
    role: str
    stage: int
    derivative_order: int
    dependency: tuple

    def __lt__(self, other):
        if isinstance(other, (SparseCertificateEffect, DerivativeAvailabilityEffect)):
            return _owned_error_effect_key(self) < _owned_error_effect_key(other)
        return checkify.ErrorEffect.__lt__(self, other)


class SparseEquationCheck(checkify.JaxException):
    def __init__(self, traceback_info, site, role, stage, derivative_order, dependency, status, eta, numerator, denominator, performed):
        super().__init__(traceback_info)
        self.site, self.role, self.stage, self.derivative_order = site, role, stage, derivative_order
        self.dependency = dependency
        self.payload = status, eta, numerator, denominator, performed

    def tree_flatten(self):
        return self.payload, (self.traceback_info, self.site, self.role, self.stage, self.derivative_order, self.dependency)

    @classmethod
    def tree_unflatten(cls, metadata, payload):
        return cls(*metadata, *payload)

    def get_effect_type(self):
        descriptors = tuple(jax.ShapeDtypeStruct(core.typeof(x).shape, core.typeof(x).dtype)
                            for x in self.payload)
        return SparseCertificateEffect(type(self), descriptors, self.site, self.role, self.stage, self.derivative_order, self.dependency)

    def __str__(self):
        return f'sparse {self.role} equation status {self.payload[0]} at site {self.site}'


@dataclass(frozen=True, slots=True)
class SparseCertificate:
    site: tuple
    role: str
    stage: int
    status: Any
    eta: Any
    residual_numerator: Any
    residual_denominator: Any
    performed: Any
    present: Any  # A checkify placeholder is not an executed equation site.
    derivative_order: int
    dependency: tuple
    basis_kind: str | None = None
    basis_indices: tuple = ()
    work_index: int | None = None
    frequency_source_index: int | None = None


def sparse_certificates(error, *, basis_kind=None, basis_indices=(), work_index=None, frequency_source_index=None) -> tuple[SparseCertificate, ...]:
    """Read actual fixed payload arrays from a returned checked computation.

    The effect keys are static site/role metadata, not mutable runtime counters.
    This includes successful equations: a false error predicate still carries
    its actual residual payload. Batched checked frames retain the batch axes.
    """
    # Payload projection can run after the execution thread leaves its
    # precision scope. Preserve returned payload bits without global config
    # changes; selected32 payloads retain their original dtype.
    with jax.enable_x64(True):
        basis_indices = tuple(basis_indices)
        if basis_kind is not None and basis_kind not in ('parameter', 'output_row', 'direction', 'functional'):
            raise ValueError('certificate basis kind must identify parameter, output row, direction or functional')
        if basis_kind is None and basis_indices:
            raise ValueError('certificate basis indices require their kind')
        certificates = []
        for effect in sorted(error._payload):
            if not isinstance(effect, SparseCertificateEffect):
                continue
            payload = error._payload[effect]
            status, eta, numerator, denominator, performed = payload[:5]
            if basis_kind is not None and (status.ndim == 0 or status.shape[0] != len(basis_indices)):
                raise ValueError('mapped certificate leading axis disagrees with actual derivative basis')
            present = jnp.asarray(error._code[effect]) != checkify._INACTIVE_CODE
            # Outer vmap adds lane axes to the code; the primitive's own batched
            # certificate can have further RHS/direction lanes in its payload.
            present = jnp.broadcast_to(present.reshape(present.shape + (1,) *
                                      (status.ndim - present.ndim)), status.shape)
            if isinstance(effect, ArchivedSparseEffect):
                present = present & payload[5]
            # Placeholder payloads are all ones in pinned checkify. They have no
            # equation meaning; expose absence rather than a fabricated failure or
            # residual. A genuinely called prior-skipped equation remains present.
            certificates.append(SparseCertificate(effect.site, effect.role, effect.stage,
                jnp.where(present, status, jnp.int32(0)),
                *(jnp.where(present, value, jnp.asarray(jnp.nan, dtype=value.dtype))
                  for value in (eta, numerator, denominator)),
                present & performed, present, effect.derivative_order, effect.dependency, basis_kind, basis_indices, work_index, frequency_source_index))
        return tuple(certificates)


@dataclass(frozen=True, slots=True)
class DerivativeAvailabilityEffect(checkify.ErrorEffect):
    site: tuple
    dependency: tuple
    reason: str

    def __lt__(self, other):
        if isinstance(other, (SparseCertificateEffect, DerivativeAvailabilityEffect)):
            return _owned_error_effect_key(self) < _owned_error_effect_key(other)
        return checkify.ErrorEffect.__lt__(self, other)


class DerivativeUnavailableCheck(checkify.JaxException):
    def __init__(self, traceback_info, dependency, reason, available):
        super().__init__(traceback_info)
        self.dependency, self.reason, self.available = dependency, reason, available

    def tree_flatten(self):
        return (self.available,), (self.traceback_info, self.dependency, self.reason)

    @classmethod
    def tree_unflatten(cls, metadata, payload):
        return cls(*metadata, *payload)

    def get_effect_type(self):
        return DerivativeAvailabilityEffect(type(self),
            (jax.ShapeDtypeStruct(core.typeof(self.available).shape, core.typeof(self.available).dtype),),
            (), self.dependency, self.reason)

    def __str__(self):
        return self.reason


def derivative_available(flag, *, dependency, reason):
    """Record one EXISTING local derivative-domain predicate, only when called.

    The caller invokes this in its native JVP rule, never as primal admission.
    Owned sites/dependencies preserve availability without a global Error gate.
    """
    flag = jnp.asarray(flag, dtype=jnp.bool_)
    exception = DerivativeUnavailableCheck(checkify.get_traceback(), tuple(dependency), reason, flag)
    payload, metadata = jax.tree.flatten(exception)
    effect = exception.get_effect_type()
    # Match checkify's context-owned integer code, including cond placeholders.
    error = checkify.Error({effect: ~flag}, {effect: jnp.asarray(0)}, {0: metadata}, {effect: payload})
    checkify._check_error(error, debug=True)


@dataclass(frozen=True, slots=True)
class DerivativeAvailability:
    site: tuple
    dependency: tuple
    reason: str
    available: Any
    present: Any


def derivative_availability(error):
    # Payload projection can run after the execution thread leaves its
    # precision scope. Preserve returned payload bits without global config
    # changes; selected32 payloads retain their original dtype.
    with jax.enable_x64(True):
        facts = []
        for effect in sorted(error._payload):
            if not isinstance(effect, DerivativeAvailabilityEffect):
                continue
            available = error._payload[effect][0]
            present = jnp.asarray(error._code[effect]) != checkify._INACTIVE_CODE
            present = jnp.broadcast_to(present.reshape(present.shape +
                (1,) * (available.ndim - present.ndim)), available.shape)
            if isinstance(effect, ArchivedAvailabilityEffect):
                present = present & error._payload[effect][1]
            facts.append(DerivativeAvailability(effect.site, effect.dependency, effect.reason,
                available, present))
        return tuple(facts)


def _availability_tree(tree, operands, site):
    # Re-key static check metadata on an ALREADY transformed JAX equation.
    # Numeric variables/operations are untouched and no expression is evaluated.
    from dataclasses import replace
    error = tree.unflatten(operands)
    keys = {effect: replace(effect, site=site) if isinstance(effect, DerivativeAvailabilityEffect)
            else effect for effect in error._pred}
    error = error._replace(_pred={keys[key]: value for key, value in error._pred.items()},
        _code={keys[key]: value for key, value in error._code.items()},
        _payload={keys[key]: value for key, value in error._payload.items()})
    return jax.tree.flatten(error)[1]


def _annotate_sparse_sites(jaxpr, path=()):
    """Label only sparse binds in an ALREADY transformed static JAX program.

    No operation is evaluated or differentiated by this pass. JAX remains the
    tracing/AD/control-flow authority. Paths stay stable across checkify's loop
    effect probe and actual tracing, unlike its mutable error-code allocator.
    """
    # JAX 0.11.2 unifies ClosedJaxpr with Jaxpr; its .jaxpr is self.
    # Only nested equation parameters recurse. replace retains actual consts.
    def nested(value, prefix):
        if isinstance(value, core.Jaxpr):
            return _annotate_sparse_sites(value, prefix)
        if isinstance(value, tuple):
            items = tuple(nested(item, prefix + (i,)) for i, item in enumerate(value))
            return type(value)(*items) if hasattr(value, '_fields') else items
        if isinstance(value, list):
            return [nested(item, prefix + (i,)) for i, item in enumerate(value)]
        return value
    equations = []
    for index, equation in enumerate(jaxpr.eqns):
        site = path + (index,)
        params = {key: nested(value, site + (key,)) for key, value in equation.params.items()}
        if equation.primitive in _certificate_primitives:
            params['certificate_site'] = site
        elif equation.primitive is checkify.check_p:
            params['err_tree'] = _availability_tree(params['err_tree'], equation.invars, site)
        equations.append(equation.replace(params=params))
    return jaxpr.replace(eqns=equations)


def checked_native(function):
    """Functionalize numerical checks AND actual certificate payloads.

    Pass the complete transformed body (including executed VJP pullback).
    For a mapped wave apply this wrapper to the transformed single frame before
    mapping: each lane then returns its own Error/payload, rather than relying
    on checkify's ordinary scan error reduction to archive iteration history.
    """
    def checked(*args):
        # Trace and functionalize in one local precision scope: checkify
        # error codes/placeholders must agree with the owned assertion rules
        # even when the raw caller has ambient x64 disabled.
        with jax.enable_x64(True):
            program, shape = jax.make_jaxpr(function, return_shape=True)(*args)
            program = _annotate_sparse_sites(program)
            flat_args, _ = jax.tree.flatten(args)
            _, output_tree = jax.tree.flatten(shape)
            error, values = public_checkify.checkify(core.jaxpr_as_fun(program),
                errors=public_checkify.user_checks | frozenset((SparseEquationCheck, DerivativeUnavailableCheck, ArchivedPrimalCheck,)))(*flat_args)
            return error, output_tree.unflatten(values)
    return checked


def checked_sequential_primal(function):
    """Preserve owned Newton certificates across an opaque sequential vmap.

    Functionalize the scalar primal before custom batching hides its body.
    Error arrays leave that call alongside the packet and are reintroduced at
    the visible outer boundary. The debug bridge remains inert for raw callers;
    checked callers retain all actual inner effects with their lane axes.
    """
    checked = checked_native(function)
    def primal(*operands):
        with jax.enable_x64(True):
            # Pinned custom_vmap rejects batched captured constants. Inner
            # checkify can create such bool/code operands under an outer cond;
            # expose ALL constants, including nondifferentiable ones, as normal
            # mapped inputs rather than closing over runtime Error arrays.
            from jax._src.interpreters import partial_eval as pe
            program, shape = jax.make_jaxpr(checked, return_shape=True)(*operands)
            explicit_program, constants = pe.separate_consts(program)
            flat_operands, _ = jax.tree.flatten(operands)
            _, output_tree = jax.tree.flatten(shape)
            mapped = jax.custom_batching.sequential_vmap(core.jaxpr_as_fun(explicit_program))
            error, packet = output_tree.unflatten(mapped(*constants, *flat_operands))
            _archive_primal_error(error)
            return packet
    return primal


_primal_archive_p = core.Primitive('scnsim_owned_primal_equation_archive')
_primal_archive_p.multiple_results = True
_primal_archive_p.is_effectful = lambda _: True


@dataclass(frozen=True, slots=True)
class ArchivedSparseEffect(SparseCertificateEffect):
    """Five equation arrays plus exact lane presence/code archive."""


@dataclass(frozen=True, slots=True)
class ArchivedAvailabilityEffect(DerivativeAvailabilityEffect):
    """Availability plus exact lane presence/code archive."""


class ArchivedPrimalCheck(checkify.JaxException):
    """Finite payload adapter; retain original exception metadata and meaning."""
    def __init__(self, original_tree, payload):
        self.original_tree, self.payload = original_tree, tuple(payload)
        original = original_tree.unflatten(self.payload[:-2])
        super().__init__(original.traceback_info)

    def tree_flatten(self):
        return self.payload, self.original_tree

    @classmethod
    def tree_unflatten(cls, original_tree, payload):
        return cls(original_tree, payload)

    def __str__(self):
        return str(self.original_tree.unflatten(self.payload[:-2]))


def _archive_layout(error):
    from dataclasses import replace
    keys, payloads, predicates, codes = {}, {}, {}, {}
    for effect in error._pred:
        payload = tuple(error._payload[effect])
        lane_code = jnp.asarray(error._code[effect])
        lane_pred = jnp.asarray(error._pred[effect], dtype=jnp.bool_)
        archived = isinstance(effect, (ArchivedSparseEffect, ArchivedAvailabilityEffect))
        owned = isinstance(effect, (SparseCertificateEffect, DerivativeAvailabilityEffect))
        if owned and not archived:
            payload += (lane_code != checkify._INACTIVE_CODE, lane_code)
        descriptors = tuple(jax.ShapeDtypeStruct(core.typeof(value).shape, core.typeof(value).dtype)
                            for value in payload)
        if isinstance(effect, SparseCertificateEffect):
            key = ArchivedSparseEffect(ArchivedPrimalCheck, descriptors, effect.site, effect.role,
                effect.stage, effect.derivative_order, effect.dependency)
        elif isinstance(effect, DerivativeAvailabilityEffect):
            key = ArchivedAvailabilityEffect(ArchivedPrimalCheck, descriptors, effect.site,
                effect.dependency, effect.reason)
        else:
            key = replace(effect, shape_dtypes=descriptors)
        keys[effect], payloads[key] = key, payload
        # Scalar first-failure control fits pinned cond placeholders. The full
        # owned lane codes/presence remain payload DATA, never any-lane reduction.
        failed = jnp.any(lane_pred)
        sentinel = jnp.iinfo(lane_code.dtype).max
        failure_code = jnp.min(jnp.where(lane_pred, lane_code, sentinel), initial=sentinel)
        present_code = jnp.min(jnp.where(lane_code != checkify._INACTIVE_CODE,
                                        lane_code, sentinel), initial=sentinel)
        code = jnp.where(failed, failure_code, present_code)
        predicates[key] = failed
        codes[key] = jnp.where(code == sentinel, jnp.asarray(checkify._INACTIVE_CODE, dtype=code.dtype), code)
    metadata = {}
    for code, tree in error._metadata.items():
        kind, detail = tree.node_data()
        if kind is SparseEquationCheck:
            _, site, role, stage, order, dependency = detail
            effect = next(effect for effect in error._pred if isinstance(effect, SparseCertificateEffect)
                and (effect.site, effect.role, effect.stage, effect.derivative_order, effect.dependency)
                == (site, role, stage, order, dependency))
            metadata[code] = jax.tree.structure(ArchivedPrimalCheck(tree, payloads[keys[effect]]))
        elif kind is DerivativeUnavailableCheck:
            _, dependency, reason = detail
            effect = next(effect for effect in error._pred if isinstance(effect, DerivativeAvailabilityEffect)
                and (effect.dependency, effect.reason) == (dependency, reason))
            metadata[code] = jax.tree.structure(ArchivedPrimalCheck(tree, payloads[keys[effect]]))
        else:
            metadata[code] = tree
    return error._replace(_pred=predicates, _code=codes, _metadata=metadata, _payload=payloads)


def _archive_primal_error(error):
    error = _archive_layout(error)
    operands, tree = jax.tree.flatten(error)
    _primal_archive_p.bind(*operands, error_tree=tree)


_primal_archive_p.def_impl(lambda *operands, error_tree: [])


@_primal_archive_p.def_effectful_abstract_eval
def _archive_abstract(*operands, error_tree):
    return [], set(error_tree.unflatten(operands)._pred)


def _archive_lower(context, *operands, error_tree):
    # Debug-only archive for raw callers; checked_native functionalizes it.
    return []


mlir.register_lowering(_primal_archive_p, _archive_lower)


def _archive_batch(operands, axes, *, error_tree):
    size = next(value.shape[axis] for value, axis in zip(operands, axes) if axis is not None)
    front = tuple(batching.bdim_at_front(value, axis, size)
                  for value, axis in zip(operands, axes))
    _archive_primal_error(error_tree.unflatten(front))
    return [], []


batching.primitive_batchers[_primal_archive_p] = _archive_batch


def _archive_jvp(primals, tangents, *, error_tree):
    _primal_archive_p.bind(*primals, error_tree=error_tree)
    return [], []


ad.primitive_jvps[_primal_archive_p] = _archive_jvp
ad.primitive_transposes[_primal_archive_p] = lambda cotangents, *operands, error_tree: (None,) * len(operands)


def _archive_error_check(context, outer, enabled_errors, *operands, error_tree):
    """Merge actual lane archives without ordinary check_p's any-lane reduction."""
    with jax.enable_x64(True):
        incoming = error_tree.unflatten(operands)
        count = max(incoming._metadata, default=-1) + 1
        offset = context.current_code
        context.advance(count)
        metadata = {code + offset: value for code, value in incoming._metadata.items()}
        for effect in sorted(incoming._pred):
            if effect.error_type not in enabled_errors:
                continue
            pred = jnp.asarray(incoming._pred[effect], dtype=jnp.bool_)
            code = jnp.asarray(incoming._code[effect])
            # Inactive conditional sites remain absent after namespace remap.
            code = jnp.where(code == checkify._INACTIVE_CODE, code, code + offset)
            old_pred = jnp.broadcast_to(jnp.asarray(outer._pred.get(effect, False)), pred.shape)
            old_code = jnp.broadcast_to(jnp.asarray(outer._code.get(effect, checkify._INACTIVE_CODE),
                                                   dtype=code.dtype), code.shape)
            payload = incoming._payload[effect]
            if effect in outer._payload:
                payload = tuple(jnp.where(old_pred.reshape(old_pred.shape +
                    (1,) * (value.ndim - old_pred.ndim)), previous, value)
                    for previous, value in zip(outer._payload[effect], payload, strict=True))
            outer = outer._update(effect, old_pred | pred, jnp.where(old_pred, old_code, code),
                                  metadata, payload)
        return outer, []


checkify.error_checks[_primal_archive_p] = _archive_error_check


_solve_p = core.Primitive('scnsim_checked_sparse_equation')
_solve_p.multiple_results = True
_batch_solve_p = core.Primitive('scnsim_checked_sparse_equation_batch')
_batch_solve_p.multiple_results = True


def _callback(values, rhs, prior, *, equation, stage, mode, role, certificate_site=(), backend='sparse', derivative_order=0, dependency=(), parameter_ad_depth=0):
    if backend == 'dense':
        return _dense_equation(values, rhs, prior, equation=equation, stage=stage, mode=mode)
    if backend != 'sparse':
        raise ValueError('unknown checked equation backend')
    rd = jnp.empty((), dtype=rhs.dtype).real.dtype
    shape = SolveOutcome(jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
                         SolveAux(jax.ShapeDtypeStruct((), jnp.int32),
                                  *(jax.ShapeDtypeStruct((), rd) for _ in range(3))))
    result = equation.call_callback(('native', stage, mode, role), _host_solve, shape,
                                    values, rhs, prior, stage=stage, mode=mode)
    return (result.value, *result.aux, prior == 0)



def _dense_equation(values, rhs, prior, *, equation, stage, mode):
    """Small selected-port equation on device with the existing certificate.

    This shares the equation primitive's implicit AD/checkify rules. Only the
    numerical implementation differs: no host callback or sparse factor is
    needed for the already-small port matrix. Prior failure skips the solve.
    """
    n = equation.pattern.shape[0]
    rd = rhs.real.dtype
    inf = jnp.asarray(jnp.inf, dtype=rd)
    def unavailable(code, performed):
        return jnp.zeros_like(rhs), code, inf, inf, inf, performed
    def attempted(_):
        matrix = jnp.zeros(equation.pattern.shape, dtype=values.dtype).at[
            jnp.asarray(equation.pattern.rows), jnp.asarray(equation.pattern.cols)].set(values)
        actual = matrix if mode == 'N' else matrix.T if mode == 'T' else jnp.conj(matrix.T)
        if n == 0:
            zero = jnp.asarray(0, dtype=rd)
            return jnp.zeros_like(rhs), jnp.int32(0), zero, zero, zero, jnp.bool_(True)
        def finite(_):
            x = jnp.linalg.solve(actual, rhs)
            num = jnp.max(jnp.abs(actual @ x - rhs), initial=0)
            den = jnp.max(jnp.abs(actual) @ jnp.abs(x) + jnp.abs(rhs), initial=0)
            eta = jnp.where(jnp.isfinite(num) & jnp.isfinite(den),
                            jnp.where(den == 0, jnp.where(num == 0, 0, inf), num / den), inf)
            tau = jnp.asarray(256 * (n + 1), dtype=rd) * jnp.finfo(rd).eps
            ok = jnp.all(jnp.isfinite(x)) & jnp.isfinite(eta) & (eta <= tau)
            return x, jnp.where(ok, jnp.int32(0), jnp.int32(stage)), eta, num, den, jnp.bool_(True)
        return jax.lax.cond(jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(rhs)),
                            finite, lambda _: unavailable(jnp.int32(stage), jnp.bool_(True)), None)
    return jax.lax.cond(prior == 0, attempted,
                        lambda _: unavailable(prior, jnp.bool_(False)), None)


_static = ('equation', 'stage', 'mode', 'role', 'certificate_site', 'backend', 'derivative_order', 'dependency', 'parameter_ad_depth')
_solve_p.def_impl(partial(jax.jit, static_argnames=_static)(_callback))


def _batch_callback(values, rhs, prior, **params):
    return jax.lax.map(lambda items: _callback(*items, **params), (values, rhs, prior))


_batch_solve_p.def_impl(partial(jax.jit, static_argnames=_static)(_batch_callback))


def _abstract(values, rhs, prior, *, equation, stage, mode, role, certificate_site=(), backend='sparse', derivative_order=0, dependency=(), parameter_ad_depth=0):
    del stage, mode, role, certificate_site, backend, derivative_order, dependency, parameter_ad_depth
    if values.shape != (len(equation.pattern.rows),):
        raise TypeError('sparse coefficient shape disagrees with declared CSC pattern')
    if rhs.shape[0] != equation.pattern.shape[0] or rhs.ndim not in (1, 2):
        raise TypeError('sparse RHS requires the declared node axis and optional RHS axis')
    if values.dtype != rhs.dtype or prior.shape != () or prior.dtype != np.dtype('int32'):
        raise TypeError('sparse equation operands require one selected dtype and scalar int32 status')
    rd = np.empty((), dtype=rhs.dtype).real.dtype
    return (rhs, core.ShapedArray((), np.dtype('int32')),
            *(core.ShapedArray((), rd) for _ in range(3)), core.ShapedArray((), np.dtype('bool')))


_solve_p.def_abstract_eval(_abstract)
def equation_lowering(function):
    """Lower owned equations in their tracing precision scope, including dense.

    MLIR invokes lower_fun after a caller's Python precision context exits.
    Keeping this local prevents linalg/callback scaffolding recanonicalizing
    selected64 operands; explicit selected32 operands remain selected32.
    """
    lowered = mlir.lower_fun(function, multiple_results=True)
    def lower(context, *operands, **params):
        with jax.enable_x64(True):
            return lowered(context, *operands, **params)
    return lower


mlir.register_lowering(_solve_p, equation_lowering(_callback))


def _batch_abstract(values, rhs, prior, **params):
    size = values.shape[0]
    avals = _abstract(core.ShapedArray(values.shape[1:], values.dtype),
                      core.ShapedArray(rhs.shape[1:], rhs.dtype),
                      core.ShapedArray(prior.shape[1:], prior.dtype), **params)
    return tuple(core.ShapedArray((size,) + value.shape, value.dtype) for value in avals)


_batch_solve_p.def_abstract_eval(_batch_abstract)
mlir.register_lowering(_batch_solve_p, equation_lowering(_batch_callback))


def _matmul(equation, values, x, mode):
    rows, cols = equation.pattern.rows, equation.pattern.cols
    if mode != 'N':
        rows, cols = cols, rows
    if mode == 'H':
        values = jnp.conj(values)
    weights = values.reshape((len(rows),) + (1,) * (x.ndim - 1))
    return jnp.zeros_like(x).at[jnp.asarray(rows)].add(weights * x[jnp.asarray(cols)])


def _jvp(primals, tangents, *, primitive=_solve_p, **params):
    values, rhs, prior = primals
    values_dot, rhs_dot, _ = map(ad.instantiate_zeros, tangents)
    primal = primitive.bind(values, rhs, prior, **params)
    product = partial(_matmul, params['equation'], mode=params['mode'])
    if primitive is _batch_solve_p:
        product = jax.vmap(product)
    residual = rhs_dot - product(values_dot, primal[0])
    order = params.get('derivative_order', 0) + 1
    tangent_params = dict(params, stage=4, role='tangent' if order == 1 else 'mixed', derivative_order=order)
    tangent = primitive.bind(values, residual, primal[1], **tangent_params)
    # Certificates are checked by the primitive's own checkify rule; their
    # numerical derivatives are not physical observables.
    return primal, (tangent[0], *(ad.Zero(core.typeof(x).to_tangent_aval()) for x in primal[1:]))


ad.primitive_jvps[_solve_p] = _jvp
ad.primitive_jvps[_batch_solve_p] = partial(_jvp, primitive=_batch_solve_p)


def _transpose(cotangents, values, rhs, prior, *, primitive=_solve_p, **params):
    if ad.is_undefined_primal(values) or ad.is_undefined_primal(prior):
        raise TypeError('sparse transpose requires the primal matrix and checked status')
    if not ad.is_undefined_primal(rhs):
        return None, None, None
    cotangent = cotangents[0]
    if isinstance(cotangent, ad.Zero):
        return None, ad.Zero(rhs.aval), None
    # JAX encodes a complex cotangent bilinearly. The Hermitian real-loss
    # convention is equivalent after conjugating at that external boundary.
    mode = params['mode']
    if mode == 'H':
        values, mode = jnp.conj(values), 'N'
    else:
        mode = 'T' if mode == 'N' else 'N'
    adjoint = primitive.bind(values, cotangent, prior,
                           **dict(params, stage=4, mode=mode, role='adjoint',
                                  derivative_order=max(1, params.get('derivative_order', 0))))
    return None, adjoint[0], None


ad.primitive_transposes[_solve_p] = _transpose
ad.primitive_transposes[_batch_solve_p] = partial(_transpose, primitive=_batch_solve_p)


def _batch(args, axes, **params):
    size = next(value.shape[axis] for value, axis in zip(args, axes)
                if axis is not None)
    operands = tuple(batching.bdim_at_front(value, axis, size)
                     for value, axis in zip(args, axes))
    result = _batch_solve_p.bind(*operands, **params)
    return result, (0,) * len(result)


batching.primitive_batchers[_solve_p] = _batch


def _nested_batch(args, axes, **params):
    size = next(value.shape[axis] for value, axis in zip(args, axes) if axis is not None)
    operands = tuple(batching.bdim_at_front(value, axis, size) for value, axis in zip(args, axes))
    inner = operands[0].shape[1]
    flat = tuple(value.reshape((size * inner,) + value.shape[2:]) for value in operands)
    result = _batch_solve_p.bind(*flat, **params)
    return tuple(value.reshape((size, inner) + value.shape[1:]) for value in result), (0,) * len(result)


batching.primitive_batchers[_batch_solve_p] = _nested_batch


def _error_check(ctx, error, enabled_errors, *operands, primitive=_solve_p,
                 value_count=1, certificate_groups=(), **params):
    # checkify manufactures select placeholders here, outside the caller
    # arithmetic scope. Preserve actual wide certificate payload dtypes;
    # selected32 outputs remain32 and no global configuration is changed.
    with jax.enable_x64(True):
        result = primitive.bind(*operands, **params)
        if checkify.FailedCheckError in enabled_errors:
            groups = certificate_groups or ((None, None, None),)
            if len(result) != value_count + 5 * len(groups):
                raise TypeError('owned equation certificate schema disagrees with primitive outputs')
            for ordinal, (role, stage, order) in enumerate(groups):
                offset = value_count + 5 * ordinal
                payload = result[offset:offset + 5]
                site = params.get('certificate_site', ())
                if certificate_groups:
                    site = site + ('equation', ordinal)
                base_order = params.get('derivative_order', 0)
                group_order = base_order + 1 if order == 'next' else base_order if order is None else order
                group_role = ('tangent' if group_order == 1 else 'mixed') if role == 'derivative' else params['role'] if role is None else role
                failure = SparseEquationCheck(checkify.get_traceback(), site,
                    group_role, params['stage'] if stage is None else stage, group_order, params.get('dependency', ()), *payload)
                error = checkify.assert_func(ctx, error, jnp.any(payload[0] != 0), failure)
        return error, result


def register_equation_certificate(primitive, *, value_count, certificate_groups=()):
    """Register an owned single equation or finite first/mixed equation jet.

    Grouped outputs start with all logical value representation channels, then
    five arrays per ORDERED equation: status, eta, numerator, denominator and
    performed. A group gets its own effect slot; no aggregate can replace its
    failure authority. ``None`` metadata inherits the original bind parameter.
    ``derivative``/``next`` denote the next implicit equation role/order.
    Registration stores only finite primitive/schema metadata, never points,
    coefficients, factors, callbacks or derivative tapes.
    """
    groups = tuple(tuple(group) for group in certificate_groups)
    if any(len(group) != 3 for group in groups):
        raise TypeError('equation certificate group requires role, stage and derivative order')
    _certificate_primitives.add(primitive)
    checkify.error_checks[primitive] = partial(_error_check, primitive=primitive,
        value_count=value_count, certificate_groups=groups)


_certificate_primitives = set()
register_equation_certificate(_solve_p, value_count=1)
register_equation_certificate(_batch_solve_p, value_count=1)


def checked_sparse_solve(equation: SparseEquation, values, rhs, *, stage=3,
                         prior_status=0, mode='N', role='primal', derivative_order=0, dependency=()) -> NativeSparseOutcome:
    """Array-only equation; selected precision is the caller's explicit scope."""
    if mode not in ('N', 'T', 'H'):
        raise ValueError('sparse solve mode must be N, T or H')
    dtype = jnp.result_type(values, rhs)
    result = _solve_p.bind(jnp.asarray(values, dtype=dtype), jnp.asarray(rhs, dtype=dtype),
                           jnp.asarray(prior_status, dtype=jnp.int32),
                           equation=equation, stage=stage, mode=mode, role=role, derivative_order=derivative_order, dependency=dependency)
    return NativeSparseOutcome(result[0], NativeSparseAux(*result[1:]))


def checked_dense_solve(equation: SparseEquation, matrix, rhs, *, stage, prior_status=0, mode='N',
                        role='primal', derivative_order=0, dependency=()):
    """Small dense equation using the SAME native implicit/checkify authority.

    The immutable equation pattern must enumerate every matrix entry in CSC
    order. The caller supplies its existing small-port structural equation.
    """
    dtype = jnp.result_type(matrix, rhs)
    matrix, rhs = jnp.asarray(matrix, dtype=dtype), jnp.asarray(rhs, dtype=dtype)
    values = matrix[jnp.asarray(equation.pattern.rows), jnp.asarray(equation.pattern.cols)]
    result = _solve_p.bind(values, rhs, jnp.asarray(prior_status, dtype=jnp.int32),
                           equation=equation, stage=stage, mode=mode, role=role,
                           derivative_order=derivative_order, dependency=dependency, backend='dense')
    return NativeSparseOutcome(result[0], NativeSparseAux(*result[1:]))


# Install finite jet rules after the shared primitive and certificate boundary
# exist. The module contains only static rules, never operation/candidate state.
from . import native_equation_jet as _native_equation_jet
