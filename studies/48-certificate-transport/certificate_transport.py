"""Study scratch: carry SCNSim equation certificates out of traced JAX programs
through ordinary outputs, using public JAX interfaces only.

This is the candidate transport of the design study. It is NOT product code.
It imports nothing from ``jax._src``.

Idea
----
The certificate arrays are already ordinary outputs of SCNSim's equation
primitives. What is missing is a way to bring the outputs of equations that sit
inside ``cond`` / ``while`` / ``scan`` bodies, or that JAX's own JVP and
transpose rules created, to the boundary of the program. ``certified`` does it
with SCNSim's own pass over the finished program:

1. trace the complete transformed function with ``jax.make_jaxpr``;
2. re-evaluate that program equation by equation; an equation site adds its
   certificate arrays to a ledger keyed by static site metadata;
3. ``cond``, ``while`` and ``scan`` whose bodies hold sites are emitted again
   through the public ``lax.switch`` / ``lax.while_loop`` / ``lax.scan`` with
   the ledger slots as extra branch outputs or loop carry;
4. the ledger is returned as an ordinary pytree next to the values.

Foreign checks (``checkify.check`` / ``debug_check`` written by anyone) stay
with JAX's public ``checkify.checkify`` and come back as its public ``Error``.

JAX surface used: ``jax.make_jaxpr``, ``jax.eval_shape``, ``jax.lax`` control
flow, ``jax.tree``, ``jax.experimental.checkify`` (public names only),
``jax.extend.core`` (``Primitive``, ``Literal``, ``jaxpr_as_fun``, primitive
objects), ``jax.interpreters`` (``ad``, ``batching``, ``mlir``).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import checkify as public_checkify
from jax.extend.core import Literal, Primitive, jaxpr_as_fun
from jax.extend.core import primitives as jax_primitives
from jax.interpreters import ad, batching, mlir


class CertificateTransportError(TypeError):
    """An equation site sits where this pass cannot bring its certificate out."""


# --------------------------------------------------------------------------
# Ledger: SCNSim's own evidence value. Static keys, array slots.
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EntryKey:
    kind: str                 # 'equation' or 'availability'
    archived: bool            # came through an attached (sequentially mapped) ledger
    site: tuple
    role: str
    stage: int
    derivative_order: int
    dependency: tuple
    reason: str
    descriptors: tuple        # ((shape, dtype), ...) when recorded; ordering only


class Slot(NamedTuple):
    failed: Any               # bool; "this slot already holds a failure"
    present: Any              # bool; false for a site that was not executed
    values: tuple             # equation: status, eta, numerator, denominator, performed
                              # availability: (available,)


@jax.tree_util.register_pytree_node_class
class Ledger:
    """Certificates of one traced frame. Keys are static, slots are arrays."""

    def __init__(self, keys, slots):
        self.keys, self.slots = tuple(keys), tuple(slots)

    def tree_flatten(self):
        return (self.slots,), self.keys

    @classmethod
    def tree_unflatten(cls, keys, children):
        return cls(keys, children[0])

    def items(self):
        return zip(self.keys, self.slots)

    def any_failed(self):
        """Traced predicate for in-program gating (replaces reading Error._pred)."""
        result = jnp.bool_(False)
        for slot in self.slots:
            result = result | jnp.any(slot.failed)
        return result


class Evidence(NamedTuple):
    ledger: Ledger            # SCNSim-owned certificates and availability facts
    foreign: Any              # public checkify.Error for checks SCNSim does not own


_LEGACY_MODULE = 'scnsim.numerics.native_sparse'
_LEGACY_NAMES = {  # (kind, archived) -> (exception class name, effect class name)
    ('equation', False): ('SparseEquationCheck', 'SparseCertificateEffect'),
    ('availability', False): ('DerivativeUnavailableCheck', 'DerivativeAvailabilityEffect'),
    ('equation', True): ('ArchivedPrimalCheck', 'ArchivedSparseEffect'),
    ('availability', True): ('ArchivedPrimalCheck', 'ArchivedAvailabilityEffect'),
}


def entry_order(key: EntryKey):
    """Row order of the stored evidence.

    It reproduces the order of the current mechanism (its effect sort key) so
    that stored evidence keeps its row order. Every component is SCNSim-owned
    static metadata.
    """
    exception, effect = _LEGACY_NAMES[key.kind, key.archived]
    equation = key.kind == 'equation'
    return (f"<class '{_LEGACY_MODULE}.{exception}'>", key.descriptors, _LEGACY_MODULE, effect,
            repr(key.site), repr(key.dependency), key.role if equation else '',
            key.stage if equation else -1, key.derivative_order if equation else -1, key.reason)


def _descriptors(values):
    return tuple((tuple(jnp.shape(value)), str(jnp.result_type(value))) for value in values)


def _placeholder(shape_slot: Slot) -> Slot:
    zeros = lambda item: jnp.zeros(item.shape, item.dtype)
    return Slot(zeros(shape_slot.failed), zeros(shape_slot.present),
                tuple(zeros(item) for item in shape_slot.values))


def _select(keep, old, new):
    keep = jnp.asarray(keep)
    new = jnp.asarray(new)
    old = jnp.asarray(old, dtype=new.dtype)
    keep = keep.reshape(keep.shape + (1,) * (max(old.ndim, new.ndim) - keep.ndim))
    return jnp.where(keep, old, new)


def _write(ledger: dict, key: EntryKey, failed, present, values) -> dict:
    """First failure stays; otherwise the latest execution replaces the slot."""
    old = ledger.get(key)
    failed = jnp.asarray(failed, dtype=jnp.bool_)
    present = jnp.asarray(present, dtype=jnp.bool_)
    if old is None:
        slot = Slot(failed, present, tuple(values))
    else:
        slot = Slot(old.failed | failed, _select(old.failed, old.present, present),
                    tuple(_select(old.failed, previous, value)
                          for previous, value in zip(old.values, values, strict=True)))
    return {**ledger, key: slot}


def _export(ledger: dict) -> Ledger:
    keys = sorted(ledger, key=entry_order)
    return Ledger(keys, tuple(ledger[key] for key in keys))


# --------------------------------------------------------------------------
# Registration of equation primitives (schema only, no JAX internals).
# --------------------------------------------------------------------------

class _Schema(NamedTuple):
    value_count: int
    groups: tuple


_equation_schemas: dict[Primitive, _Schema] = {}


def register_equation(primitive, *, value_count, certificate_groups=()):
    _equation_schemas[primitive] = _Schema(value_count, tuple(tuple(g) for g in certificate_groups))


def _record_equation(ledger, primitive, params, site, outs):
    schema = _equation_schemas[primitive]
    groups = schema.groups or ((None, None, None),)
    if len(outs) != schema.value_count + 5 * len(groups):
        raise TypeError('owned equation certificate schema disagrees with primitive outputs')
    for ordinal, (role, stage, order) in enumerate(groups):
        offset = schema.value_count + 5 * ordinal
        payload = tuple(outs[offset:offset + 5])
        group_site = site + ('equation', ordinal) if schema.groups else site
        base = params.get('derivative_order', 0)
        group_order = base + 1 if order == 'next' else base if order is None else order
        if role == 'derivative':
            group_role = ('frequency_tangent' if params.get('frequency_jet', False)
                          else 'tangent' if group_order == 1 else 'mixed')
        else:
            group_role = params['role'] if role is None else role
        key = EntryKey('equation', False, group_site, group_role,
                       params['stage'] if stage is None else stage, group_order,
                       tuple(params.get('dependency', ())), '', _descriptors(payload))
        ledger = _write(ledger, key, jnp.any(payload[0] != 0), True, payload)
    return ledger


# --------------------------------------------------------------------------
# Two small pure primitives that keep a fact on the data path.
# No JAX effect is used: an effect type would have to be registered in JAX's
# private allow-lists before it may appear inside control flow.
# --------------------------------------------------------------------------

# value = gate(value, unavailable): identity on `value`; records a
# derivative-availability fact where the caller's JVP rule ran.
gate_p = Primitive('scnsim_derivative_gate')
gate_p.def_impl(lambda value, unavailable, **_: value)
gate_p.def_abstract_eval(lambda value, unavailable, **_: value)
mlir.register_lowering(gate_p, mlir.lower_fun(lambda value, unavailable, **_: value,
                                              multiple_results=False))
ad.primitive_jvps[gate_p] = lambda primals, tangents, **params: (
    gate_p.bind(*primals, **params), tangents[0])
ad.primitive_transposes[gate_p] = lambda cotangent, value, unavailable, **params: (cotangent, None)


def _gate_batch(operands, axes, **params):
    value, unavailable = operands
    size = next(x.shape[axis] for x, axis in zip(operands, axes) if axis is not None)
    return gate_p.bind(batching.bdim_at_front(value, axes[0], size),
                       batching.bdim_at_front(unavailable, axes[1], size), **params), 0


batching.primitive_batchers[gate_p] = _gate_batch


def derivative_gate(value, available, *, dependency, reason):
    """Mark ``value`` as depending on one local derivative-domain predicate.

    Called from a JVP rule on a value the rule returns. Two equations are
    emitted (``not`` and the gate), the same count as the current
    ``derivative_available`` (``not`` and ``check``), so equation paths of the
    surrounding program do not move.
    """
    unavailable = ~jnp.asarray(available, dtype=jnp.bool_)
    return gate_p.bind(value, unavailable, dependency=tuple(dependency), reason=reason)


# anchor = attach(anchor, *ledger_leaves): identity on `anchor`; carries a
# ledger that was produced behind an opaque boundary (sequential map) to the
# next enclosing certified frame.
attach_p = Primitive('scnsim_ledger_attach')
attach_p.def_impl(lambda anchor, *leaves, **_: anchor)
attach_p.def_abstract_eval(lambda anchor, *leaves, **_: anchor)
mlir.register_lowering(attach_p, mlir.lower_fun(lambda anchor, *leaves, **_: anchor,
                                                multiple_results=False))
ad.primitive_jvps[attach_p] = lambda primals, tangents, **params: (
    attach_p.bind(*primals, **params), tangents[0])
ad.primitive_transposes[attach_p] = lambda cotangent, anchor, *leaves, **params: (
    cotangent, *(None,) * len(leaves))


def _attach_batch(operands, axes, **params):
    size = next(x.shape[axis] for x, axis in zip(operands, axes) if axis is not None)
    front = tuple(batching.bdim_at_front(x, axis, size) for x, axis in zip(operands, axes))
    return attach_p.bind(*front, **params), 0


batching.primitive_batchers[attach_p] = _attach_batch


def attach_ledger(anchor, ledger: Ledger):
    leaves, tree = jax.tree.flatten(ledger)
    return attach_p.bind(anchor, *leaves, ledger_tree=tree)


def _record_attached(ledger, params, leaves):
    inner: Ledger = params['ledger_tree'].unflatten(leaves)
    for key, slot in inner.items():
        extra = () if key.archived else ((tuple(jnp.shape(slot.present)), 'bool'),
                                         (tuple(jnp.shape(slot.present)), 'int64'))
        archived = replace(key, archived=True, descriptors=_descriptors(slot.values) + extra
                           if not key.archived else
                           _descriptors(slot.values) + key.descriptors[len(slot.values):])
        ledger = _write(ledger, archived, jnp.any(slot.failed), slot.present, slot.values)
    return ledger


# --------------------------------------------------------------------------
# The pass.
# --------------------------------------------------------------------------

def _is_jaxpr(value):
    return hasattr(value, 'eqns') and hasattr(value, 'outvars')


def _sub_jaxprs(value):
    if _is_jaxpr(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _sub_jaxprs(item)


def _owned(primitive):
    return primitive in _equation_schemas or primitive is gate_p or primitive is attach_p


def _holds_sites(jaxpr, _seen=None):
    for eqn in jaxpr.eqns:
        if _owned(eqn.primitive):
            return True
        if any(_holds_sites(sub) for value in eqn.params.values() for sub in _sub_jaxprs(value)):
            return True
    return False


def _eqn_holds_sites(eqn):
    return any(_holds_sites(sub) for value in eqn.params.values() for sub in _sub_jaxprs(value))


def _consts(jaxpr):
    return tuple(getattr(jaxpr, 'consts', ()))


def _bind(eqn, invals):
    """Evaluate one foreign equation unchanged, as jax.core.eval_jaxpr does."""
    params = eqn.primitive.get_bind_params(eqn.params)
    subfuns, params = params if isinstance(params, tuple) else ((), params)
    context = getattr(getattr(eqn, 'ctx', None), 'manager', None)
    if context is None:
        out = eqn.primitive.bind(*subfuns, *invals, **params)
    else:
        with context:
            out = eqn.primitive.bind(*subfuns, *invals, **params)
    return list(out) if eqn.primitive.multiple_results else [out]


def _run(jaxpr, consts, args, path, ledger):
    env = {}

    def read(atom):
        return atom.val if isinstance(atom, Literal) else env[atom]

    for variable, value in zip(jaxpr.constvars, consts, strict=True):
        env[variable] = value
    for variable, value in zip(jaxpr.invars, args, strict=True):
        env[variable] = value
    for index, eqn in enumerate(jaxpr.eqns):
        site = path + (index,)
        invals = [read(atom) for atom in eqn.invars]
        primitive = eqn.primitive
        if primitive in _equation_schemas:
            outs = list(primitive.bind(*invals, **eqn.params))
            ledger = _record_equation(ledger, primitive, eqn.params, site, outs)
        elif primitive is gate_p:
            value, unavailable = invals
            key = EntryKey('availability', False, site, '', -1, -1, eqn.params['dependency'],
                           eqn.params['reason'], _descriptors((unavailable,)))
            ledger = _write(ledger, key, jnp.any(unavailable), True, (~unavailable,))
            outs = [value]
        elif primitive is attach_p:
            ledger = _record_attached(ledger, eqn.params, invals[1:])
            outs = [invals[0]]
        elif not _eqn_holds_sites(eqn):
            outs = _bind(eqn, invals)
        elif primitive is jax_primitives.cond_p:
            outs, ledger = _cond(eqn, invals, site, ledger)
        elif primitive is jax_primitives.while_p:
            outs, ledger = _while(eqn, invals, site, ledger)
        elif primitive is jax_primitives.scan_p:
            outs, ledger = _scan(eqn, invals, site, ledger)
        elif primitive in _INLINE:
            outs, ledger = _inline(eqn, invals, site, ledger)
        elif primitive.name in _OPAQUE_AFTER_CERTIFICATION:
            outs = _bind(eqn, invals)
        else:
            raise CertificateTransportError(
                f'equation site inside {primitive.name!r} at {site}: this primitive has no '
                'certificate rule; certify its body or add a rule')
        for variable, value in zip(eqn.outvars, outs, strict=True):
            env[variable] = value
    return [read(atom) for atom in jaxpr.outvars], ledger


_INLINE = tuple(getattr(jax_primitives, name) for name in ('jit_p', 'pjit_p', 'custom_jvp_call_p')
                if hasattr(jax_primitives, name))
# A sequentially mapped body is certified before it is hidden (see
# certified_sequential_primal); its ledger arrives through attach_p.
_OPAQUE_AFTER_CERTIFICATION = ('custom_vmap_call',)


def _shape_of(value):
    aval = jax.typeof(value)
    return jax.ShapeDtypeStruct(aval.shape, aval.dtype)


def _probe(run, operands):
    """Keys and slot shapes a body writes, from one abstract evaluation."""
    found = []

    def body(*values):
        local = run(values)
        found[:] = list(local)
        return tuple(local[key] for key in found)

    shapes = jax.eval_shape(body, *map(_shape_of, operands))
    return dict(zip(found, shapes))


def _incoming(ledger, schema):
    keys = sorted(schema, key=entry_order)
    return keys, tuple(ledger[key] if key in ledger else _placeholder(schema[key]) for key in keys)


def _cond(eqn, invals, site, ledger):
    index, *operands = invals
    branches = eqn.params['branches']
    paths = [site + ('branches', ordinal) for ordinal in range(len(branches))]

    def body(branch, path, slots, values):
        return _run(branch, _consts(branch), values, path, slots)

    schema = {}
    for branch, path in zip(branches, paths):
        schema.update(_probe(lambda values, b=branch, p=path: body(b, p, {}, values)[1], operands))
    keys, slots = _incoming(ledger, schema)

    def emit(branch, path):
        def function(carried, *values):
            outs, local = body(branch, path, dict(zip(keys, carried)), values)
            return tuple(outs), tuple(local[key] for key in keys)
        return function

    outs, slots = lax.switch(index, [emit(b, p) for b, p in zip(branches, paths)], slots, *operands)
    return list(outs), {**ledger, **dict(zip(keys, slots))}


def _while(eqn, invals, site, ledger):
    params = eqn.params
    n_cond, n_body = params['cond_nconsts'], params['body_nconsts']
    cond_consts, body_consts, init = invals[:n_cond], invals[n_cond:n_cond + n_body], invals[n_cond + n_body:]
    cond_jaxpr, body_jaxpr = params['cond_jaxpr'], params['body_jaxpr']
    if _holds_sites(cond_jaxpr):
        raise CertificateTransportError(f'equation site inside a while_loop predicate at {site}')
    path = site + ('body_jaxpr',)

    def body(slots, carry):
        return _run(body_jaxpr, _consts(body_jaxpr), (*body_consts, *carry), path, slots)

    schema = _probe(lambda carry: body({}, carry)[1], init)
    keys, slots = _incoming(ledger, schema)
    predicate = jaxpr_as_fun(cond_jaxpr)

    def step(state):
        carry, carried = state
        outs, local = body(dict(zip(keys, carried)), carry)
        return tuple(outs), tuple(local[key] for key in keys)

    carry, slots = lax.while_loop(lambda state: predicate(*cond_consts, *state[0])[0],
                                  step, (tuple(init), slots))
    return list(carry), {**ledger, **dict(zip(keys, slots))}


def _scan_arity(eqn):
    """(consts, carry, xs) counts from shapes alone.

    scan :: (c, a) -> (c, b) keeps the carry's type and gives every mapped
    input and stacked output one extra leading axis. The count therefore
    follows from comparing body and equation shapes, without reading the
    private bookkeeping parameters (they changed in JAX 0.10 and again in 0.11).
    """
    body = eqn.params['jaxpr']
    rank = lambda atom: len(atom.aval.shape)
    carry = 0
    for inner, outer in zip(body.outvars, eqn.outvars):
        if rank(inner) != rank(outer):
            break
        carry += 1
    mapped = 0
    for inner, outer in zip(reversed(body.invars), reversed(eqn.invars)):
        if rank(outer) != rank(inner) + 1:
            break
        mapped += 1
    return len(eqn.invars) - carry - mapped, carry, mapped


def _scan(eqn, invals, site, ledger):
    params = eqn.params
    body_jaxpr = params['jaxpr']
    n_consts, n_carry, n_mapped = _scan_arity(eqn)
    consts, init, mapped = invals[:n_consts], invals[n_consts:n_consts + n_carry], invals[n_consts + n_carry:]
    path = site + ('jaxpr',)

    def body(slots, carry, row):
        return _run(body_jaxpr, _consts(body_jaxpr), (*consts, *carry, *row), path, slots)

    rows = tuple(jax.ShapeDtypeStruct(jax.typeof(x).shape[1:], jax.typeof(x).dtype) for x in mapped)
    found = []

    def probe(carry, row):
        local = body({}, carry, row)[1]
        found[:] = list(local)
        return tuple(local[key] for key in found)

    shapes = jax.eval_shape(probe, tuple(map(_shape_of, init)), rows)
    keys, slots = _incoming(ledger, dict(zip(found, shapes)))

    def step(state, row):
        carry, carried = state
        outs, local = body(dict(zip(keys, carried)), carry, row)
        return (tuple(outs[:n_carry]), tuple(local[key] for key in keys)), tuple(outs[n_carry:])

    (carry, slots), stacked = lax.scan(step, (tuple(init), slots), tuple(mapped),
                                       length=params['length'], reverse=params['reverse'],
                                       unroll=params['unroll'])
    return [*carry, *stacked], {**ledger, **dict(zip(keys, slots))}


def _inline(eqn, invals, site, ledger):
    named = [(name, value) for name, value in eqn.params.items() if _is_jaxpr(value)]
    if len(named) != 1 or len(named[0][1].invars) != len(invals):
        raise CertificateTransportError(f'cannot inline {eqn.primitive.name!r} at {site}')
    name, body = named[0]
    return _run(body, _consts(body), invals, site + (name,), ledger)


# Study-only hook. The comparison harness runs the candidate beside a product
# that still registers its primitives in JAX's private checkify table; the
# harness sets this to a context that hides those entries from the public
# checkify call below. A redesigned product has no such entries and no hook.
_foreign_check_scope = contextlib.nullcontext


def certified(function):
    """Return ``(Evidence, values)`` for the complete transformed ``function``.

    Apply it last, after jvp / vjp / grad, exactly where the current
    ``checked_native`` is applied; map it with ``jax.vmap`` afterwards.
    """
    def run(*args):
        with jax.enable_x64(True):
            program, shape = jax.make_jaxpr(function, return_shape=True)(*args)
            flat_args, _ = jax.tree.flatten(args)
            _, output_tree = jax.tree.flatten(shape)

            def functional(*flat):
                outs, ledger = _run(program, _consts(program), flat, (), {})
                return _export(ledger), tuple(outs)

            with _foreign_check_scope():
                foreign, (ledger, values) = public_checkify.checkify(
                    functional, errors=public_checkify.user_checks)(*flat_args)
            return Evidence(ledger, foreign), output_tree.unflatten(values)
    return run


def certified_sequential_primal(function):
    """Keep scalar early exits under vmap and still deliver the certificates.

    The scalar body is certified before ``sequential_vmap`` hides it. Its
    ledger leaves the mapped call as ordinary outputs and is attached to the
    first leaf of the returned packet; foreign checks go back through the
    public ``checkify.check_error``.
    """
    inner = certified(function)

    def primal(*operands):
        with jax.enable_x64(True):
            program, shape = jax.make_jaxpr(inner, return_shape=True)(*operands)
            constants = _consts(program)
            flat_operands, _ = jax.tree.flatten(operands)
            _, output_tree = jax.tree.flatten(shape)
            count = len(constants)
            explicit = lambda *flat: tuple(jax.core.eval_jaxpr(program, flat[:count], *flat[count:]))
            mapped = jax.custom_batching.sequential_vmap(explicit)
            evidence, packet = output_tree.unflatten(mapped(*constants, *flat_operands))
            public_checkify.check_error(evidence.foreign)
            leaves, tree = jax.tree.flatten(packet)
            leaves[0] = attach_ledger(leaves[0], evidence.ledger)
            return tree.unflatten(leaves)
    return primal


# --------------------------------------------------------------------------
# Host projection (the replacement of sparse_certificates / derivative_availability).
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Certificate:
    site: tuple
    role: str
    stage: int
    status: Any
    eta: Any
    residual_numerator: Any
    residual_denominator: Any
    performed: Any
    present: Any
    derivative_order: int
    dependency: tuple
    basis_kind: str | None = None
    basis_indices: tuple = ()
    work_index: int | None = None
    frequency_source_index: int | None = None


def _lanes(present, like):
    present = jnp.asarray(present)
    return jnp.broadcast_to(present.reshape(present.shape + (1,) * (like.ndim - present.ndim)), like.shape)


def ledger_certificates(ledger: Ledger, *, basis_kind=None, basis_indices=(), work_index=None,
                        frequency_source_index=None):
    with jax.enable_x64(True):
        rows = []
        for key, slot in ledger.items():
            if key.kind != 'equation':
                continue
            status, eta, numerator, denominator, performed = slot.values
            present = _lanes(slot.present, status)
            rows.append(Certificate(key.site, key.role, key.stage,
                jnp.where(present, status, jnp.int32(0)),
                *(jnp.where(present, value, jnp.asarray(jnp.nan, dtype=value.dtype))
                  for value in (eta, numerator, denominator)),
                present & performed, present, key.derivative_order, key.dependency,
                basis_kind, tuple(basis_indices), work_index, frequency_source_index))
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class Availability:
    site: tuple
    dependency: tuple
    reason: str
    available: Any
    present: Any


def ledger_availability(ledger: Ledger):
    with jax.enable_x64(True):
        return tuple(Availability(key.site, key.dependency, key.reason, slot.values[0],
                                  _lanes(slot.present, slot.values[0]))
                     for key, slot in ledger.items() if key.kind == 'availability')
