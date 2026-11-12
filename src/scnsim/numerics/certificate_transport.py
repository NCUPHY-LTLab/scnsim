"""Equation certificates leave a compiled JAX program as ordinary outputs.

Every SCNSim equation primitive returns its certificate beside its value: a
status, eta, the residual numerator and denominator, and whether the solve was
performed. This module brings those certificates to the boundary of a complete
transformed function, wherever the equation sits: at top level, inside
``cond``, ``while_loop`` or ``scan``, behind a sequential map, or in a tangent
or adjoint solve that JAX's own derivative rules created.

- ``certified`` is the pass. It traces the finished function, evaluates the
  traced program equation by equation and records every certificate in a
  ledger. Control flow that holds an equation is emitted again through
  ``jax.lax`` with the ledger slots as extra branch outputs or loop carry.
- ``Ledger`` is a pytree of static entry keys and array slots. It is an output
  and an input of ``jax.jit``, ``jax.tree.map`` slices one lane of it, and
  ``absent_evidence`` builds the value of a frame that did not run.
- ``derivative_gate`` and ``attach_evidence`` are identities that keep a fact
  on the data path where no equation output carries it.
- ``ledger_certificates`` and ``ledger_availability`` read a returned ledger
  on the host, as the rows the current transport gives.

Checks SCNSim does not own (``checkify.check``, ``checkify.debug_check``) stay
with JAX's public ``checkify`` and return as its public ``Error``.

Only public JAX interfaces are used. What the pass relies on that JAX does not
document: the parameter names of ``cond``, ``while`` and ``scan`` equations,
the operand order of ``scan``, and the primitive name ``custom_vmap_call``.

Not owned here: the equation primitives and their rules, the status reducers
that decide failure kind, stage and order, and what consumers make of the
rows. Issue 48 replaces the transport in ``native_sparse.py`` with this
module; until its last step both exist and no caller uses this one.
"""
from __future__ import annotations

from collections.abc import Iterator
import contextlib
import contextvars
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import checkify
from jax.extend.core import Jaxpr, Literal, Primitive, jaxpr_as_fun
from jax.extend.core.primitives import cond_p, custom_jvp_call_p, jit_p, scan_p, while_p
from jax.interpreters import ad, batching, mlir


class CertificateTransportError(TypeError):
    """An equation sits where the pass has no rule to bring its certificate out."""


# ---------------------------------------------------------------------------
# The pass.
# ---------------------------------------------------------------------------
def certified(function):
    """Return a function that gives ``(Evidence, values)`` for ``function``.

    Pass the complete transformed function, an executed reverse pullback
    included: only then are its tangent and adjoint equations part of the
    program. Map the result with ``jax.vmap`` for lanes; each lane then has
    its own entries.
    """
    def certified_function(*operands):
        # One precision scope for the trace and for what is emitted from it,
        # whatever the caller's: 64-bit operands keep their dtype and 32-bit
        # operands stay 32-bit.
        with jax.enable_x64(True):
            program, output_shape = jax.make_jaxpr(function, return_shape=True)(*operands)
            flat_operands, _ = jax.tree.flatten(operands)
            _, output_tree = jax.tree.flatten(output_shape)

            def with_ledger(*flat):
                outputs, slots = _evaluate(program, flat, path=(), slots={})
                return _ledger_of(slots), tuple(outputs)

            foreign, (ledger, outputs) = _functionalize_foreign_checks(with_ledger, flat_operands)
            return Evidence(ledger, foreign), output_tree.unflatten(outputs)
    return certified_function


def certified_sequential_primal(function):
    """Keep a scalar body's early exit under ``vmap`` and still return its evidence.

    ``jax.custom_batching.sequential_vmap`` runs the body once per lane, so a
    Newton loop stops when its own lane has converged. It also hides the body
    from an enclosing pass. The body is therefore certified here, before it is
    hidden: its evidence leaves the mapped call as ordinary outputs and is
    attached to the packet the body returns, where the enclosing ``certified``
    frame reads it with its lane axis. A caller outside any certified frame
    gets the packet alone.
    """
    certified_body = certified(function)

    def primal(*operands):
        with jax.enable_x64(True):
            program, output_shape = jax.make_jaxpr(certified_body, return_shape=True)(*operands)
            flat_operands, _ = jax.tree.flatten(operands)
            _, output_tree = jax.tree.flatten(output_shape)
            # sequential_vmap refuses a batched value that the mapped function
            # closes over, so every constant of the body becomes an operand.
            constants = tuple(program.consts)

            def body(*flat):
                return tuple(jax.core.eval_jaxpr(program, flat[:len(constants)],
                                                 *flat[len(constants):]))

            mapped = jax.custom_batching.sequential_vmap(body)
            evidence, packet = output_tree.unflatten(mapped(*constants, *flat_operands))
            return attach_evidence(packet, evidence)
    return primal


def _functionalize_foreign_checks(function, operands):
    """Let public ``checkify`` return the checks SCNSim does not own as its ``Error``."""
    with _recording_equations():
        return checkify.checkify(function, errors=checkify.user_checks)(*operands)


def _evaluate(program, operands, *, path, slots):
    """Evaluate ``program`` as ``jax.core.eval_jaxpr`` does, recording what it holds.

    ``path`` is the position of ``program`` in the traced function and
    ``slots`` what has been recorded so far. Returns the outputs of the
    program and the slots after it.
    """
    values = dict(zip(program.constvars, program.consts, strict=True))
    values.update(zip(program.invars, operands, strict=True))

    def read(atom):
        return atom.val if isinstance(atom, Literal) else values[atom]

    for index, equation in enumerate(program.eqns):
        inputs = [read(atom) for atom in equation.invars]
        outputs, slots = _evaluate_equation(equation, inputs, _equation_site(path, index), slots)
        values.update(zip(equation.outvars, outputs, strict=True))
    return [read(atom) for atom in program.outvars], slots


def _evaluate_equation(equation, inputs, site, slots):
    """Evaluate one equation at ``site``; return its outputs and the slots after it."""
    primitive = equation.primitive
    if primitive in _equation_schemas:
        outputs = list(primitive.bind(*inputs, **equation.params))
        return outputs, _record_equation(slots, equation, site, outputs)
    if primitive is derivative_gate_p:
        value, unavailable = inputs
        return [value], _record_gate(slots, equation, site, unavailable)
    if primitive is evidence_attach_p:
        return _read_attached_evidence(equation, inputs, slots)
    if not _encloses_recorded(equation):
        return _rebind(equation, inputs), slots
    if primitive in _ENCLOSING_RULES:
        return _ENCLOSING_RULES[primitive](equation, inputs, site, slots)
    if primitive.name in _CERTIFIED_BEFORE_HIDDEN:
        return _rebind(equation, inputs), slots
    raise CertificateTransportError(
        f'{primitive.name!r} at {site} encloses an equation and has no certificate rule: '
        'certify its body before it is hidden, or add a rule')


def _rebind(equation, inputs):
    """Evaluate one equation unchanged, in the configuration it was traced in."""
    primitive = equation.primitive
    with equation.ctx.manager:
        outputs = primitive.bind(*inputs, **primitive.get_bind_params(equation.params))
    return list(outputs) if primitive.multiple_results else [outputs]


def _encloses_recorded(equation) -> bool:
    return any(_holds_recorded(program) for program in _programs_in(equation.params))


def _holds_recorded(program) -> bool:
    """Whether ``program`` holds, at any depth, something the pass records."""
    return any(_is_recorded(equation.primitive) or _encloses_recorded(equation)
               for equation in program.eqns)


def _is_recorded(primitive) -> bool:
    return (primitive in _equation_schemas or primitive is derivative_gate_p
            or primitive is evidence_attach_p)


def _programs_in(parameters) -> Iterator[Jaxpr]:
    """The programs an equation holds in its parameters, however they are nested."""
    for value in parameters.values() if isinstance(parameters, dict) else parameters:
        if isinstance(value, Jaxpr):
            yield value
        elif isinstance(value, (tuple, list, dict)):
            yield from _programs_in(value)


# ---------------------------------------------------------------------------
# Rules for the primitives that may enclose an equation. Each emits the
# primitive again through its public function, with the slots its body writes
# as extra outputs or carry.
# ---------------------------------------------------------------------------
def _cond(equation, inputs, site, slots):
    index, *operands = inputs
    branches = [(branch, _program_path(site, 'branches', ordinal))
                for ordinal, branch in enumerate(equation.params['branches'])]

    shapes = {}
    for branch, path in branches:
        shapes.update(_slot_shapes(
            lambda *values, branch=branch, path=path:
                _evaluate(branch, values, path=path, slots={})[1],
            *map(_shape_of, operands)))
    keys, carried = _carried_slots(slots, shapes)

    def emitted(branch, path):
        def function(carried, *values):
            outputs, written = _evaluate(branch, values, path=path, slots=dict(zip(keys, carried)))
            return tuple(outputs), tuple(written[key] for key in keys)
        return function

    outputs, carried = lax.switch(index, [emitted(*branch) for branch in branches],
                                  carried, *operands)
    return list(outputs), {**slots, **dict(zip(keys, carried))}


def _while(equation, inputs, site, slots):
    params = equation.params
    predicate, body = params['cond_jaxpr'], params['body_jaxpr']
    if _holds_recorded(predicate):
        raise CertificateTransportError(
            f'the predicate of the while_loop at {site} holds an equation: '
            'a predicate has no output that could carry its certificate')
    predicate_end = params['cond_nconsts']
    body_end = predicate_end + params['body_nconsts']
    predicate_constants = inputs[:predicate_end]
    body_constants = inputs[predicate_end:body_end]
    initial = inputs[body_end:]
    path = _program_path(site, 'body_jaxpr')

    def trip(carry, carried):
        return _evaluate(body, (*body_constants, *carry), path=path, slots=carried)

    shapes = _slot_shapes(lambda *carry: trip(carry, {})[1], *map(_shape_of, initial))
    keys, carried = _carried_slots(slots, shapes)
    continues = jaxpr_as_fun(predicate)

    def step(state):
        carry, carried = state
        outputs, written = trip(carry, dict(zip(keys, carried)))
        return tuple(outputs), tuple(written[key] for key in keys)

    carry, carried = lax.while_loop(
        lambda state: continues(*predicate_constants, *state[0])[0], step,
        (tuple(initial), carried))
    return list(carry), {**slots, **dict(zip(keys, carried))}


def _scan(equation, inputs, site, slots):
    params = equation.params
    body = params['jaxpr']
    constant_count, carry_count = _scan_operand_counts(equation)
    constants = inputs[:constant_count]
    initial = inputs[constant_count:constant_count + carry_count]
    mapped = inputs[constant_count + carry_count:]
    path = _program_path(site, 'jaxpr')

    def trip(carry, row, carried):
        return _evaluate(body, (*constants, *carry, *row), path=path, slots=carried)

    shapes = _slot_shapes(lambda carry, row: trip(carry, row, {})[1],
                          tuple(map(_shape_of, initial)), tuple(map(_row_shape_of, mapped)))
    keys, carried = _carried_slots(slots, shapes)

    def step(state, row):
        carry, carried = state
        outputs, written = trip(carry, row, dict(zip(keys, carried)))
        return ((tuple(outputs[:carry_count]), tuple(written[key] for key in keys)),
                tuple(outputs[carry_count:]))

    (carry, carried), stacked = lax.scan(
        step, (tuple(initial), carried), tuple(mapped),
        length=params['length'], reverse=params['reverse'], unroll=params['unroll'])
    return [*carry, *stacked], {**slots, **dict(zip(keys, carried))}


def _scan_operand_counts(equation) -> tuple[int, int]:
    """How many operands of a ``scan`` equation are constants and how many carry.

    The operands are constants, carry, then mapped inputs; the outputs are
    carry, then stacked outputs. A carry keeps its type through the body,
    while a mapped input and a stacked output have one more leading axis
    outside the body than inside. Comparing ranks therefore gives the counts
    without the bookkeeping parameters of ``scan``, which JAX keeps private.
    """
    body = equation.params['jaxpr']

    def rank(atom):
        return len(atom.aval.shape)

    if (len(body.invars), len(body.outvars)) != (len(equation.invars), len(equation.outvars)):
        raise CertificateTransportError('scan body and scan equation disagree in arity')
    carry_count = 0
    for inner, outer in zip(body.outvars, equation.outvars):
        if rank(inner) != rank(outer):
            break
        carry_count += 1
    mapped_count = 0
    for inner, outer in zip(reversed(body.invars), reversed(equation.invars)):
        if rank(outer) != rank(inner) + 1:
            break
        mapped_count += 1
    constant_count = len(equation.invars) - carry_count - mapped_count
    if constant_count < 0:
        raise CertificateTransportError('scan operands are not constants, carry and mapped inputs')
    return constant_count, carry_count


def _inline(equation, inputs, site, slots):
    """Evaluate the one program of a ``jit`` or ``custom_jvp_call`` equation in place."""
    held = [(name, value) for name, value in equation.params.items() if isinstance(value, Jaxpr)]
    if len(held) != 1 or len(held[0][1].invars) != len(inputs):
        raise CertificateTransportError(
            f'{equation.primitive.name!r} at {site} does not hold one program over its operands')
    name, body = held[0]
    return _evaluate(body, inputs, path=_program_path(site, name), slots=slots)


_ENCLOSING_RULES = {
    cond_p: _cond,
    while_p: _while,
    scan_p: _scan,
    # A custom JVP rule has already run when the pass sees the finished
    # program, so the call is the plain primal body.
    custom_jvp_call_p: _inline,
    jit_p: _inline,
}
# The body of a sequential map is certified before the map hides it
# (certified_sequential_primal); its evidence arrives through attach_evidence.
# JAX does not export this primitive, so it is known by name.
_CERTIFIED_BEFORE_HIDDEN = frozenset({'custom_vmap_call'})


def _slot_shapes(written_by, *operand_shapes):
    """The entries a body writes and the shape of each slot, from one abstract evaluation.

    ``written_by(*values)`` returns the slots the body writes when it starts
    with none.
    """
    keys = []

    def slots_in_key_order(*values):
        written = written_by(*values)
        keys[:] = written
        return tuple(written[key] for key in keys)

    return dict(zip(keys, jax.eval_shape(slots_in_key_order, *operand_shapes)))


def _carried_slots(slots, shapes):
    """The slots a body may write, in a fixed order: those held so far, else absent."""
    keys = sorted(shapes, key=_row_order)
    return keys, tuple(slots[key] if key in slots else Slot.absent(shapes[key]) for key in keys)


def _shape_of(value) -> jax.ShapeDtypeStruct:
    aval = jax.typeof(value)
    return jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)


def _row_shape_of(mapped) -> jax.ShapeDtypeStruct:
    """The shape of one row of an input that ``scan`` maps over its leading axis."""
    aval = jax.typeof(mapped)
    return jax.ShapeDtypeStruct(aval.shape[1:], aval.dtype, weak_type=aval.weak_type)


# ---------------------------------------------------------------------------
# The ledger: what a certified call returns beside its values.
# ---------------------------------------------------------------------------
class Evidence(NamedTuple):
    """The evidence of one certified frame."""
    ledger: Ledger  # SCNSim's certificates and derivative-availability facts.
    # jax.experimental.checkify.Error of the checks SCNSim does not own. Read
    # it with its own get() and throw().
    foreign: Any


@dataclass(frozen=True, slots=True)
class EquationKey:
    """The static identity of one equation's certificate."""
    site: tuple
    role: str
    stage: int
    derivative_order: int
    dependency: tuple
    # Shape and dtype text of each array of the slot; part of today's row order.
    descriptors: tuple
    # The entry reached this frame through attached evidence.
    archived: bool = False


@dataclass(frozen=True, slots=True)
class AvailabilityKey:
    """The static identity of one derivative-availability fact."""
    site: tuple
    dependency: tuple
    reason: str
    descriptors: tuple
    archived: bool = False


class Slot(NamedTuple):
    """The arrays of one ledger entry."""
    failed: Any  # Bool: the slot holds a failure, which later executions do not replace.
    present: Any  # Bool: the site was executed.
    # An equation: status, eta, residual numerator, residual denominator,
    # performed. An availability fact: available.
    values: tuple

    @classmethod
    def absent(cls, shape: Slot) -> Slot:
        """The slot of a site that was not executed, with the arrays of ``shape``."""
        return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shape)

    def followed_by(self, later: Slot) -> Slot:
        """The slot after a further execution of its site.

        A site inside a loop is executed many times and keeps one slot: its
        first failure, otherwise its latest execution.
        """
        def kept(earlier, latest):
            return jnp.where(self.failed, earlier, latest)

        return Slot(self.failed | later.failed, kept(self.present, later.present),
                    tuple(kept(earlier, latest) for earlier, latest
                          in zip(self.values, later.values, strict=True)))


@jax.tree_util.register_pytree_node_class
class Ledger:
    """The certificates and availability facts of one certified frame.

    The keys are static and the slots are arrays, so a ledger crosses
    ``jax.jit`` in both directions and ``jax.tree.map`` reaches every array.
    Entries are in row order.
    """

    def __init__(self, keys, slots):
        self.keys, self.slots = tuple(keys), tuple(slots)

    def tree_flatten(self):
        return (self.slots,), self.keys

    @classmethod
    def tree_unflatten(cls, keys, children):
        return cls(keys, children[0])

    def entries(self):
        """Each key with its slot."""
        return zip(self.keys, self.slots, strict=True)

    def any_failed(self):
        """Whether any entry holds a failure; usable inside a traced program."""
        failed = jnp.bool_(False)
        for slot in self.slots:
            failed = failed | jnp.any(slot.failed)
        return failed


def absent_evidence(shape):
    """The evidence of a frame that did not run.

    ``shape`` is what ``jax.eval_shape`` returns for the evidence of a
    certified function. No site is present and no foreign check has failed.
    """
    def zeros(leaf):
        if not isinstance(leaf, jax.ShapeDtypeStruct):
            raise TypeError('absent evidence is built from the shapes jax.eval_shape returns')
        return jnp.zeros(leaf.shape, leaf.dtype)

    return jax.tree.map(zeros, shape)


def _ledger_of(slots) -> Ledger:
    keys = sorted(slots, key=_row_order)
    return Ledger(keys, tuple(slots[key] for key in keys))


# ---------------------------------------------------------------------------
# Reading a returned ledger on the host.
# ---------------------------------------------------------------------------
# What ``available`` reads where a fact is absent: no rule said the derivative
# is unavailable.
AVAILABLE_WHERE_ABSENT = True
_BASIS_KINDS = ('parameter', 'output_row', 'direction', 'functional')


def ledger_certificates(ledger: Ledger, *, basis_kind=None, basis_indices=(), work_index=None,
                        frequency_source_index=None):
    """The certificate rows of ``ledger``, successful equations included.

    A row keeps the lane axes of its frame. Where a site was not executed its
    status is 0, its residual numbers are NaN and ``performed`` is false.
    """
    # The row types stay with the current transport until Issue 48 removes it.
    # That module imports this one, so this import cannot be at the top.
    from .native_sparse import SparseCertificate

    basis_indices = _checked_basis(basis_kind, basis_indices)
    # A projection can run after the execution thread has left its precision
    # scope. The local scope keeps the returned bits: 32-bit stays 32-bit.
    with jax.enable_x64(True):
        rows = []
        for key, slot in ledger.entries():
            if not isinstance(key, EquationKey):
                continue
            status, eta, numerator, denominator, performed = slot.values
            if basis_kind is not None and (status.ndim == 0 or status.shape[0] != len(basis_indices)):
                raise ValueError('mapped certificate leading axis disagrees with actual derivative basis')
            present = _over_lanes(slot.present, status)
            rows.append(SparseCertificate(
                key.site, key.role, key.stage,
                jnp.where(present, status, jnp.int32(0)),
                *(jnp.where(present, value, jnp.asarray(jnp.nan, dtype=value.dtype))
                  for value in (eta, numerator, denominator)),
                present & performed, present, key.derivative_order, key.dependency,
                basis_kind, basis_indices, work_index, frequency_source_index))
        return tuple(rows)


def ledger_availability(ledger: Ledger):
    """The derivative-availability facts of ``ledger``.

    ``present`` tells where the rule that states the fact ran. Where it did
    not, ``available`` reads ``AVAILABLE_WHERE_ABSENT``.
    """
    from .native_sparse import DerivativeAvailability  # See ledger_certificates.

    with jax.enable_x64(True):
        facts = []
        for key, slot in ledger.entries():
            if not isinstance(key, AvailabilityKey):
                continue
            available, = slot.values
            present = _over_lanes(slot.present, available)
            facts.append(DerivativeAvailability(
                key.site, key.dependency, key.reason,
                jnp.where(present, available, AVAILABLE_WHERE_ABSENT), present))
        return tuple(facts)


def _checked_basis(basis_kind, basis_indices) -> tuple:
    basis_indices = tuple(basis_indices)
    if basis_kind is not None and basis_kind not in _BASIS_KINDS:
        raise ValueError('certificate basis kind must identify parameter, output row, direction or functional')
    if basis_kind is None and basis_indices:
        raise ValueError('certificate basis indices require their kind')
    return basis_indices


def _over_lanes(present, like):
    """``present`` with the shape of ``like``.

    ``present`` has the lane axes of the frame; the arrays of a batched
    equation have further axes behind them.
    """
    present = jnp.asarray(present)
    trailing = (1,) * (like.ndim - present.ndim)
    return jnp.broadcast_to(present.reshape(present.shape + trailing), like.shape)


# ---------------------------------------------------------------------------
# Which primitives are equations, and how each lays out its certificates.
# ---------------------------------------------------------------------------
CERTIFICATE_ARRAYS = 5  # status, eta, residual numerator, residual denominator, performed


class CertificateGroup(NamedTuple):
    """How one equation of a primitive that solves several is labelled.

    ``None`` inherits the parameter the primitive was bound with. A role of
    ``'derivative'`` and an order of ``'next'`` name the next implicit
    equation: one order above the bound one, with the role that order has.
    """
    role: str | None
    stage: int | None
    derivative_order: int | str | None


class _EquationSchema(NamedTuple):
    value_count: int
    groups: tuple[CertificateGroup, ...]


_equation_schemas: dict[Primitive, _EquationSchema] = {}


def register_equation(primitive, *, value_count, certificate_groups=()):
    """Tell the pass that ``primitive`` is an equation that returns certificates.

    Its outputs start with ``value_count`` value arrays, followed by five
    certificate arrays for each equation it solves, in order. A primitive
    that solves one equation leaves ``certificate_groups`` empty and is
    labelled by its own parameters. Only this static layout is stored.
    """
    groups = tuple(tuple(group) for group in certificate_groups)
    if any(len(group) != 3 for group in groups):
        raise TypeError('equation certificate group requires role, stage and derivative order')
    _equation_schemas[primitive] = _EquationSchema(
        value_count, tuple(CertificateGroup(*group) for group in groups))


# ---------------------------------------------------------------------------
# Recording into the slots.
# ---------------------------------------------------------------------------
def _record_equation(slots, equation, site, outputs):
    schema = _equation_schemas[equation.primitive]
    groups = schema.groups or (CertificateGroup(None, None, None),)
    if len(outputs) != schema.value_count + CERTIFICATE_ARRAYS * len(groups):
        raise TypeError('owned equation certificate schema disagrees with primitive outputs')
    for ordinal, group in enumerate(groups):
        start = schema.value_count + CERTIFICATE_ARRAYS * ordinal
        certificate = tuple(outputs[start:start + CERTIFICATE_ARRAYS])
        role, stage, derivative_order = _group_label(group, equation.params)
        key = EquationKey(
            _group_site(site, ordinal) if schema.groups else site, role, stage, derivative_order,
            equation.params.get('dependency', ()), _descriptors(certificate))
        status = certificate[0]
        slots = _record(slots, key, Slot(jnp.any(status != 0), jnp.bool_(True), certificate))
    return slots


def _group_label(group: CertificateGroup, params) -> tuple[str, int, int]:
    bound_order = params.get('derivative_order', 0)
    if group.derivative_order == 'next':
        derivative_order = bound_order + 1
    elif group.derivative_order is None:
        derivative_order = bound_order
    else:
        derivative_order = group.derivative_order
    if group.role == 'derivative':
        if params.get('frequency_jet', False):
            role = 'frequency_tangent'
        else:
            role = 'tangent' if derivative_order == 1 else 'mixed'
    else:
        role = params['role'] if group.role is None else group.role
    stage = params['stage'] if group.stage is None else group.stage
    return role, stage, derivative_order


def _record_gate(slots, equation, site, unavailable):
    key = AvailabilityKey(site, equation.params['dependency'], equation.params['reason'],
                          _descriptors((unavailable,)))
    return _record(slots, key, Slot(unavailable, jnp.bool_(True), (~unavailable,)))


def _read_attached_evidence(equation, inputs, slots):
    """Record attached evidence in this frame and return the packet it travelled with."""
    packet_count = equation.params['packet_count']
    evidence = equation.params['evidence_tree'].unflatten(inputs[packet_count:])
    for key, slot in evidence.ledger.entries():
        # The attached arrays have a lane axis. Which execution a loop of this
        # frame keeps is decided for all lanes together, as for any other slot.
        archived = replace(key, archived=True, descriptors=_archived_descriptors(slot))
        slots = _record(slots, archived, Slot(jnp.any(slot.failed), slot.present, slot.values))
    if jax.tree.leaves(evidence.foreign):
        checkify.check_error(evidence.foreign)
    return list(inputs[:packet_count]), slots


def _record(slots, key, execution: Slot):
    """``slots`` after one more execution of the entry ``key``."""
    earlier = slots.get(key)
    return {**slots, key: execution if earlier is None else earlier.followed_by(execution)}


# ---------------------------------------------------------------------------
# Site labels and row order, as the current transport defines them. Issue 48
# (decision 1) redefines both in its last step; only this section changes then.
# ---------------------------------------------------------------------------
def _equation_site(path, index) -> tuple:
    """The label of equation ``index`` of the program at ``path``.

    Today ``index`` counts all equations of the traced program, JAX's own
    included.
    """
    return path + (index,)


def _program_path(site, parameter, ordinal=None) -> tuple:
    """The path of a program that the equation at ``site`` holds in ``parameter``."""
    return site + (parameter,) if ordinal is None else site + (parameter, ordinal)


def _group_site(site, ordinal) -> tuple:
    """The label of equation ``ordinal`` of a primitive that solves several."""
    return site + ('equation', ordinal)


# Today's rows are sorted by the text of the check and effect classes of the
# current transport, then by the slot's array descriptors, and only then by
# site, dependency and the remaining labels. The names are kept as text so
# that stored rows keep their order while both transports exist.
_CURRENT_TRANSPORT = 'scnsim.numerics.native_sparse'
_CURRENT_CLASS_NAMES = {  # (key type, archived) -> (check class, effect class)
    (EquationKey, False): ('SparseEquationCheck', 'SparseCertificateEffect'),
    (AvailabilityKey, False): ('DerivativeUnavailableCheck', 'DerivativeAvailabilityEffect'),
    (EquationKey, True): ('ArchivedPrimalCheck', 'ArchivedSparseEffect'),
    (AvailabilityKey, True): ('ArchivedPrimalCheck', 'ArchivedAvailabilityEffect'),
}


def _row_order(key: EquationKey | AvailabilityKey) -> tuple:
    """The sort key that puts the entries of a frame in row order."""
    check, effect = _CURRENT_CLASS_NAMES[type(key), key.archived]
    if isinstance(key, EquationKey):
        labels = (key.role, key.stage, key.derivative_order, '')
    else:
        labels = ('', -1, -1, key.reason)
    return (f"<class '{_CURRENT_TRANSPORT}.{check}'>", key.descriptors, _CURRENT_TRANSPORT, effect,
            repr(key.site), repr(key.dependency), *labels)


def _descriptors(arrays) -> tuple:
    return tuple(_descriptor(jax.typeof(array).shape, jax.typeof(array).dtype) for array in arrays)


def _archived_descriptors(slot: Slot) -> tuple:
    """The descriptors of an attached slot.

    The current transport archives two more arrays with a slot, the lane
    presence and the lane check code (64-bit in the pass's precision scope),
    and they take part in its row order.
    """
    lanes = jax.typeof(slot.present).shape
    return _descriptors(slot.values) + (_descriptor(lanes, 'bool'), _descriptor(lanes, 'int64'))


def _descriptor(shape, dtype) -> tuple:
    return tuple(shape), str(dtype)


# ---------------------------------------------------------------------------
# Standing beside the current transport.
# ---------------------------------------------------------------------------
_pass_is_recording = contextvars.ContextVar('scnsim_certificate_pass_is_recording', default=False)


def pass_records_equations() -> bool:
    """Whether the pass is functionalizing a program whose equations it has recorded.

    The current transport registers rules for the equation primitives in
    JAX's check table. Public ``checkify`` runs them too, and they would
    return an equation failure as if it were a foreign check. They ask this
    function and stand down. Issue 48 removes them, and this function, in its
    last step.
    """
    return _pass_is_recording.get()


@contextlib.contextmanager
def _recording_equations():
    # JAX caches the functionalized form of a loop body or branch by the
    # program object, not by this state. That is safe here: a body that holds
    # an equation was emitted by the pass itself, so no other caller has it.
    token = _pass_is_recording.set(True)
    try:
        yield
    finally:
        _pass_is_recording.reset(token)


# ---------------------------------------------------------------------------
# Two identity primitives that keep a fact on the data path. They are pure:
# a JAX effect would have to be admitted to JAX's private allow-lists before
# it could appear inside control flow or a custom derivative.
# ---------------------------------------------------------------------------
def derivative_gate(value, available, *, dependency, reason):
    """Return ``value`` unchanged and state whether its derivative is available.

    Call it in a derivative rule, on a value the rule returns or feeds into
    its tangent equation: the fact is recorded only where the rule ran and
    travels with that value. ``dependency`` names what the fact is about;
    ``reason`` says why the derivative is unavailable where ``available`` is
    false. ``available`` is one boolean; lanes of an inner ``vmap`` share one
    fact, which holds only if it holds in every lane.
    """
    # Two equations, `not` and the gate: as many as the current transport's
    # derivative_available emits, so the equations around a call site keep
    # their position when it changes form.
    if jnp.shape(available) != ():
        raise TypeError('derivative gate takes one boolean, not an array of them')
    unavailable = ~jnp.asarray(available, dtype=jnp.bool_)
    return derivative_gate_p.bind(value, unavailable, dependency=tuple(dependency), reason=reason)


def attach_evidence(packet, evidence: Evidence):
    """Return ``packet`` unchanged, carrying ``evidence`` to the enclosing certified frame.

    The evidence travels with every leaf of ``packet``, so it is read wherever
    any of them is used. Outside a certified frame it is dropped.
    """
    if not isinstance(evidence, Evidence):
        raise TypeError('attach_evidence carries the Evidence a certified call returns')
    leaves, packet_tree = jax.tree.flatten(packet)
    if not leaves:
        raise CertificateTransportError('evidence needs a value to travel with')
    evidence_leaves, evidence_tree = jax.tree.flatten(evidence)
    carried = evidence_attach_p.bind(*leaves, *evidence_leaves, packet_count=len(leaves),
                                     evidence_tree=evidence_tree)
    return packet_tree.unflatten(carried)


derivative_gate_p = Primitive('scnsim_derivative_gate')
evidence_attach_p = Primitive('scnsim_evidence_attach')
evidence_attach_p.multiple_results = True


def _gate_value(value, unavailable, *, dependency, reason):
    return value


def _gate_jvp(primals, tangents, **params):
    return derivative_gate_p.bind(*primals, **params), tangents[0]


def _gate_transpose(cotangent, value, unavailable, **params):
    return (cotangent if ad.is_undefined_primal(value) else None), None


def _gate_batch(operands, axes, **params):
    value, unavailable = operands
    value_axis, unavailable_axis = axes
    if unavailable_axis is not None:
        # As the current transport does: the lanes of an inner vmap share one
        # fact, and the derivative is unavailable if it is in any of them.
        unavailable = jnp.any(unavailable, axis=unavailable_axis)
    return derivative_gate_p.bind(value, unavailable, **params), value_axis


derivative_gate_p.def_impl(_gate_value)
derivative_gate_p.def_abstract_eval(_gate_value)
mlir.register_lowering(derivative_gate_p, mlir.lower_fun(_gate_value, multiple_results=False))
ad.primitive_jvps[derivative_gate_p] = _gate_jvp
ad.primitive_transposes[derivative_gate_p] = _gate_transpose
batching.primitive_batchers[derivative_gate_p] = _gate_batch


def _attach_packet(*operands, packet_count, evidence_tree):
    return list(operands[:packet_count])


def _attach_jvp(primals, tangents, *, packet_count, **params):
    packet = evidence_attach_p.bind(*primals, packet_count=packet_count, **params)
    return packet, list(tangents[:packet_count])


def _attach_transpose(cotangents, *operands, packet_count, evidence_tree):
    packet = [cotangent if ad.is_undefined_primal(operand) else None
              for cotangent, operand in zip(cotangents, operands[:packet_count], strict=True)]
    evidence = [ad.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
                for operand in operands[packet_count:]]
    return packet + evidence


def _attach_batch(operands, axes, *, packet_count, evidence_tree):
    size = next(operand.shape[axis] for operand, axis in zip(operands, axes) if axis is not None)
    # The enclosing frame reads the evidence with its lane axis in front.
    evidence = [batching.bdim_at_front(operand, axis, size)
                for operand, axis in zip(operands[packet_count:], axes[packet_count:])]
    packet = evidence_attach_p.bind(*operands[:packet_count], *evidence,
                                    packet_count=packet_count, evidence_tree=evidence_tree)
    return packet, list(axes[:packet_count])


evidence_attach_p.def_impl(_attach_packet)
evidence_attach_p.def_abstract_eval(_attach_packet)
mlir.register_lowering(evidence_attach_p, mlir.lower_fun(_attach_packet, multiple_results=True))
ad.primitive_jvps[evidence_attach_p] = _attach_jvp
ad.primitive_transposes[evidence_attach_p] = _attach_transpose
batching.primitive_batchers[evidence_attach_p] = _attach_batch
