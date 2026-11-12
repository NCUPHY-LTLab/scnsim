"""The evidence of a checked frame, as its consumers use it.

``checked_native`` returns ``(evidence, values)`` for a complete transformed
function. The evidence holds the certificate of every equation of the frame,
its derivative-availability facts and the checks SCNSim does not own. To a
consumer it is opaque: it is handed to the operations of this module, and to
JAX as a pytree of arrays, and to nothing else.

What a consumer may rely on:

- The evidence is a pytree of arrays. It is an output of ``jax.jit`` and an
  input of a later compiled call, and ``vmap``, ``scan`` and ``lax.map`` stack
  it with their lane axes in front.
- On the host, ``lane`` and ``lane_shapes`` read lanes, ``certificates`` and
  ``availability`` project rows and facts, ``for_dependency`` keeps the
  entries of one dependency, ``raise_foreign`` and ``raise_any_failure``
  raise, and ``structure`` is the text a compile key needs.
- Inside a traced program, ``absent`` is the evidence of a frame that did not
  run and ``any_failed`` tells whether a frame holds a failure.

Not owned here: building a checked frame (``checked_native``), the equation
primitives and their certificates, and the status reducers that decide the
kind, stage and order of a failure.

Today the evidence is JAX's ``checkify.Error`` with SCNSim's effects in it.
This module and ``native_sparse.py`` are the only places that read that
object. The last step of Issue 48 gives each operation a body on the
``Evidence`` of ``certificate_transport.py``; its consumers stay as they are.
"""
from __future__ import annotations

import enum

import jax
import jax.numpy as jnp
from jax._src import checkify
from jax.tree_util import PyTreeDef
import numpy as np

from .native_sparse import (
    ArchivedPrimalCheck, DerivativeAvailabilityEffect, SparseCertificateEffect,
    SparseEquationCheck, derivative_availability, root_error_frame, sparse_certificates,
)


# ---------------------------------------------------------------------------
# Lanes, on the host.
# ---------------------------------------------------------------------------
def lane(evidence, index):
    """The evidence of lane ``index`` of the leading axis, as host arrays."""
    return jax.tree.map(lambda value: np.asarray(value)[index], evidence)


def lane_shapes(evidence) -> tuple[tuple[int, ...], ...]:
    """The lane axes of every entry of ``evidence``, as one shape per entry.

    The entries of a frame that was mapped have the axes of the mapping, in
    order; those of a frame that was not have the shape ``()``. A consumer
    compares them with the locators it is about to attach to the lanes.
    """
    return tuple(tuple(np.shape(code)) for code in evidence._code.values())


# ---------------------------------------------------------------------------
# Inside a traced program.
# ---------------------------------------------------------------------------
def absent(shape):
    """The evidence of a frame that did not run.

    ``shape`` is what ``jax.eval_shape`` returns for the evidence of a checked
    function. No equation and no fact is present in the result, and nothing in
    it has failed.
    """
    def zeros(leaf):
        if not isinstance(leaf, jax.ShapeDtypeStruct):
            raise TypeError('ordered tail outcome must contain only typed array leaves')
        return jnp.zeros(leaf.shape, dtype=leaf.dtype)

    def inactive_code(leaf):
        if not isinstance(leaf, jax.ShapeDtypeStruct):
            raise TypeError('ordered tail Error codes must contain only typed array leaves')
        # The code JAX gives a check that cannot fail. The projections read it
        # as "not present", so a lane that did not run has no rows.
        return jnp.full(leaf.shape, checkify._INACTIVE_CODE, dtype=leaf.dtype)

    return shape._replace(
        _pred=jax.tree.map(zeros, shape._pred),
        _code={effect: inactive_code(code) for effect, code in shape._code.items()},
        _payload=jax.tree.map(zeros, shape._payload),
    )


def any_failed(evidence):
    """Whether ``evidence`` holds a failure, as a traced boolean.

    Today that is any failure, a check SCNSim does not own included.
    """
    failed = jnp.asarray(False)
    for predicate in evidence._pred.values():
        failed = failed | jnp.any(predicate)
    return failed


# ---------------------------------------------------------------------------
# Rows and facts, on the host.
# ---------------------------------------------------------------------------
def certificates(evidence, *, basis_kind=None, basis_indices=(), work_index=None,
                 frequency_source_index=None):
    """The certificate rows of ``evidence``, successful equations included.

    A row keeps the lane axes of its frame. Where an equation did not run its
    status is 0, its residual numbers are NaN and ``performed`` is false. The
    locators say which work row, frequency and derivative basis the frame
    belongs to; they are copied onto every row.
    """
    return sparse_certificates(
        evidence, basis_kind=basis_kind, basis_indices=basis_indices,
        work_index=work_index, frequency_source_index=frequency_source_index)


def availability(evidence):
    """The derivative-availability facts of ``evidence``.

    ``present`` tells where the rule that states a fact ran. Only a fact that
    is present can make a derivative unavailable.
    """
    return derivative_availability(evidence)


def for_dependency(evidence, dependency):
    """The evidence of the equations and facts of one ``dependency`` alone.

    A value frame holds several root occurrences. Each one continues from its
    own packet, so the frame that completes it is handed its entries only.
    """
    return root_error_frame(evidence, dependency)


# ---------------------------------------------------------------------------
# Compile keys.
# ---------------------------------------------------------------------------
def structure(evidence) -> str:
    """The static structure of ``evidence``, or of a tuple of them, as text.

    A program that takes evidence as an operand is compiled for that
    structure, so the text belongs in its compile key.
    """
    return str(jax.tree.structure(evidence))


# ---------------------------------------------------------------------------
# Raising on the host.
# ---------------------------------------------------------------------------
class FrameKind(enum.Enum):
    """What the caller of ``raise_foreign`` holds.

    The current transport keeps owned and foreign failures in one object, and
    each of its readers tells them apart in its own way. The kind names the
    reader, so that each one keeps its rule. The ledger of Issue 48 keeps the
    foreign checks apart and raises them first for every kind.
    """

    # One frame without lanes, straight from its call. Its first failure
    # decides: nothing is raised when that is an equation failure, even if a
    # foreign check failed after it. Evidence with lanes is refused, so any
    # failure in it is raised.
    SCALAR = 'scalar'
    # Frames stacked along lane axes, straight from their call. The first
    # failure of every failing lane decides in the same way.
    LANES = 'lanes'
    # A frame, or lanes of one, that a host projection reads as a primal
    # value. A foreign check is raised whatever else failed. So is a
    # derivative-availability fact that does not hold: the projection of a
    # primal value reads certificates only.
    PRIMAL_PROJECTION = 'primal_projection'
    # The same for a transformed frame, whose projection reads the
    # availability facts as well.
    DERIVATIVE_PROJECTION = 'derivative_projection'


def raise_foreign(evidence, *, frame: FrameKind) -> None:
    """Raise a failed check that SCNSim does not own, with the check's own message.

    The exception is JAX's ``JaxRuntimeError``. An equation failure is
    evidence and is not raised here: its kind, stage and order come from the
    status reducers of the frame's values.
    """
    if frame is FrameKind.SCALAR:
        _raise_unless_an_equation_failed_first(evidence)
    elif frame is FrameKind.LANES:
        _raise_unless_an_equation_failed_first_in_each_lane(evidence)
    elif frame is FrameKind.PRIMAL_PROJECTION:
        _raise_failures_outside(evidence, owned=(SparseCertificateEffect,))
    elif frame is FrameKind.DERIVATIVE_PROJECTION:
        _raise_failures_outside(
            evidence, owned=(SparseCertificateEffect, DerivativeAvailabilityEffect))
    else:
        raise TypeError('raise_foreign needs the FrameKind of its evidence')


def raise_any_failure(evidence) -> None:
    """Raise the first failure of a frame, an equation failure included.

    For a frame without lanes that is not expected to fail at all. The
    exception is JAX's ``JaxRuntimeError`` with the message of the failure.
    """
    if evidence.get_exception() is not None:
        evidence.throw()


def _raise_unless_an_equation_failed_first(evidence):
    exception = evidence.get_exception()
    # An equation failure of a sequentially mapped body arrives archived.
    while isinstance(exception, ArchivedPrimalCheck):
        exception = exception.original_tree.unflatten(exception.payload[:-2])
    if exception is not None and not isinstance(exception, SparseEquationCheck):
        evidence.throw()


def _raise_unless_an_equation_failed_first_in_each_lane(evidence):
    if _may_hold_a_failure(evidence):
        exception = evidence.get_exception()
        if exception is not None and not _is_equation_failure(exception):
            evidence.throw()


def _may_hold_a_failure(evidence) -> bool:
    """Skip building an exception for evidence in which nothing failed.

    Entries of one lane shape with no failure among them cannot raise. Entries
    of different shapes are left to JAX's own reading.
    """
    predicates = tuple(np.asarray(value) for value in evidence._pred.values())
    if len({value.shape for value in predicates}) > 1:
        return True
    return any(bool(value.any()) for value in predicates)


def _is_equation_failure(exception) -> bool:
    """Whether ``exception`` holds equation failures only, in every lane it names."""
    if isinstance(exception, SparseEquationCheck):
        return True
    if type(exception) is checkify.BatchedError:
        lanes = exception.error_mapping
        return (type(lanes) is dict and bool(lanes)
                and all(type(index) is tuple and all(type(i) is int and i >= 0 for i in index)
                        and _is_equation_failure(failure)
                        for index, failure in lanes.items()))
    if type(exception) is ArchivedPrimalCheck:
        if not isinstance(exception.original_tree, PyTreeDef) or len(exception.payload) < 2:
            return False
        try:
            original = exception.original_tree.unflatten(exception.payload[:-2])
        except (TypeError, ValueError):
            return False
        return _is_equation_failure(original)
    return False


def _raise_failures_outside(evidence, *, owned):
    """Raise a failure of any entry that is not of an ``owned`` kind."""
    foreign = evidence._replace(_pred={
        effect: np.zeros_like(predicate, dtype=bool) if isinstance(effect, owned) else predicate
        for effect, predicate in evidence._pred.items()})
    if foreign.get_exception() is not None:
        foreign.throw()
