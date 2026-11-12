"""Which failures of a checked frame are raised on the host, and which stay evidence.

A failed equation is evidence. Its kind, stage and order come from the status
reducers of the frame's values, so the host must not raise it and must leave
its certificate as it is. A failed check that SCNSim does not own means an
internal invariant broke: it must be raised, with its own message and cause,
also when an equation failed in another lane of the same call.

Every frame here is a real one: a compiled checked call on the fixture system
of ``test_certificate_transport.py``, with a singular matrix where an equation
is to fail and ``checkify.check`` where a foreign check is to fail. The
evidence object is opaque. It is handed to ``scnsim.numerics.frame_evidence``
and compared as a pytree of arrays, and its rows are named by role, stage,
derivative order and dependency.

``CurrentTransportWrapperTests`` is the one exception. It tests a private
function of the interface's implementation on JAX's ``Error`` with wrappers
built by hand, because no checked call returns a malformed one. Issue 48
deletes that function and the class in its last step.
"""
from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np

from scnsim.numerics import frame_evidence
from scnsim.numerics.frame_evidence import FrameKind
from scnsim.numerics.native_sparse import checked_native, checked_sequential_primal
from tests import test_certificate_transport as pinned
from tests.jax_runtime import initialize_jax_through_scnsim


def setUpModule():
    # These tests call JAX directly. SCNSim has to initialize it first, or the
    # SCNSim tests that run after this module in the same process fail.
    initialize_jax_through_scnsim()


VALUES, SINGULAR, RHS = pinned.VALUES, pinned.SINGULAR, pinned.RHS
SOLVED, failed = pinned.SOLVED, pinned.failed
RUN, SINGULAR_MATRIX = pinned.RUN, pinned.SINGULAR_MATRIX

# The equation of a response: its dependency names the response it belongs to,
# and a singular matrix fails it with the status of its stage.
RESPONSE = ('responses', 0, ('Z',))
RESPONSE_STAGE = 7
RESPONSE_ROW = pinned.RowName('primal', RESPONSE_STAGE, 0, RESPONSE)
# The equation inside the Newton body of the fixture, which fails at stage 3.
NEWTON_ROW = pinned.RowName('primal', 3, 0, ('loop',))
CONVERGING = pinned.padded((RUN, RUN))
FAILING = pinned.padded((RUN, SINGULAR_MATRIX, RUN))
FOREIGN_MESSAGE = 'a check that SCNSim does not own'
EVERY_KIND = tuple(FrameKind)
# The kinds that accept evidence with lanes.
LANE_KINDS = (FrameKind.LANES, FrameKind.PRIMAL_PROJECTION, FrameKind.DERIVATIVE_PROJECTION)


def response_solve(values, rhs):
    return pinned.solve(values, rhs, stage=RESPONSE_STAGE, dependency=RESPONSE).value


def response_solve_beside_check(values, rhs, holds):
    solution = response_solve(values, rhs)
    checkify.check(holds, FOREIGN_MESSAGE)
    return solution


def newton_with_foreign_check(values, rhs, plan, count, limit):
    """The Newton body of the fixture with a debug check that fails from trip ``limit`` on."""
    def trip(state):
        index, _ = state
        checkify.debug_check(index < limit, 'trip {index} is beyond the limit', index=index)
        return index + 1, pinned.loop_trip(values, rhs, plan[index]).value
    return jax.lax.while_loop(lambda state: state[0] < count, trip,
                              (jnp.int32(0), jnp.zeros_like(rhs)))


newton = checked_sequential_primal(pinned.while_trips)


def frame(function, *operands, lanes=False):
    """The evidence of one compiled checked call; ``lanes`` maps it over the leading axis."""
    checked = checked_native(function)
    if lanes:
        checked = jax.vmap(checked)
    with jax.enable_x64(True):
        evidence, _ = jax.jit(checked)(*operands)
    return evidence


def matrices(failing):
    """One matrix, or one per lane: singular where ``failing`` holds."""
    if isinstance(failing, bool):
        return SINGULAR if failing else VALUES
    return np.stack([SINGULAR if lane_fails else VALUES for lane_fails in failing])


def right_hand_sides(failing):
    return RHS if isinstance(failing, bool) else np.stack([RHS] * len(failing))


def newton_frame(plans, *, inner_lanes=False):
    """A frame of the sequentially mapped Newton body: one plan, or one per lane."""
    if isinstance(plans, np.ndarray):
        return frame(newton, VALUES, RHS, plans, np.int32(pinned.LONGEST_PLAN))
    count = len(plans)
    operands = (np.stack([VALUES] * count), np.stack([RHS] * count), np.stack(plans),
                np.full(count, pinned.LONGEST_PLAN, dtype=np.int32))
    if inner_lanes:
        return frame(jax.vmap(newton), *operands)
    return frame(newton, *operands, lanes=True)


def bits(evidence) -> tuple:
    """Every array of the evidence: dtype, shape and bytes."""
    return tuple((np.asarray(leaf).dtype.str, np.asarray(leaf).shape, np.asarray(leaf).tobytes())
                 for leaf in jax.tree.leaves(evidence))


def rows_of(evidence) -> pinned.CheckedCall:
    return pinned.CheckedCall(None, frame_evidence.certificates(evidence),
                              frame_evidence.availability(evidence))


def raised_by(operation, *operands, **options):
    """The message and the class of the cause of what ``operation`` raises, or None."""
    try:
        operation(*operands, **options)
    except checkify.JaxRuntimeError as error:
        return str(error), type(error.__cause__).__name__
    return None


class NativeFailureRoutingTests(pinned.CertificateTestCase):
    def assert_not_raised_and_unchanged(self, evidence, kinds):
        before = bits(evidence)
        for kind in kinds:
            with self.subTest(kind=kind.name):
                self.assertIsNone(frame_evidence.raise_foreign(evidence, frame=kind))
        self.assertEqual(bits(evidence), before)

    def test_equation_failures_stay_evidence_with_their_status_lanes_and_response(self):
        for failing in (True, [True], [False, True, False]):
            with self.subTest(failing=failing):
                evidence = frame(response_solve, matrices(failing), right_hand_sides(failing),
                                 lanes=not isinstance(failing, bool))
                self.assert_not_raised_and_unchanged(
                    evidence, EVERY_KIND if isinstance(failing, bool) else LANE_KINDS)
                call = rows_of(evidence)
                # One row, which names the response and holds the stage as
                # status in exactly the lanes that failed.
                self.assert_row_names(call, [RESPONSE_ROW])
                expected = (failed(RESPONSE_STAGE) if failing is True else
                            [failed(RESPONSE_STAGE) if lane_fails else SOLVED for lane_fails in failing])
                self.assert_entries(self.row(call, RESPONSE_ROW), expected)

    def test_equation_failures_of_a_sequentially_mapped_body_stay_evidence(self):
        frames = (
            ('one call', newton_frame(FAILING), EVERY_KIND, failed(3)),
            ('lanes of the call', newton_frame((CONVERGING, FAILING)), LANE_KINDS,
             [SOLVED, failed(3)]),
            # The lanes are inside the frame here, so the frame itself has none.
            ('lanes inside the call', newton_frame((CONVERGING, FAILING), inner_lanes=True),
             EVERY_KIND, [SOLVED, failed(3)]),
        )
        for label, evidence, kinds, expected in frames:
            with self.subTest(frame=label):
                self.assert_not_raised_and_unchanged(evidence, kinds)
                call = rows_of(evidence)
                self.assert_row_names(call, [NEWTON_ROW])
                self.assert_entries(self.row(call, NEWTON_ROW), expected)

    def test_foreign_and_mixed_failures_are_raised_as_the_frame_raises_them(self):
        frames = (
            ('a foreign check',
             frame(response_solve_beside_check, VALUES, RHS, np.bool_(False))),
            ('a foreign check in the only lane',
             frame(response_solve_beside_check, matrices([False]), right_hand_sides([False]),
                   np.array([False]), lanes=True)),
            ('an equation failure in lane 0 and a foreign check in lane 1',
             frame(response_solve_beside_check, matrices([True, False]),
                   right_hand_sides([True, False]), np.array([True, False]), lanes=True)),
            ('a foreign check in a sequentially mapped body',
             frame(checked_sequential_primal(newton_with_foreign_check), VALUES, RHS,
                   CONVERGING, np.int32(2), np.int32(1))),
        )
        for label, evidence in frames:
            with self.subTest(frame=label):
                before = bits(evidence)
                # What the frame raises when nothing is set aside as evidence.
                original = raised_by(frame_evidence.raise_any_failure, evidence)
                self.assertIsNotNone(original)
                self.assertEqual(
                    raised_by(frame_evidence.raise_foreign, evidence, frame=FrameKind.LANES),
                    original)
                self.assertEqual(bits(evidence), before)

    def test_frame_without_a_failure_raises_nothing(self):
        frames = (
            frame(response_solve_beside_check, VALUES, RHS, np.bool_(True)),
            frame(response_solve_beside_check, matrices([False, False]),
                  right_hand_sides([False, False]), np.array([True, True]), lanes=True),
            newton_frame((CONVERGING, CONVERGING)),
        )
        for evidence in frames:
            self.assert_not_raised_and_unchanged(evidence, LANE_KINDS)
            self.assertIsNone(frame_evidence.raise_any_failure(evidence))

    def test_raising_any_failure_raises_an_equation_failure_with_its_text(self):
        evidence = frame(response_solve, SINGULAR, RHS)
        with self.assertRaises(checkify.JaxRuntimeError) as raised:
            frame_evidence.raise_any_failure(evidence)
        # The message ends with the site of the equation, which no test but
        # SiteAndRowOrderTests pins.
        self.assertRegex(str(raised.exception),
                         rf'^sparse primal equation status {RESPONSE_STAGE} at site \(.*\)$')
        self.assertEqual(type(raised.exception.__cause__).__name__, 'SparseEquationCheck')


class CurrentTransportWrapperTests(unittest.TestCase):
    """A wrapper that cannot be read as equation failures is not admitted as one.

    On JAX's ``Error`` a failure with lanes arrives as a ``BatchedError`` and
    one from a sequentially mapped body as an ``ArchivedPrimalCheck``. The
    interface admits them as owned only when every failure inside is an
    equation failure, and refuses what it cannot read.
    """

    def setUp(self):
        from jax._src.checkify import BatchedError
        from scnsim.numerics.native_sparse import ArchivedPrimalCheck, SparseEquationCheck
        self.batched, self.archived = BatchedError, ArchivedPrimalCheck
        self.equation = SparseEquationCheck(
            None, (256,), 'primal', RESPONSE_STAGE, 0, RESPONSE, np.int32(RESPONSE_STAGE),
            np.float64(np.inf), np.float64(np.inf), np.float64(np.inf), np.bool_(True))

    def archive(self, exception):
        values, tree = jax.tree.flatten(exception)
        # An archive adds the lane presence and the lane code to the payload.
        return self.archived(tree, (*values, np.bool_(True), np.int32(1)))

    def test_wrappers_of_equation_failures_are_admitted(self):
        admitted = frame_evidence._is_equation_failure
        self.assertTrue(admitted(self.equation))
        self.assertTrue(admitted(self.batched({(0,): self.equation, (2,): self.equation})))
        self.assertTrue(admitted(self.archive(self.equation)))
        self.assertTrue(admitted(self.batched({(0,): self.archive(self.equation)})))

    def test_unreadable_wrappers_and_other_failures_are_refused(self):
        admitted = frame_evidence._is_equation_failure
        self.assertFalse(admitted(None))
        empty = self.batched({(0,): self.equation})
        empty.error_mapping.clear()
        self.assertFalse(admitted(empty))
        mislabelled = self.batched({(0,): self.equation})
        mislabelled.error_mapping['not a lane'] = self.equation
        self.assertFalse(admitted(mislabelled))
        truncated = self.archive(self.equation)
        truncated.payload = ()
        self.assertFalse(admitted(truncated))
        foreign = frame(response_solve_beside_check, VALUES, RHS, np.bool_(False))
        with self.assertRaises(checkify.JaxRuntimeError) as raised:
            frame_evidence.raise_any_failure(foreign)
        check = raised.exception.__cause__
        self.assertFalse(admitted(check))
        self.assertFalse(admitted(self.batched({(0,): self.equation, (1,): check})))


if __name__ == '__main__':
    unittest.main()
