"""The evidence of a checked frame, read through the interface its consumers use.

``scnsim.numerics.frame_evidence`` names what the product may do with the
evidence object a checked call returns (Issue 48). The tests of
``test_certificate_transport.py`` hand that object to two projections only.
These tests cover what they could not reach without the object's private
state:

- the lanes an ordered chunk skips after a failure are absent;
- the certificates of a completed root branch belong to the frame that
  consumes it;
- the evidence is an input of a later compiled call, and one lane of it is
  the evidence of that lane computed alone;
- the frames of one result are appended in call order;
- the text of a foreign check's message;
- which reader raises what, where the current transport decides by an
  accident of JAX's internals;
- no product module beside the transport and the interface reads the object.

The fixture system and its functions are those of
``test_certificate_transport.py``. The evidence object is opaque here too: it
goes to the interface and to JAX as a pytree of arrays. A row is named by its
role, stage, derivative order and dependency, never by ``site`` or position.
No tolerance is used.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, NamedTuple
import unittest

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import checkify
import numpy as np

import scnsim
from scnsim import (
    CircuitPlan, CircuitRun, ExecutionPolicy, JVP, VJP, OutputCoordinate, ParameterDefinitions,
    ParameterSpec, ResponseElementSpec, SensitivitySpec, components, units as u,
)
from scnsim.numerics import frame_evidence
from scnsim.numerics.frame_evidence import FrameKind
from scnsim.numerics.native_program import _compile_chunk
from scnsim.numerics.native_sparse import (
    archive_completed_primal_errors, checked_native, checked_sequential_primal,
)
from tests import test_certificate_transport as pinned
from tests.jax_runtime import initialize_jax_through_scnsim


def setUpModule():
    # These tests call JAX directly. SCNSim has to initialize it first, or the
    # SCNSim tests that run after this module in the same process fail.
    initialize_jax_through_scnsim()


VALUES, SINGULAR, RHS, DIRECTION = pinned.VALUES, pinned.SINGULAR, pinned.RHS, pinned.DIRECTION
LANE_VALUES, LANE_RHS, LANE_FLAGS = pinned.LANE_VALUES, pinned.LANE_RHS, pinned.LANE_FLAGS
SOLVED, ABSENT, failed, RowName = pinned.SOLVED, pinned.ABSENT, pinned.failed, pinned.RowName
RUN, SINGULAR_MATRIX = pinned.RUN, pinned.SINGULAR_MATRIX
ROW_ARRAYS = ('status', 'eta', 'residual_numerator', 'residual_denominator', 'performed', 'present')
FOREIGN_MESSAGE = 'a check that SCNSim does not own'
# How JAX words a failed ``checkify.check`` or ``checkify.debug_check``.
CHECK_FAILED = ' (`check` failed)'


def frame(function, *operands, lanes=False):
    """The evidence of one compiled checked call; ``lanes`` maps it over the leading axis."""
    checked = checked_native(function)
    if lanes:
        checked = jax.vmap(checked)
    with jax.enable_x64(True):
        evidence, _ = jax.jit(checked)(*operands)
    return evidence


def rows_of(evidence) -> pinned.CheckedCall:
    return pinned.CheckedCall(None, frame_evidence.certificates(evidence),
                              frame_evidence.availability(evidence))


def bits(array) -> tuple:
    """Everything that makes two arrays the same array: dtype, shape and bytes."""
    array = np.asarray(array)
    return array.dtype.str, array.shape, array.tobytes()


def row_bits(row) -> tuple:
    return tuple(bits(getattr(row, field)) for field in ROW_ARRAYS)


def named_row_bits(evidence) -> list:
    """Every row of ``evidence`` by name with all its arrays, in a fixed order."""
    return sorted((pinned.name_of(row), row_bits(row)) for row in frame_evidence.certificates(evidence))


def fact_bits(evidence) -> list:
    return sorted((fact.dependency, fact.reason, bits(fact.available), bits(fact.present))
                  for fact in frame_evidence.availability(evidence))


def message_of(operation, *operands, **options) -> str | None:
    """The message of the ``JaxRuntimeError`` that ``operation`` raises, or None."""
    try:
        operation(*operands, **options)
    except checkify.JaxRuntimeError as error:
        return str(error)
    return None


# ---------------------------------------------------------------------------
# The ordered tail of a frequency chunk.
# ---------------------------------------------------------------------------
POINT_ROW = RowName('primal', 3, 0, ('point',))


class Point(NamedTuple):
    """What one frequency point returns: the two statuses a chunk reduces, and its value."""
    check_status: Any
    certificate_status: Any
    solution: Any


def frequency_point(values, scale, prior_check, prior_certificate):
    """One point of a chunk: the fixture matrix times ``scale``. A zero scale is singular."""
    outcome = pinned.solve(scale * values, RHS, dependency=('point',))
    return Point(jnp.asarray(prior_check, dtype=jnp.int32), outcome.aux.status, outcome.value)


def run_chunk(scales, *, ordered_tail, prior_certificate_status=0):
    """Compile the product's chunk of ``frequency_point`` over ``scales`` and run it."""
    chunk = _compile_chunk(frequency_point, len(scales), ordered_tail=ordered_tail)
    with jax.enable_x64(True):
        return jax.jit(chunk)(jnp.asarray(VALUES), jnp.asarray(scales, dtype=jnp.complex128),
                              jnp.int32(0), jnp.int32(prior_certificate_status))


class OrderedTailTests(pinned.CertificateTestCase):
    # The third point is singular.
    SCALES = (1, 2, 0, 3, 4)

    def assert_chunk(self, outcome, *, executed, entries, certificate_status):
        self.assertEqual(np.asarray(outcome.executed).tolist(), executed)
        self.assertEqual(int(outcome.certificate_status), certificate_status)
        call = rows_of(outcome.errors)
        self.assert_row_names(call, [POINT_ROW])
        self.assert_entries(self.row(call, POINT_ROW), entries)
        self.assertEqual(call.availability, ())

    def test_lanes_after_the_first_failure_did_not_run_and_are_absent(self):
        self.assert_chunk(run_chunk(self.SCALES, ordered_tail=True),
                          executed=[True, True, True, False, False],
                          entries=[SOLVED, SOLVED, failed(3), ABSENT, ABSENT],
                          certificate_status=3)

    def test_full_chunk_runs_the_same_lanes_to_the_end(self):
        # The same points as independent lanes: the last two are present, so
        # the ordered tail left them absent because it skipped them.
        self.assert_chunk(run_chunk(self.SCALES, ordered_tail=False),
                          executed=[True] * 5,
                          entries=[SOLVED, SOLVED, failed(3), SOLVED, SOLVED],
                          certificate_status=3)

    def test_tail_without_a_failure_runs_every_lane(self):
        self.assert_chunk(run_chunk((1, 2, 3), ordered_tail=True), executed=[True] * 3,
                          entries=[SOLVED] * 3, certificate_status=0)

    def test_tail_that_starts_after_a_failure_runs_no_lane(self):
        self.assert_chunk(run_chunk((1, 2, 3), ordered_tail=True, prior_certificate_status=7),
                          executed=[False] * 3, entries=[ABSENT] * 3, certificate_status=7)

    def test_skipped_lanes_do_not_raise(self):
        evidence = run_chunk(self.SCALES, ordered_tail=True).errors
        self.assertIsNone(frame_evidence.raise_foreign(evidence, frame=FrameKind.LANES))


class AbsentEvidenceTests(pinned.CertificateTestCase):
    """The value the ordered tail gives a lane that did not run, built directly."""

    def absent(self, function, *operands, lanes=False):
        checked = checked_native(function)
        if lanes:
            checked = jax.vmap(checked)
        with jax.enable_x64(True):
            shape, _ = jax.eval_shape(checked, *operands)
            return frame_evidence.absent(shape)

    def test_no_equation_is_present_and_nothing_has_failed(self):
        direct_row = RowName('primal', 3, 0, ('direct',))
        for lanes, entries in ((False, ABSENT), (True, [ABSENT] * len(LANE_VALUES))):
            with self.subTest(lanes=lanes):
                operands = (LANE_VALUES, LANE_RHS) if lanes else (SINGULAR, RHS)
                evidence = self.absent(pinned.direct, *operands, lanes=lanes)
                call = rows_of(evidence)
                self.assert_row_names(call, [direct_row])
                self.assert_entries(self.row(call, direct_row), entries)
                self.assertFalse(bool(frame_evidence.any_failed(evidence)))
                for kind in (FrameKind.LANES, FrameKind.PRIMAL_PROJECTION,
                             FrameKind.DERIVATIVE_PROJECTION):
                    self.assertIsNone(frame_evidence.raise_foreign(evidence, frame=kind))
                self.assertIsNone(frame_evidence.raise_any_failure(evidence))

    def test_no_fact_is_present(self):
        evidence = self.absent(pinned.chart_forward, pinned.CHART_MATRIX, pinned.TIED_LARGEST,
                               pinned.CHART_DIRECTION)
        fact, = frame_evidence.availability(evidence)
        self.assertEqual((fact.dependency, fact.reason),
                         (pinned.CHART_DEPENDENCY, pinned.CHART_REASON))
        self.assertEqual(np.asarray(fact.present).tolist(), False)
        chart_tangent = RowName('tangent', 13, 1, pinned.CHART_DEPENDENCY)
        self.assert_entries(self.row(rows_of(evidence), chart_tangent), ABSENT)

    def test_absent_evidence_has_the_structure_of_the_frame_it_stands_for(self):
        # A branch that did not run and one that did must return the same pytree.
        operands = (pinned.CHART_MATRIX, pinned.UNIQUE_LARGEST, pinned.CHART_DIRECTION)
        with jax.enable_x64(True):
            checked = checked_native(pinned.chart_forward)
            shape, _ = jax.eval_shape(checked, *operands)
            ran, _ = jax.jit(checked)(*operands)
            absent = frame_evidence.absent(shape)
        self.assertEqual([(bits(leaf)[:2]) for leaf in jax.tree.leaves(absent)],
                         [(bits(leaf)[:2]) for leaf in jax.tree.leaves(ran)])
        self.assertEqual(jax.tree.structure(absent).num_leaves, jax.tree.structure(ran).num_leaves)


class AnyFailedTests(unittest.TestCase):
    def traced(self, evidence):
        with jax.enable_x64(True):
            flag = jax.jit(frame_evidence.any_failed)(evidence)
        self.assertEqual((flag.dtype, flag.shape), (np.bool_, ()))
        return bool(flag)

    def test_failed_equation_is_seen_inside_a_program(self):
        self.assertTrue(self.traced(frame(pinned.direct, SINGULAR, RHS)))
        self.assertFalse(self.traced(frame(pinned.direct, VALUES, RHS)))

    def test_one_failing_lane_is_enough(self):
        # Lane 1 of the fixture lanes is singular.
        self.assertTrue(self.traced(frame(pinned.direct, LANE_VALUES, LANE_RHS, lanes=True)))
        self.assertFalse(self.traced(frame(pinned.direct, LANE_VALUES[2:], LANE_RHS[2:], lanes=True)))

    def test_skipped_equation_counts_as_a_failure(self):
        # A skipped equation carries the failure it was handed.
        def skipped(values, rhs):
            return pinned.solve(values, rhs, prior_status=9, dependency=('skipped',)).value
        self.assertTrue(self.traced(frame(skipped, VALUES, RHS)))


# ---------------------------------------------------------------------------
# A completed root branch and the frame that consumes it.
# ---------------------------------------------------------------------------
ROOT_A, ROOT_B, CONSUMER = ('root a',), ('root b',), ('consumer',)
ROOT_A_ROW = RowName('primal', 3, 0, ROOT_A)
ROOT_B_ROW = RowName('primal', 7, 0, ROOT_B)
CONSUMER_ROW = RowName('primal', 9, 0, CONSUMER)


def newton_of_root_a(values, rhs, plan, count):
    """The Newton body of root occurrence ``a``: one equation site, run ``count`` times."""
    def trip(state):
        index, _ = state
        coefficients = jnp.where(plan[index] == SINGULAR_MATRIX, jnp.zeros_like(values), values)
        return index + 1, pinned.solve(coefficients, rhs, dependency=ROOT_A).value
    return lax.while_loop(lambda state: state[0] < count, trip,
                          (jnp.int32(0), jnp.zeros_like(rhs)))


root_a = checked_sequential_primal(newton_of_root_a)


def value_frame_with_two_roots(values, rhs, plan, count):
    """A value frame with two root occurrences: a Newton body and a single solve."""
    return (root_a(values, rhs, plan, count),
            pinned.solve(2 * values, rhs, stage=7, dependency=ROOT_B).value)


def consuming_frame(values, rhs, branches):
    """A later frame that takes completed branches as operands and solves its own equation."""
    # Today's form of handing the evidence of a completed branch to the frame
    # that consumes it; step 4 of Issue 48 moves the product's call sites.
    archive_completed_primal_errors(branches)
    return pinned.solve(values, rhs, stage=9, dependency=CONSUMER).value


class CompletedBranchTests(pinned.CertificateTestCase):
    def value_frame(self, plan):
        return frame(value_frame_with_two_roots, VALUES, RHS, pinned.padded(plan),
                     np.int32(len(plan)))

    def test_one_occurrence_is_cut_out_of_its_value_frame(self):
        evidence = self.value_frame((RUN, RUN))
        whole = dict(named_row_bits(evidence))
        self.assertCountEqual(whole, [ROOT_A_ROW, ROOT_B_ROW])
        for dependency, name in ((ROOT_A, ROOT_A_ROW), (ROOT_B, ROOT_B_ROW)):
            with self.subTest(dependency=dependency):
                branch = frame_evidence.for_dependency(evidence, dependency)
                # Its own row, with the arrays it has in the value frame.
                self.assertEqual(named_row_bits(branch), [(name, whole[name])])
        self.assertEqual(frame_evidence.certificates(
            frame_evidence.for_dependency(evidence, ('no such occurrence',))), ())

    def test_facts_follow_their_dependency(self):
        def chart_beside_solve(matrix, vector, direction, values, rhs):
            return pinned.chart_forward(matrix, vector, direction), pinned.direct(values, rhs)
        evidence = frame(chart_beside_solve, pinned.CHART_MATRIX, pinned.TIED_LARGEST,
                         pinned.CHART_DIRECTION, VALUES, RHS)
        chart = frame_evidence.for_dependency(evidence, pinned.CHART_DEPENDENCY)
        self.assertEqual(fact_bits(chart), fact_bits(evidence))
        self.assertEqual(len(frame_evidence.availability(chart)), 1)
        self.assert_row_names(rows_of(chart), [RowName('tangent', 13, 1, pinned.CHART_DEPENDENCY)])
        solve = frame_evidence.for_dependency(evidence, ('direct',))
        self.assertEqual(frame_evidence.availability(solve), ())
        self.assert_row_names(rows_of(solve), [RowName('primal', 3, 0, ('direct',))])

    def test_certificates_of_completed_branches_belong_to_the_consuming_frame(self):
        for plan, root_a_entry in (((RUN, RUN), SOLVED), ((RUN, SINGULAR_MATRIX, RUN), failed(3))):
            with self.subTest(plan=plan):
                value_evidence = self.value_frame(plan)
                branches = tuple(frame_evidence.for_dependency(value_evidence, dependency)
                                 for dependency in (ROOT_A, ROOT_B))
                # The branches are operands of the later compiled call.
                consumer = frame(consuming_frame, 3 * VALUES, RHS, branches)
                call = rows_of(consumer)
                self.assert_row_names(call, [ROOT_A_ROW, ROOT_B_ROW, CONSUMER_ROW])
                self.assert_entries(self.row(call, ROOT_A_ROW), root_a_entry)
                self.assert_entries(self.row(call, ROOT_B_ROW), SOLVED)
                self.assert_entries(self.row(call, CONSUMER_ROW), SOLVED)
                # The branch rows are the ones the value frame recorded, bit for bit.
                recorded = dict(named_row_bits(value_evidence))
                consumed = dict(named_row_bits(consumer))
                for name in (ROOT_A_ROW, ROOT_B_ROW):
                    self.assertEqual(consumed[name], recorded[name], name)
                # A failed branch equation stays evidence in the consuming frame.
                self.assertIsNone(frame_evidence.raise_foreign(consumer, frame=FrameKind.SCALAR))
                self.assertIsNone(
                    frame_evidence.raise_foreign(consumer, frame=FrameKind.PRIMAL_PROJECTION))

    def test_compile_key_text_follows_the_branches_a_frame_consumes(self):
        evidence = self.value_frame((RUN, RUN))
        branch_a, branch_b = (frame_evidence.for_dependency(evidence, dependency)
                              for dependency in (ROOT_A, ROOT_B))
        both = frame_evidence.structure((branch_a, branch_b))
        self.assertIsInstance(both, str)
        self.assertEqual(frame_evidence.structure((branch_a, branch_b)), both)
        self.assertNotEqual(frame_evidence.structure((branch_a,)), both)
        self.assertNotEqual(frame_evidence.structure((branch_b, branch_a)), both)


# ---------------------------------------------------------------------------
# The evidence as data: through a compiled call, and one lane of it.
# ---------------------------------------------------------------------------
class EvidenceAsDataTests(unittest.TestCase):
    def test_evidence_passes_through_a_later_compiled_call_unchanged(self):
        frames = {
            'one frame': frame(pinned.one_equation_of_each_kind, VALUES, RHS, np.bool_(False)),
            'lanes': frame(pinned.one_of_two_branches, LANE_FLAGS, LANE_VALUES, LANE_RHS, lanes=True),
            'a fact': frame(pinned.chart_forward, pinned.CHART_MATRIX, pinned.TIED_LARGEST,
                            pinned.CHART_DIRECTION),
        }
        for label, evidence in frames.items():
            with self.subTest(frame=label):
                with jax.enable_x64(True):
                    returned = jax.jit(lambda operand: operand)(evidence)
                self.assertEqual(frame_evidence.structure(returned), frame_evidence.structure(evidence))
                self.assertEqual([bits(leaf) for leaf in jax.tree.leaves(returned)],
                                 [bits(leaf) for leaf in jax.tree.leaves(evidence)])
                self.assertEqual(named_row_bits(returned), named_row_bits(evidence))
                self.assertEqual(fact_bits(returned), fact_bits(evidence))

    def test_one_lane_is_the_evidence_of_that_lane_computed_alone(self):
        cases = (
            (pinned.one_of_two_branches, (LANE_FLAGS, LANE_VALUES, LANE_RHS)),
            (pinned.chart_forward, (np.stack([pinned.CHART_MATRIX] * 3),
                                    np.stack([pinned.UNIQUE_LARGEST, pinned.TIED_LARGEST,
                                              pinned.UNIQUE_LARGEST]),
                                    np.stack([pinned.CHART_DIRECTION] * 3))),
        )
        for function, operands in cases:
            lanes = frame(function, *operands, lanes=True)
            count = len(operands[0])
            self.assertEqual(set(frame_evidence.lane_shapes(lanes)), {(count,)})
            for index in range(count):
                with self.subTest(function=function.__name__, lane=index):
                    alone = frame(function, *(operand[index] for operand in operands))
                    cut = frame_evidence.lane(lanes, index)
                    self.assertEqual(named_row_bits(cut), named_row_bits(alone))
                    self.assertEqual(fact_bits(cut), fact_bits(alone))
                    self.assertEqual(set(frame_evidence.lane_shapes(cut)), {()})

    def test_two_lane_axes_are_cut_one_after_the_other(self):
        # Work lanes in front, direction lanes behind them, as a response chunk has them.
        def over_directions(values, rhs, directions):
            return jax.vmap(checked_native(pinned.forward(pinned.squared_norm)),
                            in_axes=(None, None, 0))(values, rhs, directions)
        directions = np.stack([DIRECTION, 2 * DIRECTION, -DIRECTION])
        with jax.enable_x64(True):
            evidence, _ = jax.jit(jax.vmap(over_directions, in_axes=(0, 0, None)))(
                LANE_VALUES[:2], LANE_RHS[:2], directions)
        self.assertEqual(set(frame_evidence.lane_shapes(evidence)), {(2, 3)})
        work_lane = frame_evidence.lane(evidence, 1)
        self.assertEqual(set(frame_evidence.lane_shapes(work_lane)), {(3,)})
        alone = frame(pinned.forward(pinned.squared_norm), LANE_VALUES[1], LANE_RHS[1], directions[2])
        self.assertEqual(named_row_bits(frame_evidence.lane(work_lane, 2)), named_row_bits(alone))


# ---------------------------------------------------------------------------
# The frames of one result, in call order.
# ---------------------------------------------------------------------------
FREQUENCIES_GHZ = (5.0, 6.0, 7.0)
RESPONSE = ('responses', 0, ('Y',))
RESPONSE_PRIMAL = RowName('primal', 15, 0, RESPONSE)
RESPONSE_TANGENT = RowName('tangent', 4, 1, RESPONSE)
RESPONSE_ADJOINT = RowName('adjoint', 4, 1, RESPONSE)


class FrameLocator(NamedTuple):
    """Where the frame of a certificate row belongs in its result."""
    work_index: int | None
    frequency_source_index: int | None
    basis_kind: str | None
    basis_indices: tuple


def frames_of(certificates) -> list[tuple[FrameLocator, list]]:
    """The rows of a result as runs of one locator, each with the sorted names of its rows.

    A run is one frame, or neighbouring frames that carry the same locator.
    """
    runs = []
    for row in certificates:
        locator = FrameLocator(row.work_index, row.frequency_source_index, row.basis_kind,
                               tuple(row.basis_indices))
        if not runs or runs[-1][0] != locator:
            runs.append((locator, []))
        runs[-1][1].append(pinned.name_of(row))
    return [(locator, sorted(names)) for locator, names in runs]


def lossy_resonator_plan():
    """A lossy parallel resonator behind a coupling capacitor, with its capacitance as parameter."""
    plan = CircuitPlan(id="frame_order")
    definitions = ParameterDefinitions(id="frame_order_parameters")
    capacitance = definitions.parameter(id="capacitance", baseline=200.0 * u.fF,
                                        spec=ParameterSpec(unit=u.F))
    port, resonator = plan.bus(id="p"), plan.bus(id="resonator")
    capacitor = plan.add(components.capacitor(id="cap", capacitance=capacitance))
    inductor = plan.add(components.inductor(id="ind", inductance=3.0 * u.nH))
    resistor = plan.add(components.resistor(id="loss", resistance=10.0 * u.kohm))
    coupling = plan.add(components.capacitor(id="coupling", capacitance=1.0 * u.fF))
    plan.parallel(id="resonator", start=resonator,
                  branches=((capacitor,), (inductor,), (resistor,)), end=plan.ground)
    plan.series(id="coupling", start=port, elements=(coupling,), end=resonator)
    plan.add_port(id="p", at=port, role="terminated", reference_impedance=50.0 * u.ohm)
    return plan, capacitance


class FrameOrderTests(unittest.TestCase):
    """The certificate rows of a sensitivity are the rows of its frames, in call order.

    Three frequencies with two per chunk give one chunk of two lanes and a
    tail of one. Frames without a derivative basis come first, in the order
    they were computed; the frames of the derivative directions follow, in
    work order.
    """

    @classmethod
    def setUpClass(cls):
        plan, cls.capacitance = lossy_resonator_plan()
        cls.run_ = CircuitRun(plan=plan, storage="memory", precision="float64",
                              execution_policy=ExecutionPolicy(frequency_chunk_size=2))
        cls.quantities = {
            f"loss{index}": ResponseElementSpec(
                family="Y", input_coordinate="p", output_coordinate="p",
                frequency=frequency * u.GHz).real.on(cls.run_.original)
            for index, frequency in enumerate(FREQUENCIES_GHZ)}

    @classmethod
    def tearDownClass(cls):
        cls.run_.close()

    def frames(self, derivative):
        result = self.run_.sensitivity(SensitivitySpec(
            quantities=self.quantities, wrt=(self.capacitance,), derivative=derivative))
        for row in result.certificates:
            self.assertTrue(np.all(np.asarray(row.present)))
            self.assertTrue(np.all(np.asarray(row.status) == 0))
        return frames_of(result.certificates)

    def test_forward_mode_frames_follow_the_work_rows(self):
        frames = self.frames(JVP(direction={self.capacitance: 200.0 * u.fF}))
        work = [0, 1, 2]
        self.assertEqual(frames, [
            # The point frame of each work row: its solve and the equation of its linearization.
            *((FrameLocator(index, index, None, ()), [RESPONSE_PRIMAL, RESPONSE_TANGENT])
              for index in work),
            # The frame of the one direction, for each work row.
            *((FrameLocator(index, index, 'direction', (0,)), [RESPONSE_TANGENT])
              for index in work),
        ])

    def test_reverse_mode_frames_follow_the_value_pass_then_the_pullback_pass(self):
        frames = self.frames(VJP(weights={OutputCoordinate(quantity_label="loss0"): 1 / u.S}))
        chunk = (FrameLocator(None, None, None, ()), [RESPONSE_PRIMAL])

        def lane(index):
            return FrameLocator(index, index, None, ()), [RESPONSE_PRIMAL]

        self.assertEqual(frames, [
            # The value pass: the chunk of two lanes, then the tail of one. Each
            # chunk has a frame of its own in front of the frames of its lanes.
            chunk, lane(0), lane(1), chunk, lane(2),
            # The pullback pass replays each point and solves its adjoint equation.
            *((FrameLocator(index, index, None, ()), [RESPONSE_ADJOINT, RESPONSE_PRIMAL])
              for index in (0, 1, 2)),
            # The frame of the one functional, for each work row.
            *((FrameLocator(index, index, 'functional', (0,)), [RESPONSE_ADJOINT])
              for index in (0, 1, 2)),
        ])


# ---------------------------------------------------------------------------
# Raising on the host.
# ---------------------------------------------------------------------------
def solve_then_check(values, rhs, holds):
    solution = pinned.direct(values, rhs)
    checkify.check(holds, FOREIGN_MESSAGE)
    return solution


def check_then_solve(values, rhs, holds):
    checkify.check(holds, FOREIGN_MESSAGE)
    return pinned.direct(values, rhs)


def newton_with_trip_limit(values, rhs, plan, count, limit):
    """The Newton body of the fixture with a debug check that fails from trip ``limit`` on."""
    def trip(state):
        index, _ = state
        checkify.debug_check(index < limit, 'trip {index} is beyond the limit {limit}',
                             index=index, limit=limit)
        return index + 1, pinned.loop_trip(values, rhs, plan[index]).value
    return lax.while_loop(lambda state: state[0] < count, trip,
                          (jnp.int32(0), jnp.zeros_like(rhs)))


class ForeignCheckMessageTests(unittest.TestCase):
    """A foreign check is raised as ``JaxRuntimeError`` with the text of the check."""

    def assert_message(self, evidence, message, kinds=tuple(FrameKind)):
        for kind in kinds:
            with self.subTest(kind=kind.name):
                with self.assertRaises(checkify.JaxRuntimeError) as raised:
                    frame_evidence.raise_foreign(evidence, frame=kind)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(type(raised.exception.__cause__).__name__, 'FailedCheckError')

    def test_message_of_a_check_in_one_frame(self):
        self.assert_message(frame(solve_then_check, VALUES, RHS, np.bool_(False)),
                            FOREIGN_MESSAGE + CHECK_FAILED)

    def test_message_of_a_check_that_failed_in_one_of_several_lanes(self):
        evidence = frame(solve_then_check, LANE_VALUES[2:], LANE_RHS[2:],
                         np.array([True, False, True]), lanes=True)
        # The text names neither the lane nor the number of lanes.
        self.assert_message(evidence, FOREIGN_MESSAGE + CHECK_FAILED)

    def test_message_carries_the_values_the_check_was_given(self):
        body = checked_sequential_primal(newton_with_trip_limit)
        evidence = frame(body, VALUES, RHS, pinned.padded((RUN, RUN)), np.int32(2), np.int32(1))
        self.assert_message(evidence, 'trip 1 is beyond the limit 1' + CHECK_FAILED)

    def test_message_of_a_check_in_a_sequentially_mapped_body_lists_every_lane(self):
        # Three lanes of the body inside one frame. The limits let lane 0 pass,
        # fail lane 1 at trip 1 and lane 2 at trip 0. Today the text holds the
        # values of all three lanes, the passing one included. Issue 48 sets
        # the rule that it reports the failing lane's values; its last step
        # changes this expectation.
        body = jax.vmap(checked_sequential_primal(newton_with_trip_limit))
        evidence = frame(body, np.stack([VALUES] * 3), np.stack([RHS] * 3),
                         np.stack([pinned.padded((RUN, RUN))] * 3),
                         np.array([2, 2, 2], dtype=np.int32), np.array([5, 1, 0], dtype=np.int32))
        self.assert_message(evidence, 'trip [1 1 0] is beyond the limit [5 1 0]' + CHECK_FAILED)


class ReaderRuleTests(unittest.TestCase):
    """What each kind of reader raises when owned and foreign failures meet.

    The current transport keeps both in one object, and JAX's internal order
    of checks decides between them for the two readers that take a frame
    straight from its call. Issue 48 sets one rule for every reader in its
    last step, a foreign check is raised first, and changes this class to it.
    """
    EQUATION_TEXT = r'^sparse primal equation status 3 at site \(.*\)$'
    FOREIGN_TEXT = FOREIGN_MESSAGE + CHECK_FAILED
    PROJECTIONS = (FrameKind.PRIMAL_PROJECTION, FrameKind.DERIVATIVE_PROJECTION)

    def raised(self, evidence, kind):
        return message_of(frame_evidence.raise_foreign, evidence, frame=kind)

    def test_equation_failure_before_a_foreign_check_hides_it_from_a_frame_reader(self):
        evidence = frame(solve_then_check, SINGULAR, RHS, np.bool_(False))
        self.assertIsNone(self.raised(evidence, FrameKind.SCALAR))
        self.assertIsNone(self.raised(evidence, FrameKind.LANES))
        for kind in self.PROJECTIONS:
            self.assertEqual(self.raised(evidence, kind), self.FOREIGN_TEXT)

    def test_foreign_check_before_an_equation_failure_is_raised_by_every_reader(self):
        evidence = frame(check_then_solve, SINGULAR, RHS, np.bool_(False))
        for kind in FrameKind:
            self.assertEqual(self.raised(evidence, kind), self.FOREIGN_TEXT)

    def test_scalar_reader_refuses_lanes(self):
        # Lane 1 of the fixture lanes is singular: an equation failure only.
        evidence = frame(pinned.direct, LANE_VALUES, LANE_RHS, lanes=True)
        self.assertRegex(self.raised(evidence, FrameKind.SCALAR), self.EQUATION_TEXT)
        for kind in (FrameKind.LANES, *self.PROJECTIONS):
            self.assertIsNone(self.raised(evidence, kind))

    def test_lanes_reader_raises_a_mixed_chunk_with_the_text_of_its_first_check(self):
        # An equation failure in lane 0 and a foreign check in lane 1.
        evidence = frame(solve_then_check, np.stack([SINGULAR, VALUES]), np.stack([RHS] * 2),
                         np.array([True, False]), lanes=True)
        self.assertRegex(self.raised(evidence, FrameKind.LANES), self.EQUATION_TEXT)
        for kind in self.PROJECTIONS:
            self.assertEqual(self.raised(evidence, kind), self.FOREIGN_TEXT)

    def test_only_a_derivative_projection_owns_an_availability_fact(self):
        operands = (pinned.CHART_MATRIX, pinned.TIED_LARGEST, pinned.CHART_DIRECTION)
        unavailable = frame(pinned.chart_forward, *operands)
        self.assertIsNone(self.raised(unavailable, FrameKind.DERIVATIVE_PROJECTION))
        for kind in (FrameKind.SCALAR, FrameKind.LANES, FrameKind.PRIMAL_PROJECTION):
            self.assertEqual(self.raised(unavailable, kind), pinned.CHART_REASON)
        available = frame(pinned.chart_forward, pinned.CHART_MATRIX, pinned.UNIQUE_LARGEST,
                          pinned.CHART_DIRECTION)
        for kind in FrameKind:
            self.assertIsNone(self.raised(available, kind))

    def test_kind_of_frame_is_required(self):
        evidence = frame(pinned.direct, VALUES, RHS)
        with self.assertRaises(TypeError):
            frame_evidence.raise_foreign(evidence, frame='scalar')


# ---------------------------------------------------------------------------
# Who reads the evidence object.
# ---------------------------------------------------------------------------
PACKAGE = Path(scnsim.__file__).parent
# The current transport and the interface's implementation on it.
EVIDENCE_READERS = {'numerics/native_sparse.py', 'numerics/frame_evidence.py'}
# One query of JAX's runtime state, which is not part of the transport (Issue 48).
RUNTIME_QUERY = 'execution/config.py'
# The modules that hold or pass on the evidence of a checked frame.
CONSUMERS = {
    'numerics/evidence.py', 'numerics/native_program.py', 'numerics/native_root.py',
    'execution/physical_analysis.py', 'execution/native_ad.py', 'execution/warmup.py',
    'execution/physical_optimization.py', 'compilation/objective_program.py',
}
PRIVATE_FIELDS = {'_pred', '_code', '_payload', '_metadata'}
# How JAX's ``Error`` gives its exception; a consumer that calls them knows the type.
ERROR_METHODS = {'get_exception', 'throw'}
# The check and effect classes of the current transport.
TRANSPORT_CLASSES = {
    'SparseEquationCheck', 'SparseCertificateEffect', 'DerivativeUnavailableCheck',
    'DerivativeAvailabilityEffect', 'ArchivedPrimalCheck', 'ArchivedSparseEffect',
    'ArchivedAvailabilityEffect',
}


def product_sources() -> dict[str, str]:
    return {path.relative_to(PACKAGE).as_posix(): path.read_text(encoding='utf-8')
            for path in sorted(PACKAGE.rglob('*.py'))}


def own_attribute(node: ast.Attribute) -> bool:
    """``self._metadata`` and the like: an attribute a class keeps for itself."""
    return isinstance(node.value, ast.Name) and node.value.id == 'self'


class EvidenceReaderTests(unittest.TestCase):
    """Only the transport and the interface look inside the evidence object."""

    @classmethod
    def setUpClass(cls):
        cls.sources = product_sources()
        cls.trees = {name: ast.parse(source) for name, source in cls.sources.items()}

    def test_scan_reads_the_consumers_and_the_exempt_modules(self):
        self.assertLessEqual(CONSUMERS | EVIDENCE_READERS | {RUNTIME_QUERY}, set(self.sources))

    def test_private_jax_package_is_named_only_by_the_transport_and_the_interface(self):
        naming = {name for name, source in self.sources.items() if 'jax._src' in source}
        self.assertEqual(naming, EVIDENCE_READERS | {RUNTIME_QUERY})

    def test_private_fields_of_the_evidence_are_read_only_there(self):
        found = []
        for name, tree in self.trees.items():
            if name in EVIDENCE_READERS:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and node.attr in PRIVATE_FIELDS
                        and not own_attribute(node)):
                    found.append((name, node.lineno, node.attr))
                # ``error._replace(_pred=...)`` writes a field by keyword.
                if isinstance(node, ast.keyword) and node.arg in PRIVATE_FIELDS:
                    found.append((name, node.value.lineno, node.arg))
        self.assertEqual(found, [])

    def test_interface_and_transport_do_read_them(self):
        # The scan would pass on any source if it could not see such a read.
        for name in EVIDENCE_READERS:
            reads = {node.attr for node in ast.walk(self.trees[name])
                     if isinstance(node, ast.Attribute) and node.attr in PRIVATE_FIELDS
                     and not own_attribute(node)}
            self.assertLessEqual({'_pred', '_code', '_payload'}, reads, name)

    def test_consumers_use_neither_the_methods_of_jax_error_nor_the_transport_classes(self):
        found = []
        for name, tree in self.trees.items():
            if name in EVIDENCE_READERS:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in ERROR_METHODS | TRANSPORT_CLASSES:
                    found.append((name, node.lineno, node.attr))
                if isinstance(node, ast.ImportFrom):
                    found.extend((name, node.lineno, alias.name) for alias in node.names
                                 if alias.name in TRANSPORT_CLASSES)
        self.assertEqual(found, [])


if __name__ == '__main__':
    unittest.main()
