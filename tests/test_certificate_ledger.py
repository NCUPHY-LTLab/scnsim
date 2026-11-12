"""The certificate ledger and the pass that fills it.

``scnsim.numerics.certificate_transport`` brings the certificate of every
equation out of a compiled JAX program as an ordinary output (Issue 48). No
caller uses it yet. ``test_certificate_transport_parity.py`` requires that it
returns the rows of the current transport; these tests cover what that
comparison cannot show:

- the module reaches JAX through public interfaces only;
- an equation the pass cannot reach stops the trace with a typed error;
- one slot per site: its first failure, otherwise its latest execution;
- the value of a frame that did not run, built from ``jax.eval_shape``;
- 32-bit certificate arrays stay 32-bit inside the ledger;
- a ledger is an input of ``jax.jit``, and ``jax.tree.map`` slices one lane;
- an equation failure is not returned as a foreign check, and the current
  transport still records its own frames;
- the two identity primitives are inert outside a certified frame.

The fixture system and its functions are those of
``test_certificate_transport.py``. No tolerance is used.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import checkify
import numpy as np

from scnsim.numerics import certificate_transport
from scnsim.numerics.certificate_transport import (
    AVAILABLE_WHERE_ABSENT, AvailabilityKey, CertificateTransportError, EquationKey, Evidence,
    Slot, absent_evidence, attach_evidence, certified, certified_sequential_primal,
    derivative_gate, ledger_availability, ledger_certificates, pass_records_equations,
    register_equation,
)
from scnsim.numerics.native_sparse import checked_native, sparse_certificates
from tests import test_certificate_transport as pinned
from tests.jax_runtime import initialize_jax_through_scnsim


def setUpModule():
    # These tests call JAX directly. SCNSim has to initialize it first, or the
    # SCNSim tests that run after this module in the same process fail.
    initialize_jax_through_scnsim()


VALUES, SINGULAR, RHS = pinned.VALUES, pinned.SINGULAR, pinned.RHS
VALUES32, RHS32 = VALUES.astype(np.complex64), RHS.astype(np.complex64)
LANE_VALUES, LANE_RHS, LANE_FLAGS = pinned.LANE_VALUES, pinned.LANE_RHS, pinned.LANE_FLAGS
ROW_ARRAYS = ('status', 'eta', 'residual_numerator', 'residual_denominator', 'performed', 'present')
FACT_DEPENDENCY = ('a fact',)
FACT_REASON = 'the slope of this test is unavailable'


def run(function, *operands, x64=True):
    """Compile ``function`` and run it once in its precision scope."""
    with jax.enable_x64(x64):
        return jax.jit(function)(*operands)


def bits(array) -> tuple:
    """Everything that makes two arrays the same array: dtype, shape and bytes."""
    array = np.asarray(array)
    return array.dtype.str, array.shape, array.tobytes()


def row_bits(row) -> tuple:
    return tuple(bits(getattr(row, field)) for field in ROW_ARRAYS)


def solve_with_fact(values, rhs, available):
    """One equation, and a fact stated on its solution."""
    solution = pinned.direct(values, rhs)
    return derivative_gate(solution, available, dependency=FACT_DEPENDENCY, reason=FACT_REASON)


def branch_with_fact(flag, values, rhs, available):
    """The equation and its fact sit in a branch that only ``flag`` takes."""
    return lax.cond(flag, lambda: solve_with_fact(values, rhs, available),
                    lambda: jnp.zeros_like(rhs))


def solve_beside_two_checks(values, rhs, first_holds, second_holds):
    solution = pinned.direct(values, rhs)
    checkify.check(first_holds, 'the first foreign check')
    checkify.check(second_holds, 'the second foreign check')
    return solution


# ---------------------------------------------------------------------------
# Public JAX interfaces only.
# ---------------------------------------------------------------------------
PUBLIC_JAX_MODULES = {'jax', 'jax.numpy', 'jax.experimental', 'jax.extend.core',
                      'jax.extend.core.primitives', 'jax.interpreters'}
# Every name the module reaches as an attribute of ``jax`` itself.
JAX_NAMES = {'ShapeDtypeStruct', 'core', 'custom_batching', 'enable_x64', 'eval_shape',
             'make_jaxpr', 'tree', 'tree_util', 'typeof'}


class PublicInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(certificate_transport.__file__).read_text(encoding='utf-8')
        self.tree = ast.parse(self.source)

    def test_source_does_not_name_the_private_jax_package(self):
        self.assertNotIn('jax._src', self.source)

    def test_every_jax_import_is_a_public_name_of_a_public_module(self):
        modules, names = set(), set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules.add(node.module)
                if node.module.split('.')[0] == 'jax':
                    names.update(alias.name for alias in node.names)
        from_jax = {module for module in modules if module.split('.')[0] == 'jax'}
        self.assertTrue(from_jax)
        self.assertLessEqual(from_jax, PUBLIC_JAX_MODULES)
        self.assertEqual(sorted(name for name in names if name.startswith('_')), [])

    def test_names_reached_through_jax_are_the_declared_ones(self):
        reached = {node.attr for node in ast.walk(self.tree)
                   if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                   and node.value.id == 'jax'}
        self.assertEqual(sorted(reached), sorted(JAX_NAMES))

    def test_no_underscore_attribute_is_read(self):
        private = sorted({node.attr for node in ast.walk(self.tree)
                          if isinstance(node, ast.Attribute) and node.attr.startswith('_')})
        self.assertEqual(private, [])


# ---------------------------------------------------------------------------
# Fail closed.
# ---------------------------------------------------------------------------
class FailClosedTests(unittest.TestCase):
    def assert_refused(self, function, *operands, names):
        with jax.enable_x64(True), self.assertRaises(CertificateTransportError) as refused:
            certified(function)(*operands)
        self.assertIn(names, str(refused.exception))
        # The refusal ends the pass; the current transport records again.
        self.assertFalse(pass_records_equations())

    def test_equation_inside_a_primitive_without_a_rule_is_refused(self):
        def inside_checkpoint(values, rhs):
            return jax.checkpoint(lambda coefficients: pinned.direct(coefficients, rhs))(values)
        self.assert_refused(inside_checkpoint, VALUES, RHS, names='has no certificate rule')

    def test_fact_inside_a_primitive_without_a_rule_is_refused(self):
        def inside_checkpoint(rhs):
            return jax.checkpoint(lambda value: derivative_gate(
                value, True, dependency=FACT_DEPENDENCY, reason=FACT_REASON))(rhs)
        self.assert_refused(inside_checkpoint, RHS, names='has no certificate rule')

    def test_equation_in_a_loop_predicate_is_refused(self):
        def solves_in_its_predicate(values, rhs):
            return lax.while_loop(
                lambda state: jnp.abs(pinned.direct(values, state)[0]) > 1, lambda state: state / 2,
                rhs)
        self.assert_refused(solves_in_its_predicate, VALUES, RHS, names='predicate')

    def test_primitive_without_a_rule_is_left_alone_when_it_holds_no_equation(self):
        def beside_checkpoint(values, rhs):
            return pinned.direct(values, jax.checkpoint(lambda value: 2 * value)(rhs))
        evidence, solution = run(certified(beside_checkpoint), VALUES, RHS)
        self.assertEqual(len(ledger_certificates(evidence.ledger)), 1)
        self.assertEqual(bits(solution), bits(run(beside_checkpoint, VALUES, RHS)))


# ---------------------------------------------------------------------------
# The slot rule and the absent placeholder.
# ---------------------------------------------------------------------------
def execution(status, eta, *, performed=True) -> Slot:
    """The slot one execution of an equation writes."""
    return Slot(jnp.bool_(status != 0), jnp.bool_(True),
                (jnp.int32(status), jnp.float64(eta), jnp.float64(eta), jnp.float64(1.0),
                 jnp.bool_(performed)))


def slot_bits(slot: Slot) -> tuple:
    return tuple(bits(leaf) for leaf in jax.tree.leaves(slot))


class SlotRuleTests(unittest.TestCase):
    SOLVED, SOLVED_LATER = (0, 0.125), (0, 0.25)
    FAILED, FAILED_LATER = (3, np.inf), (7, np.inf)

    def assert_kept(self, executions, kept):
        with jax.enable_x64(True):
            slot = execution(*executions[0])
            for later in executions[1:]:
                slot = slot.followed_by(execution(*later))
            self.assertEqual(slot_bits(slot), slot_bits(execution(*kept)))

    def test_latest_execution_replaces_an_earlier_success(self):
        self.assert_kept([self.SOLVED, self.SOLVED_LATER], self.SOLVED_LATER)

    def test_failure_replaces_an_earlier_success(self):
        self.assert_kept([self.SOLVED, self.FAILED], self.FAILED)

    def test_failure_stays_when_later_executions_succeed(self):
        self.assert_kept([self.SOLVED, self.FAILED, self.SOLVED_LATER, self.SOLVED], self.FAILED)

    def test_first_failure_stays_when_a_later_execution_fails_too(self):
        self.assert_kept([self.FAILED, self.FAILED_LATER], self.FAILED)
        self.assert_kept([self.FAILED_LATER, self.FAILED], self.FAILED_LATER)

    def test_execution_replaces_an_absent_slot(self):
        with jax.enable_x64(True):
            solved = execution(*self.SOLVED)
            absent = Slot.absent(jax.eval_shape(lambda: solved))
            self.assertEqual(slot_bits(absent.followed_by(solved)), slot_bits(solved))

    def test_absent_slot_is_all_zero_in_the_dtypes_of_its_shape(self):
        with jax.enable_x64(True):
            shape = jax.eval_shape(lambda: Slot(
                jnp.bool_(True), jnp.ones((3,), jnp.bool_),
                (jnp.ones((3,), jnp.int32), jnp.ones((3, 2), jnp.float32), jnp.ones((3,), jnp.bool_))))
            absent = Slot.absent(shape)
        self.assertEqual(
            [(np.asarray(leaf).dtype.name, np.asarray(leaf).shape, bool(np.any(np.asarray(leaf))))
             for leaf in jax.tree.leaves(absent)],
            [('bool', (), False), ('bool', (3,), False), ('int32', (3,), False),
             ('float32', (3, 2), False), ('bool', (3,), False)])


class AbsentEvidenceTests(unittest.TestCase):
    def frame_and_its_absent_value(self, function, *operands, x64=True):
        frame = certified(function)
        with jax.enable_x64(x64):
            evidence_shape, _ = jax.eval_shape(frame, *operands)
            return run(frame, *operands, x64=x64)[0], absent_evidence(evidence_shape)

    def test_absent_evidence_has_the_structure_of_the_frame_and_no_site_present(self):
        evidence, absent = self.frame_and_its_absent_value(
            solve_with_fact, SINGULAR, RHS, np.bool_(False))
        self.assertEqual(jax.tree.structure(absent), jax.tree.structure(evidence))
        self.assertEqual([bits(leaf)[:2] for leaf in jax.tree.leaves(absent)],
                         [bits(leaf)[:2] for leaf in jax.tree.leaves(evidence)])
        self.assertFalse(bool(absent.ledger.any_failed()))
        self.assertIsNone(absent.foreign.get())
        row, = ledger_certificates(absent.ledger)
        self.assertEqual(
            [np.asarray(getattr(row, field)).tolist() for field in ('present', 'performed', 'status')],
            [False, False, 0])
        for field in ('eta', 'residual_numerator', 'residual_denominator'):
            self.assertTrue(np.isnan(np.asarray(getattr(row, field))), field)
        fact, = ledger_availability(absent.ledger)
        self.assertEqual((np.asarray(fact.present).tolist(), np.asarray(fact.available).tolist()),
                         (False, AVAILABLE_WHERE_ABSENT))

    def test_absent_evidence_is_the_other_branch_of_a_frame_that_may_not_run(self):
        frame = certified(pinned.direct)

        def frame_if(active, values, rhs):
            evidence_shape, solution_shape = jax.eval_shape(frame, values, rhs)
            skipped = absent_evidence(evidence_shape), jnp.zeros(solution_shape.shape, solution_shape.dtype)
            return lax.cond(active, lambda: frame(values, rhs), lambda: skipped)

        for active, present in ((True, True), (False, False)):
            with self.subTest(active=active):
                evidence, _ = run(frame_if, np.bool_(active), SINGULAR, RHS)
                row, = ledger_certificates(evidence.ledger)
                self.assertEqual((np.asarray(row.present).tolist(), np.asarray(row.status).tolist()),
                                 (present, 3 if present else 0))
                self.assertEqual(bool(evidence.ledger.any_failed()), present)

    def test_absent_evidence_refuses_anything_but_shapes(self):
        evidence, _ = self.frame_and_its_absent_value(pinned.direct, VALUES, RHS)
        with self.assertRaises(TypeError):
            absent_evidence(evidence)

    def test_available_reads_true_where_a_fact_is_absent(self):
        self.assertIs(AVAILABLE_WHERE_ABSENT, True)
        # Lanes 0 and 2 run the rule, lane 1 does not; in lane 2 the derivative is unavailable.
        evidence, _ = run(jax.vmap(certified(branch_with_fact)), np.array([True, False, True]),
                          np.stack([VALUES] * 3), np.stack([RHS] * 3), np.array([True, False, False]))
        fact, = ledger_availability(evidence.ledger)
        self.assertEqual((fact.dependency, fact.reason), (FACT_DEPENDENCY, FACT_REASON))
        self.assertEqual(np.asarray(fact.present).tolist(), [True, False, True])
        self.assertEqual(np.asarray(fact.available).tolist(), [True, True, False])
        # What a consumer reads: unavailable only where the fact is present and says so.
        unavailable = np.asarray(fact.present) & ~np.asarray(fact.available)
        self.assertEqual(unavailable.tolist(), [False, False, True])


# ---------------------------------------------------------------------------
# The arrays the ledger holds.
# ---------------------------------------------------------------------------
class SinglePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.ambient_x64 = jax.config.jax_enable_x64

    def assert_single_precision_slots(self, evidence):
        for key, slot in evidence.ledger.entries():
            self.assertIsInstance(key, EquationKey)
            self.assertEqual(
                [np.asarray(leaf).dtype.name for leaf in jax.tree.leaves(slot)],
                ['bool', 'bool', 'int32', 'float32', 'float32', 'float32', 'bool'])
            # The static description of the slot names the same dtypes.
            self.assertEqual([dtype for _, dtype in key.descriptors],
                             ['int32', 'float32', 'float32', 'float32', 'bool'])

    def test_complex64_lanes_keep_float32_slots_present_or_absent(self):
        with jax.enable_x64(False):
            evidence, _ = run(
                jax.vmap(certified(pinned.one_of_two_branches)), np.array([True, False, False]),
                np.stack([VALUES32, VALUES32, np.zeros_like(VALUES32)]), np.stack([RHS32] * 3),
                x64=False)
            self.assertFalse(jax.config.jax_enable_x64)
        self.assertEqual(jax.config.jax_enable_x64, self.ambient_x64)
        self.assertEqual(len(evidence.ledger.keys), 2)
        self.assert_single_precision_slots(evidence)
        for row in ledger_certificates(evidence.ledger):
            self.assertEqual(np.asarray(row.eta).dtype, np.float32)

    def test_absent_evidence_of_a_complex64_frame_keeps_float32_slots(self):
        frame = certified(pinned.reverse(pinned.squared_norm))
        with jax.enable_x64(False):
            evidence_shape, _ = jax.eval_shape(frame, VALUES32, RHS32)
            absent = absent_evidence(evidence_shape)
        self.assertEqual(jax.config.jax_enable_x64, self.ambient_x64)
        self.assertEqual(len(absent.ledger.keys), 2)
        self.assert_single_precision_slots(absent)


class LedgerAsPytreeTests(unittest.TestCase):
    def lanes(self):
        return run(jax.vmap(certified(pinned.one_of_two_branches)), LANE_FLAGS, LANE_VALUES, LANE_RHS)

    def test_ledger_passes_through_a_later_compiled_call_unchanged(self):
        evidence, _ = self.lanes()
        returned = run(lambda ledger: ledger, evidence.ledger)
        self.assertEqual(returned.keys, evidence.ledger.keys)
        self.assertEqual([bits(leaf) for leaf in jax.tree.leaves(returned)],
                         [bits(leaf) for leaf in jax.tree.leaves(evidence.ledger)])
        self.assertEqual([row_bits(row) for row in ledger_certificates(returned)],
                         [row_bits(row) for row in ledger_certificates(evidence.ledger)])

    def test_compiled_call_reads_whether_a_ledger_holds_a_failure(self):
        def any_failed(ledger):
            return run(lambda held: held.any_failed(), ledger)

        for values, failed in ((VALUES, False), (SINGULAR, True)):
            with self.subTest(failed=failed):
                evidence, _ = run(certified(pinned.direct), values, RHS)
                answer = any_failed(evidence.ledger)
                self.assertEqual((np.asarray(answer).dtype, bool(answer)), (np.bool_, failed))
        # Lane 1 fails; the ledger of all lanes holds a failure.
        self.assertTrue(bool(any_failed(self.lanes()[0].ledger)))

    def test_slice_of_one_lane_is_the_ledger_of_that_lane_alone(self):
        evidence, _ = self.lanes()
        for lane in range(len(LANE_FLAGS)):
            with self.subTest(lane=lane):
                sliced = jax.tree.map(lambda leaf, lane=lane: leaf[lane], evidence.ledger)
                alone, _ = run(certified(pinned.one_of_two_branches),
                               LANE_FLAGS[lane], LANE_VALUES[lane], LANE_RHS[lane])
                self.assertEqual(sliced.keys, alone.ledger.keys)
                self.assertEqual([row_bits(row) for row in ledger_certificates(sliced)],
                                 [row_bits(row) for row in ledger_certificates(alone.ledger)])
                self.assertEqual(bool(sliced.any_failed()), bool(alone.ledger.any_failed()))

    def test_entries_are_keyed_by_what_identifies_a_row(self):
        evidence, _ = run(certified(solve_with_fact), VALUES, RHS, np.bool_(True))
        equation, = (key for key in evidence.ledger.keys if isinstance(key, EquationKey))
        fact, = (key for key in evidence.ledger.keys if isinstance(key, AvailabilityKey))
        self.assertEqual(
            (type(equation), equation.role, equation.stage, equation.derivative_order,
             equation.dependency, equation.archived),
            (EquationKey, 'primal', 3, 0, ('direct',), False))
        self.assertEqual((type(fact), fact.dependency, fact.reason, fact.archived),
                         (AvailabilityKey, FACT_DEPENDENCY, FACT_REASON, False))


# ---------------------------------------------------------------------------
# Reading rows on the host.
# ---------------------------------------------------------------------------
class RowProjectionTests(unittest.TestCase):
    BASIS = dict(basis_kind='parameter', basis_indices=(4, 1, 3, 0, 2), work_index=6,
                 frequency_source_index=9)

    def test_basis_labels_reach_every_row_as_they_do_today(self):
        with jax.enable_x64(True):
            error, _ = jax.jit(jax.vmap(checked_native(pinned.direct)))(LANE_VALUES, LANE_RHS)
            evidence, _ = jax.jit(jax.vmap(certified(pinned.direct)))(LANE_VALUES, LANE_RHS)
        today, = sparse_certificates(error, **self.BASIS)
        row, = ledger_certificates(evidence.ledger, **self.BASIS)
        self.assertEqual(type(row), type(today))
        self.assertEqual(
            (row.basis_kind, row.basis_indices, row.work_index, row.frequency_source_index),
            (today.basis_kind, today.basis_indices, today.work_index, today.frequency_source_index))
        self.assertEqual(row.basis_indices, (4, 1, 3, 0, 2))
        self.assertEqual(row_bits(row), row_bits(today))

    def test_basis_that_does_not_fit_the_rows_is_refused_as_today(self):
        with jax.enable_x64(True):
            error, _ = jax.jit(jax.vmap(checked_native(pinned.direct)))(LANE_VALUES, LANE_RHS)
            evidence, _ = jax.jit(jax.vmap(certified(pinned.direct)))(LANE_VALUES, LANE_RHS)
        for options in (dict(basis_kind='lane', basis_indices=(0, 1, 2, 3, 4)),
                        dict(basis_indices=(0, 1, 2, 3, 4)),
                        dict(basis_kind='parameter', basis_indices=(0, 1))):
            with self.subTest(options=options):
                with self.assertRaises(ValueError) as today:
                    sparse_certificates(error, **options)
                with self.assertRaises(ValueError) as refused:
                    ledger_certificates(evidence.ledger, **options)
                self.assertEqual(str(refused.exception), str(today.exception))


# ---------------------------------------------------------------------------
# Beside the current transport.
# ---------------------------------------------------------------------------
class ForeignCheckTests(unittest.TestCase):
    def test_equation_failure_is_not_returned_as_a_foreign_check(self):
        evidence, _ = run(certified(pinned.direct), SINGULAR, RHS)
        self.assertTrue(bool(evidence.ledger.any_failed()))
        self.assertIsNone(evidence.foreign.get())
        # The foreign value holds no entry for the equation at all.
        self.assertEqual(jax.tree.leaves(evidence.foreign), [])
        evidence.foreign.throw()

    def test_foreign_check_is_returned_with_its_message_beside_an_equation_failure(self):
        for values, failed in ((VALUES, False), (SINGULAR, True)):
            with self.subTest(equation_failed=failed):
                evidence, _ = run(certified(pinned.solve_beside_foreign_check), values, RHS,
                                  np.bool_(False))
                self.assertEqual(bool(evidence.ledger.any_failed()), failed)
                self.assertEqual(evidence.foreign.get(),
                                 'a check that SCNSim does not own (`check` failed)')
                with self.assertRaises(checkify.JaxRuntimeError):
                    evidence.foreign.throw()

    def test_first_failing_foreign_check_is_the_one_reported(self):
        for holds, message in (((True, True), None),
                               ((False, True), 'the first foreign check (`check` failed)'),
                               ((True, False), 'the second foreign check (`check` failed)'),
                               ((False, False), 'the first foreign check (`check` failed)')):
            with self.subTest(holds=holds):
                evidence, _ = run(certified(solve_beside_two_checks), SINGULAR, RHS,
                                  *map(np.bool_, holds))
                self.assertEqual(evidence.foreign.get(), message)

    def test_foreign_check_inside_a_sequentially_mapped_body_reaches_the_frame(self):
        def trips_with_invariant(values, rhs, plan, count):
            def trip(state):
                index, _ = state
                checkify.debug_check(index < 1, 'trip {index} is beyond the first', index=index)
                return index + 1, pinned.loop_trip(values, rhs, plan[index]).value
            return lax.while_loop(lambda state: state[0] < count, trip,
                                  (jnp.int32(0), jnp.zeros_like(rhs)))

        body = certified_sequential_primal(trips_with_invariant)
        plan = pinned.padded((pinned.RUN, pinned.RUN))
        for trips, message in ((1, None), (2, 'trip 1 is beyond the first (`check` failed)')):
            with self.subTest(trips=trips):
                evidence, (count, _) = run(certified(body), VALUES, RHS, plan, np.int32(trips))
                self.assertEqual(evidence.foreign.get(), message)
                self.assertEqual(int(count), trips)
                row, = ledger_certificates(evidence.ledger)
                self.assertEqual(np.asarray(row.status).tolist(), 0)


class CurrentTransportTests(unittest.TestCase):
    """The current transport keeps recording its own frames while both exist."""

    def assert_current_rows(self, function, *operands):
        """``function`` returns today's rows whether or not a certified frame is inside it."""
        with jax.enable_x64(True):
            error, _ = jax.jit(checked_native(function))(*operands)
        rows = sparse_certificates(error)
        self.assertEqual([(row.role, row.stage, row.dependency, np.asarray(row.status).tolist())
                          for row in rows], [('primal', 3, ('direct',), 3)])
        return rows

    def test_current_frame_records_after_a_certified_frame_was_built(self):
        run(certified(pinned.direct), SINGULAR, RHS)
        self.assertFalse(pass_records_equations())
        self.assert_current_rows(pinned.direct, SINGULAR, RHS)

    def test_current_frame_around_a_certified_frame_records_its_equation(self):
        def around(values, rhs):
            evidence, solution = certified(pinned.direct)(values, rhs)
            return solution
        inside, = self.assert_current_rows(around, SINGULAR, RHS)
        alone, = self.assert_current_rows(pinned.direct, SINGULAR, RHS)
        self.assertEqual(row_bits(inside), row_bits(alone))

    def test_certified_frame_around_a_current_frame_records_its_equation(self):
        def around(values, rhs):
            error, solution = checked_native(pinned.direct)(values, rhs)
            return solution
        evidence, _ = run(certified(around), SINGULAR, RHS)
        row, = ledger_certificates(evidence.ledger)
        alone, = self.assert_current_rows(pinned.direct, SINGULAR, RHS)
        self.assertEqual(row_bits(row), row_bits(alone))
        self.assertIsNone(evidence.foreign.get())


# ---------------------------------------------------------------------------
# The identity primitives and the registry.
# ---------------------------------------------------------------------------
class IdentityPrimitiveTests(unittest.TestCase):
    def test_gate_returns_its_value_outside_a_certified_frame(self):
        def gated(value, available):
            return derivative_gate(value, available, dependency=FACT_DEPENDENCY, reason=FACT_REASON)
        with jax.enable_x64(True):
            for label, call in (('eager', gated), ('jit', jax.jit(gated))):
                with self.subTest(call=label):
                    self.assertEqual(bits(call(jnp.asarray(RHS), jnp.bool_(False))), bits(RHS))
            mapped = jax.jit(jax.vmap(gated))(jnp.stack([RHS, 2 * RHS]), jnp.array([True, False]))
            self.assertEqual(bits(mapped), bits(np.stack([RHS, 2 * RHS])))

    def test_gate_takes_one_boolean(self):
        with self.assertRaises(TypeError):
            derivative_gate(jnp.asarray(RHS), jnp.array([True, False]), dependency=FACT_DEPENDENCY,
                            reason=FACT_REASON)

    def test_gate_records_its_fact_where_it_ran(self):
        for available in (True, False):
            with self.subTest(available=available):
                evidence, solution = run(certified(solve_with_fact), VALUES, RHS, np.bool_(available))
                fact, = ledger_availability(evidence.ledger)
                self.assertEqual((fact.dependency, fact.reason), (FACT_DEPENDENCY, FACT_REASON))
                self.assertEqual((np.asarray(fact.present).tolist(), np.asarray(fact.available).tolist()),
                                 (True, available))
                self.assertEqual(bits(solution), bits(run(pinned.direct, VALUES, RHS)))
                # An unavailable derivative is a failure of the frame; the equation solved.
                self.assertEqual(bool(evidence.ledger.any_failed()), not available)

    def test_sequential_body_runs_outside_a_certified_frame(self):
        body = certified_sequential_primal(pinned.while_trips)
        plan = pinned.padded((pinned.RUN, pinned.ZERO_RHS))
        with jax.enable_x64(True):
            expected = jax.jit(pinned.while_trips)(VALUES, RHS, plan, np.int32(2))
            for label, call in (('eager', body), ('jit', jax.jit(body))):
                with self.subTest(call=label):
                    count, solution = call(VALUES, RHS, plan, np.int32(2))
                    self.assertEqual((bits(count), bits(solution)), tuple(map(bits, expected)))
            counts, _ = jax.jit(jax.vmap(body))(np.stack([VALUES] * 2), np.stack([RHS] * 2),
                                                np.stack([plan] * 2), np.array([2, 0], dtype=np.int32))
            self.assertEqual(np.asarray(counts).tolist(), [2, 0])

    def test_evidence_needs_a_value_to_travel_with(self):
        evidence, _ = run(certified(pinned.direct), VALUES, RHS)
        self.assertIsInstance(evidence, Evidence)
        with self.assertRaises(CertificateTransportError):
            attach_evidence((), evidence)
        # A bare ledger is not what a certified call returns.
        with self.assertRaises(TypeError):
            attach_evidence(RHS, evidence.ledger)

    def test_attached_evidence_is_read_wherever_a_leaf_of_its_packet_is_used(self):
        body = certified_sequential_primal(pinned.while_trips)
        plan = pinned.padded((pinned.RUN, pinned.SINGULAR_MATRIX))
        for used in (0, 1):
            with self.subTest(used_leaf=used):
                evidence, _ = run(certified(lambda *operands, used=used: body(*operands)[used]),
                                  VALUES, RHS, plan, np.int32(2))
                row, = ledger_certificates(evidence.ledger)
                self.assertEqual((row.dependency, np.asarray(row.status).tolist()), (('loop',), 3))
                self.assertTrue(evidence.ledger.keys[0].archived)


class RegistrationTests(unittest.TestCase):
    def test_certificate_group_needs_role_stage_and_derivative_order(self):
        with self.assertRaises(TypeError):
            register_equation(object(), value_count=1, certificate_groups=(('tangent', 4),))


if __name__ == '__main__':
    unittest.main()
