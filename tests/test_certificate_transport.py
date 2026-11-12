"""Equation certificates as a compiled JAX program returns them.

Every SCNSim equation returns a certificate with its value: a status, eta, the
residual numerator and denominator, and whether the solve was performed.
``checked_native`` brings the certificates of a complete transformed function
out of the compiled program; ``sparse_certificates`` and
``derivative_availability`` read them on the host. Issue 48 replaces the way
they travel. These tests pin what a reader of the rows relies on, so that the
replacement can be judged against them:

- an equation that did not run, one that was skipped and one that ran are
  told apart;
- a site inside ``cond``, ``while_loop`` or ``scan`` keeps one row: its first
  failure, otherwise its latest execution;
- ``vmap`` gives one entry per lane;
- adjoint, tangent, mixed and frequency-tangent equations have rows of their
  own;
- a derivative-availability fact is recorded where its rule ran;
- a sequentially mapped Newton body returns its rows with a lane axis;
- 32-bit payloads stay 32-bit and the x64 setting is left alone.

The evidence object a checked call returns is opaque here: it is only handed
to the two projections. A row is named by its role, stage, derivative order
and dependency, never by ``site`` or by its position. The one exception is
``SiteAndRowOrderTests``, which pins ``site`` and the row order as they are
today because Issue 48 is due to change them.

No tolerance is asserted. What an entry reports is structural: its status
code, its two flags, and whether its residual numbers are finite, infinite,
NaN or exactly zero.
"""
from __future__ import annotations

from typing import Any, NamedTuple
import unittest

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import checkify
import numpy as np

from scnsim.compilation.program import SparsePattern
from scnsim.numerics.native_equation_jet import checked_sparse_frequency_jet
from scnsim.numerics.native_root import compile_native_null_vector
from scnsim.numerics.native_sparse import (
    checked_native, checked_sequential_primal, checked_sparse_solve,
    derivative_availability, sparse_certificates,
)
from scnsim.numerics.sparse_solve import SparseEquation
from tests.jax_runtime import initialize_jax_through_scnsim


def setUpModule():
    # These tests call JAX directly. SCNSim has to initialize it first, or the
    # SCNSim tests that run after this module in the same process fail.
    initialize_jax_through_scnsim()


# ---------------------------------------------------------------------------
# The fixture system: four nodes, a banded complex matrix.
# ---------------------------------------------------------------------------
NODE_COUNT = 4
_ROW, _COLUMN = np.indices((NODE_COUNT, NODE_COUNT))
# Banded: every entry except the two diagonals at distance 2.
_BAND = np.abs(_ROW - _COLUMN) != 2
PATTERN, _ = SparsePattern.from_coo((NODE_COUNT, NODE_COUNT), *np.nonzero(_BAND))
EQUATION = SparseEquation(PATTERN)
# Strictly diagonally dominant: a diagonal of magnitude at least 4 against two
# off-diagonal entries per row that sum to less than 2. Every solve of it, of a
# multiple of it and of its transpose therefore succeeds in both precisions.
MATRIX = np.where(_ROW == _COLUMN, 4.0 + 1j * _ROW,
                  np.where(_BAND, 0.25 * (1 + _ROW - _COLUMN) + 0.125j * (_ROW + _COLUMN), 0))
VALUES = MATRIX[np.asarray(PATTERN.rows), np.asarray(PATTERN.cols)]
# The all-zero matrix: its factorization stops at an exactly singular factor.
SINGULAR = np.zeros_like(VALUES)
RHS = np.array([1 + 0.5j, -0.75 + 0.25j, 0.5 - 1j, 0.25 + 0.75j])
DIRECTION = 0.125 * (np.arange(VALUES.size) - 5) + 0j

# Five lanes; lane 1 holds the singular matrix.
LANE_VALUES = np.stack([VALUES, SINGULAR, 2 * VALUES, VALUES, 3 * VALUES])
LANE_RHS = np.stack([RHS, RHS, 2 * RHS, -RHS, 1j * RHS])
LANE_FLAGS = np.array([True, False, True, False, False])


def solve(values, rhs, **options):
    """One certified equation on the fixture pattern; the options name its row."""
    return checked_sparse_solve(EQUATION, values, rhs, **options)


def direct(values, rhs):
    return solve(values, rhs, dependency=('direct',)).value


def squared_norm(values, rhs):
    """A real loss of one solution, so that a derivative has an equation to solve."""
    return jnp.sum(jnp.abs(solve(values, rhs, dependency=('loss',)).value) ** 2)


def reverse(function):
    """Value and gradient of a real ``function`` in the real part of the coefficients."""
    def value_and_gradient(values, *rest):
        return jax.value_and_grad(
            lambda real: function(real + 1j * jnp.imag(values), *rest))(jnp.real(values))
    return value_and_gradient


def forward(function):
    """Value and derivative of ``function`` along a direction of the coefficients."""
    def value_and_tangent(values, rhs, direction):
        return jax.jvp(lambda coefficients: function(coefficients, rhs), (values,), (direction,))
    return value_and_tangent


def second_forward(values, rhs, direction):
    """The second directional derivative of the solution along one direction."""
    def first_derivative(coefficients):
        return jax.jvp(lambda inner: direct(inner, rhs), (coefficients,), (direction,))[1]
    return jax.jvp(first_derivative, (values,), (direction,))


def forward_over_reverse(values, rhs, direction):
    """A Hessian-vector product of the loss in the real part of the coefficients."""
    gradient = jax.grad(lambda real: squared_norm(real + 1j * jnp.imag(values), rhs))
    return jax.jvp(gradient, (jnp.real(values),), (jnp.real(direction),))


def frequency_jet(values, rhs):
    """A solution with its frequency derivative, from one call-local factor."""
    jet = checked_sparse_frequency_jet(EQUATION, values, rhs, 0.5j * values, 0.25 * rhs,
                                       dependency=('jet',))
    return jet.primal.value, jet.frequency.value


def one_of_two_branches(flag, values, rhs):
    """Two sites of which a call runs one: stage 3 when ``flag`` holds, else stage 7."""
    return lax.cond(
        flag,
        lambda: solve(values, rhs, dependency=('true branch',)).value,
        lambda: solve(2 * values, rhs, stage=7, dependency=('false branch',)).value)


def branch_loss(values, flag, rhs):
    return jnp.sum(jnp.abs(one_of_two_branches(flag, values, rhs)) ** 2)


def one_equation_of_each_kind(values, rhs, run_last):
    """A frame with a solved, a failed, a skipped and a conditional equation."""
    first = solve(values, rhs, dependency=('solved',))
    singular = solve(jnp.zeros_like(values), rhs, dependency=('failed',))
    after_failure = solve(values, rhs, prior_status=singular.aux.status, stage=7,
                          dependency=('skipped',))
    conditional = lax.cond(
        run_last,
        lambda: solve(values, rhs, stage=9, dependency=('conditional',)).value,
        lambda: jnp.zeros_like(rhs))
    return first.value, singular.aux.status, after_failure.aux.status, conditional


def solve_beside_foreign_check(values, rhs, holds):
    solution = direct(values, rhs)
    checkify.check(holds, 'a check that SCNSim does not own')
    return solution


# ---------------------------------------------------------------------------
# Loops: one equation site, and a plan of what each trip makes of it.
# ---------------------------------------------------------------------------
RUN, ZERO_RHS, SINGULAR_MATRIX, PRIOR_FAILURE = range(4)
# The status handed to a PRIOR_FAILURE trip. Any nonzero status skips the solve.
PRIOR_STATUS = 9
# Trip plans of the while loop are padded to one length, so its operands keep one shape.
LONGEST_PLAN = 4


def loop_trip(values, rhs, kind):
    """The equation site shared by both loops; ``kind`` decides this trip's outcome."""
    coefficients = jnp.where(kind == SINGULAR_MATRIX, jnp.zeros_like(values), values)
    right_hand_side = jnp.where(kind == ZERO_RHS, jnp.zeros_like(rhs), rhs)
    prior = jnp.where(kind == PRIOR_FAILURE, jnp.int32(PRIOR_STATUS), jnp.int32(0))
    return solve(coefficients, right_hand_side, prior_status=prior, dependency=('loop',))


def while_trips(values, rhs, plan, count):
    """Run the first ``count`` trips of ``plan``. The trip count is data, as in Newton."""
    def trip(state):
        index, _ = state
        return index + 1, loop_trip(values, rhs, plan[index]).value
    return lax.while_loop(lambda state: state[0] < count, trip,
                          (jnp.int32(0), jnp.zeros_like(rhs)))


def scan_trips(values, rhs, plan):
    def trip(count, kind):
        return count + 1, loop_trip(values, rhs, kind).value
    return lax.scan(trip, jnp.int32(0), plan)


def padded(plan):
    return np.array(tuple(plan) + (RUN,) * (LONGEST_PLAN - len(plan)), dtype=np.int32)


# ---------------------------------------------------------------------------
# Derivative availability: SCNSim's null-vector chart states one fact in its JVP rule.
# ---------------------------------------------------------------------------
CHART_DEPENDENCY = ('root_occurrence', 0, 0)
# The reason the chart's rule gives when the largest coordinate of the vector is not unique.
CHART_REASON = 'root_slope_unresolved: slope_certificate null-vector phase chart is unavailable'
null_vector = compile_native_null_vector(3, dependency=CHART_DEPENDENCY)
CHART_MATRIX = np.array([[1 + 1j, 2, 0.5j], [0.25, 3 - 0.5j, 0.5], [-0.5, 0.25, 3]])
CHART_DIRECTION = 0.125 * np.arange(9).reshape(3, 3) + 0j
CHART_LEFT = np.array([0.25 + 0j, 1, 0.125])
UNIQUE_LARGEST = np.array([1 + 0j, 0.5, 0.25])
TIED_LARGEST = np.array([1 + 0j, 1, 0.25])


def chart_forward(matrix, vector, direction):
    """The chart and its derivative along a matrix direction: its JVP rule runs."""
    return jax.jvp(lambda variable: null_vector(variable, vector, CHART_LEFT), (matrix,), (direction,))


def chart_forward_in_branch(flag, matrix, vector, direction):
    """The same derivative, but the chart sits in a branch only ``flag`` takes."""
    return jax.jvp(
        lambda variable: lax.cond(flag, lambda: null_vector(variable, vector, CHART_LEFT),
                                  lambda: variable[0] * vector),
        (matrix,), (direction,))


def loss_gradient_beside_chart(scale, values, rhs, matrix, vector):
    """Reverse mode in one real scale, with the chart of the scaled matrix as auxiliary output."""
    def loss_and_chart(variable):
        return squared_norm(variable * values, rhs), null_vector(variable * matrix, vector, CHART_LEFT)
    return jax.value_and_grad(loss_and_chart, has_aux=True)(scale)


# ---------------------------------------------------------------------------
# Reading one checked call.
# ---------------------------------------------------------------------------
class CheckedCall(NamedTuple):
    """What one compiled checked call returned, read on the host."""
    values: Any
    certificates: tuple
    availability: tuple


def checked_call(function, *operands, lanes=False, x64=True) -> CheckedCall:
    """Compile ``function`` with its certificates and run it once.

    ``lanes`` maps the checked function over the leading axis of every operand.
    Operands are NumPy arrays, so the precision scope decides their JAX dtype.
    """
    checked = checked_native(function)
    if lanes:
        checked = jax.vmap(checked)
    with jax.enable_x64(x64):
        evidence, values = jax.jit(checked)(*operands)
    # Read after the precision scope has closed: a projection must not depend on
    # the ambient x64 setting.
    return CheckedCall(values, sparse_certificates(evidence), derivative_availability(evidence))


class RowName(NamedTuple):
    """The identity of a certificate row that does not depend on its position."""
    role: str
    stage: int
    derivative_order: int
    dependency: tuple


def name_of(row) -> RowName:
    return RowName(row.role, row.stage, row.derivative_order, row.dependency)


class Entry(NamedTuple):
    """What one equation, or one lane of it, reports in its row."""
    present: bool
    performed: bool
    status: int
    residual: str  # A name returned by residual_kind.


# Ran and was certified. The right-hand side was not zero, so the residual scale is positive.
SOLVED = Entry(present=True, performed=True, status=0, residual='finite')
# Ran on a zero right-hand side: the solution, its residual and its scale are exactly zero.
SOLVED_ZERO = Entry(present=True, performed=True, status=0, residual='zero')
# The site exists in the program but did not run.
ABSENT = Entry(present=False, performed=False, status=0, residual='nan')


def failed(status) -> Entry:
    """Ran and failed. A singular matrix reports the stage of its equation as status."""
    return Entry(present=True, performed=True, status=status, residual='infinite')


def skipped(status) -> Entry:
    """Was reached with an earlier failure, whose status it carries; the solve did not run."""
    return Entry(present=True, performed=False, status=status, residual='infinite')


def residual_kind(eta, numerator, denominator) -> str:
    """Name what the three residual numbers of one entry say; no magnitude is judged."""
    numbers = np.array([eta, numerator, denominator])
    if np.all(np.isnan(numbers)):
        return 'nan'
    if np.all(np.isposinf(numbers)):
        return 'infinite'
    if np.all(numbers == 0):
        return 'zero'
    if np.all(np.isfinite(numbers)) and denominator > 0:
        return 'finite'
    return f'unexpected {numbers}'


def flattened(expected) -> tuple[tuple[int, ...], list[Entry]]:
    """The shape and the entries of one Entry or of nested sequences of them."""
    if isinstance(expected, Entry):
        return (), [expected]
    parts = [flattened(item) for item in expected]
    return (len(parts),) + parts[0][0], [entry for _, entries in parts for entry in entries]


class CertificateTestCase(unittest.TestCase):
    def row(self, call, name: RowName):
        """The one row with this name."""
        rows = [row for row in call.certificates if name_of(row) == name]
        self.assertEqual(len(rows), 1, f'{name} names {len(rows)} rows')
        return rows[0]

    def assert_row_names(self, call, names):
        """The call returned exactly these rows, whatever their order."""
        self.assertCountEqual([name_of(row) for row in call.certificates], names)

    def assert_entries(self, row, expected, *, real_dtype=np.float64):
        """Every entry of ``row`` reports ``expected``, in the certificate dtypes."""
        shape, entries = flattened(expected)
        arrays = {field: np.asarray(getattr(row, field)) for field in (
            'status', 'eta', 'residual_numerator', 'residual_denominator', 'performed', 'present')}
        self.assertEqual({field: array.shape for field, array in arrays.items()},
                         dict.fromkeys(arrays, shape))
        real = np.dtype(real_dtype).name
        self.assertEqual(
            {field: array.dtype.name for field, array in arrays.items()},
            {'status': 'int32', 'eta': real, 'residual_numerator': real,
             'residual_denominator': real, 'performed': 'bool', 'present': 'bool'})
        flat = {field: array.reshape(-1) for field, array in arrays.items()}
        self.assertEqual(flat['present'].tolist(), [entry.present for entry in entries])
        self.assertEqual(flat['performed'].tolist(), [entry.performed for entry in entries])
        self.assertEqual(flat['status'].tolist(), [entry.status for entry in entries])
        self.assertEqual(
            [residual_kind(*numbers) for numbers in zip(
                flat['eta'], flat['residual_numerator'], flat['residual_denominator'])],
            [entry.residual for entry in entries])


# ---------------------------------------------------------------------------
# The tests.
# ---------------------------------------------------------------------------
class AbsentSkippedExecutedTests(CertificateTestCase):
    def test_solved_failed_skipped_and_absent_equations_are_told_apart(self):
        for run_last, conditional in ((False, ABSENT), (True, SOLVED)):
            with self.subTest(run_last=run_last):
                call = checked_call(one_equation_of_each_kind, VALUES, RHS, np.bool_(run_last))
                names = {label: RowName('primal', stage, 0, (label,)) for label, stage in (
                    ('solved', 3), ('failed', 3), ('skipped', 7), ('conditional', 9))}
                self.assert_row_names(call, names.values())
                self.assert_entries(self.row(call, names['solved']), SOLVED)
                # A singular matrix fails at the stage of its own equation.
                self.assert_entries(self.row(call, names['failed']), failed(3))
                # The skipped equation carries the failure it was handed, not its own stage 7.
                self.assert_entries(self.row(call, names['skipped']), skipped(3))
                self.assert_entries(self.row(call, names['conditional']), conditional)
                # The certificate is also an ordinary output of the equation.
                self.assertEqual([int(status) for status in call.values[1:3]], [3, 3])


class ForeignCheckTests(CertificateTestCase):
    def test_foreign_check_in_the_frame_leaves_the_owned_row_as_it_is(self):
        direct_row = RowName('primal', 3, 0, ('direct',))
        for values, holds, expected in ((VALUES, True, SOLVED), (VALUES, False, SOLVED),
                                        (SINGULAR, False, failed(3))):
            with self.subTest(holds=holds, expected=expected):
                # The compiled call returns whether or not the foreign check holds.
                call = checked_call(solve_beside_foreign_check, values, RHS, np.bool_(holds))
                self.assert_row_names(call, [direct_row])
                self.assert_entries(self.row(call, direct_row), expected)


class ControlFlowSlotTests(CertificateTestCase):
    LOOP = RowName('primal', 3, 0, ('loop',))

    def assert_loop_row(self, plan, expected):
        """After ``plan`` the row of the loop site reports ``expected`` in both kinds of loop."""
        calls = {
            'while_loop': checked_call(while_trips, VALUES, RHS, padded(plan), np.int32(len(plan))),
            'scan': checked_call(scan_trips, VALUES, RHS, np.array(plan, dtype=np.int32)),
        }
        for loop, call in calls.items():
            with self.subTest(loop=loop, plan=plan):
                # However many trips ran, the site has one row.
                self.assert_row_names(call, [self.LOOP])
                self.assertEqual(int(call.values[0]), len(plan))
                self.assert_entries(self.row(call, self.LOOP), expected)

    def test_cond_reports_the_branch_taken_and_leaves_the_other_absent(self):
        true_branch = RowName('primal', 3, 0, ('true branch',))
        false_branch = RowName('primal', 7, 0, ('false branch',))
        for flag in (True, False):
            with self.subTest(flag=flag):
                call = checked_call(one_of_two_branches, np.bool_(flag), VALUES, RHS)
                self.assert_row_names(call, [true_branch, false_branch])
                self.assert_entries(self.row(call, true_branch), SOLVED if flag else ABSENT)
                self.assert_entries(self.row(call, false_branch), ABSENT if flag else SOLVED)

    def test_loop_without_a_trip_leaves_its_site_absent(self):
        self.assert_loop_row((), ABSENT)

    def test_loop_reports_its_latest_execution(self):
        # Only the trip on the zero right-hand side has an exactly zero residual,
        # so the row tells which trip it reports.
        self.assert_loop_row((RUN, RUN, ZERO_RHS), SOLVED_ZERO)
        self.assert_loop_row((ZERO_RHS, RUN, RUN), SOLVED)

    def test_loop_keeps_a_failure_when_later_trips_succeed(self):
        self.assert_loop_row((RUN, SINGULAR_MATRIX, RUN, RUN), failed(3))

    def test_loop_keeps_its_first_failure_when_a_later_trip_fails_too(self):
        self.assert_loop_row((RUN, SINGULAR_MATRIX, PRIOR_FAILURE, RUN), failed(3))
        self.assert_loop_row((RUN, PRIOR_FAILURE, SINGULAR_MATRIX, RUN), skipped(PRIOR_STATUS))


class LaneTests(CertificateTestCase):
    DIRECT = RowName('primal', 3, 0, ('direct',))
    ONE_SINGULAR_LANE = [SOLVED, failed(3), SOLVED, SOLVED, SOLVED]

    def test_vmap_of_a_checked_function_gives_one_entry_per_lane(self):
        call = checked_call(direct, LANE_VALUES, LANE_RHS, lanes=True)
        self.assert_row_names(call, [self.DIRECT])
        self.assert_entries(self.row(call, self.DIRECT), self.ONE_SINGULAR_LANE)

    def test_checking_a_vmapped_function_gives_the_same_lanes(self):
        call = checked_call(jax.vmap(direct), LANE_VALUES, LANE_RHS)
        self.assert_row_names(call, [self.DIRECT])
        self.assert_entries(self.row(call, self.DIRECT), self.ONE_SINGULAR_LANE)

    def test_vmap_inside_a_lane_adds_a_further_axis(self):
        def two_right_hand_sides(values, pair):
            return jax.vmap(lambda rhs: direct(values, rhs))(pair)
        call = checked_call(two_right_hand_sides, LANE_VALUES,
                            np.stack([LANE_RHS, 2 * LANE_RHS], axis=1), lanes=True)
        self.assert_row_names(call, [self.DIRECT])
        self.assert_entries(self.row(call, self.DIRECT),
                            [[entry, entry] for entry in self.ONE_SINGULAR_LANE])

    def test_lanes_are_absent_in_the_branch_they_did_not_take(self):
        call = checked_call(one_of_two_branches, LANE_FLAGS, LANE_VALUES, LANE_RHS, lanes=True)
        self.assert_entries(self.row(call, RowName('primal', 3, 0, ('true branch',))),
                            [SOLVED, ABSENT, SOLVED, ABSENT, ABSENT])
        # Lane 1 takes the false branch with the singular matrix: it fails at stage 7.
        self.assert_entries(self.row(call, RowName('primal', 7, 0, ('false branch',))),
                            [ABSENT, failed(7), ABSENT, SOLVED, SOLVED])


class DerivativeEquationTests(CertificateTestCase):
    PRIMAL = RowName('primal', 3, 0, ('loss',))
    TANGENT = RowName('tangent', 4, 1, ('loss',))
    ADJOINT = RowName('adjoint', 4, 1, ('loss',))
    MIXED = RowName('mixed', 4, 2, ('loss',))

    def test_reverse_mode_adds_an_adjoint_row_beside_the_primal_row(self):
        call = checked_call(reverse(squared_norm), VALUES, RHS)
        self.assert_row_names(call, [self.PRIMAL, self.ADJOINT])
        self.assert_entries(self.row(call, self.PRIMAL), SOLVED)
        self.assert_entries(self.row(call, self.ADJOINT), SOLVED)

    def test_failed_primal_skips_the_adjoint_equation(self):
        call = checked_call(reverse(squared_norm), SINGULAR, RHS)
        self.assert_row_names(call, [self.PRIMAL, self.ADJOINT])
        self.assert_entries(self.row(call, self.PRIMAL), failed(3))
        self.assert_entries(self.row(call, self.ADJOINT), skipped(3))

    def test_forward_mode_adds_a_tangent_row_beside_the_primal_row(self):
        call = checked_call(forward(squared_norm), VALUES, RHS, DIRECTION)
        self.assert_row_names(call, [self.PRIMAL, self.TANGENT])
        self.assert_entries(self.row(call, self.PRIMAL), SOLVED)
        self.assert_entries(self.row(call, self.TANGENT), SOLVED)

    def test_failed_primal_skips_the_tangent_equation(self):
        call = checked_call(forward(squared_norm), SINGULAR, RHS, DIRECTION)
        self.assert_row_names(call, [self.PRIMAL, self.TANGENT])
        self.assert_entries(self.row(call, self.PRIMAL), failed(3))
        self.assert_entries(self.row(call, self.TANGENT), skipped(3))

    def test_second_forward_derivative_adds_a_mixed_row(self):
        primal, tangent, mixed = (RowName(role, stage, order, ('direct',)) for role, stage, order in (
            ('primal', 3, 0), ('tangent', 4, 1), ('mixed', 4, 2)))
        call = checked_call(second_forward, VALUES, RHS, DIRECTION)
        # Each of the two differentiations solves a first-order tangent equation.
        self.assert_row_names(call, [primal, tangent, tangent, mixed])
        for row in call.certificates:
            self.assert_entries(row, SOLVED)

    def test_forward_over_reverse_has_one_row_of_each_role(self):
        call = checked_call(forward_over_reverse, VALUES, RHS, DIRECTION)
        self.assert_row_names(call, [self.PRIMAL, self.TANGENT, self.ADJOINT, self.MIXED])
        for row in call.certificates:
            self.assert_entries(row, SOLVED)

    def test_frequency_jet_has_a_frequency_tangent_row(self):
        primal = RowName('primal', 3, 0, ('jet',))
        frequency_tangent = RowName('frequency_tangent', 4, 1, ('jet',))
        for values, first, second in ((VALUES, SOLVED, SOLVED), (SINGULAR, failed(3), skipped(3))):
            with self.subTest(first=first):
                call = checked_call(frequency_jet, values, RHS)
                self.assert_row_names(call, [primal, frequency_tangent])
                self.assert_entries(self.row(call, primal), first)
                self.assert_entries(self.row(call, frequency_tangent), second)

    def test_adjoint_rows_follow_the_branch_each_lane_took(self):
        call = checked_call(reverse(branch_loss), LANE_VALUES, LANE_FLAGS, LANE_RHS, lanes=True)
        took_true = [SOLVED, ABSENT, SOLVED, ABSENT, ABSENT]
        expected = {
            RowName('primal', 3, 0, ('true branch',)): took_true,
            RowName('adjoint', 4, 1, ('true branch',)): took_true,
            RowName('primal', 7, 0, ('false branch',)): [ABSENT, failed(7), ABSENT, SOLVED, SOLVED],
            RowName('adjoint', 4, 1, ('false branch',)): [ABSENT, skipped(7), ABSENT, SOLVED, SOLVED],
        }
        self.assert_row_names(call, expected)
        for name, entries in expected.items():
            with self.subTest(name=name):
                self.assert_entries(self.row(call, name), entries)


class DerivativeAvailabilityTests(CertificateTestCase):
    # The chart's rule solves its tangent equation at the stage of the root's implicit equation.
    CHART_TANGENT = RowName('tangent', 13, 1, CHART_DEPENDENCY)

    def fact(self, call):
        """The one availability fact of the call, which carries the chart's dependency and reason."""
        self.assertEqual([(fact.dependency, fact.reason) for fact in call.availability],
                         [(CHART_DEPENDENCY, CHART_REASON)])
        return call.availability[0]

    def assert_fact(self, fact, *, available, present):
        for field, expected in (('available', available), ('present', present)):
            array = np.asarray(getattr(fact, field))
            self.assertEqual(array.dtype, np.bool_, field)
            self.assertEqual(array.tolist(), expected, field)

    def test_no_fact_is_recorded_when_the_rule_did_not_run(self):
        call = checked_call(null_vector, CHART_MATRIX, UNIQUE_LARGEST, CHART_LEFT)
        self.assertEqual(call.availability, ())
        self.assertEqual(call.certificates, ())

    def test_fact_carries_its_dependency_and_reason_where_the_rule_ran(self):
        for vector, available in ((UNIQUE_LARGEST, True), (TIED_LARGEST, False)):
            with self.subTest(available=available):
                call = checked_call(chart_forward, CHART_MATRIX, vector, CHART_DIRECTION)
                self.assert_fact(self.fact(call), available=available, present=True)
                self.assert_row_names(call, [self.CHART_TANGENT])
                self.assert_entries(self.row(call, self.CHART_TANGENT), SOLVED)

    def test_fact_holds_or_fails_per_lane(self):
        call = checked_call(chart_forward, np.stack([CHART_MATRIX] * 3),
                            np.stack([UNIQUE_LARGEST, TIED_LARGEST, UNIQUE_LARGEST]),
                            np.stack([CHART_DIRECTION] * 3), lanes=True)
        self.assert_fact(self.fact(call), available=[True, False, True], present=[True] * 3)

    def test_fact_is_absent_in_a_lane_whose_branch_did_not_run_the_rule(self):
        call = checked_call(chart_forward_in_branch, np.array([True, False, True]),
                            np.stack([CHART_MATRIX] * 3),
                            np.stack([UNIQUE_LARGEST, TIED_LARGEST, TIED_LARGEST]),
                            np.stack([CHART_DIRECTION] * 3), lanes=True)
        fact = self.fact(call)
        present = np.asarray(fact.present)
        self.assertEqual(present.tolist(), [True, False, True])
        # What an absent lane holds in ``available`` has no meaning; only present lanes are read.
        self.assertEqual(np.asarray(fact.available)[present].tolist(), [True, False])
        self.assert_entries(self.row(call, self.CHART_TANGENT), [SOLVED, ABSENT, SOLVED])

    def test_fact_survives_reverse_mode(self):
        primal, adjoint = RowName('primal', 3, 0, ('loss',)), RowName('adjoint', 4, 1, ('loss',))
        for vector, available in ((UNIQUE_LARGEST, True), (TIED_LARGEST, False)):
            with self.subTest(available=available):
                call = checked_call(loss_gradient_beside_chart, np.float64(1.0), VALUES, RHS,
                                    CHART_MATRIX, vector)
                self.assert_fact(self.fact(call), available=available, present=True)
                # Nothing consumes the chart's tangent here, so its equation is not part
                # of the program: only the loss has rows.
                self.assert_row_names(call, [primal, adjoint])
                self.assert_entries(self.row(call, primal), SOLVED)
                self.assert_entries(self.row(call, adjoint), SOLVED)


class SequentialMapTests(CertificateTestCase):
    LOOP = RowName('primal', 3, 0, ('loop',))
    # Five lanes that stop after 3, 0, 2, 4 and 1 trips.
    PLANS = ((RUN, RUN, ZERO_RHS), (), (ZERO_RHS, RUN), (RUN, SINGULAR_MATRIX, RUN, RUN), (RUN,))
    EXPECTED = [SOLVED_ZERO, ABSENT, SOLVED, failed(3), SOLVED]

    def lane_operands(self):
        lanes = len(self.PLANS)
        return (np.stack([VALUES] * lanes), np.stack([RHS] * lanes),
                np.stack([padded(plan) for plan in self.PLANS]),
                np.array([len(plan) for plan in self.PLANS], dtype=np.int32))

    def assert_lanes(self, call):
        self.assert_row_names(call, [self.LOOP])
        self.assert_entries(self.row(call, self.LOOP), self.EXPECTED)
        self.assertEqual(np.asarray(call.values[0]).tolist(), [len(plan) for plan in self.PLANS])

    def test_scalar_call_returns_the_row_of_its_body(self):
        body = checked_sequential_primal(while_trips)
        for plan, expected in zip(self.PLANS, self.EXPECTED):
            with self.subTest(plan=plan):
                call = checked_call(body, VALUES, RHS, padded(plan), np.int32(len(plan)))
                self.assert_row_names(call, [self.LOOP])
                self.assert_entries(self.row(call, self.LOOP), expected)

    def test_checking_the_mapped_body_returns_a_lane_axis(self):
        self.assert_lanes(checked_call(jax.vmap(checked_sequential_primal(while_trips)),
                                       *self.lane_operands()))

    def test_mapping_the_checked_body_returns_the_same_lanes(self):
        self.assert_lanes(checked_call(checked_sequential_primal(while_trips),
                                       *self.lane_operands(), lanes=True))


class SinglePrecisionTests(CertificateTestCase):
    VALUES32 = VALUES.astype(np.complex64)
    RHS32 = RHS.astype(np.complex64)

    def setUp(self):
        self.ambient_x64 = jax.config.jax_enable_x64

    def single_precision_call(self, function, *operands, **options):
        with jax.enable_x64(False):
            call = checked_call(function, *operands, x64=False, **options)
            # Neither the compiled call nor the projection left a precision scope open.
            self.assertFalse(jax.config.jax_enable_x64)
        self.assertEqual(jax.config.jax_enable_x64, self.ambient_x64)
        return call

    def test_complex64_solve_keeps_float32_payloads(self):
        call = self.single_precision_call(direct, self.VALUES32, self.RHS32)
        self.assertEqual(np.asarray(call.values).dtype, np.complex64)
        direct_row = RowName('primal', 3, 0, ('direct',))
        self.assert_row_names(call, [direct_row])
        self.assert_entries(self.row(call, direct_row), SOLVED, real_dtype=np.float32)

    def test_complex64_adjoint_keeps_float32_payloads(self):
        call = self.single_precision_call(reverse(squared_norm), self.VALUES32, self.RHS32)
        self.assertEqual([np.asarray(value).dtype for value in call.values], [np.float32] * 2)
        for name in (RowName('primal', 3, 0, ('loss',)), RowName('adjoint', 4, 1, ('loss',))):
            self.assert_entries(self.row(call, name), SOLVED, real_dtype=np.float32)

    def test_absent_and_failed_complex64_lanes_keep_float32_payloads(self):
        call = self.single_precision_call(
            one_of_two_branches, np.array([True, False, False]),
            np.stack([self.VALUES32, self.VALUES32, np.zeros_like(self.VALUES32)]),
            np.stack([self.RHS32] * 3), lanes=True)
        self.assert_entries(self.row(call, RowName('primal', 3, 0, ('true branch',))),
                            [SOLVED, ABSENT, ABSENT], real_dtype=np.float32)
        self.assert_entries(self.row(call, RowName('primal', 7, 0, ('false branch',))),
                            [ABSENT, SOLVED, failed(7)], real_dtype=np.float32)


class SiteAndRowOrderTests(CertificateTestCase):
    """The one place that pins ``site`` and the order of rows.

    Today ``site`` is the path of an equation in the traced program: its index
    among all equations of that program, JAX's own included, preceded for a
    nested program by the path of the enclosing equation, the name of the
    parameter that holds the nested program and, for ``cond``, the branch
    index. A primitive that solves several equations appends
    ``('equation', ordinal)``. Rows are sorted by the text of ``site``, so
    ``(22,)`` comes before ``(7,)``. Issue 48 decides what ``site`` means next;
    its last step changes this test and no other.
    """

    def test_site_labels_and_row_order_as_of_today(self):
        direct_row = RowName('primal', 3, 0, ('direct',))
        loss_row = RowName('primal', 3, 0, ('loss',))
        loop_row = RowName('primal', 3, 0, ('loop',))
        one_trip = (padded((RUN,)), np.int32(1))
        frames = (
            ('one solve', checked_call(direct, VALUES, RHS),
             [((0,), direct_row)]),
            ('reverse mode', checked_call(reverse(squared_norm), VALUES, RHS),
             [((22,), RowName('adjoint', 4, 1, ('loss',))), ((7,), loss_row)]),
            ('forward mode', checked_call(forward(squared_norm), VALUES, RHS, DIRECTION),
             [((1, 'equation', 0), loss_row),
              ((1, 'equation', 1), RowName('tangent', 4, 1, ('loss',)))]),
            ('second forward derivative', checked_call(second_forward, VALUES, RHS, DIRECTION),
             [((20,), RowName('tangent', 4, 1, ('direct',))),
              ((4, 'equation', 0), direct_row),
              ((4, 'equation', 1), RowName('tangent', 4, 1, ('direct',))),
              ((70,), RowName('mixed', 4, 2, ('direct',)))]),
            ('cond', checked_call(one_of_two_branches, np.bool_(True), VALUES, RHS),
             [((1, 'branches', 0, 1), RowName('primal', 7, 0, ('false branch',))),
              ((1, 'branches', 1, 0), RowName('primal', 3, 0, ('true branch',)))]),
            ('while_loop', checked_call(while_trips, VALUES, RHS, *one_trip),
             [((1, 'body_jaxpr', 14), loop_row)]),
            ('scan', checked_call(scan_trips, VALUES, RHS, np.array((RUN,), dtype=np.int32)),
             [((0, 'jaxpr', 9), loop_row)]),
            # The row of a sequentially mapped body keeps the path it has inside that body.
            ('sequential map',
             checked_call(checked_sequential_primal(while_trips), VALUES, RHS, *one_trip),
             [((1, 'body_jaxpr', 14), loop_row)]),
        )
        for label, call, expected in frames:
            with self.subTest(frame=label):
                self.assertEqual([(row.site, name_of(row)) for row in call.certificates], expected)
        # An availability fact is labelled with the path of the check its rule emitted.
        chart = checked_call(chart_forward, CHART_MATRIX, UNIQUE_LARGEST, CHART_DIRECTION)
        self.assertEqual([row.site for row in chart.certificates], [(61,)])
        self.assertEqual([fact.site for fact in chart.availability], [(9,)])


if __name__ == '__main__':
    unittest.main()
