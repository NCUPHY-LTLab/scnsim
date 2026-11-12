"""The certificate ledger returns the rows the current transport returns.

Issue 48 moves equation certificates out of JAX's private check machinery and
into SCNSim's own ledger (``scnsim.numerics.certificate_transport``). Until the
switch both transports exist, and no caller uses the ledger. Each call here
compiles one function with both and requires the same result:

- the same number of certificate rows in the same order, with the same
  ``site``, role, stage, derivative order and dependency;
- the six arrays of every row bit for bit: dtype, shape and bytes;
- the same returned values, bit for bit;
- the same derivative-availability facts: ``site``, dependency and reason, and
  ``present`` bit for bit. ``available`` is compared where the fact is present:
  what an absent lane holds is a placeholder of JAX today and a defined value
  in the ledger.

The calls are those of ``test_certificate_transport.py``, which pins today's
behaviour. ``PINNED_CALLS`` lists them under the test that makes them, with the
functions and operands of that module, and one test requires that none of its
tests is missing. ``FURTHER_CALLS`` adds what that module does not reach, and
``SITE_MOVES`` the two places where the transports agree in everything but
``site``.

A function differs between the two transports in two places only, and never
in what it computes:

- a sequentially mapped body is wrapped by ``checked_sequential_primal`` or by
  ``certified_sequential_primal``;
- a derivative-availability fact is stated by ``derivative_available`` or by
  ``derivative_gate``. The product's null-vector chart states its fact in
  today's form, which the ledger does not read, so for the ledger the chart is
  written out here with the gate (``null_vector_chart``). The same chart
  written with today's form is first required to return exactly what the
  product's chart returns.

No tolerance is used.
"""
from __future__ import annotations

from collections.abc import Callable
import json
from typing import NamedTuple
import unittest

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import checkify
import numpy as np

from scnsim.compilation.program import SparsePattern
from scnsim.numerics.certificate_transport import (
    certified, certified_sequential_primal, derivative_gate, ledger_availability,
    ledger_certificates,
)
from scnsim.numerics.native_sparse import (
    checked_dense_solve, checked_native, checked_sequential_primal, derivative_availability,
    derivative_available, sparse_certificates,
)
from scnsim.numerics.sparse_solve import SparseEquation
from tests import test_certificate_transport as pinned
from tests.jax_runtime import initialize_jax_through_scnsim


def setUpModule():
    # These tests call JAX directly. SCNSim has to initialize it first, or the
    # SCNSim tests that run after this module in the same process fail.
    initialize_jax_through_scnsim()


# ---------------------------------------------------------------------------
# The two transports.
# ---------------------------------------------------------------------------
class Transport(NamedTuple):
    """One way of bringing certificates out of a compiled program."""
    certify: Callable  # A function -> a function that returns (evidence, values).
    sequential_primal: Callable  # A scalar body -> the body that keeps its early exit per lane.
    certificates: Callable  # The returned evidence -> its certificate rows.
    availability: Callable  # The returned evidence -> its availability facts.
    state_fact: Callable  # (value, available, dependency=, reason=) -> value, inside a JVP rule.


def _state_fact_as_today(value, available, *, dependency, reason):
    derivative_available(available, dependency=dependency, reason=reason)
    return value


CURRENT = Transport(checked_native, checked_sequential_primal, sparse_certificates,
                    derivative_availability, _state_fact_as_today)
LEDGER = Transport(certified, certified_sequential_primal,
                   lambda evidence: ledger_certificates(evidence.ledger),
                   lambda evidence: ledger_availability(evidence.ledger), derivative_gate)


class Call(NamedTuple):
    """One compiled call."""
    function: Callable[[Transport], Callable]  # The function a transport certifies.
    operands: tuple
    lanes: bool = False  # Map the certified function over the leading axis of every operand.
    x64: bool = True


def call(function, *operands, lanes=False, x64=True) -> Call:
    """A call of one function that both transports certify as it is."""
    return Call(lambda transport: function, operands, lanes, x64)


def call_built_by(build, *operands, lanes=False) -> Call:
    """A call whose function is written in the form of the transport that certifies it."""
    return Call(build, operands, lanes)


def compiled(transport: Transport, made: Call) -> pinned.CheckedCall:
    """Compile and run ``made`` with ``transport``; read the result on the host."""
    certified_function = transport.certify(made.function(transport))
    if made.lanes:
        certified_function = jax.vmap(certified_function)
    with jax.enable_x64(made.x64):
        evidence, values = jax.jit(certified_function)(*made.operands)
    # Read after the precision scope has closed, as a consumer does.
    return pinned.CheckedCall(values, transport.certificates(evidence),
                              transport.availability(evidence))


# ---------------------------------------------------------------------------
# The null-vector chart, written out so that its fact can take either form.
# ---------------------------------------------------------------------------
CHART_WIDTH = 3


def null_vector_chart(state_fact):
    """The chart of ``native_root.compile_native_null_vector``, stating its fact by ``state_fact``.

    The rule is that function's, statement for statement, so that both forms
    trace the same program around the one statement that differs.
    """
    indices = np.arange(CHART_WIDTH - 1)
    equation = SparseEquation(SparsePattern.from_coo(
        (CHART_WIDTH - 1, CHART_WIDTH - 1),
        np.tile(indices, CHART_WIDTH - 1), np.repeat(indices, CHART_WIDTH - 1))[0])

    @jax.custom_jvp
    def vector(matrix, primal_vector, primal_left):
        return primal_vector

    @vector.defjvp
    def direction(primals, tangents):
        matrix, primal_vector, primal_left = primals
        matrix_dot, _, _ = tangents
        v = vector(matrix, primal_vector, primal_left)
        magnitude = jnp.abs(v)
        v = state_fact(v, jnp.count_nonzero(magnitude == jnp.max(magnitude)) == 1,
                       dependency=pinned.CHART_DEPENDENCY, reason=pinned.CHART_REASON)
        pivot = jnp.argmax(magnitude)
        omitted_row = jnp.argmax(jnp.abs(primal_left))
        local_indices = jnp.arange(CHART_WIDTH - 1, dtype=jnp.int32)
        rows = local_indices + (local_indices >= omitted_row)
        columns = local_indices + (local_indices >= pivot)
        chart = v / v[pivot]
        minor = matrix[rows[:, None], columns[None, :]]
        rhs = -(matrix_dot @ chart)[rows]
        solution = checked_dense_solve(equation, minor, rhs, stage=13, role='tangent',
                                       derivative_order=1, dependency=pinned.CHART_DEPENDENCY)
        chart_dot = jnp.zeros_like(chart).at[columns].set(solution.value)
        _, vector_dot = jax.jvp(lambda value: value / jnp.linalg.norm(value), (chart,), (chart_dot,))
        return v, vector_dot

    return vector


class ChartFunctions(NamedTuple):
    """The functions of the pinned module that use the chart, over one chart."""
    primal: Callable
    forward: Callable
    forward_in_branch: Callable
    beside_loss_gradient: Callable


def functions_of(chart) -> ChartFunctions:
    """``chart_forward``, ``chart_forward_in_branch`` and ``loss_gradient_beside_chart`` of the
    pinned module, over ``chart`` in place of the product's."""
    def forward(matrix, vector, direction):
        return jax.jvp(lambda variable: chart(variable, vector, pinned.CHART_LEFT),
                       (matrix,), (direction,))

    def forward_in_branch(flag, matrix, vector, direction):
        return jax.jvp(
            lambda variable: lax.cond(flag, lambda: chart(variable, vector, pinned.CHART_LEFT),
                                      lambda: variable[0] * vector),
            (matrix,), (direction,))

    def beside_loss_gradient(scale, values, rhs, matrix, vector):
        def loss_and_chart(variable):
            return (pinned.squared_norm(variable * values, rhs),
                    chart(variable * matrix, vector, pinned.CHART_LEFT))
        return jax.value_and_grad(loss_and_chart, has_aux=True)(scale)

    return ChartFunctions(chart, forward, forward_in_branch, beside_loss_gradient)


PRODUCT_CHART = ChartFunctions(pinned.null_vector, pinned.chart_forward,
                               pinned.chart_forward_in_branch, pinned.loss_gradient_beside_chart)
CHART_AS_TODAY = functions_of(null_vector_chart(CURRENT.state_fact))
CHART_WITH_GATE = functions_of(null_vector_chart(LEDGER.state_fact))


def chart_call(function_name, *operands, lanes=False) -> Call:
    """A call of a chart function: the product's for today's transport, the gate's for the ledger."""
    def build(transport):
        chart = PRODUCT_CHART if transport is CURRENT else CHART_WITH_GATE
        return getattr(chart, function_name)
    return call_built_by(build, *operands, lanes=lanes)


# ---------------------------------------------------------------------------
# The calls of test_certificate_transport.py, under the test that makes them.
# ---------------------------------------------------------------------------
VALUES, SINGULAR, RHS, DIRECTION = pinned.VALUES, pinned.SINGULAR, pinned.RHS, pinned.DIRECTION
LANE_VALUES, LANE_RHS, LANE_FLAGS = pinned.LANE_VALUES, pinned.LANE_RHS, pinned.LANE_FLAGS
RUN, ZERO_RHS, SINGULAR_MATRIX, PRIOR_FAILURE = (
    pinned.RUN, pinned.ZERO_RHS, pinned.SINGULAR_MATRIX, pinned.PRIOR_FAILURE)
VALUES32, RHS32 = VALUES.astype(np.complex64), RHS.astype(np.complex64)
CHART_MATRIX, CHART_DIRECTION = pinned.CHART_MATRIX, pinned.CHART_DIRECTION
UNIQUE_LARGEST, TIED_LARGEST = pinned.UNIQUE_LARGEST, pinned.TIED_LARGEST
THREE_CHART_MATRICES = np.stack([CHART_MATRIX] * 3)
THREE_CHART_DIRECTIONS = np.stack([CHART_DIRECTION] * 3)
SEQUENTIAL_PLANS = pinned.SequentialMapTests.PLANS
SEQUENTIAL_LANES = (
    np.stack([VALUES] * len(SEQUENTIAL_PLANS)), np.stack([RHS] * len(SEQUENTIAL_PLANS)),
    np.stack([pinned.padded(plan) for plan in SEQUENTIAL_PLANS]),
    np.array([len(plan) for plan in SEQUENTIAL_PLANS], dtype=np.int32))


def both_loops(plan) -> tuple[Call, Call]:
    """The while loop and the scan of the pinned module over one plan of trips."""
    return (call(pinned.while_trips, VALUES, RHS, pinned.padded(plan), np.int32(len(plan))),
            call(pinned.scan_trips, VALUES, RHS, np.array(plan, dtype=np.int32)))


def two_right_hand_sides(values, pair):
    return jax.vmap(lambda rhs: pinned.direct(values, rhs))(pair)


def sequential_body(transport):
    return transport.sequential_primal(pinned.while_trips)


def mapped_sequential_body(transport):
    return jax.vmap(transport.sequential_primal(pinned.while_trips))


PINNED_CALLS: dict[str, tuple[Call, ...]] = {
    'AbsentSkippedExecutedTests.test_solved_failed_skipped_and_absent_equations_are_told_apart': (
        call(pinned.one_equation_of_each_kind, VALUES, RHS, np.bool_(False)),
        call(pinned.one_equation_of_each_kind, VALUES, RHS, np.bool_(True))),
    'ForeignCheckTests.test_foreign_check_in_the_frame_leaves_the_owned_row_as_it_is': (
        call(pinned.solve_beside_foreign_check, VALUES, RHS, np.bool_(True)),
        call(pinned.solve_beside_foreign_check, VALUES, RHS, np.bool_(False)),
        call(pinned.solve_beside_foreign_check, SINGULAR, RHS, np.bool_(False))),
    'ControlFlowSlotTests.test_cond_reports_the_branch_taken_and_leaves_the_other_absent': (
        call(pinned.one_of_two_branches, np.bool_(True), VALUES, RHS),
        call(pinned.one_of_two_branches, np.bool_(False), VALUES, RHS)),
    'ControlFlowSlotTests.test_loop_without_a_trip_leaves_its_site_absent': both_loops(()),
    'ControlFlowSlotTests.test_loop_reports_its_latest_execution': (
        *both_loops((RUN, RUN, ZERO_RHS)), *both_loops((ZERO_RHS, RUN, RUN))),
    'ControlFlowSlotTests.test_loop_keeps_a_failure_when_later_trips_succeed':
        both_loops((RUN, SINGULAR_MATRIX, RUN, RUN)),
    'ControlFlowSlotTests.test_loop_keeps_its_first_failure_when_a_later_trip_fails_too': (
        *both_loops((RUN, SINGULAR_MATRIX, PRIOR_FAILURE, RUN)),
        *both_loops((RUN, PRIOR_FAILURE, SINGULAR_MATRIX, RUN))),
    'LaneTests.test_vmap_of_a_checked_function_gives_one_entry_per_lane': (
        call(pinned.direct, LANE_VALUES, LANE_RHS, lanes=True),),
    'LaneTests.test_checking_a_vmapped_function_gives_the_same_lanes': (
        call(jax.vmap(pinned.direct), LANE_VALUES, LANE_RHS),),
    'LaneTests.test_vmap_inside_a_lane_adds_a_further_axis': (
        call(two_right_hand_sides, LANE_VALUES, np.stack([LANE_RHS, 2 * LANE_RHS], axis=1),
             lanes=True),),
    'LaneTests.test_lanes_are_absent_in_the_branch_they_did_not_take': (
        call(pinned.one_of_two_branches, LANE_FLAGS, LANE_VALUES, LANE_RHS, lanes=True),),
    'DerivativeEquationTests.test_reverse_mode_adds_an_adjoint_row_beside_the_primal_row': (
        call(pinned.reverse(pinned.squared_norm), VALUES, RHS),),
    'DerivativeEquationTests.test_failed_primal_skips_the_adjoint_equation': (
        call(pinned.reverse(pinned.squared_norm), SINGULAR, RHS),),
    'DerivativeEquationTests.test_forward_mode_adds_a_tangent_row_beside_the_primal_row': (
        call(pinned.forward(pinned.squared_norm), VALUES, RHS, DIRECTION),),
    'DerivativeEquationTests.test_failed_primal_skips_the_tangent_equation': (
        call(pinned.forward(pinned.squared_norm), SINGULAR, RHS, DIRECTION),),
    'DerivativeEquationTests.test_second_forward_derivative_adds_a_mixed_row': (
        call(pinned.second_forward, VALUES, RHS, DIRECTION),),
    'DerivativeEquationTests.test_forward_over_reverse_has_one_row_of_each_role': (
        call(pinned.forward_over_reverse, VALUES, RHS, DIRECTION),),
    'DerivativeEquationTests.test_frequency_jet_has_a_frequency_tangent_row': (
        call(pinned.frequency_jet, VALUES, RHS), call(pinned.frequency_jet, SINGULAR, RHS)),
    'DerivativeEquationTests.test_adjoint_rows_follow_the_branch_each_lane_took': (
        call(pinned.reverse(pinned.branch_loss), LANE_VALUES, LANE_FLAGS, LANE_RHS, lanes=True),),
    'DerivativeAvailabilityTests.test_no_fact_is_recorded_when_the_rule_did_not_run': (
        chart_call('primal', CHART_MATRIX, UNIQUE_LARGEST, pinned.CHART_LEFT),),
    'DerivativeAvailabilityTests.test_fact_carries_its_dependency_and_reason_where_the_rule_ran': (
        chart_call('forward', CHART_MATRIX, UNIQUE_LARGEST, CHART_DIRECTION),
        chart_call('forward', CHART_MATRIX, TIED_LARGEST, CHART_DIRECTION)),
    'DerivativeAvailabilityTests.test_fact_holds_or_fails_per_lane': (
        chart_call('forward', THREE_CHART_MATRICES,
                   np.stack([UNIQUE_LARGEST, TIED_LARGEST, UNIQUE_LARGEST]),
                   THREE_CHART_DIRECTIONS, lanes=True),),
    'DerivativeAvailabilityTests.test_fact_is_absent_in_a_lane_whose_branch_did_not_run_the_rule': (
        chart_call('forward_in_branch', np.array([True, False, True]), THREE_CHART_MATRICES,
                   np.stack([UNIQUE_LARGEST, TIED_LARGEST, TIED_LARGEST]),
                   THREE_CHART_DIRECTIONS, lanes=True),),
    'DerivativeAvailabilityTests.test_fact_survives_reverse_mode': (
        chart_call('beside_loss_gradient', np.float64(1.0), VALUES, RHS, CHART_MATRIX,
                   UNIQUE_LARGEST),
        chart_call('beside_loss_gradient', np.float64(1.0), VALUES, RHS, CHART_MATRIX,
                   TIED_LARGEST)),
    'SequentialMapTests.test_scalar_call_returns_the_row_of_its_body': tuple(
        call_built_by(sequential_body, VALUES, RHS, pinned.padded(plan), np.int32(len(plan)))
        for plan in SEQUENTIAL_PLANS),
    'SequentialMapTests.test_checking_the_mapped_body_returns_a_lane_axis': (
        call_built_by(mapped_sequential_body, *SEQUENTIAL_LANES),),
    'SequentialMapTests.test_mapping_the_checked_body_returns_the_same_lanes': (
        call_built_by(sequential_body, *SEQUENTIAL_LANES, lanes=True),),
    'SinglePrecisionTests.test_complex64_solve_keeps_float32_payloads': (
        call(pinned.direct, VALUES32, RHS32, x64=False),),
    'SinglePrecisionTests.test_complex64_adjoint_keeps_float32_payloads': (
        call(pinned.reverse(pinned.squared_norm), VALUES32, RHS32, x64=False),),
    'SinglePrecisionTests.test_absent_and_failed_complex64_lanes_keep_float32_payloads': (
        call(pinned.one_of_two_branches, np.array([True, False, False]),
             np.stack([VALUES32, VALUES32, np.zeros_like(VALUES32)]), np.stack([RHS32] * 3),
             lanes=True, x64=False),),
    # The frames of the site test that no other test makes with the same operands.
    'SiteAndRowOrderTests.test_site_labels_and_row_order_as_of_today': (
        call(pinned.direct, VALUES, RHS),
        *both_loops((RUN,)),
        call_built_by(sequential_body, VALUES, RHS, pinned.padded((RUN,)), np.int32(1))),
}


# ---------------------------------------------------------------------------
# Calls the pinned module does not make.
# ---------------------------------------------------------------------------
ROOT_DEPENDENCY = ('root_occurrence', 0, 0)


def root_like_rule(transport):
    """The shape of the product's root: a sequentially mapped body under a custom JVP rule
    that states a fact and solves its own tangent equation."""
    body = transport.sequential_primal(pinned.while_trips)

    @jax.custom_jvp
    def root(values, rhs, plan, count):
        return body(values, rhs, plan, count)

    @root.defjvp
    def direction(primals, tangents):
        values, rhs, plan, count = primals
        trips, solution = root(*primals)
        solution = transport.state_fact(solution, trips > 1, dependency=ROOT_DEPENDENCY,
                                        reason='the body made fewer than two trips')
        tangent = pinned.solve(values, tangents[1] - solution, stage=13, role='tangent',
                               derivative_order=1, dependency=ROOT_DEPENDENCY)
        return (trips, solution), (np.zeros((), dtype=jax.dtypes.float0), tangent.value)

    def forward(values, rhs, plan, count, rhs_direction):
        return jax.jvp(lambda variable: root(values, variable, plan, count), (rhs,), (rhs_direction,))

    return forward


def nested_sequential_body(transport):
    """A sequentially mapped body inside another: its rows are attached twice."""
    inner = transport.sequential_primal(pinned.while_trips)
    return jax.vmap(transport.sequential_primal(lambda *operands: inner(*operands)))


def with_foreign_check_in_the_body(transport):
    def trips_with_invariant(values, rhs, plan, count):
        def trip(state):
            index, _ = state
            checkify.debug_check(index < pinned.LONGEST_PLAN, 'trip {index} is beyond the plan',
                                 index=index)
            return index + 1, pinned.loop_trip(values, rhs, plan[index]).value
        return lax.while_loop(lambda state: state[0] < count, trip,
                              (jnp.int32(0), jnp.zeros_like(rhs)))
    return jax.vmap(transport.sequential_primal(trips_with_invariant))


def frame_inside_a_batched_map(transport):
    """A certified frame inside ``lax.map``: the enclosing frame records its equation again."""
    def function(values, right_hand_sides):
        return lax.map(lambda rhs: transport.certify(pinned.direct)(values, rhs)[1],
                       right_hand_sides, batch_size=2)
    return function


def root_like_rule_in_a_branch(transport):
    forward = root_like_rule(transport)

    def function(flag, values, rhs, plan, count, rhs_direction):
        return lax.cond(flag, lambda: forward(values, rhs, plan, count, rhs_direction)[1][1],
                        lambda: jnp.zeros_like(rhs))
    return function


def reverse_through_root_like_rule(transport):
    forward = root_like_rule(transport)

    def function(values, rhs, plan, count):
        def loss(variable):
            (_, solution), _ = forward(values, variable, plan, count, jnp.ones_like(variable))
            return jnp.sum(jnp.abs(solution) ** 2)
        return jax.value_and_grad(lambda real: loss(real + 1j * jnp.imag(rhs)))(jnp.real(rhs))
    return function


def mapped_pushforward(values, rhs, directions):
    solution, push = jax.linearize(lambda coefficients: pinned.direct(coefficients, rhs), values)
    return solution, lax.map(push, directions)


ROOT_LANES = (*SEQUENTIAL_LANES, np.stack([RHS] * len(SEQUENTIAL_PLANS)))
TWO_TRIPS = (pinned.padded((RUN, RUN)), np.int32(2))
FURTHER_CALLS: dict[str, tuple[Call, ...]] = {
    'a sequentially mapped body under a custom JVP rule with a fact and a tangent equation': (
        call_built_by(root_like_rule, VALUES, RHS, *TWO_TRIPS, RHS),
        call_built_by(root_like_rule, VALUES, RHS, pinned.padded((RUN,)), np.int32(1), RHS),
        call_built_by(root_like_rule, *ROOT_LANES, lanes=True)),
    'that rule inside a branch that some lanes take': (
        call_built_by(root_like_rule_in_a_branch, np.array([True, False, True, True, False]),
                      *ROOT_LANES, lanes=True),),
    'that rule under reverse mode': (
        call_built_by(reverse_through_root_like_rule, VALUES, RHS, *TWO_TRIPS),),
    'a sequentially mapped body inside another': (
        call_built_by(nested_sequential_body, *SEQUENTIAL_LANES),),
    'a foreign debug check inside a sequentially mapped body': (
        call_built_by(with_foreign_check_in_the_body, *SEQUENTIAL_LANES),),
    'a certified frame inside a batched lax.map': (
        call_built_by(frame_inside_a_batched_map, VALUES,
                      np.stack([RHS, 2 * RHS, 3 * RHS, 4 * RHS, 5 * RHS])),),
    'a linearized solve pushed forward along mapped directions': (
        call(mapped_pushforward, VALUES, RHS, np.stack([DIRECTION, 2 * DIRECTION, 1j * DIRECTION])),),
}
# How many rows and facts the first call of each group returns: the comparison
# is not between two empty results.
FURTHER_ROWS_AND_FACTS = {
    'a sequentially mapped body under a custom JVP rule with a fact and a tangent equation': (2, 1),
    'that rule inside a branch that some lanes take': (2, 1),
    'that rule under reverse mode': (3, 2),
    'a sequentially mapped body inside another': (1, 0),
    'a foreign debug check inside a sequentially mapped body': (1, 0),
    'a certified frame inside a batched lax.map': (2, 0),
    'a linearized solve pushed forward along mapped directions': (2, 0),
}


# ---------------------------------------------------------------------------
# Where only ``site`` differs.
#
# Today ``site`` counts every equation of the traced program, JAX's own
# included. In two places the transports emit a different number of equations
# of their own, so an equation behind such a place has another ``site``. Its
# arrays and its other labels are the same. Issue 48 (decision 1) makes
# ``site`` count SCNSim's equations only in its last step; neither place can
# move it then.
# ---------------------------------------------------------------------------
def equation_after_a_sequential_body(transport):
    """An equation behind a sequentially mapped body, in the same program.

    Today's transport returns the body's rows through an archive that takes
    15 equations; the ledger attaches them with one. The product calls such a
    body inside a custom JVP function, where those equations sit in a program
    of their own and move nothing.
    """
    body = transport.sequential_primal(pinned.while_trips)

    def function(values, rhs, plan, count):
        trips, solution = body(values, rhs, plan, count)
        return trips, pinned.solve(values, solution, stage=7, dependency=('behind',)).value
    return function


def chart_under_an_inner_vmap(transport):
    """A fact stated in a rule that is mapped inside the frame.

    Both transports reduce the lanes to one fact: the derivative is
    unavailable if it is in any lane. JAX's batching rule for its check
    emits 19 equations to do so, the gate's one.
    """
    chart = PRODUCT_CHART if transport is CURRENT else CHART_WITH_GATE
    return jax.vmap(chart.forward)


class SiteMove(NamedTuple):
    made: Call
    current_sites: list  # Of the rows, then of the facts.
    ledger_sites: list


BODY_SITE = (1, 'body_jaxpr', 14)
SITE_MOVES = (
    SiteMove(call_built_by(equation_after_a_sequential_body, VALUES, RHS,
                           pinned.padded((RUN,)), np.int32(1)),
             current_sites=[BODY_SITE, (16,)], ledger_sites=[BODY_SITE, (2,)]),
    SiteMove(call_built_by(chart_under_an_inner_vmap, THREE_CHART_MATRICES,
                           np.stack([UNIQUE_LARGEST, TIED_LARGEST, UNIQUE_LARGEST]),
                           THREE_CHART_DIRECTIONS),
             current_sites=[(88,), (29,)], ledger_sites=[(70,), (11,)]),
    SiteMove(call_built_by(chart_under_an_inner_vmap, THREE_CHART_MATRICES,
                           np.stack([UNIQUE_LARGEST] * 3), THREE_CHART_DIRECTIONS),
             current_sites=[(88,), (29,)], ledger_sites=[(70,), (11,)]),
)


# ---------------------------------------------------------------------------
# Comparing two results.
# ---------------------------------------------------------------------------
ROW_LABELS = ('role', 'stage', 'derivative_order', 'dependency', 'basis_kind', 'basis_indices',
              'work_index', 'frequency_source_index')
ROW_ARRAYS = ('status', 'eta', 'residual_numerator', 'residual_denominator', 'performed', 'present')
FACT_LABELS = ('dependency', 'reason')
FACT_ARRAYS = ('available', 'present')
# What was compared between the two transports, by table; printed once per run.
compared = {table: dict.fromkeys(('calls', 'rows', 'facts', 'arrays'), 0)
            for table in ('pinned', 'further', 'site moves')}


def bits(array) -> tuple:
    """Everything that makes two arrays the same array: dtype, shape and bytes."""
    array = np.asarray(array)
    return array.dtype.str, array.shape, array.tobytes()


def sites_of(result) -> list:
    """The ``site`` of every row, then of every fact, in the order returned."""
    return [entry.site for entry in (*result.certificates, *result.availability)]


def record_comparison(table, result):
    tally = compared[table]
    tally['calls'] += 1
    tally['rows'] += len(result.certificates)
    tally['facts'] += len(result.availability)
    tally['arrays'] += (len(jax.tree.leaves(result.values))
                        + len(ROW_ARRAYS) * len(result.certificates)
                        + len(FACT_ARRAYS) * len(result.availability))


def tearDownModule():
    print('CERTIFICATE_PARITY ' + json.dumps(compared, sort_keys=True), flush=True)


class ParityTestCase(unittest.TestCase):
    def assert_same_result(self, first, second, *, sites=True, available_everywhere=False):
        """Two compiled calls returned the same values, rows and facts, in the same order."""
        self.assert_same_arrays(first.values, second.values, 'returned values')
        self.assertEqual(len(first.certificates), len(second.certificates), 'row count')
        self.assertEqual(len(first.availability), len(second.availability), 'fact count')
        if sites:
            self.assertEqual(sites_of(first), sites_of(second), 'site')
        for index, (row, other) in enumerate(zip(first.certificates, second.certificates)):
            for label in ROW_LABELS:
                self.assertEqual(getattr(row, label), getattr(other, label), f'row {index} {label}')
            for field in ROW_ARRAYS:
                self.assert_same_arrays(getattr(row, field), getattr(other, field),
                                        f'row {index} at {row.site} {field}')
        for index, (fact, other) in enumerate(zip(first.availability, second.availability)):
            for label in FACT_LABELS:
                self.assertEqual(getattr(fact, label), getattr(other, label), f'fact {index} {label}')
            self.assert_same_arrays(fact.present, other.present, f'fact {index} present')
            self.assert_same_available(fact, other, index, everywhere=available_everywhere)

    def assert_same_arrays(self, first, second, what):
        first_leaves, first_tree = jax.tree.flatten(first)
        second_leaves, second_tree = jax.tree.flatten(second)
        self.assertEqual(first_tree, second_tree, what)
        for leaf, other in zip(first_leaves, second_leaves, strict=True):
            self.assertEqual(bits(leaf), bits(other), what)

    def assert_same_available(self, fact, other, index, *, everywhere):
        available, other_available = np.asarray(fact.available), np.asarray(other.available)
        what = f'fact {index} available'
        self.assertEqual((available.dtype, available.shape),
                         (other_available.dtype, other_available.shape), what)
        where = np.ones(available.shape, dtype=bool) if everywhere else np.asarray(fact.present)
        self.assertEqual(available[where].tolist(), other_available[where].tolist(), what)

    def assert_transports_agree(self, made: Call, *, table, sites=True):
        """Both transports return the same result for ``made``; returns the two results."""
        current, ledger = compiled(CURRENT, made), compiled(LEDGER, made)
        self.assert_same_result(current, ledger, sites=sites)
        record_comparison(table, current)
        return current, ledger

    def assert_transports_agree_on_every_call(self, calls, *, table):
        for group, made in calls.items():
            for ordinal, one in enumerate(made):
                with self.subTest(calls=group, ordinal=ordinal):
                    self.assert_transports_agree(one, table=table)


# ---------------------------------------------------------------------------
# The tests.
# ---------------------------------------------------------------------------
class PinnedCallTests(ParityTestCase):
    def test_every_test_of_the_pinned_module_has_its_calls_here(self):
        tests_of_the_pinned_module = {
            f'{name}.{method}'
            for name, case in vars(pinned).items()
            if isinstance(case, type) and issubclass(case, unittest.TestCase)
            for method in unittest.defaultTestLoader.getTestCaseNames(case)}
        self.assertEqual(sorted(PINNED_CALLS), sorted(tests_of_the_pinned_module))
        for test, made in PINNED_CALLS.items():
            self.assertTrue(made, test)

    def test_ledger_returns_the_rows_of_the_current_transport(self):
        self.assert_transports_agree_on_every_call(PINNED_CALLS, table='pinned')


class ChartWrittenOutTests(ParityTestCase):
    """The chart written out here is the product's chart.

    Both are certified by the current transport, so every array is compared,
    ``available`` in absent lanes included.
    """

    def test_chart_written_with_todays_form_returns_what_the_products_chart_returns(self):
        chart_calls = [(test, one) for test, made in PINNED_CALLS.items()
                       if test.startswith('DerivativeAvailabilityTests.') for one in made]
        self.assertEqual(len(chart_calls), 7)
        for test, one in chart_calls:
            with self.subTest(test=test):
                # A chart call builds the product's function for the current transport.
                product = one.function(CURRENT)
                name = PRODUCT_CHART._fields[PRODUCT_CHART.index(product)]
                written_out = one._replace(function=lambda transport: getattr(CHART_AS_TODAY, name))
                self.assert_same_result(compiled(CURRENT, one), compiled(CURRENT, written_out),
                                        available_everywhere=True)


class FurtherCallTests(ParityTestCase):
    def test_ledger_returns_the_rows_of_the_current_transport(self):
        self.assert_transports_agree_on_every_call(FURTHER_CALLS, table='further')

    def test_further_calls_return_rows_and_facts(self):
        self.assertEqual(sorted(FURTHER_ROWS_AND_FACTS), sorted(FURTHER_CALLS))
        for group, rows_and_facts in FURTHER_ROWS_AND_FACTS.items():
            with self.subTest(calls=group):
                result = compiled(LEDGER, FURTHER_CALLS[group][0])
                self.assertEqual((len(result.certificates), len(result.availability)), rows_and_facts)


class SiteMoveTests(ParityTestCase):
    def test_everything_but_site_is_the_same_where_site_moves(self):
        for ordinal, move in enumerate(SITE_MOVES):
            with self.subTest(move=ordinal):
                current, ledger = self.assert_transports_agree(move.made, table='site moves',
                                                               sites=False)
                self.assertEqual(sites_of(current), move.current_sites)
                self.assertEqual(sites_of(ledger), move.ledger_sites)

    def test_lanes_of_an_inner_vmap_share_one_fact(self):
        """What both transports agree on above: one fact, unavailable if any lane is."""
        for move, available in zip(SITE_MOVES[1:], (False, True)):
            with self.subTest(available=available):
                fact, = compiled(LEDGER, move.made).availability
                self.assertEqual((np.asarray(fact.present).tolist(), np.asarray(fact.available).tolist()),
                                 (True, available))


if __name__ == '__main__':
    unittest.main()
