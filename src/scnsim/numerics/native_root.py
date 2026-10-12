"""Compiled local root control over the provider's exact selected equations.

Assembly, View/loading, selected Schur equations and sparse certification are
owned by state_function. This module translates the existing 32-step Newton,
bit-stagnation and closure policy; it never assembles another operator or owns a
factor. Only retained matrices and full equation action vectors are dense.
Implicit derivatives use the closed physical equation, never Newton history.
Packets retain the exact final selected state as call-local arrays for coupling;
no factor, tape, Run or static cache belongs to that state.
"""
from __future__ import annotations

from typing import Any, NamedTuple


class RootPacket(NamedTuple):
    omega: Any
    slope: Any
    status: Any
    failure_stage: Any
    steps: Any
    closure: Any
    certificate: Any
    right_vector: Any
    left_vector: Any
    singular_values: Any
    determinant_mantissa: Any
    determinant_exponent: Any
    determinant_derivative_mantissa: Any
    determinant_derivative_exponent: Any
    row_shifts: Any
    normalizer_mantissa: Any
    normalizer_exponent: Any
    check_status: Any
    certificate_status: Any
    certificates: tuple
    selected_state: Any


# Fixed array stage vocabulary for the host typed-failure adapter.
ROOT_STAGES = ('newton', 'newton_certificate', 'slope_certificate',
               'rank_certificate', 'determinant_pivot', 'determinant_scaling')


def _shift(value, power):
    import jax.numpy as jnp
    return jnp.ldexp(value.real, power) + 1j * jnp.ldexp(value.imag, power)


def _scaled(value):
    import jax.numpy as jnp
    magnitude = jnp.abs(value)
    _, power = jnp.frexp(magnitude)
    power = jnp.where(magnitude == 0, 0, power - 1)
    mantissa = jnp.where(magnitude == 0, jnp.zeros_like(value), _shift(value, -power))
    return mantissa, power, ~jnp.isfinite(magnitude)


def _multiply(a, b):
    import jax.numpy as jnp
    value, power, bad = _scaled(a[0] * b[0])
    return value, jnp.where(value == 0, 0, a[1] + b[1] + power), bad


def _add(a, b):
    import jax.numpy as jnp
    power = jnp.maximum(a[1], b[1])
    value = _shift(a[0], a[1] - power) + _shift(b[0], b[1] - power)
    mantissa, shift, bad = _scaled(value)
    result = mantissa, jnp.where(mantissa == 0, 0, power + shift)
    result = (jnp.where(a[0] == 0, b[0], jnp.where(b[0] == 0, a[0], result[0])),
              jnp.where(a[0] == 0, b[1], jnp.where(b[0] == 0, a[1], result[1])))
    return *result, bad


def _determinant(matrix):
    """Same pivoted local determinant, with mantissa/exponent products."""
    import jax.numpy as jnp
    from .jax_core import first
    work = matrix
    determinant = jnp.ones((), dtype=matrix.dtype), jnp.int32(0)
    status = jnp.int32(0)
    import jax
    for column in range(matrix.shape[0]):
        def step(carry):
            work, dm, de, status = carry
            candidates = jnp.abs(work[column:, column])
            status = first(status, ~jnp.all(jnp.isfinite(candidates)), 11)
            row = column + jnp.argmax(candidates)
            old = work[column]
            work = work.at[column].set(work[row]).at[row].set(old)
            dm = jnp.where(row == column, dm, -dm)
            pivot = work[column, column]
            factor, power, bad = _scaled(pivot)
            dm, de, product_bad = _multiply((dm, de), (factor, power))
            status = first(status, bad | product_bad, 11)
            def eliminate(work):
                for next_row in range(column + 1, matrix.shape[0]):
                    multiplier = work[next_row, column] / pivot
                    work = work.at[next_row, column + 1:].add(-multiplier * work[column, column + 1:])
                    work = work.at[next_row, column].set(0)
                return work
            work = jax.lax.cond((pivot != 0) & (status == 0), eliminate, lambda x: x, work)
            return work, dm, de, status
        # The original returns at a zero pivot; unused columns cannot fail it.
        work, dm, de, status = jax.lax.cond((determinant[0] != 0) & (status == 0),
            step, lambda x: x, (work, *determinant, status))
        determinant = dm, de
    return determinant[0], determinant[1], status


def _local_pair(matrix, prime):
    """Existing row-scaled determinant/cofactor derivative, not det*trace(inv)."""
    import jax.numpy as jnp
    from .jax_core import first
    width = matrix.shape[0]
    maxima = jnp.max(jnp.abs(matrix), axis=1)
    _, powers = jnp.frexp(maxima)
    shifts = jnp.where(maxima == 0, 0, 1 - powers)
    scaled, scaled_prime = _shift(matrix, shifts[:, None]), _shift(prime, shifts[:, None])
    status = first(jnp.int32(0), ~jnp.all(jnp.isfinite(maxima)), 11)
    status = first(status, ~(jnp.all(jnp.isfinite(scaled)) & jnp.all(jnp.isfinite(scaled_prime))), 11)
    stage = jnp.where(status != 0, jnp.int32(5), jnp.int32(0))
    determinant_m, determinant_e, determinant_status = _determinant(scaled)
    stage = jnp.where((status == 0) & (determinant_status != 0), jnp.int32(4), stage)
    status = first(status, determinant_status != 0, determinant_status)
    derivative = jnp.zeros((), dtype=matrix.dtype), jnp.int32(0)
    for row in range(width):
        for column in range(width):
            rr = tuple(i for i in range(width) if i != row)
            cc = tuple(i for i in range(width) if i != column)
            minor = scaled[jnp.asarray(rr, dtype=jnp.int32)[:, None], jnp.asarray(cc, dtype=jnp.int32)[None, :]]
            cofactor_m, cofactor_e, code = _determinant(minor)
            cofactor_m = cofactor_m if (row + column) % 2 == 0 else -cofactor_m
            prime_m, prime_e, bad = _scaled(scaled_prime[row, column])
            term_m, term_e, product_bad = _multiply((cofactor_m, cofactor_e), (prime_m, prime_e))
            sum_m, sum_e, sum_bad = _add(derivative, (term_m, term_e))
            derivative = sum_m, sum_e
            stage = jnp.where((status == 0) & ((code != 0) | bad | product_bad | sum_bad), jnp.int32(4), stage)
            status = first(status, code != 0, code)
            status = first(status, bad | product_bad | sum_bad, 11)
    common = -jnp.sum(shifts, dtype=jnp.int32)
    determinant_e = jnp.where(determinant_m == 0, 0, determinant_e + common)
    derivative_e = jnp.where(derivative[0] == 0, 0, derivative[1] + common)
    h, hp = _shift(determinant_m, determinant_e), _shift(derivative[0], derivative_e)
    bad_restore = ((determinant_m != 0) & ((h == 0) | ~jnp.isfinite(h))) | ((derivative[0] != 0) & ((hp == 0) | ~jnp.isfinite(hp)))
    stage = jnp.where((status == 0) & bad_restore, jnp.int32(5), stage)
    status = first(status, bad_restore, 11)
    return h, hp, determinant_m, determinant_e, derivative[0], derivative_e, shifts, status, stage


def _row_norm_product(matrix):
    import jax.numpy as jnp
    rd = matrix.real.dtype
    mantissa, exponent = jnp.asarray(1, dtype=rd), jnp.int32(0)
    zero = jnp.asarray(False)
    for row in matrix:
        scale = jnp.maximum(jnp.max(jnp.abs(row.real)), jnp.max(jnp.abs(row.imag)))
        divisor = jnp.where(scale == 0, jnp.asarray(1, dtype=rd), scale)
        norm = jnp.sqrt(jnp.sum((row.real / divisor) ** 2 + (row.imag / divisor) ** 2, dtype=rd))
        for factor in (scale, norm):
            fm, fe = jnp.frexp(factor)
            product_m, shift = jnp.frexp(mantissa * fm)
            zero = zero | (factor == 0)
            mantissa = jnp.where(zero, jnp.asarray(0, dtype=rd), product_m)
            exponent = jnp.where(zero, jnp.int32(0), exponent + fe + shift)
    return mantissa, exponent


def _scaled_ratio(value, normalizer):
    import jax.numpy as jnp
    vm, ve = jnp.frexp(value)
    m, shift = jnp.frexp(vm / normalizer[0])
    m = jnp.where(normalizer[0] == 0, jnp.where(value == 0, 0, jnp.inf), jnp.where(value == 0, 0, m))
    e = jnp.where((value == 0) | (normalizer[0] == 0), 0, ve - normalizer[1] + shift)
    return m, e


def compile_native_root(declaration, state_function, *, precision, node_count,
                        selected, eliminated, pattern, availability_dependency):
    """Bind one exact root declaration to compiled local control flow.

    state_function(candidate, omega) returns NativeSelectedJet with sparse
    values parallel to pattern. Derivatives of that state use the same physical
    primitive and checked sparse derivative authority supplied by its owner.
    availability_dependency is the provider's exact View/branch occurrence
    namespace; it is preserved independently of the physical equation.
    """
    import jax
    import jax.numpy as jnp
    from .jax_core import first, tau
    selected, eliminated = tuple(selected), tuple(eliminated)
    rows, columns = jnp.asarray(pattern.rows), jnp.asarray(pattern.cols)
    # Canonical SparsePattern contains one slot per physical matrix entry.
    # Certificate rows gather those fixed slots instead of expanding one
    # full-nnz mask for every eliminated coordinate. Runtime Q/Qp stay arrays.
    entry_slots = {(int(row), int(column)): slot for slot, (row, column) in
                   enumerate(zip(pattern.rows, pattern.cols, strict=True))}
    def entry_gather(values, target_row, target_columns):
        slots = tuple(entry_slots.get((target_row, node), -1) for node in target_columns)
        if not pattern.rows.size:
            return jnp.zeros((len(slots),), dtype=values.dtype)
        safe = jnp.asarray(tuple(max(slot, 0) for slot in slots), dtype=jnp.int32)
        present = jnp.asarray(tuple(slot >= 0 for slot in slots), dtype=jnp.bool_)
        return jnp.where(present, values[safe], jnp.asarray(0, dtype=values.dtype))
    rd, cd = (jnp.float32, jnp.complex64) if precision == 'float32' else (jnp.float64, jnp.complex128)
    width = len(selected)
    hybrid = declaration.kind == 'hybridized_pole'
    if declaration.kind not in ('diagonal_root', 'operator_element_root', 'hybridized_pole'):
        raise TypeError(f'native root kind requires its numerical provider: {declaration.kind}')
    coordinate = declaration.coordinate_index if declaration.kind == 'diagonal_root' else declaration.row_index
    column = coordinate if declaration.kind == 'diagonal_root' else declaration.column_index

    def quotient(num, den):
        return jnp.where(den == 0, jnp.where(num == 0, jnp.asarray(0, dtype=rd), jnp.inf), num / den)

    def action(values, vector):
        return jnp.zeros((node_count,), dtype=values.dtype).at[rows].add(values * vector[columns])

    def same_bits(left, right):
        bits_type = jnp.uint32 if precision == 'float32' else jnp.uint64
        return ((jax.lax.bitcast_convert_type(left.real, bits_type) == jax.lax.bitcast_convert_type(right.real, bits_type)) &
                (jax.lax.bitcast_convert_type(left.imag, bits_type) == jax.lax.bitcast_convert_type(right.imag, bits_type)))

    def equation(state):
        if hybrid:
            return _local_pair(state.F, state.Fp)
        return state.F[coordinate, column], state.Fp[coordinate, column], jnp.zeros((), dtype=cd), jnp.int32(0), jnp.zeros((), dtype=cd), jnp.int32(0), jnp.zeros((width,), dtype=jnp.int32), jnp.int32(0), jnp.int32(0)

    def element_certificate(state, omega):
        row, column_node = selected[coordinate], selected[column]
        x = jnp.zeros((node_count,), dtype=cd).at[column_node].set(1)
        if eliminated:
            x = x.at[jnp.asarray(eliminated)].set(-state.X[:, column])
        slope_scale = jnp.abs(entry_gather(state.Qp_values, row, (column_node,))[0])
        if eliminated:
            q_row = entry_gather(state.Q_values, row, eliminated)
            qp_row = entry_gather(state.Qp_values, row, eliminated)
            slope_scale = (slope_scale + jnp.sum(jnp.abs(qp_row) * jnp.abs(state.X[:, column]))
                           + jnp.sum(jnp.abs(q_row) * jnp.abs(state.Xp[:, column])))
        qx, bx = action(state.Q_values, x), action(state.bound_values, jnp.abs(x))
        checked = jnp.asarray((row, *eliminated))
        f, fp = state.F[coordinate, column], state.Fp[coordinate, column]
        eta_q = quotient(jnp.max(jnp.abs(qx[checked])), jnp.max(jnp.abs(bx[checked])))
        eta_f = quotient(jnp.abs(f), bx[row])
        correction = jnp.where(fp == 0, jnp.inf, jnp.abs(f / fp) / jnp.abs(omega))
        normalized = jnp.where(slope_scale == 0, jnp.inf, jnp.abs(fp) / slope_scale)
        return jnp.asarray((state.elimination_eta, eta_q, eta_f, correction, normalized, slope_scale), dtype=rd)

    def finalize(candidate, omega, last_state, steps, status, failure_stage):
        """Original final selected-state/closure authority, without Newton.

        All inputs are call-local arrays. Alternative private methods supply
        their actual terminal point, last state, steps and prior failure; no
        branch selection, hidden iteration or substituted certificate occurs.
        Production Newton calls this at its original point and in its order.
        """
        state = (jax.lax.cond(status == 0, lambda _: state_function(candidate, omega),
                             lambda _: last_state, operand=None)
                 if hybrid else state_function(candidate, omega))
        code = jnp.where(state.check_status != 0, state.check_status, state.certificate_status)
        status = first(status, code != 0, code)
        if not hybrid:
            certificate = element_certificate(state, omega)
            slope = state.Fp[coordinate, column]
            closed = jnp.isfinite(omega) & (omega.real > 0) & jnp.all(certificate[:4] <= tau(node_count, rd))
            status = first(status, ~closed, 12)
            slope_bad = ~(certificate[4] > tau(node_count, rd))
            status = first(status, slope_bad, 13)
            if declaration.kind == 'diagonal_root':
                status = first(status, omega.imag > 0, 12)
            stage = jnp.where(status == 13, 2, jnp.where(status == 12, 1, 0))
            return RootPacket(omega, slope, status, jnp.int32(stage), steps, status == 0,
                certificate, jnp.zeros((width,), dtype=cd), jnp.zeros((width,), dtype=cd),
                jnp.zeros((width,), dtype=rd), jnp.zeros((), dtype=cd), jnp.int32(0),
                jnp.zeros((), dtype=cd), jnp.int32(0), jnp.zeros((width,), dtype=jnp.int32),
                jnp.zeros((), dtype=rd), jnp.int32(0), state.check_status,
                first(state.certificate_status, status != 0, status), state.certificates, state)
        h, hp, dm, de, dpm, dpe, shifts, determinant_status, determinant_stage = equation(state)
        failure_stage = jnp.where((status == 0) & (determinant_status != 0), determinant_stage, failure_stage)
        status = first(status, determinant_status != 0, determinant_status)
        prior_status = status
        U, singular, Vh = jax.lax.cond(status == 0,
            lambda matrix: jnp.linalg.svd(matrix, full_matrices=False),
            lambda matrix: (jnp.zeros_like(matrix), jnp.zeros((width,), dtype=rd), jnp.zeros_like(matrix)),
            state.F)
        rank_ok = jnp.asarray(False) if width < 2 else ((singular[0] > 0) &
            (singular[-1] / singular[0] <= tau(width, rd)) &
            (singular[-2] / singular[0] > tau(width, rd)))
        status = first(status, ~rank_ok, 12)
        v, u = Vh[-1].conj(), U[:, -1]
        pivot = jnp.argmax(jnp.abs(v))
        phase = jnp.exp(-jnp.asarray(1j, dtype=cd) * jnp.angle(v[pivot]))
        v, u = v * phase, u * phase
        right = quotient(jnp.max(jnp.abs(state.F @ v)), jnp.max(jnp.abs(state.F) @ jnp.abs(v)))
        left = quotient(jnp.max(jnp.abs(u.conj() @ state.F)), jnp.max(jnp.abs(u) @ jnp.abs(state.F)))
        normalizer = _row_norm_product(state.F)
        eta_m, eta_e = _scaled_ratio(jnp.abs(h), normalizer)
        tm, te = jnp.frexp(tau(width, rd))
        det_closed = jnp.isfinite(eta_m) & ((eta_m == 0) | (eta_e < te) | ((eta_e == te) & (eta_m <= tm)))
        slope = jnp.vdot(u, state.Fp @ v)
        scale = jnp.sum(jnp.abs(u) * (jnp.abs(state.Fp) @ jnp.abs(v)), dtype=rd)
        correction = jnp.where(hp == 0, jnp.inf, jnp.abs(h / hp) / jnp.abs(omega))
        x = jnp.zeros((node_count,), dtype=cd).at[jnp.asarray(selected)].set(v)
        if eliminated:
            x = x.at[jnp.asarray(eliminated)].set(-state.X @ v)
        eta_q = quotient(jnp.max(jnp.abs(action(state.Q_values, x))),
                         jnp.max(action(state.bound_values, jnp.abs(x))))
        closed = (jnp.isfinite(h) & jnp.isfinite(hp) & det_closed & jnp.isfinite(right) &
            (right <= tau(width, rd)) & jnp.isfinite(left) & (left <= tau(width, rd)) &
            jnp.isfinite(eta_q) & (eta_q <= tau(node_count, rd)) & (scale > 0) &
            (jnp.abs(slope) / scale > tau(width, rd)) & jnp.isfinite(correction) &
            (correction <= tau(width, rd)) & (omega.real > 0) & (omega.imag <= 0))
        status = first(status, ~closed, 12)
        stage = jnp.where(prior_status != 0, failure_stage, jnp.where(~rank_ok, 3, jnp.where(status == 12, 1, 0)))
        certificate = jnp.asarray((right, left, eta_m, eta_e, eta_q, scale, correction), dtype=rd)
        return RootPacket(omega, slope, status, jnp.int32(stage), steps, status == 0,
            certificate, v, u, singular, dm, de, dpm, dpe, shifts, normalizer[0], normalizer[1],
            state.check_status, first(state.certificate_status, status != 0, status), state.certificates, state)

    def primal(candidate, seed):
        initial = jnp.asarray(seed, dtype=cd)
        status = first(jnp.int32(0), ~jnp.isfinite(initial), 14)
        if hybrid:
            status = first(status, initial.real <= 0, 14)
        # Hybrid checked_state aborts immediately on initialization/local
        # failure. An abstract state supplies only fixed failed-packet shapes;
        # no extra state evaluation or factorization is performed there.
        state_shape = jax.eval_shape(state_function, candidate, initial)
        empty_state = jax.tree_util.tree_map(lambda item: jnp.zeros(item.shape, dtype=item.dtype), state_shape)
        def condition(loop):
            omega, steps, stopped, status, stage, previous_state = loop
            return (steps < 32) & ~stopped & (status == 0)
        def body(loop):
            omega, steps, _, status, stage, previous_state = loop
            state = state_function(candidate, omega)
            code = jnp.where(state.check_status != 0, state.check_status, state.certificate_status)
            h, hp, _, _, _, _, _, determinant_status, determinant_stage = equation(state)
            stage = jnp.where((code == 0) & (determinant_status != 0), determinant_stage, stage)
            code = first(code, determinant_status != 0, determinant_status)
            code = first(code, hp == 0, 11)
            updated = jnp.asarray(omega - jnp.asarray(h / hp, dtype=cd), dtype=cd)
            stopped = same_bits(updated, omega)
            return jnp.where(code == 0, updated, omega), steps + 1, stopped, code, stage, state
        omega, steps, _, status, failure_stage, last_state = jax.lax.while_loop(condition, body,
            (initial, jnp.int32(0), jnp.asarray(False), status, jnp.int32(0), empty_state))
        return finalize(candidate, omega, last_state, steps, status, failure_stage)

    # The scalar implicit equation uses the same selected-dtype checked solve
    # as the previous derivative frame. Its transpose and mixed jets remain
    # the native sparse primitive's authority, not an AD of Newton history.
    import numpy as np
    from ..compilation.program import SparsePattern
    from .sparse_solve import SparseEquation
    from .native_sparse import checked_dense_solve
    from .determinant_ad import determinant
    from .native_sparse import derivative_available

    dependency = availability_dependency

    def dense_equation(size):
        indices = np.arange(size)
        return SparseEquation(SparsePattern.from_coo((size, size),
            np.tile(indices, size), np.repeat(indices, size))[0])

    implicit_equation = dense_equation(1)

    def physical_equation(candidate, omega):
        state = state_function(candidate, omega)
        return determinant(state.F) if hybrid else state.F[coordinate, column]

    def zero_tangent(value):
        if jnp.issubdtype(value.dtype, jnp.inexact):
            return jnp.zeros_like(value)
        return np.zeros(value.shape, dtype=jax.dtypes.float0)

    # Carry the functionalized scalar Error through opaque sequential mapping
    # and reintroduce its owned checks outside; retain the implicit AD rule.
    from .native_sparse import checked_sequential_primal
    primal = checked_sequential_primal(primal)

    @jax.custom_jvp
    def root(candidate, seed):
        return primal(candidate, seed)

    @root.defjvp
    def root_direction(primals, tangents):
        candidate, seed = primals
        candidate_dot, _ = tangents
        packet = root(candidate, seed)
        omega = packet.omega
        # Seed selects the local branch; it is not a continuous physical input
        # to the machine-closed implicit equation.
        _, rhs = jax.jvp(lambda value: physical_equation(value, omega),
                         (candidate,), (candidate_dot,))
        _, denominator = jax.jvp(lambda value: physical_equation(candidate, value),
                                 (omega,), (jnp.ones_like(omega),))
        derivative_available(packet.status == 0, dependency=dependency,
            reason='root implicit derivative requires original closed root')
        derivative_available(denominator != 0, dependency=dependency,
            reason='root_slope_unresolved: slope_certificate implicit denominator is zero')
        solution = checked_dense_solve(implicit_equation, denominator.reshape((1, 1)),
            (-rhs).reshape((1,)), stage=13, role='tangent', derivative_order=1, dependency=dependency)
        omega_dot = solution.value[0]
        tangent = jax.tree_util.tree_map(zero_tangent, packet)
        # The selected equations carry the TOTAL physical directional jet at
        # the actual implicit root. Coupling can reuse this state instead of
        # assembling another final-root frame; no Newton history is traced.
        _, state_dot = jax.jvp(state_function, (candidate, omega), (candidate_dot, omega_dot))
        if not hybrid:
            return packet, tangent._replace(omega=omega_dot, selected_state=state_dot,
                slope=state_dot.Fp[coordinate, column])

        # Gauge vectors and u^H Fp v remain primal metadata. Coupling installs
        # its separate right-vector chart on this selected F/Fp directional
        # state; chart certification cannot block frequency-only root AD.
        return packet, tangent._replace(omega=omega_dot, selected_state=state_dot)

    # Internal finite equation/finalization interfaces for the private method
    # comparison. The root's public call remains the SAME Newton trajectory.
    root.finalize = finalize
    root.state_function = state_function
    root.equation = equation
    return root


def compile_native_null_vector(width, *, dependency=()):
    """Existing right-null implicit chart, consumed only by coupling.

    The vector/left witness are detached certified primal metadata. F is the
    same selected physical matrix at the implicitly differentiated root.
    Largest-coordinate phase ties retain the existing derivative-unavailable
    condition here, rather than restricting root-frequency differentiation.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    from .native_sparse import derivative_available
    from ..compilation.program import SparsePattern
    from .sparse_solve import SparseEquation
    from .native_sparse import checked_dense_solve
    indices = np.arange(width - 1)
    equation = SparseEquation(SparsePattern.from_coo((width - 1, width - 1),
        np.tile(indices, width - 1), np.repeat(indices, width - 1))[0])

    @jax.custom_jvp
    def vector(matrix, primal_vector, primal_left):
        return primal_vector

    @vector.defjvp
    def direction(primals, tangents):
        matrix, primal_vector, primal_left = primals
        matrix_dot, _, _ = tangents
        v = vector(matrix, primal_vector, primal_left)
        magnitude = jnp.abs(v)
        derivative_available(jnp.count_nonzero(magnitude == jnp.max(magnitude)) == 1,
            dependency=tuple(dependency),
            reason='root_slope_unresolved: slope_certificate null-vector phase chart is unavailable')
        pivot = jnp.argmax(magnitude)
        omitted_row = jnp.argmax(jnp.abs(primal_left))
        local_indices = jnp.arange(width - 1, dtype=jnp.int32)
        rows = local_indices + (local_indices >= omitted_row)
        columns = local_indices + (local_indices >= pivot)
        chart = v / v[pivot]
        minor = matrix[rows[:, None], columns[None, :]]
        rhs = -(matrix_dot @ chart)[rows]
        solution = checked_dense_solve(equation, minor, rhs, stage=13,
            role='tangent', derivative_order=1, dependency=tuple(dependency))
        chart_dot = jnp.zeros_like(chart).at[columns].set(solution.value)
        _, vector_dot = jax.jvp(lambda value: value / jnp.linalg.norm(value),
                                (chart,), (chart_dot,))
        return v, vector_dot

    return vector
