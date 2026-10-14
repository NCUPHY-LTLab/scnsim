"""Native family transfer roots over the original network/cofactor equations.

Z Newton uses the RF cofactor of the same effective operator (-i omega)Y.
Y/S retain the actual family entry iteration. Original full-transfer/cofactor
closure and complete denominator norms remain the success authority.
The local implicit derivative cancels only the certified nonzero denominator;
it is not differentiation of finite Newton history. Native denominator factors
belong to one sparse certificate callback and never cross calls or consumers.
"""
from __future__ import annotations
from typing import NamedTuple, Any
from contextlib import nullcontext
import numpy as np


Z_ITERATION_RESIDUAL = 'scnsim.jax.transfer_zero.rf_cofactor.v1'


TRANSFER_STAGES = ('transfer_frequency', 'transfer_numerator_scale',
                   'transfer_slope_rank', 'transfer_denominator', 'transfer_correction')


class NativeDenominatorPacket(NamedTuple):
    value: Any
    derivative: Any
    status: Any
    identity_numerator: Any
    identity_denominator: Any
    identity_residual: Any
    derivative_residual: Any
    finite: Any
    closed: Any
    mantissa: Any
    exponent: Any
    row_shifts: Any
    row_permutation: Any
    column_permutation: Any


class NativeTransferPacket(NamedTuple):
    omega: Any
    slope: Any
    numerator_slope: Any
    denominator: Any
    status: Any
    failure_stage: Any
    source_stage: Any  # Original family formation status; zero for root-owned failures.
    steps: Any
    closure: Any
    certificate: Any
    check_status: Any
    certificate_status: Any
    certificates: tuple
    selected_state: Any
    denominator_certificate: Any
    numerator_mantissa: Any
    numerator_exponent: Any
    slope_mantissa: Any
    slope_exponent: Any
    row_shifts: Any
    singular_values: Any
    iteration_mantissa: Any
    iteration_exponent: Any
    iteration_slope_mantissa: Any
    iteration_slope_exponent: Any


class _DenominatorPhases:
    """Callback-local factor ownership has no recorder or clock capture."""
    def phase(self, *args, **kwargs):
        return nullcontext()


def _host_denominator(pattern, values, primes, prior, *, ordering):
    from scipy.sparse import csc_matrix
    from types import SimpleNamespace
    from .determinants import Denominator, QuantityError
    from .sparse_direct import tau
    dtype = np.asarray(values).dtype
    cd, rd = dtype.type, np.empty((), dtype=dtype).real.dtype.type
    size = pattern.shape[0]
    empty = NativeDenominatorPacket(cd(0), cd(0), np.int32(prior),
        *(rd(np.inf) for _ in range(4)), np.bool_(False), np.bool_(False), cd(0), np.int32(0),
        *(np.zeros((size,), dtype=np.int32) for _ in range(3)))
    if np.asarray(prior).item():
        return empty
    # Register the object before initialization: even an initialization traceback
    # cannot retain its allocated native factor past this callback's finally.
    owner = Denominator.__new__(Denominator)
    owner.factor = None
    try:
        matrix = csc_matrix((values, pattern.rows, pattern.indptr), shape=pattern.shape)
        prime = csc_matrix((primes, pattern.rows, pattern.indptr), shape=pattern.shape)
        owner.__init__(matrix, prime, SimpleNamespace(real_dtype=rd, complex_dtype=cd), _DenominatorPhases())
        value, derivative, facts = owner.certificate_pair(verify=False)
        closed = bool(facts['finite'] and facts['identity_solve_residual'] <= tau(size, rd)
                      and facts['derivative_solve_residual'] <= tau(size, rd))
        return NativeDenominatorPacket(cd(value), cd(derivative), np.int32(0),
            rd(facts['identity_solve_numerator']), rd(facts['identity_solve_denominator']),
            rd(facts['identity_solve_residual']), rd(facts['derivative_solve_residual']),
            np.bool_(facts['finite']), np.bool_(closed), cd(facts['determinant']['mantissa']),
            np.int32(facts['determinant']['exponent']),
            np.asarray(facts['row_shifts'], dtype=np.int32),
            np.asarray(facts['row_permutation'], dtype=np.int32),
            np.asarray(facts['column_permutation'], dtype=np.int32))
    except QuantityError:
        # Only the established typed denominator-domain failure is a numerical
        # outcome. Arbitrary native/backend exceptions escape unchanged.
        return empty._replace(status=np.int32(12))
    finally:
        owner.factor = None


def compile_native_transfer(declaration, frequency_program, *, precision, view_index, capability, availability_dependency):
    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify
    from .jax_core import first, tau
    from .native_root import _local_pair, _determinant, _scaled, _multiply, _shift, _scaled_ratio
    from .native_sparse import checked_dense_solve, checked_sparse_solve
    from .determinant_ad import determinant
    from .sparse_structure import _full_pattern
    from .sparse_solve import SparseEquation
    rd, cd = (jnp.float32, jnp.complex64) if precision == 'float32' else (jnp.float64, jnp.complex128)
    output, input_ = declaration.output_index, declaration.input_index
    family = declaration.family
    view_program = frequency_program.view_program
    width = len(view_program.terminal_ids) if capability else len(view_program.selected)
    rows, columns = tuple(i for i in range(width) if i != input_), tuple(i for i in range(width) if i != output)
    sign = -1 if (input_ + output) % 2 else 1
    minor_width = 1 if family == 'Y' else width + 1 if family == 'S' else max(1, len(rows))
    denominator_width = len(view_program.eliminated) if family == 'Y' else width
    denominator_equation = frequency_program.ee if family == 'Y' else frequency_program.small
    def state_function(candidate, omega):
        return frequency_program.transfer_family_jet(candidate.operands[view_index],
            candidate.views[view_index], omega, family=family, capability=capability,
            dependency=availability_dependency)
    small = frequency_program.small
    implicit = SparseEquation(_full_pattern(1))
    scalar = lambda dtype: jax.ShapeDtypeStruct((), dtype)
    denominator_shape = NativeDenominatorPacket(scalar(cd), scalar(cd), scalar(jnp.int32),
        *(scalar(rd) for _ in range(4)), scalar(jnp.bool_), scalar(jnp.bool_), scalar(cd), scalar(jnp.int32),
        *(jax.ShapeDtypeStruct((denominator_width,), jnp.int32) for _ in range(3)))

    def same_bits(left, right):
        bits = jnp.uint32 if precision == 'float32' else jnp.uint64
        return ((jax.lax.bitcast_convert_type(left.real, bits) == jax.lax.bitcast_convert_type(right.real, bits)) &
                (jax.lax.bitcast_convert_type(left.imag, bits) == jax.lax.bitcast_convert_type(right.imag, bits)))

    def positive_product(values):
        mantissa, exponent = jnp.asarray(1, dtype=rd), jnp.int32(0)
        zero = jnp.bool_(False)
        for value in values:
            fm, fe = jnp.frexp(jnp.asarray(value, dtype=rd))
            mantissa, shift = jnp.frexp(mantissa * fm)
            zero = zero | (value == 0)
            exponent = exponent + fe + shift
        return jnp.where(zero, jnp.asarray(0, dtype=rd), mantissa), jnp.where(zero, jnp.int32(0), exponent)

    def ratio_le(pair, limit):
        lm, le = jnp.frexp(jnp.asarray(limit, dtype=rd))
        return jnp.isfinite(pair[0]) & ((pair[0] == 0) | (pair[1] < le) | ((pair[1] == le) & (pair[0] <= lm)))

    def positive_sum(terms):
        mantissa, exponent = jnp.asarray(0, dtype=rd), jnp.int32(0)
        for value, power in terms:
            common = jnp.maximum(exponent, power)
            left, right = jnp.ldexp(mantissa, exponent-common), jnp.ldexp(value, power-common)
            total = jnp.nextafter(left + right, jnp.asarray(jnp.inf, dtype=rd))
            updated, shift = jnp.frexp(total)
            new_m = jnp.where(mantissa == 0, value, updated)
            new_e = jnp.where(mantissa == 0, power, common + shift)
            mantissa = jnp.where(value == 0, mantissa, new_m)
            exponent = jnp.where(value == 0, exponent, new_e)
        return mantissa, exponent

    def matrix_pair(state, omega):
        return state.Y, state.Yp

    def denominator_packet(values, primes, status):
        stop = jax.lax.stop_gradient
        return denominator_equation.call_callback(('native_transfer_denominator', family, denominator_width),
            _host_denominator, denominator_shape, stop(values), stop(primes), stop(status))

    def empty_denominator():
        zero, one = jnp.asarray(0, dtype=rd), jnp.asarray(1, dtype=cd)
        return NativeDenominatorPacket(one, jnp.asarray(0, dtype=cd), jnp.int32(0),
            zero, zero, zero, zero, jnp.bool_(True), jnp.bool_(True), one, jnp.int32(0),
            *(jnp.zeros((0,), dtype=jnp.int32) for _ in range(3)))

    def at(candidate, omega):
        state = state_function(candidate, omega)
        checks, certificates = state.check_status, state.certificate_status
        Y, Yp = state.Y, state.Yp
        value, slope = state.V[output, input_], state.Vp[output, input_]
        status = first(checks, certificates != 0, certificates)
        stage = jnp.int32(3)
        source_stage = status
        extra_certificates = []
        def fail(bad, code, failure_stage):
            nonlocal status, stage
            stage = jnp.where((status == 0) & bad, jnp.int32(failure_stage), stage)
            status = first(status, bad, code)
        fail(~jnp.isfinite(value), 12, 1)
        fail(~jnp.isfinite(slope), 13, 2)
        if family == 'Y':
            high = tuple(state.H_values[jnp.asarray(sources)] for _, sources in frequency_program.blocks)
            prime = tuple(state.Hp_values[jnp.asarray(sources)] for _, sources in frequency_program.blocks)
            from .sparse_structure import _dense
            HRR, HRRp = _dense(frequency_program.blocks[0][0], high[0]), _dense(frequency_program.blocks[0][0], prime[0])
            if denominator_width:
                denominator = denominator_packet(high[3], prime[3], status)
                fail(denominator.status != 0, 12, 3)
                rhs = _dense(frequency_program.blocks[2][0], high[2])
                solved = checked_sparse_solve(frequency_program.ee, high[3], rhs,
                    stage=12, prior_status=status, dependency=availability_dependency)
                extra_certificates.append(solved.aux)
                fail(solved.aux.performed & (solved.aux.status != 0), 12, 3)
                rhs_prime = _dense(frequency_program.blocks[2][0], prime[2]) - frequency_program.ee.matmul(prime[3], solved.value)
                solved_prime = checked_sparse_solve(frequency_program.ee, high[3], rhs_prime,
                    stage=12, prior_status=status, role='frequency_tangent', derivative_order=1, dependency=availability_dependency)
                extra_certificates.append(solved_prime.aux)
                fail(solved_prime.aux.performed & (solved_prime.aux.status != 0), 12, 3)
                X, Xp = solved.value, solved_prime.value
                detd, detdp = denominator.value, denominator.derivative
                N = HRR * detd - frequency_program.re.matmul(high[1], detd * X)
                Np = (HRRp * detd + HRR * detdp - frequency_program.re.matmul(prime[1], detd * X)
                      - frequency_program.re.matmul(high[1], detdp * X + detd * Xp))
            else:
                denominator = empty_denominator()
                N, Np = HRR, HRRp
                X = jnp.zeros((0, width), dtype=cd)
            AN, ANp = N[output, input_].reshape((1, 1)), Np[output, input_].reshape((1, 1))
            bound = state.bound_values
            fail(~(jnp.all(jnp.isfinite(bound)) & jnp.isfinite(jnp.abs(omega)) & (jnp.abs(omega) > 0)), 12, 1)
            BH = bound / jnp.abs(omega)
            fail(~jnp.all(jnp.isfinite(BH)) | jnp.any((bound > 0) & (BH == 0)), 12, 1)
            if capability:
                view = candidate.views[view_index]
                pattern = view_program.patterns[3]
                weights = jnp.abs(view.model.B)
                cols = jnp.asarray(pattern.cols)
                extra = weights[:, None] * jnp.abs(view.Go)[cols[:, None], cols[None, :]] * weights[None, :]
                BH = BH.at[jnp.asarray(frequency_program.assembly.load_slots).ravel()].add(extra.ravel())
            BF = _dense(frequency_program.blocks[0][0], BH[jnp.asarray(frequency_program.blocks[0][1])])
            if denominator_width:
                BF = BF + frequency_program.re.matmul(BH[jnp.asarray(frequency_program.blocks[1][1])], jnp.abs(X))
            fail(~jnp.all(jnp.isfinite(BF)), 12, 1)
            normalizer = positive_product((jnp.abs(denominator.value), jnp.linalg.norm(BF[output])))
        elif family == 'Z':
            if rows:
                AN = Y[jnp.asarray(rows)[:, None], jnp.asarray(columns)[None, :]]
                ANp = Yp[jnp.asarray(rows)[:, None], jnp.asarray(columns)[None, :]]
                AN, ANp = AN.at[0].multiply(sign), ANp.at[0].multiply(sign)
            else:
                AN, ANp = jnp.eye(1, dtype=cd), jnp.zeros((1, 1), dtype=cd)
            normalizer = positive_product(tuple(jnp.linalg.norm(Y[row]) for row in rows))
        else:
            view = candidate.views[view_index]
            eye = jnp.eye(width, dtype=cd)
            Pp = view.Dk @ Yp @ view.Dk
            P, N = eye + view.Dk @ Y @ view.Dk, eye - view.Dk @ Y @ view.Dk
            AN, ANp = jnp.zeros((minor_width, minor_width), dtype=cd), jnp.zeros((minor_width, minor_width), dtype=cd)
            AN = AN.at[:-1, :-1].set(P).at[:-1, -1].set(N[:, input_]).at[-1, output].set(jnp.asarray(-1, dtype=cd))
            ANp = ANp.at[:-1, :-1].set(Pp).at[:-1, -1].set(-Pp[:, input_])
            normalizer = positive_product(tuple(jnp.linalg.norm(row) for row in AN))
        h, hp, hm, he, hpm, hpe, shifts, code, _ = _local_pair(AN, ANp)
        fail(code != 0, 13, 1)
        # Same nonzero-frequency physical operator; no alternate View/loading
        # or DC eligibility rule. The original Y cofactor above still certifies.
        if family == 'Z':
            if rows:
                if not capability:
                    effective, effective_prime = state.selected_state.F, state.selected_state.Fp
                else:
                    from .native_compensated import multiply, add, project, from_value
                    divisor = -jnp.asarray(1j, dtype=cd) * omega
                    constant = -jnp.asarray(1j, dtype=cd)
                    effective = project(multiply(from_value(divisor, xp=jnp), from_value(Y, xp=jnp)))
                    effective_prime = project(add(
                        multiply(from_value(divisor, xp=jnp), from_value(Yp, xp=jnp)),
                        multiply(from_value(constant, xp=jnp), from_value(Y, xp=jnp))))
                RF = effective[jnp.asarray(rows)[:, None], jnp.asarray(columns)[None, :]]
                RFp = effective_prime[jnp.asarray(rows)[:, None], jnp.asarray(columns)[None, :]]
                RF, RFp = RF.at[0].multiply(sign), RFp.at[0].multiply(sign)
            else:
                RF, RFp = jnp.eye(1, dtype=cd), jnp.zeros((1, 1), dtype=cd)
            _, _, rm, re, rpm, rpe, _, iteration_code, _ = _local_pair(
                RF, RFp, restore_required=False)
            fail(iteration_code != 0, 13, 1)
        else:
            rm, re, _ = _scaled(value)
            rpm, rpe, _ = _scaled(slope)
        if family != 'Y':
            denominator = denominator_packet((P if family == 'S' else Y).T.ravel(),
                (Pp if family == 'S' else Yp).T.ravel(), status)
            fail(denominator.status != 0, 12, 3)
        fail(~(jnp.isfinite(h) & jnp.isfinite(hp)), 12, 1)
        fail(~jnp.isfinite(normalizer[0]), 12, 1)
        eta = _scaled_ratio(jnp.abs(h), normalizer)
        scaled, prime = _shift(AN, shifts[:, None]), _shift(ANp, shifts[:, None])
        terms = []
        for row in range(minor_width):
            for column in range(minor_width):
                rr, cc = tuple(i for i in range(minor_width) if i != row), tuple(i for i in range(minor_width) if i != column)
                cm, ce, _ = _determinant(scaled[jnp.asarray(rr, dtype=jnp.int32)[:, None], jnp.asarray(cc, dtype=jnp.int32)[None, :]])
                tm, te = positive_product((jnp.abs(cm), jnp.abs(prime[row, column])))
                terms.append((tm, te + ce))
        slope_scale = positive_sum(terms)
        slope_ratio = _scaled_ratio(jnp.abs(hpm), slope_scale)
        slope_ratio = slope_ratio[0], slope_ratio[1] + hpe + jnp.sum(shifts, dtype=jnp.int32)
        singular = jax.lax.cond(status == 0,
            lambda matrix: jnp.linalg.svd(matrix, compute_uv=False),
            lambda matrix: jnp.zeros((minor_width,), dtype=rd), AN)
        rank_pass = (jnp.bool_(True) if minor_width == 1 else
            (singular[0] != 0) & (singular[-1] / singular[0] <= tau(minor_width, rd)) &
            (singular[-2] / singular[0] > tau(minor_width, rd)))
        slope_pass = (slope_scale[0] > 0) & ~ratio_le(slope_ratio, tau(minor_width, rd))
        fail(~denominator.closed, 12, 3)
        correction = _scaled_ratio(jnp.abs(value), positive_product((jnp.abs(slope), jnp.asarray(2*jnp.pi, dtype=rd))))
        residual_pass = ratio_le(eta, tau(minor_width, rd))
        correction_pass = ratio_le(correction, jnp.asarray(0.01, dtype=rd))
        closed = (status == 0) & residual_pass & slope_pass & rank_pass & correction_pass & (omega.real > 0)
        certificate = jnp.asarray((eta[0], eta[1], slope_ratio[0], slope_ratio[1],
            slope_scale[0], slope_scale[1], correction[0], correction[1]), dtype=rd)
        # Stage classification is deferred until the original final closure.
        final_stage = jnp.where(~(jnp.isfinite(omega) & (omega.real > 0)), 0,
            jnp.where(~residual_pass, 1, jnp.where(~(slope_pass & rank_pass), 2, 4)))
        final_code = jnp.where(final_stage == 2, jnp.int32(13), jnp.int32(12))
        failure_stage = jnp.where(status == 0, final_stage, stage)
        closure_code = jnp.where(closed, jnp.int32(0), jnp.where(status != 0, status, final_code))
        return NativeTransferPacket(omega, slope, hp, denominator.value, status, failure_stage, source_stage,
            jnp.int32(0), closed, certificate, checks, certificates,
            (*state.certificates, *extra_certificates), state, denominator, hm, he, hpm, hpe, shifts, singular,
            rm, re, rpm, rpe), value, closure_code

    def primal(candidate, seed):
        initial = jnp.asarray(seed, dtype=cd)
        shape = jax.eval_shape(at, candidate, initial)
        empty_packet, _, _ = jax.tree.map(lambda item: jnp.zeros(item.shape, dtype=item.dtype), shape)
        initial_code = jnp.where(jnp.isfinite(initial) & (initial.real > 0), jnp.int32(0), jnp.int32(12))
        def body(loop):
            omega, step, stopped, packet, status = loop
            packet, value, closure_code = at(candidate, omega)
            checkify.debug_check(~(stopped & packet.closure),
                'stagnant transfer certificate returned without a failure')
            final = (step == 32) | stopped
            status = packet.status
            status = first(status, final & ~packet.closure, closure_code)
            iteration_slope = packet.iteration_slope_mantissa if family == 'Z' else packet.slope
            zero_slope = (status == 0) & ~packet.closure & (iteration_slope == 0)
            status = first(status, zero_slope, 13)
            if family == 'Z':
                correction = _shift(packet.iteration_mantissa / packet.iteration_slope_mantissa,
                    packet.iteration_exponent - packet.iteration_slope_exponent)
            else:
                correction = value / packet.slope
            updated = jnp.asarray(omega - jnp.asarray(correction, dtype=cd), dtype=cd)
            bad_update = (status == 0) & ~packet.closure & ~jnp.isfinite(updated)
            status = first(status, bad_update, 12)
            packet = packet._replace(status=status, steps=step,
                failure_stage=jnp.where(zero_slope, 2, jnp.where(bad_update, 0, packet.failure_stage)),
                certificate_status=first(packet.certificate_status, status != 0, status))
            stagnant = same_bits(updated, omega)
            done = packet.closure | (status != 0) | final
            return jnp.where(done, omega, updated), step+1, stagnant, packet, status
        def condition(loop):
            return (loop[1] < 33) & (loop[4] == 0) & ~loop[3].closure
        failed = empty_packet._replace(omega=initial, status=initial_code, certificate_status=initial_code)
        _, _, _, packet, _ = jax.lax.while_loop(condition, body,
            (initial, jnp.int32(0), jnp.bool_(False), failed, initial_code))
        return packet

    def physical_equation(candidate, omega):
        state = state_function(candidate, omega)
        Y, _ = matrix_pair(state, omega)
        if family == 'Y':
            # Certified nonzero eliminated multiplier defines the same local
            # root; no determinant callback participates in its derivative.
            return Y[output, input_]
        if family == 'S':
            view = candidate.views[view_index]
            eye = jnp.eye(width, dtype=cd)
            P, N = eye + view.Dk @ Y @ view.Dk, eye - view.Dk @ Y @ view.Dk
            bordered = jnp.zeros((width + 1, width + 1), dtype=cd)
            bordered = bordered.at[:-1, :-1].set(P).at[:-1, -1].set(N[:, input_]).at[-1, output].set(jnp.asarray(-1, dtype=cd))
            return determinant(bordered)
        if not rows:
            return jnp.asarray(sign, dtype=cd)
        return sign * determinant(Y[jnp.asarray(rows)[:, None], jnp.asarray(columns)[None, :]])

    # Keep scalar Newton early exits under vmap. Pinned checkify supports
    # scan-of-scalar-while, while a batched while predicate is unsupported.
    # The outer implicit custom JVP remains the derivative authority; no AD
    # transformation differentiates this primal iteration implementation.
    from .native_sparse import checked_sequential_primal
    primal = checked_sequential_primal(primal)

    @jax.custom_jvp
    def root(candidate, seed):
        return primal(candidate, seed)

    def implicit_direction(candidate, candidate_dot, packet):
        _, rhs = jax.jvp(lambda value: physical_equation(value, packet.omega), (candidate,), (candidate_dot,))
        _, denominator = jax.jvp(lambda omega: physical_equation(candidate, omega),
                                 (packet.omega,), (jnp.ones_like(packet.omega),))
        from .native_sparse import derivative_available
        dependency = availability_dependency
        derivative_available(packet.status == 0, dependency=dependency,
            reason='root implicit derivative requires original closed root')
        derivative_available(denominator != 0, dependency=dependency,
            reason='implicit transfer denominator is unresolved')
        solved = checked_dense_solve(implicit, denominator.reshape((1, 1)), (-rhs).reshape((1,)), stage=13,
            role='tangent', derivative_order=1, dependency=availability_dependency)
        omega_dot = solved.value[0]
        def zero(value):
            return jnp.zeros_like(value) if jnp.issubdtype(value.dtype, jnp.inexact) else np.zeros(value.shape, dtype=jax.dtypes.float0)
        tangent = jax.tree.map(zero, packet)
        _, state_dot = jax.jvp(state_function, (candidate, packet.omega), (candidate_dot, omega_dot))
        return packet, tangent._replace(omega=omega_dot, selected_state=state_dot)
    @root.defjvp
    def direction(primals, tangents):
        candidate, seed = primals
        candidate_dot, _ = tangents
        return implicit_direction(candidate, candidate_dot, root(candidate, seed))

    @jax.custom_jvp
    def from_packet(candidate, packet):
        """Same-operation completed root with the same local implicit authority."""
        return packet

    @from_packet.defjvp
    def packet_direction(primals, tangents):
        candidate, packet = primals
        candidate_dot, _ = tangents
        # Preserve the root-induced omega dependence when the implicit rule is
        # itself differentiated, matching direction's root(candidate, seed).
        return implicit_direction(candidate, candidate_dot, from_packet(candidate, packet))

    root.from_packet = from_packet
    return root
