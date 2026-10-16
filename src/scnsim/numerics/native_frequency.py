"""Array-only candidate preparation and ordered native frequency equations.

Candidate fields, View weights and local inverse stencils are runtime arrays
prepared once. This module contains no CheckedGraph, Workspace, native handle,
point cache or operation observer. Sparse factors remain callback-local; the
small port equations use the same checked primitive on the JAX device. S source
reconciliation is deliberately AFTER the S small solve, never speculative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
import numpy as np

from .assembly_program import compile_assembly
from .native_sparse import checked_sparse_solve, checked_dense_solve, NativeSparseAux, NativeSparseOutcome
from .sparse_structure import _block, _dense, _full_pattern
from .sparse_solve import SparseEquation
from .native_equation_jet import checked_sparse_frequency_jet


class NativeCandidate(NamedTuple):
    model: Any
    views: tuple
    operands: tuple


class NativeFrequencyOutcome(NamedTuple):
    S: Any
    Y: Any
    Z: Any
    operator: Any
    status: Any
    check_status: Any
    certificate_status: Any
    check_position: Any
    certificate_position: Any
    certificates: tuple


class NativeSelectedJet(NamedTuple):
    F: Any
    Fp: Any
    Q_values: Any
    Qp_values: Any
    bound_values: Any
    X: Any
    Xp: Any
    elimination_eta: Any
    check_status: Any
    certificate_status: Any
    certificates: tuple



class NativeSelectedPairJet(NamedTuple):
    F: Any
    Fp: Any
    Q_values: Any
    Qp_values: Any
    bound_values: Any
    X: Any
    Xp: Any
    elimination_eta: Any
    check_status: Any
    certificate_status: Any
    certificates: tuple
    Q_low_values: Any
    Qp_low_values: Any
    X_low: Any
    Xp_low: Any


class NativeTransferFamilyJet(NamedTuple):
    V: Any
    Vp: Any
    Y: Any
    Yp: Any
    H_values: Any
    Hp_values: Any
    bound_values: Any
    check_status: Any
    certificate_status: Any
    certificates: tuple
    selected_state: Any


def selected_pair_jet_function(frequency_program, view_index, *, dependency=()):
    from functools import partial
    return partial(_selected_candidate_pair_jet, frequency_program, view_index, dependency)


def _selected_candidate_pair_jet(program, view_index, dependency, candidate, omega):
    return program.selected_pair_jet(candidate.operands[view_index], candidate.views[view_index], omega, dependency=dependency)


def selected_compensated_jet_function(frequency_program, view_index, *, dependency=()):
    """Same selected-state interface over the actual compensated Schur jet.

    Hybrid roots consume one selected-dtype physical value for every field.
    Project all represented channels together; high-only Q/X would disagree
    with the F/Fp formed from the checked pair equations. Empty elimination
    retains the producer's actual empty arrays and no-solve certificates.
    """
    from functools import partial
    return partial(_selected_candidate_compensated_jet, frequency_program, view_index, dependency)


def _selected_candidate_compensated_jet(program, view_index, dependency, candidate, omega):
    from . import native_compensated as cp
    state = _selected_candidate_pair_jet(program, view_index, dependency, candidate, omega)
    return NativeSelectedJet(state.F, state.Fp,
        cp.project((state.Q_values, state.Q_low_values)),
        cp.project((state.Qp_values, state.Qp_low_values)), state.bound_values,
        cp.project((state.X, state.X_low)), cp.project((state.Xp, state.Xp_low)),
        state.elimination_eta, state.check_status, state.certificate_status, state.certificates)


def _pair_product(pattern, values, solution):
    """Original per-row ascending-column accumulation; no dense operator."""
    import jax
    import jax.numpy as jnp
    from . import native_compensated as cp
    rows, columns = jnp.asarray(pattern.rows), jnp.asarray(pattern.cols)
    output_shape = (pattern.shape[0],) + solution[0].shape[1:]
    initial = tuple(jnp.zeros(output_shape, dtype=solution[0].dtype) for _ in range(2))
    def stamp(position, carry):
        term = cp.multiply(tuple(part[position] for part in values),
                           tuple(part[columns[position]] for part in solution))
        row = rows[position]
        update = cp.add(tuple(part[row] for part in carry), term)
        return tuple(part.at[row].set(value) for part, value in zip(carry, update, strict=True))
    return jax.lax.fori_loop(0, len(pattern.rows), stamp, initial)


def selected_jet_function(frequency_program, view_index, *, dependency=()):
    """Detach one immutable selected-state callable, never an owning registry."""
    from functools import partial
    return partial(_selected_candidate_jet, frequency_program, view_index, dependency)


def _selected_candidate_jet(program, view_index, dependency, candidate, omega):
    return program.selected_jet(candidate.operands[view_index], candidate.views[view_index], omega, dependency=dependency)


def prepare_candidate(model_program, view_programs, assemblies, theta, *, capabilities):
    """One physical build, then the declared View and stencil projections."""
    model = model_program(theta)
    views = tuple(view.apply_model(model, theta, port_realizable=capability)
                  for view, capability in zip(view_programs, capabilities, strict=True))
    operands = tuple(assembly.operands(view) for assembly, view in zip(assemblies, views, strict=True))
    return NativeCandidate(model, views, operands)


@dataclass(frozen=True, slots=True, eq=False)
class NativeFrequencyProgram:
    """One immutable View's SAME equations, consumed by a bounded frame."""
    view_program: Any
    assembly: Any
    blocks: tuple
    ee: Any
    re: Any
    W: Any
    small: Any

    @classmethod
    def create(cls, view_program):
        assembly = compile_assembly(view_program)
        r, e = view_program.selected, view_program.eliminated
        blocks = tuple(_block(assembly.pattern, left, right)
                       for left, right in ((r, r), (r, e), (e, r), (e, e)))
        q = len(view_program.terminal_ids)
        return cls(view_program, assembly, blocks,
                   SparseEquation(blocks[3][0], 'MMD_AT_PLUS_A'), SparseEquation(blocks[1][0]),
                   SparseEquation(assembly.pattern), SparseEquation(_full_pattern(q)))

    def selected_jet(self, operands, view, omega, *, dependency=()):
        """Loaded Q/Schur and analytic frequency jet for the SAME candidate.

        Q/Qp/bound slots follow assembly.pattern exactly. X and Xp have shape
        (number of eliminated coordinates, number of retained coordinates).
        Bound is the existing absolute stamp envelope, not abs(Q) or a dense
        global matrix. Elimination eta is max of actual X/Xp certificates.
        """
        import jax.numpy as jnp
        from .jax_core import first
        assembled = self.assembly.assemble_prepared(operands, omega, view_status=view.status,
                                                    loaded=True, compensated=False)
        check_status = assembled.status
        high = tuple(assembled.values[jnp.asarray(sources)] for _, sources in self.blocks)
        prime = tuple(assembled.frequency_values[jnp.asarray(sources)] for _, sources in self.blocks)
        # The existing eliminated-block reciprocity boundary precedes factor
        # creation. It is distinct from a successful solve residual certificate.
        from .jax_core import tau
        pattern = self.blocks[3][0]
        keys = np.asarray(pattern.rows) * pattern.shape[0] + np.asarray(pattern.cols)
        flipped_keys = np.asarray(pattern.cols) * pattern.shape[0] + np.asarray(pattern.rows)
        union = np.unique(np.concatenate((keys, flipped_keys)))
        direct = jnp.zeros((len(union),), dtype=omega.dtype).at[jnp.asarray(np.searchsorted(union, keys))].set(high[3])
        flipped = jnp.zeros_like(direct).at[jnp.asarray(np.searchsorted(union, flipped_keys))].set(high[3])
        zero = jnp.asarray(0, dtype=omega.real.dtype)
        numerator = jnp.max(jnp.abs(direct - flipped), initial=zero)
        denominator = jnp.max(jnp.abs(direct) + jnp.abs(flipped), initial=zero)
        asym = jnp.where(denominator == 0, jnp.where(numerator == 0, zero, jnp.inf), numerator / denominator)
        bad = ~(jnp.isfinite(asym) & (asym <= tau(pattern.shape[0], omega.real.dtype)))
        input_performed = check_status == 0
        input_status = first(jnp.int32(0), input_performed & bad, 3)
        input_certificate = NativeSparseAux(input_status, asym, numerator, denominator, input_performed)
        prior = first(check_status, input_status != 0, input_status)
        rhs = _dense(self.blocks[2][0], high[2])
        rhs_prime = _dense(self.blocks[2][0], prime[2])
        jet = checked_sparse_frequency_jet(self.ee, high[3], rhs, prime[3], rhs_prime,
                                           prior_status=prior, dependency=dependency)
        solved, derivative = jet.primal, jet.frequency
        X, Xp = solved.value, derivative.value
        F = _dense(self.blocks[0][0], high[0]) - self.re.matmul(high[1], X)
        Fp = (_dense(self.blocks[0][0], prime[0]) - self.re.matmul(prime[1], X)
              - self.re.matmul(high[1], Xp))
        cert_status = first(input_status, solved.aux.performed & (solved.aux.status != 0), solved.aux.status)
        cert_status = first(cert_status, derivative.aux.performed & (derivative.aux.status != 0), derivative.aux.status)
        return NativeSelectedJet(F, Fp, assembled.values, assembled.frequency_values, assembled.bound,
                                  X, Xp, jnp.maximum(solved.aux.eta, derivative.aux.eta), check_status,
                                  cert_status, (input_certificate, solved.aux, derivative.aux))

    def transfer_family_jet(self, operands, view, omega, *, family, capability, dependency=()):
        """Original family network and analytic frequency equation authority.

        Port H excludes reference-source loading, W includes it. Non-Port Z
        alone uses the represented compensated selected equation. All source
        reconciliation occurs after the S stage, with no speculative source.
        """
        import jax.numpy as jnp
        from .jax_core import first, tau
        dtype, rd = omega.dtype, omega.real.dtype
        divisor = -jnp.asarray(1j, dtype=dtype) * omega
        if not capability:
            if family == 'S':
                raise TypeError('S transfer requires a Port-realizable declared View')
            state = (self.selected_pair_jet(operands, view, omega, dependency=dependency) if family == 'Z'
                     else self.selected_jet(operands, view, omega, dependency=dependency))
            Y = state.F / divisor
            Yp = (state.Fp * divisor + jnp.asarray(1j, dtype=dtype) * state.F) / (divisor * divisor)
            certs = list(state.certificates)
            code = state.certificate_status
            if family == 'Z':
                prior = first(state.check_status, code != 0, code)
                solved = checked_dense_solve(self.small, Y, jnp.eye(Y.shape[0], dtype=dtype),
                                             stage=7, prior_status=prior, dependency=dependency)
                certs.append(solved.aux)
                code = first(code, solved.aux.performed & (solved.aux.status != 0), solved.aux.status)
                V, Vp = solved.value, -solved.value @ Yp @ solved.value
                H, Hp = jnp.zeros_like(state.Q_values), jnp.zeros_like(state.Qp_values)
            else:
                V, Vp = Y, Yp
                H = state.Q_values / divisor
                Hp = (state.Qp_values * divisor + jnp.asarray(1j, dtype=dtype) * state.Q_values) / (divisor * divisor)
            return NativeTransferFamilyJet(V, Vp, Y, Yp, H, Hp, state.bound_values,
                state.check_status, code, tuple(certs), state)
        assembled = self.assembly.assemble_prepared(operands, omega, view_status=view.status,
                                                    loaded=False, compensated=False)
        checks, code = assembled.status, jnp.int32(0)
        certs = []
        q = len(self.view_program.terminal_ids)
        eye = jnp.eye(q, dtype=dtype)
        def prior():
            return first(checks, code != 0, code)
        def admit(result):
            nonlocal code
            certs.append(result.aux)
            code = first(code, result.aux.performed & (result.aux.status != 0), result.aux.status)
            return result.value
        def small(matrix, rhs, stage, *, frequency=False):
            return admit(checked_dense_solve(self.small, matrix.astype(dtype), rhs.astype(dtype),
                stage=stage, prior_status=prior(), role='frequency_tangent' if frequency else 'primal',
                derivative_order=1 if frequency else 0, dependency=dependency))
        def sparse(values, rhs, stage, *, frequency=False):
            return admit(checked_sparse_solve(self.W, values, rhs.astype(dtype), stage=stage,
                prior_status=prior(), role='frequency_tangent' if frequency else 'primal',
                derivative_order=1 if frequency else 0, dependency=dependency))
        Rinv = small(view.Rk, eye, 1)
        pattern = self.view_program.patterns[3]
        B = _dense(pattern, view.model.B).astype(dtype)
        Bk = B @ view.selected_map.T
        cols = jnp.asarray(pattern.cols)
        def stamp(matrix):
            entries = view.model.B[:, None] * matrix[cols[:, None], cols[None, :]] * view.model.B[None, :]
            return jnp.zeros_like(assembled.values).at[jnp.asarray(self.assembly.load_slots).ravel()].add(entries.ravel())
        H = assembled.values / divisor + stamp(view.Go)
        Hp = (assembled.frequency_values * divisor + jnp.asarray(1j, dtype=dtype) * assembled.values) / (divisor * divisor)
        W = H + stamp(view.selected_map.T @ Rinv @ view.selected_map)
        X = sparse(W, Bk, 5)
        Ysrc = small(Bk.T @ X, eye, 6)
        Y = Ysrc - Rinv
        Xp = sparse(W, -self.W.matmul(Hp, X), 5, frequency=True)
        Yp = -Ysrc @ (Bk.T @ Xp) @ Ysrc
        empty = jnp.zeros_like(Y)
        V, Vp = Y, Yp
        if family == 'Z':
            V = small(Y, eye, 7)
            Vp = -V @ Yp @ V
        elif family == 'S':
            Pp = view.Dk @ Yp @ view.Dk
            normalized = view.Dk @ Y @ view.Dk
            P, N = eye + normalized, eye - normalized
            V = small(P, N, 8)
            Dinv = small(view.Dk, eye, 1)
            voltage = sparse(W, jnp.asarray(2, dtype=dtype) * Bk @ Dinv, 9)
            source = small(view.Dk, Bk.T @ voltage, 9) - eye
            numerator = jnp.max(jnp.abs(source - V), initial=0)
            denominator = 1 + jnp.max(jnp.abs(source), initial=0) + jnp.max(jnp.abs(V), initial=0)
            eta = numerator / denominator
            performed = prior() == 0
            status = first(prior(), ~(jnp.isfinite(eta) & (eta <= tau(len(self.view_program.node_ids), rd))), 9)
            missing = jnp.asarray(jnp.inf, dtype=rd)
            admit(NativeSparseOutcome(empty, NativeSparseAux(status, jnp.where(performed, eta, missing),
                jnp.where(performed, numerator, missing), jnp.where(performed, denominator, missing), performed)))
            # Existing final network finiteness precedes its analytic Sp solve.
            checks = first(checks, ~jnp.all(jnp.isfinite(jnp.stack((Y, V, empty)))), 10)
            Vp = small(P, -Pp - Pp @ V, 8, frequency=True)
        if family != 'S':
            checks = first(checks, ~jnp.all(jnp.isfinite(jnp.stack((Y, V, empty)))), 10)
        return NativeTransferFamilyJet(V, Vp, Y, Yp, H, Hp, assembled.bound,
            checks, code, tuple(certs), ())

    def selected_pair_jet(self, operands, view, omega, *, dependency=()):
        """Scoped retained Z authority: sparse high/low assembly and state.

        Actual represented X/Xp and both Q/Qp channels accompany projected
        F/Fp. No high-only H is substituted for this compensated equation.
        """
        import jax.numpy as jnp
        import numpy as np
        from .jax_core import first, tau
        from .native_pair import checked_pair_solve
        from . import native_compensated as cp
        assembled = self.assembly.assemble_prepared_pair(operands, omega,
                                                       view_status=view.status, loaded=True)
        check_status = assembled.status
        pairs = tuple(tuple(part[jnp.asarray(sources)] for part in
                            (assembled.values, assembled.low_values)) for _, sources in self.blocks)
        primes = tuple(tuple(part[jnp.asarray(sources)] for part in
                             (assembled.frequency_values, assembled.frequency_low_values))
                       for _, sources in self.blocks)
        rr = tuple(_dense(self.blocks[0][0], part) for part in pairs[0])
        rrp = tuple(_dense(self.blocks[0][0], part) for part in primes[0])
        if not self.view_program.eliminated:
            empty = jnp.zeros((0, len(self.view_program.selected)), dtype=omega.dtype)
            return NativeSelectedPairJet(cp.project(rr), cp.project(rrp), assembled.values,
                assembled.frequency_values, assembled.bound, empty, empty,
                jnp.asarray(0, dtype=omega.real.dtype), check_status, jnp.int32(0), (),
                assembled.low_values, assembled.frequency_low_values, empty, empty)
        pattern = self.ee.pattern
        keys = np.asarray(pattern.rows) * pattern.shape[0] + np.asarray(pattern.cols)
        transposed = np.asarray(pattern.cols) * pattern.shape[0] + np.asarray(pattern.rows)
        union = np.unique(np.concatenate((keys, transposed)))
        positions, reverse = jnp.asarray(np.searchsorted(union, keys)), jnp.asarray(np.searchsorted(union, transposed))
        direct = tuple(jnp.zeros((len(union),), dtype=omega.dtype).at[positions].set(part) for part in pairs[3])
        flipped = tuple(jnp.zeros((len(union),), dtype=omega.dtype).at[reverse].set(part) for part in pairs[3])
        skew = cp.project(cp.add(direct, cp.negate(flipped)))
        scale = jnp.abs(cp.project(direct)) + jnp.abs(cp.project(flipped))
        numerator, denominator = jnp.max(jnp.abs(skew)), jnp.max(scale)
        asym = jnp.where(denominator == 0, jnp.where(numerator == 0, 0, jnp.inf), numerator / denominator)
        bad = ~(jnp.isfinite(asym) & (asym <= tau(pattern.shape[0], omega.real.dtype)))
        bad = bad | ~jnp.all(jnp.isfinite(pairs[3][1]))
        input_performed = check_status == 0
        certificate_status = first(jnp.int32(0), input_performed & bad, 3)
        input_certificate = NativeSparseAux(certificate_status, asym, numerator, denominator, input_performed)
        prior = first(check_status, certificate_status != 0, certificate_status)
        rhs = tuple(_dense(self.blocks[2][0], part) for part in pairs[2])
        from .native_pair import checked_pair_frequency_jet
        frequency_rhs = tuple(_dense(self.blocks[2][0], part) for part in primes[2])
        solved, derivative = checked_pair_frequency_jet(self.ee, pairs[3], rhs,
            primes[3], frequency_rhs, prior_status=prior, dependency=dependency)
        F = cp.project(cp.add(rr, cp.negate(_pair_product(self.re.pattern, pairs[1], solved.value))))
        Fp = cp.project(cp.add(cp.add(rrp,
            cp.negate(_pair_product(self.re.pattern, primes[1], solved.value))),
            cp.negate(_pair_product(self.re.pattern, pairs[1], derivative.value))))
        certificate_status = first(certificate_status, solved.aux.performed & (solved.aux.status != 0), solved.aux.status)
        certificate_status = first(certificate_status, derivative.aux.performed & (derivative.aux.status != 0), derivative.aux.status)
        return NativeSelectedPairJet(F, Fp, assembled.values, assembled.frequency_values,
            assembled.bound, solved.value[0], derivative.value[0],
            jnp.maximum(solved.aux.eta, derivative.aux.eta), check_status, certificate_status,
            (input_certificate, solved.aux, derivative.aux), assembled.low_values,
            assembled.frequency_low_values, solved.value[1], derivative.value[1])

    @staticmethod
    def certificate_layout(*, families, capability):
        """Static logical roles parallel to returned primal certificates."""
        families = frozenset(families)
        roles = []
        if families - {'operator'} and capability:
            roles.extend((('reference_inverse', 1), ('source_voltage', 5), ('source_admittance', 6)))
            if 'Z' in families:
                roles.append(('impedance', 7))
            if 'S' in families:
                roles.extend((('scattering', 8), ('normalizer_inverse', 1),
                              ('reconciliation_voltage', 9), ('reconciliation_scattering', 9),
                              ('source_reconciliation', 9)))
        if 'operator' in families:
            roles.append(('selected_elimination', 3))
        return tuple(roles)

    def __call__(self, operands, view, omega, *, families, capability,
                 prior_check_status=0, prior_certificate_status=0, dependency=()):
        import jax.numpy as jnp
        from .jax_core import first
        families = frozenset(families)
        q = len(self.view_program.terminal_ids)
        dtype = omega.dtype
        eye = jnp.eye(q, dtype=dtype)
        empty = jnp.zeros((q, q), dtype=dtype)
        # Checks and equation certificates have DISTINCT semantic precedence.
        # Earlier certificate failure skips later equations, but does not hide
        # a later actually evaluated assembly check from the check reducer.
        check_status = view.status
        certificate_status = jnp.int32(0)
        check_position = jnp.where(view.status != 0, jnp.int32(0), jnp.int32(-1))
        certificate_position = jnp.int32(-1)
        certificates = []
        check_count = 1
        def combined():
            checks = first(jnp.asarray(prior_check_status, dtype=jnp.int32), check_status != 0, check_status)
            certs = first(jnp.asarray(prior_certificate_status, dtype=jnp.int32), certificate_status != 0, certificate_status)
            return first(checks, certs != 0, certs)
        def check(code):
            nonlocal check_status, check_position, check_count
            owns_first = (check_status == 0) & (code != 0)
            check_position = jnp.where(owns_first, jnp.int32(check_count), check_position)
            check_status = first(check_status, code != 0, code)
            check_count += 1
        def certificate(aux):
            nonlocal certificate_status, certificate_position
            # A skipped equation carries its prior status but is not a newly
            # performed failing equation at this semantic occurrence.
            owns_first = (certificate_status == 0) & aux.performed & (aux.status != 0)
            certificate_position = jnp.where(owns_first, jnp.int32(len(certificates)), certificate_position)
            certificate_status = first(certificate_status, aux.performed & (aux.status != 0), aux.status)
            certificates.append(aux)
        def stage_dependency(stage):
            if dependency and dependency[0] == 'responses':
                families = ('S',) if stage in (8, 9) else ('Z',) if stage == 7 else tuple(
                    family for family in dependency[2] if family != 'operator')
                return ('responses', dependency[1], families)
            return dependency
        def small(matrix, rhs, stage):
            equation = checked_dense_solve(self.small, matrix.astype(dtype), rhs.astype(dtype),
                                           stage=stage, prior_status=combined(), dependency=stage_dependency(stage))
            certificate(equation.aux)
            return equation.value
        def sparse(equation, values, rhs, stage, *, dependency_override=None):
            solved = checked_sparse_solve(equation, values.astype(dtype), rhs.astype(dtype),
                                          stage=stage, prior_status=combined(), dependency=stage_dependency(stage) if dependency_override is None else dependency_override)
            certificate(solved.aux)
            return solved.value
        def selected():
            assembled = self.assembly.assemble_prepared(operands, omega, view_status=view.status,
                                                        loaded=True, compensated=False)
            check(assembled.status)
            values = tuple(assembled.values[jnp.asarray(sources)] for _, sources in self.blocks)
            rhs = _dense(self.blocks[2][0], values[2])
            X = sparse(self.ee, values[3], rhs, 3, dependency_override=('responses', dependency[1], ('operator',)) if dependency else ())
            return _dense(self.blocks[0][0], values[0]) - self.re.matmul(values[1], X)
        S = Y = Z = operator = empty
        responses = families - {'operator'}
        if responses and capability:
            assembled = self.assembly.assemble_prepared(operands, omega, view_status=view.status,
                                                        loaded=False, compensated=False)
            check(assembled.status)
            Rinv = small(view.Rk, eye, 1)
            B = _dense(self.view_program.patterns[3], view.model.B).astype(dtype)
            Bk = B @ view.selected_map.T
            effective = view.selected_map.T @ Rinv @ view.selected_map + view.Go
            pattern = self.view_program.patterns[3]
            cols = jnp.asarray(pattern.cols)
            stamp = view.model.B[:, None] * effective[cols[:, None], cols[None, :]] * view.model.B[None, :]
            load = jnp.zeros_like(assembled.values).at[jnp.asarray(self.assembly.load_slots).ravel()].add(stamp.ravel())
            W = assembled.values / (-jnp.asarray(1j, dtype=dtype) * omega) + load
            X = sparse(self.W, W, Bk, 5)
            Ysrc = small(Bk.T @ X, eye, 6)
            Y = Ysrc - Rinv
            if 'Z' in families:
                Z = small(Y, eye, 7)
            if 'S' in families:
                normalized = view.Dk @ Y @ view.Dk
                S = small(eye + normalized, eye - normalized, 8)
                Dinv = small(view.Dk, eye, 1)
                # Preserve the actual S stage-before-source boundary. This is
                # a second callback with a second call-local W factor.
                voltage = sparse(self.W, W, jnp.asarray(2, dtype=dtype) * Bk @ Dinv, 9)
                source_s = small(view.Dk, Bk.T @ voltage, 9) - eye
                numerator = jnp.max(jnp.abs(source_s - S), initial=0)
                denominator = 1 + jnp.max(jnp.abs(source_s), initial=0) + jnp.max(jnp.abs(S), initial=0)
                eta = numerator / denominator
                tau = jnp.asarray(256 * (len(self.view_program.node_ids) + 1), dtype=eta.dtype) * jnp.finfo(eta.dtype).eps
                performed = combined() == 0
                reconciliation = first(combined(), ~(jnp.isfinite(eta) & (eta <= tau)), jnp.int32(9))
                missing = jnp.asarray(jnp.inf, dtype=eta.dtype)
                certificate(NativeSparseAux(reconciliation, jnp.where(performed, eta, missing),
                    jnp.where(performed, numerator, missing), jnp.where(performed, denominator, missing), performed))
        elif responses:
            if 'S' in responses or any(family in responses for family in ('Y', 'Z')):
                # Current public matrix contract requires Port realizability.
                # Retained scalar response lowering is a separate declared
                # route; it must not silently enter this matrix route.
                from .determinants import QuantityError
                raise QuantityError('port_realizability', 'selected_network',
                                    'Direct matrix response requires a Port-realizable View')
        if 'operator' in families:
            operator = selected()
        return NativeFrequencyOutcome(S, Y, Z, operator, combined(), check_status, certificate_status,
                                      check_position, certificate_position, tuple(certificates))
