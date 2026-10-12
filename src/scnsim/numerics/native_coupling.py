"""Native residue coupling over finished roots of the SAME physical core.

Root packets provide their actual final selected states. Coupling never reroots
or refactors those states. The only new selected state is its declared evaluation
point. Physical contractions use ordinary transpose; the separate native right
chart owns vector derivatives and its local gauge availability.
"""
from __future__ import annotations
from functools import partial
from typing import Any, NamedTuple
import numpy as np


class NativeReciprocity(NamedTuple):
    performed: Any
    closed: Any
    numerator: Any
    denominator: Any
    ratio: Any
    corrected_defect_norm: Any
    allowance: Any
    input_skew: Any
    raw_skew: Any
    residual_contribution: Any
    corrected_defect: Any
    rounding_budget: Any


class NativeCouplingPacket(NamedTuple):
    coupling_rad_s: Any
    residue_a: Any
    residue_b: Any
    branch_roots_rad_s: Any
    evaluation_omega_rad_s: Any
    branch_slopes: Any
    branch_slope_scales: Any
    branch_independence: Any
    reciprocity: tuple
    check_status: Any
    certificate_status: Any
    status: Any
    failure_stage: Any
    derivative_available: Any
    certificates: tuple


COUPLING_STAGES = ('residue_slope', 'residue_rank', 'residue_coupling', 'reciprocity')


def _host_reciprocity(pattern, matrix, values, solution, *, selected, eliminated, ordering):
    # The established signed-residual certificate is reused exactly. This
    # callback performs no solve/model realization and owns no native factor.
    from scipy.sparse import csc_matrix
    from types import SimpleNamespace
    from .sparse_direct import schur_reciprocity
    dtype = np.asarray(matrix).dtype
    rd = np.empty((), dtype=dtype).real.dtype.type
    # Sparse indexing treats a two-entry tuple as (row, column), not an
    # ordered row selection. Keep the declared coordinate order as index arrays
    # at this host boundary; the reciprocity equations remain unchanged.
    system = SimpleNamespace(real_dtype=rd, complex_dtype=dtype.type,
                             selected=np.asarray(selected, dtype=np.int64),
                             eliminated=np.asarray(eliminated, dtype=np.int64))
    q = csc_matrix((values, pattern.rows, pattern.indptr), shape=pattern.shape)
    closed, facts = schur_reciprocity(system, (matrix, None, q, None, None, solution, None, None))
    skew = facts['input_skew']
    return NativeReciprocity(np.bool_(True), np.bool_(closed),
        *(np.asarray(facts[key], dtype=rd) for key in
          ('numerator', 'denominator', 'ratio', 'corrected_defect_norm', 'allowance')),
        np.asarray((skew['retained'], skew['eliminated'], skew['cross']), dtype=rd),
        *(np.asarray(facts[key], dtype=dtype) for key in
          ('raw_skew', 'residual_contribution', 'corrected_defect')),
        np.asarray(facts['rounding_budget'], dtype=rd))


def compile_native_coupling(frequency_program, state_function, branch_declarations,
                            *, precision, midpoint, availability_dependency):
    """Return (candidate, finished_branch_packets, evaluation_omega)->packet."""
    import jax
    import jax.numpy as jnp
    from .jax_core import first, tau
    from .native_root import compile_native_null_vector
    branch_declarations = tuple(branch_declarations)
    if len(branch_declarations) != 2:
        raise TypeError('declared residue coupling requires its two finished branches')
    width = len(frequency_program.view_program.selected)
    rd, cd = (jnp.float32, jnp.complex64) if precision == 'float32' else (jnp.float64, jnp.complex128)
    dependency = availability_dependency
    vector_chart = compile_native_null_vector(width, dependency=dependency)
    selected = tuple(frequency_program.view_program.selected)
    eliminated = tuple(frequency_program.view_program.eliminated)
    scalar = lambda dtype: jax.ShapeDtypeStruct((), dtype)
    shape = NativeReciprocity(scalar(jnp.bool_), scalar(jnp.bool_),
        *(scalar(rd) for _ in range(5)), jax.ShapeDtypeStruct((3,), rd),
        *(jax.ShapeDtypeStruct((width, width), cd) for _ in range(3)),
        jax.ShapeDtypeStruct((width, width), rd))

    def reciprocity(state):
        # Protect wide bit transport while tracing this callback boundary;
        # host arithmetic remains NumPy-only and selected32 stays selected32.
        with jax.enable_x64(True):
            return frequency_program.W.call_callback(('native_reciprocity', selected, eliminated),
                _host_reciprocity, shape, jax.lax.stop_gradient(state.F),
                jax.lax.stop_gradient(state.Q_values), jax.lax.stop_gradient(state.X),
                selected=selected, eliminated=eliminated)

    @jax.custom_jvp
    def branch_sqrt(value):
        return jnp.sqrt(value)

    @branch_sqrt.defjvp
    def branch_sqrt_direction(primals, tangents):
        from .native_sparse import derivative_available
        value, = primals
        direction, = tangents
        result = branch_sqrt(value)
        available = ~((value.imag == 0) & (value.real < 0))
        derivative_available(available, dependency=dependency,
            reason='residue coupling principal-square-root branch derivative unavailable')
        return result, direction / (jnp.asarray(2, dtype=cd) * result)

    def same_location(left, right):
        # Equality is selected-dtype bit equality, including signed zeros.
        with jax.enable_x64(True):
            bits = jnp.uint32 if precision == 'float32' else jnp.uint64
            return ((jax.lax.bitcast_convert_type(left.real, bits) == jax.lax.bitcast_convert_type(right.real, bits)) &
                    (jax.lax.bitcast_convert_type(left.imag, bits) == jax.lax.bitcast_convert_type(right.imag, bits)))

    def active(candidate, packets, evaluation_omega):
        roots = jnp.stack(tuple(packet.omega for packet in packets))
        checks = jnp.int32(0)
        certificates = jnp.int32(0)
        for packet in packets:
            checks = first(checks, packet.check_status != 0, packet.check_status)
            certificates = first(certificates, packet.certificate_status != 0, packet.certificate_status)
        status = first(checks, certificates != 0, certificates)
        failure_stage = jnp.int32(0)
        vectors, slopes, scales, residues = [], [], [], []
        gauge_available = jnp.bool_(True)
        for declaration, packet in zip(branch_declarations, packets, strict=True):
            state = packet.selected_state
            if declaration.kind == 'diagonal_root':
                vector = jnp.zeros((width,), dtype=cd).at[declaration.coordinate_index].set(1)
                status = first(status, packet.omega.imag > 0, 12)
            elif declaration.kind == 'hybridized_pole':
                magnitude = jnp.abs(packet.right_vector)
                gauge_available = gauge_available & (jnp.count_nonzero(magnitude == jnp.max(magnitude)) == 1)
                vector = vector_chart(state.F, packet.right_vector, packet.left_vector)
            else:
                raise TypeError('residue branch declaration requires diagonal or hybridized root')
            vector = vector / jnp.linalg.norm(vector)
            slope = -vector.T @ state.Fp @ vector
            scale = jnp.sum(jnp.abs(vector) * (jnp.abs(state.Fp) @ jnp.abs(vector)), dtype=rd)
            good = jnp.isfinite(slope) & jnp.isfinite(scale) & (scale > 0) & (jnp.abs(slope) / scale > tau(width, rd))
            status = first(status, ~good, 13)
            residue = -jnp.asarray(1, dtype=cd) / slope
            status = first(status, ~jnp.isfinite(residue), 13)
            vectors.append(vector); slopes.append(slope); scales.append(scale); residues.append(residue)
        singular = jnp.linalg.svd(jnp.stack(vectors, axis=1), compute_uv=False)
        rank = (singular[1] / singular[0] > tau(width, rd)) if width >= 2 else jnp.bool_(False)
        failure_stage = jnp.where((status == 0) & ~rank, jnp.int32(1), failure_stage)
        status = first(status, ~rank, 13)
        evaluation = (roots[0] + roots[1]) / jnp.asarray(2, dtype=cd) if midpoint else jnp.asarray(evaluation_omega, dtype=cd)
        failure_stage = jnp.where((status == 0) & ~jnp.isfinite(evaluation), jnp.int32(2), failure_stage)
        status = first(status, ~jnp.isfinite(evaluation), 13)
        reciprocity_facts = []
        locations = roots[0], roots[1], evaluation
        # Preserve original root-reciprocity-before-evaluation-state order.
        # A later native fatal cannot precede a failed earlier required check.
        def certificate_at(state, location, ordinal, status, failure_stage):
            duplicate = jnp.bool_(False)
            for previous in locations[:ordinal]:
                duplicate = duplicate | same_location(location, previous)
            absent = jax.tree.map(lambda item: jnp.zeros(item.shape, dtype=item.dtype), shape)
            actual = jax.lax.cond((status == 0) & ~duplicate,
                lambda _: reciprocity(state), lambda _: absent, operand=None)
            failure_stage = jnp.where((status == 0) & actual.performed & ~actual.closed,
                                      jnp.int32(3), failure_stage)
            return actual, first(status, actual.performed & ~actual.closed, 13), failure_stage
        for ordinal, packet in enumerate(packets):
            actual, status, failure_stage = certificate_at(packet.selected_state,
                locations[ordinal], ordinal, status, failure_stage)
            reciprocity_facts.append(actual)
        state_shape = jax.eval_shape(state_function, candidate, evaluation)
        empty_state = jax.tree.map(lambda item: jnp.zeros(item.shape, dtype=item.dtype), state_shape)
        evaluation_state = jax.lax.cond(status == 0, lambda _: state_function(candidate, evaluation),
                                          lambda _: empty_state, operand=None)
        checks = first(checks, evaluation_state.check_status != 0, evaluation_state.check_status)
        certificates = first(certificates, evaluation_state.certificate_status != 0, evaluation_state.certificate_status)
        status = first(status, checks != 0, checks)
        status = first(status, certificates != 0, certificates)
        actual, status, failure_stage = certificate_at(evaluation_state, evaluation, 2, status, failure_stage)
        reciprocity_facts.append(actual)
        product = jnp.asarray(slopes[0] * slopes[1], dtype=cd)
        denominator = branch_sqrt(product)
        good = jnp.isfinite(denominator) & (denominator != 0)
        failure_stage = jnp.where((status == 0) & ~good, jnp.int32(2), failure_stage)
        status = first(status, ~good, 13)
        value = jnp.asarray((vectors[0].T @ evaluation_state.F @ vectors[1]) / denominator, dtype=cd)
        failure_stage = jnp.where((status == 0) & ~jnp.isfinite(value), jnp.int32(2), failure_stage)
        status = first(status, ~jnp.isfinite(value), 13)
        available = gauge_available & ~((product.imag == 0) & (product.real < 0))
        certificates = first(certificates, status != 0, status)
        return NativeCouplingPacket(value, residues[0], residues[1], roots, evaluation,
            jnp.stack(slopes), jnp.stack(scales), singular, tuple(reciprocity_facts),
            checks, certificates, status, failure_stage, available,
            tuple(evaluation_state.certificates))
    def coupling(candidate, packets, evaluation_omega):
        checks, certificates = jnp.int32(0), jnp.int32(0)
        for packet in packets:
            checks = first(checks, packet.check_status != 0, packet.check_status)
            certificates = first(certificates, packet.certificate_status != 0, packet.certificate_status)
        prior = first(checks, certificates != 0, certificates)
        result_shape = jax.eval_shape(active, candidate, packets, evaluation_omega)
        empty = jax.tree.map(lambda item: jnp.zeros(item.shape, dtype=item.dtype), result_shape)
        skipped = empty._replace(check_status=checks, certificate_status=certificates, status=prior)
        return jax.lax.cond(prior == 0, lambda _: active(candidate, packets, evaluation_omega),
                            lambda _: skipped, operand=None)
    return coupling
