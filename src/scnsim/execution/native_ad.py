"""Bounded JAX-native derivative orchestration for the physical provider.

This module owns operation-local candidate/root linear frames and composes the
provider's checked frequency chunks with its compiler-owned public projection.
It never rebuilds a physical equation or stores a frame in a program cache.
"""
from __future__ import annotations

from contextlib import nullcontext


_BASIS_WIDTH = 8


def _abstract_operands(operands):
    """Keep only the pytree's array signature in a reusable variant recipe."""
    import jax
    from ..numerics.native_program import native_operand_abstract

    def abstract(value):
        aval = native_operand_abstract(value)
        return jax.ShapeDtypeStruct(tuple(aval.shape), aval.dtype, weak_type=aval.weak_type)
    return jax.tree.map(abstract, operands)


def _bucket_sizes(count, width):
    """Yield full policy buckets followed by actual one-frequency tails."""
    while count >= width:
        yield width
        count -= width
    while count:
        yield 1
        count -= 1


def _work_groups(native):
    groups = []
    for row in native.work_layout:
        key = row['view_index'], tuple(row['demand'])
        if groups and groups[-1][0] == key:
            groups[-1][1].append(row)
        else:
            groups.append((key, [row]))
    return tuple((key, tuple(rows)) for key, rows in groups)


def _frame_key(jax, frame):
    """Include JAX's static Partial authority, not just its array shapes."""
    return jax.tree.structure(frame)


def _checked_failure(error):
    exception = error.get_exception()
    if exception is not None:
        error.throw()


def _lane_error(jax, error, index):
    import numpy as np
    return jax.tree.map(lambda value: np.asarray(value)[index], error)


def _append_value_error_frames(jax, frames, error, rows):
    from ..numerics.evidence import MappedNativeErrorFrame
    for lane, row in enumerate(rows):
        frames.append(MappedNativeErrorFrame(
            _lane_error(jax, error, lane), row['work_index'], row['source_index'],
        ))


def _append_response_error_frames(frames, result, rows, *, basis_kind, basis_indices):
    from ..numerics.evidence import response_ad_error_frames
    frames.extend(response_ad_error_frames(
        result, work_indices=tuple(row['work_index'] for row in rows),
        frequency_source_indices=tuple(row['source_index'] for row in rows),
        basis_kind=basis_kind, basis_indices=basis_indices,
    ))


def _candidate_frame(native, theta, *, backend, owner):
    variant = candidate_ad_variant(native, theta, device=backend.device)
    error, frame = backend.execute_native(
        variant.function, theta,
        precision=backend.precision, compile_key=variant.key,
        owner=owner, phase='native_ad_candidate_frame',
    )
    _checked_failure(error)
    return frame


def _candidate_tangents(native, frame, directions, *, backend, owner, basis_kind,
                        basis_indices):
    jax = backend.jax
    function = native.candidate_tangents_function
    operands = frame, directions
    key = (*native.variant_key(
        'candidate_jvp', device=backend.device, operands=operands,
        transform='candidate_frame_pushforward_block8',
    ), 'frame_treedef', _frame_key(jax, frame), 'basis', basis_kind,
        tuple(basis_indices), 'block', _BASIS_WIDTH)
    return backend.execute_native(
        function, *operands, precision=backend.precision, compile_key=key,
        owner=owner, phase='native_ad_candidate_pushforward',
        details={'basis_kind': basis_kind, 'basis_count': len(basis_indices)},
    )


def _branch_frame(native, candidate, omega, seeds, check_status, certificate_status,
                  *, backend, owner, completed_branches=None, completed_branch_errors=None):
    if completed_branches is not None:
        from ..numerics.native_program import bind_completed_branches
        completed_branches = bind_completed_branches(completed_branches)
    variant = branch_ad_variant(native, candidate, omega, seeds, check_status,
                                certificate_status, device=backend.device,
                                completed_branches=completed_branches,
                                completed_branch_errors=completed_branch_errors)
    error, frame = backend.execute_native(
        variant.function, candidate, omega, seeds, check_status, certificate_status,
        *((completed_branches,completed_branch_errors) if completed_branches is not None else ()),
        precision=backend.precision, compile_key=variant.key,
        owner=owner, phase='native_ad_joint_branches',
    )
    return error, frame


def candidate_ad_variant(native, theta, *, device):
    """Return the exact checked candidate-frame recipe shared with warmup."""
    from ..numerics.native_sparse import checked_native
    from .variant_plan import ExecutionVariant

    operands = (theta,)
    key = native.variant_key(
        'candidate_ad', device=device, operands=operands,
        transform='checked_candidate_linearization',
    )
    return ExecutionVariant('candidate_ad', key,
        checked_native(native.candidate_ad_function), _abstract_operands(operands))


def branch_ad_variant(native, candidate, omega, seeds, check_status,
                      certificate_status, *, device, completed_branches=None, completed_branch_errors=None):
    """Return the exact checked joint-branch frame recipe shared with warmup."""
    from ..numerics.native_sparse import checked_native
    from .variant_plan import ExecutionVariant
    import jax

    operands = candidate, omega, seeds, check_status, certificate_status
    if completed_branches is not None:
        from ..numerics.native_program import bind_completed_branches
        completed_branches = bind_completed_branches(completed_branches)
        operands = (*operands, completed_branches, completed_branch_errors)
    key = native.variant_key(
        'branch_ad', device=device, operands=operands,
        transform=('checked_joint_branch_linearization' if completed_branches is None
                   else ('checked_completed_branch_linearization',
                         str(jax.tree.structure(completed_branch_errors)))),
    )
    return ExecutionVariant('branch_ad', key,
        checked_native(native.branch_ad_function), _abstract_operands(operands))


def _response_slot_plan(native):
    plans = []
    for observable_index in native.response_observable_indices:
        declaration = native.occurrence_layout[observable_index]
        positions_by_work = {}
        for position, work_index in enumerate(declaration['work_indices']):
            positions_by_work.setdefault(work_index, []).append(position)
        plans.append((observable_index, declaration['view_index'], declaration['family'],
                      {key: tuple(value) for key, value in positions_by_work.items()},
                      len(declaration['work_indices'])))
    return tuple(plans)


def _response_slots(native):
    return [[None] * plan[4] for plan in _response_slot_plan(native)]


def _save_response_chunk(native, slots, selected, families, family_values,
                         family_derivatives=None):
    """Retain only each declared selected leaf, never a frequency outcome."""
    import jax
    import jax.numpy as jnp

    family_positions = {family: index for index, family in enumerate(families)}
    for plan_position, plan in enumerate(_response_slot_plan(native)):
        observable_index, view_index, family, positions_by_work, _ = plan
        if view_index != selected[0]['view_index'] or family not in family_positions:
            continue
        family_index = family_positions[family]
        values = family_values[family_index]
        selected_values = native.response_leaf_function(observable_index, values)
        if family_derivatives is None:
            selected_derivatives = None
        else:
            derivative_values = jnp.swapaxes(family_derivatives[family_index], 0, 1)
            selected_derivatives = jax.vmap(
                lambda array: native.response_leaf_function(observable_index, array)
            )(derivative_values)
        for lane, work in enumerate(selected):
            for occurrence in positions_by_work.get(work['work_index'], ()):
                slots[plan_position][occurrence] = (
                    selected_values[lane] if selected_derivatives is None
                    else selected_derivatives[:, lane]
                )


def _finish_response_slots(native, slots, *, banked=False):
    import jax.numpy as jnp
    if banked:
        return tuple(jnp.stack(tuple(values), axis=1) for values in slots)
    return tuple(jnp.stack(tuple(values), axis=0) for values in slots)


def _value_responses(native, candidate, omega, *, backend, owner, policy,
                     initial_checks, initial_certificates):
    import jax.numpy as jnp
    from ..numerics.native_sparse import checked_native

    slots = _response_slots(native)
    family_shapes = {}
    frames = []
    check_status, certificate_status = initial_checks, initial_certificates
    for (view_index, demand), rows in _work_groups(native):
        offset = 0
        for bucket in _bucket_sizes(len(rows), policy.frequency_chunk_size):
            selected = rows[offset:offset + bucket]
            offset += bucket
            frequencies = jnp.stack(tuple(omega[row['omega_index']] for row in selected))
            function = checked_native(native.chunk_function(
                view_index, demand, bucket, transform='value',
            ))
            operands = (candidate, frequencies, check_status, certificate_status)
            key = native.variant_key(
                'chunk', device=backend.device, operands=operands,
                view_index=view_index, demand=demand, bucket=bucket,
                transform='checked_value',
            )
            error, result = backend.execute_native(
                function, *operands, precision=backend.precision, compile_key=key,
                owner=owner, phase='native_ad_response_value',
                details={'view_index': view_index, 'work_count': bucket},
            )
            # The chunk returns per-frequency checked Errors as data. Preserve
            # any enclosing structural Error separately so projection can
            # distinguish it from those ordered equation records.
            from ..numerics.evidence import MappedNativeErrorFrame
            frames.append(MappedNativeErrorFrame(error, None, None))
            _append_value_error_frames(backend.jax, frames, result.errors, selected)
            families = tuple(sorted(demand))
            family_values = tuple(getattr(result.outcomes, family) for family in families)
            for family, value in zip(families, family_values, strict=True):
                family_shapes[(view_index, family)] = (tuple(value.shape[1:]), value.dtype)
            _save_response_chunk(native, slots, selected, families, family_values)
            check_status, certificate_status = result.check_status, result.certificate_status
    return (_finish_response_slots(native, slots), family_shapes, frames,
            check_status, certificate_status)


def _response_jvps(native, candidate_frame, omega, directions, *, backend, owner,
                   policy, initial_checks, initial_certificates,
                   basis_kind, basis_indices, frame_owner):
    import jax.numpy as jnp

    value_slots = _response_slots(native)
    derivative_slots = _response_slots(native)
    frames = []
    check_status, certificate_status = initial_checks, initial_certificates
    for (view_index, demand), rows in _work_groups(native):
        offset = 0
        for bucket in _bucket_sizes(len(rows), policy.frequency_chunk_size):
            selected = rows[offset:offset + bucket]
            offset += bucket
            frequencies = jnp.stack(tuple(omega[row['omega_index']] for row in selected))
            function = native.response_ad_chunk(view_index, demand, bucket, mode='jvp')
            operands = (candidate_frame, frequencies, directions,
                        check_status, certificate_status)
            key = (*native.variant_key(
                'response_ad', device=backend.device, operands=operands,
                view_index=view_index, demand=demand, bucket=bucket,
                transform='checked_jvp_block8',
            ), 'frame_treedef', _frame_key(backend.jax, candidate_frame),
                'basis_kind', basis_kind, 'basis_indices', tuple(basis_indices),
                'direction_block', _BASIS_WIDTH)
            result = backend.execute_native(
                function, *operands, precision=backend.precision,
                compile_key=key, owner=frame_owner, phase='native_ad_response_jvp',
                details={'view_index': view_index, 'work_count': bucket,
                         'basis_count': len(basis_indices)},
            )
            _append_response_error_frames(frames, result, selected,
                basis_kind=basis_kind, basis_indices=basis_indices)
            families = tuple(sorted(demand))
            _save_response_chunk(native, value_slots, selected, families,
                                 result.public_values)
            _save_response_chunk(native, derivative_slots, selected, families,
                                 result.public_values, result.derivatives)
            check_status, certificate_status = result.check_status, result.certificate_status
    return (_finish_response_slots(native, value_slots),
            _finish_response_slots(native, derivative_slots, banked=True),
            frames, check_status, certificate_status)


def _project(native, response_values, branch_frame):
    return native.compact_leaf_projection_function(
        response_values, branch_frame.outcome.values,
        branch_frame.outcome.derivative_available,
    )


def _branch_direction_frames(native, candidate_frame, branch_frame, theta_directions,
                             omega, seeds, *, backend, owner, basis_kind,
                             basis_indices, mode):
    import jax
    import jax.numpy as jnp
    from ..numerics.native_sparse import checked_native
    from ..numerics.evidence import MappedNativeErrorFrame

    candidate_dots = _candidate_tangents(
        native, candidate_frame, theta_directions, backend=backend, owner=owner,
        basis_kind=basis_kind, basis_indices=basis_indices,
    )
    zeros_omega = jnp.zeros((len(basis_indices), *omega.shape), dtype=omega.dtype)
    zeros_seeds = jnp.zeros((len(basis_indices), *seeds.shape), dtype=seeds.dtype)
    if mode == 'jvp':
        single = checked_native(lambda frame, candidate_dot, omega_dot, seeds_dot:
            native.branch_tangent_function(frame, candidate_dot, omega_dot, seeds_dot))

        def mapped(frame, candidate_values, omega_values, seed_values):
            def one(values):
                return single(frame, values[0], values[1], values[2])
            return jax.lax.map(one, (candidate_values, omega_values, seed_values),
                               batch_size=_BASIS_WIDTH)

        operands = branch_frame, candidate_dots, zeros_omega, zeros_seeds
        transform = 'checked_branch_jvp_block8'
    else:
        raise TypeError('branch pullbacks are dispatched by _branch_pullbacks')
    key = (*native.variant_key(
        'branch_ad', device=backend.device,
        operands=(candidate_frame.candidate, omega, seeds), transform=transform,
    ), 'frame_treedef', _frame_key(jax, branch_frame),
        'basis_kind', basis_kind, 'basis_indices', tuple(basis_indices),
        'direction_block', _BASIS_WIDTH)
    errors, values = backend.execute_native(
        mapped, *operands, precision=backend.precision, compile_key=key,
        owner=owner, phase='native_ad_branch_jvp',
        details={'basis_kind': basis_kind, 'basis_count': len(basis_indices)},
    )
    return values, (MappedNativeErrorFrame(
        errors, None, None, basis_kind, tuple(basis_indices),
    ),)


def _branch_pullbacks(native, branch_frame, branch_covectors, *, backend, owner,
                      basis_kind, basis_indices):
    # A response-only projection has no branch-output subtree. In that case
    # there is no branch pullback to map; JAX cannot infer a scan length from
    # an empty pytree, and manufacturing a zero cotangent would add a second
    # derivative authority.
    if not branch_covectors:
        return None, ()

    import jax
    from ..numerics.native_sparse import checked_native
    from ..numerics.evidence import MappedNativeErrorFrame

    single = checked_native(lambda frame, covectors:
        native.branch_pullback_function(frame, covectors))

    def mapped(frame, covector_values):
        return jax.lax.map(lambda covectors: single(frame, covectors),
                           covector_values, batch_size=_BASIS_WIDTH)

    key = (*native.variant_key(
        'branch_ad', device=backend.device,
        operands=(branch_frame, branch_covectors),
        transform='checked_branch_vjp_block8',
    ), 'frame_treedef', _frame_key(jax, branch_frame),
        'basis_kind', basis_kind, 'basis_indices', tuple(basis_indices),
        'direction_block', _BASIS_WIDTH)
    errors, cotangents = backend.execute_native(
        mapped, branch_frame, branch_covectors,
        precision=backend.precision, compile_key=key, owner=owner,
        phase='native_ad_branch_vjp',
        details={'basis_kind': basis_kind, 'basis_count': len(basis_indices)},
    )
    return cotangents, (MappedNativeErrorFrame(
        errors, None, None, basis_kind, tuple(basis_indices),
    ),)


def _project_jvps(native, response_values, branch_frame, response_tangents,
                  branch_tangents, *, backend, owner, frame_owner):
    import jax

    primal = (response_values, branch_frame.outcome.values)

    def function(response_bank, branch_bank, response, branch, available):
        def one(tangents):
            response_tangent, branch_tangent = tangents
            def project(response_value, branch_value):
                return native.compact_leaf_projection_function(
                    response_value, branch_value, available,
                )[0]
            _, tangent = jax.jvp(
                project, (response, branch),
                (response_tangent, branch_tangent),
            )
            return tangent
        return jax.lax.map(
            one, (response_bank, branch_bank), batch_size=_BASIS_WIDTH,
        )

    operands = (response_tangents, branch_tangents, *primal,
                branch_frame.outcome.derivative_available)
    key = (*native.variant_key(
        'value', device=backend.device, operands=primal,
        transform='compact_public_projection_jvp_block8',
    ), 'direction_block', _BASIS_WIDTH,
        'projection_frame', _frame_key(jax, (*primal, branch_frame.outcome.derivative_available)))
    return backend.execute_native(
        function, *operands,
        precision=backend.precision, compile_key=key, owner=frame_owner,
        phase='native_ad_public_projection_jvp',
    )


def objective_loss_pullback_variant(native, loss, values, coefficients, *, device):
    """Return the same compact loss/cotangent recipe for warmup and execution.

    Coefficients are explicit dynamic operands. The key carries the ordered
    static loss structure and operand signature, never target/scale/weight
    values. The callable uses the loss object's structural algebra only; its
    numeric coefficients arrive through ``runtime_coefficients``.
    """
    from .variant_plan import ExecutionVariant

    operands = (values, coefficients)
    key = (native.variant_key(
        'value', device=device, operands=operands,
        transform='objective_loss_output_pullback',
    ), ('objective_loss', loss.structural_record()))
    return ExecutionVariant('objective_loss_output_pullback', key,
        loss.output_pullback_function(), _abstract_operands(operands),
        lifetime='run_static')


def _objective_loss_payload_and_covectors(native, loss, values, coefficients, *,
                                          backend, owner):
    variant = objective_loss_pullback_variant(native, loss, values, coefficients,
                                               device=backend.device)
    return backend.execute_native(
        variant.function, values, coefficients,
        precision=backend.precision, compile_key=variant.key,
        owner=owner, phase='native_objective_loss_pullback',
        details={'objective_count': len(loss.objective_ids)},
    )


def _projection_pullback(native, response_values, branch_frame, *, backend,
                         weights=None, basis=None):
    import jax
    import jax.numpy as jnp

    available = branch_frame.outcome.derivative_available
    def function(response, branch):
        return native.compact_leaf_projection_function(response, branch, available)[0]

    flat, pullback = jax.vjp(function, response_values, branch_frame.outcome.values)
    if basis is None:
        response_cotangents, branch_cotangents = pullback(weights)
        return response_cotangents, branch_cotangents, weights, flat, None

    def one(covector):
        return pullback(covector)

    response_cotangents, branch_cotangents = jax.lax.map(
        one, basis, batch_size=_BASIS_WIDTH,
    )
    return response_cotangents, branch_cotangents, None, flat, None


def _work_vjp(native, candidate_frame, omega, response_cotangents, family_shapes, *,
              backend, policy, check_status, certificate_status,
              basis_kind, basis_indices, frame_owner):
    import jax
    import jax.numpy as jnp
    from ..numerics.evidence import response_ad_error_frames

    plans = _response_slot_plan(native)
    total_candidate = None
    frames = []
    checks, certificates = check_status, certificate_status
    for (view_index, demand), rows in _work_groups(native):
        offset = 0
        for bucket in _bucket_sizes(len(rows), policy.frequency_chunk_size):
            selected = rows[offset:offset + bucket]
            offset += bucket
            frequencies = jnp.stack(tuple(omega[row['omega_index']] for row in selected))
            families = tuple(sorted(demand))
            count = len(basis_indices)
            by_family = {
                family: jnp.zeros(
                    (count, bucket, *family_shapes[(view_index, family)][0]),
                    dtype=family_shapes[(view_index, family)][1],
                ) for family in families
            }
            for position, plan in enumerate(plans):
                observable_index, selected_view, family, work_positions, _ = plan
                if selected_view != view_index or family not in by_family:
                    continue
                leaf_cotangents = response_cotangents[position]
                lane_cotangents = []
                for row in selected:
                    positions = work_positions.get(row['work_index'], ())
                    value = None
                    for occurrence in positions:
                        contribution = leaf_cotangents[:, occurrence]
                        value = contribution if value is None else value + contribution
                    if value is None:
                        leaf_shape = tuple(leaf_cotangents.shape[2:])
                        value = jnp.zeros((count, *leaf_shape), dtype=leaf_cotangents.dtype)
                    lane_cotangents.append(value)
                selected_cotangents = jnp.stack(tuple(lane_cotangents), axis=1)
                shape, dtype = family_shapes[(view_index, family)]
                dummy = jnp.zeros((bucket, *shape), dtype=dtype)

                def transpose_selector(matrix, cotangents):
                    _, pullback = jax.vjp(
                        lambda value: native.response_leaf_function(observable_index, value),
                        matrix,
                    )
                    return jax.vmap(lambda value: pullback(value)[0])(cotangents)

                key = native.variant_key(
                    'chunk', device=backend.device,
                    operands=(dummy, selected_cotangents), view_index=view_index,
                    demand=demand, bucket=bucket,
                    transform=f'response_leaf_transpose:{observable_index}',
                    equation_signature=('response_leaf', native.aggregate_identity,
                                        observable_index),
                )
                family_cotangents = backend.execute_native(
                    transpose_selector, dummy, selected_cotangents,
                    precision=backend.precision, compile_key=key,
                    owner=frame_owner, phase='native_ad_response_leaf_transpose',
                    details={'view_index': view_index, 'observable_index': observable_index,
                             'work_count': bucket, 'basis_count': count},
                )
                by_family[family] = by_family[family] + family_cotangents

            directional = tuple(jnp.swapaxes(by_family[family], 0, 1)
                                for family in families)
            function = native.response_ad_chunk(view_index, demand, bucket, mode='vjp')
            operands = (candidate_frame, frequencies, directional, checks, certificates)
            key = (*native.variant_key(
                'response_ad', device=backend.device, operands=operands,
                view_index=view_index, demand=demand, bucket=bucket,
                transform='checked_vjp_block8',
            ), 'frame_treedef', _frame_key(jax, candidate_frame),
                'basis_kind', basis_kind, 'basis_indices', tuple(basis_indices),
                'direction_block', _BASIS_WIDTH)
            result = backend.execute_native(
                function, *operands, precision=backend.precision,
                compile_key=key, owner=frame_owner, phase='native_ad_response_vjp',
                details={'view_index': view_index, 'work_count': bucket,
                         'basis_count': len(basis_indices)},
            )
            frames.extend(response_ad_error_frames(result,
                work_indices=tuple(row['work_index'] for row in selected),
                frequency_source_indices=tuple(row['source_index'] for row in selected),
                basis_kind=basis_kind, basis_indices=basis_indices))
            total_candidate = (result.candidate_cotangents if total_candidate is None else
                native.core.add_candidate_cotangents(total_candidate,
                                                      result.candidate_cotangents))
            checks, certificates = result.check_status, result.certificate_status
    return total_candidate, frames, checks, certificates


def _candidate_gradient(native, candidate_frame, candidate_cotangents, *, backend,
                        frame_owner, basis_kind, basis_indices):
    jax = backend.jax
    operands = (candidate_frame, candidate_cotangents)
    key = (*native.variant_key(
        'candidate_vjp', device=backend.device, operands=operands,
        transform='candidate_frame_pullback_block8',
    ), 'frame_treedef', _frame_key(jax, candidate_frame),
        'basis_kind', basis_kind, 'basis_indices', tuple(basis_indices),
        'direction_block', _BASIS_WIDTH)
    return backend.execute_native(
        native.candidate_pullback_function, *operands,
        precision=backend.precision, compile_key=key, owner=frame_owner,
        phase='native_ad_candidate_pullback',
        details={'basis_kind': basis_kind, 'basis_count': len(basis_indices)},
    )


def _raw_result(native, candidate_frame, branch_frame, flat, projection, checks, certificates):
    from ..numerics.native_program import NativeProgramAux

    candidate = candidate_frame.candidate
    aux = NativeProgramAux(
        candidate.physical_checks, candidate.partition_status,
        candidate.realized_capabilities, checks, certificates, (),
        projection.derivative_available, branch_frame.outcome.branch_packets,
        branch_frame.outcome.quantity_packets, projection.observable_values,
        projection.observable_raw_values, projection.output_values,
        tuple((index, candidate.views[native.occurrence_layout[index]['view_index']].coordinate_transform)
              for index in native.branch_observable_indices
              if native.occurrence_layout[index]['family'] in ('static_capacitance', 'static_capacitance_element')),
    )
    return (flat, aux)


def _basis_directions(jnp, count, dimension, *, dtype, indices=None):
    import numpy as np

    selected = tuple(range(dimension)) if indices is None else tuple(indices)
    basis = np.zeros((len(selected), dimension), dtype=np.float64)
    if selected:
        basis[np.arange(len(selected)), np.asarray(selected, dtype=np.intp)] = 1.0
    return jnp.asarray(basis, dtype=dtype), selected


def sensitivity_derivative(native, arguments, *, derivative_kind, quantity_program,
                           inputs, prepared_analysis, backend, program_owner,
                           execution_policy, frame_owner, compact_evidence=False,
                           completed_branches=None, completed_branch_errors=None):
    """Acquire optional original-point search, then consume its local implicit AD."""
    searches = ()
    if completed_branches is None and any(row['spec'].get('search')
            for row in quantity_program.observable_metadata() if row['start_index'] is not None):
        from .physical_analysis import acquire_completed_root_frame
        native,completed,_,searches = acquire_completed_root_frame(quantity_program,inputs,
            backend=backend,program_owner=program_owner,
            operation_resources=backend.operation_resources,
            compact_evidence=compact_evidence,discover=True)
        completed_branches,completed_branch_errors = completed
    outcome = _sensitivity_derivative(native,arguments,derivative_kind=derivative_kind,
        quantity_program=quantity_program,inputs=inputs,prepared_analysis=prepared_analysis,
        backend=backend,program_owner=program_owner,execution_policy=execution_policy,
        frame_owner=frame_owner,compact_evidence=compact_evidence,
        completed_branches=completed_branches,completed_branch_errors=completed_branch_errors)
    from ..numerics.evidence import attach_root_search_metadata
    outcome = attach_root_search_metadata(outcome,searches)
    return outcome


def _sensitivity_derivative(native, arguments, *, derivative_kind, quantity_program,
                           inputs, prepared_analysis, backend, program_owner,
                           execution_policy, frame_owner, compact_evidence=False, completed_branches=None, completed_branch_errors=None):
    """Evaluate one checked JVP, VJP, or blocked Jacobian on the shared core."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from json import loads
    from ..numerics.evidence import (
        project_native_derivative_outcome, project_native_jacobian_outcome,
        project_native_jvp_outcome,
    )
    from ..numerics.evidence import MappedNativeErrorFrame

    theta, omega, seeds = arguments
    indices = tuple(quantity_program.program.model_program.fields.layout.input_indices(
        prepared_analysis.active_keys))
    owner = program_owner
    borrowed = nullcontext() if owner is None else owner.hold_exact_core(native.core)
    with borrowed, jax.enable_x64(True), jax.default_device(backend.device):
        candidate_frame = _candidate_frame(native, theta, backend=backend, owner=owner)
        candidate = candidate_frame.candidate
        from ..numerics.jax_core import first
        initial_checks = first(candidate.physical_checks.status,
            candidate.partition_status != 0, candidate.partition_status)
        initial_certificates = jnp.int32(0)
        policy = execution_policy
        error_frames = []
        output_count = sum(int(np.prod(block.shape)) * len(block.components)
                           for block in quantity_program.output_layout.blocks)

        if derivative_kind == 'jvp':
            direction = quantity_program.direction_inputs(
                inputs, loads(prepared_analysis.direction_bytes))
            directions = jnp.asarray(direction.theta, dtype=jnp.float64)[None, :]
            basis_kind, basis_indices = 'direction', (0,)
            response_values, response_derivatives, frames, checks, certificates = _response_jvps(
                native, candidate_frame, omega, directions, backend=backend, owner=owner,
                policy=policy, initial_checks=initial_checks,
                initial_certificates=initial_certificates,
                basis_kind=basis_kind, basis_indices=basis_indices,
                frame_owner=frame_owner)
            error_frames.extend(frames)
            branch_error, branch_frame = _branch_frame(native, candidate, omega, seeds,
                checks, certificates, backend=backend, owner=owner,
                completed_branches=completed_branches,
                completed_branch_errors=completed_branch_errors)
            branch_theta, branch_errors = _branch_direction_frames(
                native, candidate_frame, branch_frame, directions, omega, seeds,
                backend=backend, owner=frame_owner, basis_kind=basis_kind,
                basis_indices=basis_indices, mode='jvp')
            error_frames.extend(branch_errors)
            flat, projection = _project(native, response_values, branch_frame)
            derivative = _project_jvps(native, response_values, branch_frame,
                response_derivatives, branch_theta, backend=backend,
                owner=owner, frame_owner=frame_owner)
            derivative = derivative[0]
            raw = _raw_result(native, candidate_frame, branch_frame, flat, projection,
                              branch_frame.outcome.check_status,
                              branch_frame.outcome.certificate_status)
            return project_native_jvp_outcome(native, (raw, derivative), branch_error,
                                              error_frames=tuple(error_frames), compact_evidence=compact_evidence)

        if derivative_kind == 'jacobian':
            output_count = sum(block.row_count for block in quantity_program.output_layout.blocks)
            if len(indices) <= output_count:
                directions, basis_indices = _basis_directions(
                    jnp, len(indices), len(theta), dtype=jnp.float64, indices=indices)
                basis_kind, mode = 'parameter', 'forward'
                response_values, response_derivatives, frames, checks, certificates = _response_jvps(
                    native, candidate_frame, omega, directions, backend=backend, owner=owner,
                    policy=policy, initial_checks=initial_checks,
                    initial_certificates=initial_certificates,
                    basis_kind=basis_kind, basis_indices=basis_indices,
                    frame_owner=frame_owner)
                error_frames.extend(frames)
                branch_error, branch_frame = _branch_frame(native, candidate, omega, seeds,
                    checks, certificates, backend=backend, owner=owner,
                completed_branches=completed_branches,
                completed_branch_errors=completed_branch_errors)
                branch_tangents, branch_errors = _branch_direction_frames(
                    native, candidate_frame, branch_frame, directions, omega, seeds,
                    backend=backend, owner=frame_owner, basis_kind=basis_kind,
                    basis_indices=basis_indices, mode='jvp')
                error_frames.extend(branch_errors)
                flat, projection = _project(native, response_values, branch_frame)
                columns = _project_jvps(native, response_values, branch_frame,
                    response_derivatives, branch_tangents, backend=backend,
                    owner=owner, frame_owner=frame_owner)
                jacobian = jnp.swapaxes(columns, 0, 1)
            else:
                basis_kind, mode = 'output_row', 'reverse'
                response_values, family_shapes, value_frames, checks, certificates = _value_responses(
                    native, candidate, omega, backend=backend, owner=owner,
                    policy=policy, initial_checks=initial_checks,
                    initial_certificates=initial_certificates)
                error_frames.extend(value_frames)
                branch_error, branch_frame = _branch_frame(native, candidate, omega, seeds,
                    checks, certificates, backend=backend, owner=owner,
                completed_branches=completed_branches,
                completed_branch_errors=completed_branch_errors)
                flat, projection = _project(native, response_values, branch_frame)
                gradient_banks = []
                for start in range(0, output_count, _BASIS_WIDTH):
                    basis_indices = tuple(range(start, min(start + _BASIS_WIDTH, output_count)))
                    basis, _ = _basis_directions(
                        jnp, len(basis_indices), output_count,
                        dtype=jnp.float64, indices=basis_indices,
                    )
                    response_covectors, branch_covectors, _, flat, _ = _projection_pullback(
                        native, response_values, branch_frame, backend=backend, basis=basis)
                    total_response_candidate, frames, _, _ = _work_vjp(
                        native, candidate_frame, omega, response_covectors, family_shapes,
                        backend=backend, policy=policy,
                        check_status=checks, certificate_status=certificates,
                        basis_kind=basis_kind, basis_indices=basis_indices,
                        frame_owner=frame_owner)
                    error_frames.extend(frames)
                    branch_cotangents, branch_errors = _branch_pullbacks(
                        native, branch_frame, branch_covectors, backend=backend,
                        owner=frame_owner, basis_kind=basis_kind, basis_indices=basis_indices)
                    error_frames.extend(branch_errors)
                    # Pullback returns (candidate, omega, seeds); selecting
                    # its candidate tree keeps the leading output-row bank.
                    branch_candidate = (None if branch_cotangents is None
                                        else branch_cotangents[0])
                    if branch_candidate is None:
                        candidate_cotangents = total_response_candidate
                    elif total_response_candidate is None:
                        candidate_cotangents = branch_candidate
                    else:
                        candidate_cotangents = native.core.add_candidate_cotangents(
                            total_response_candidate, branch_candidate)
                    gradient_banks.append(_candidate_gradient(
                        native, candidate_frame, candidate_cotangents,
                        backend=backend, frame_owner=frame_owner, basis_kind=basis_kind,
                        basis_indices=basis_indices,
                    ))
                full_jacobian = jnp.concatenate(tuple(gradient_banks), axis=0)
                jacobian = full_jacobian[:, jnp.asarray(indices, dtype=jnp.int32)]
            raw = _raw_result(native, candidate_frame, branch_frame, flat, projection,
                              branch_frame.outcome.check_status,
                              branch_frame.outcome.certificate_status)
            return project_native_jacobian_outcome(native, (raw, jacobian),
                branch_error,
                error_frames=tuple(error_frames), compact_evidence=compact_evidence)

        if derivative_kind == 'vjp':
            weights = jnp.asarray(quantity_program.weight_vector(
                loads(prepared_analysis.weights_bytes)), dtype=jnp.float64)
            response_values, family_shapes, value_frames, checks, certificates = _value_responses(
                native, candidate, omega, backend=backend, owner=owner,
                policy=policy, initial_checks=initial_checks,
                initial_certificates=initial_certificates)
            error_frames.extend(value_frames)
            branch_error, branch_frame = _branch_frame(native, candidate, omega, seeds,
                checks, certificates, backend=backend, owner=owner,
                completed_branches=completed_branches,
                completed_branch_errors=completed_branch_errors)
            flat, projection = _project(native, response_values, branch_frame)
            response_covectors, branch_covectors, output_covectors, _, _ = _projection_pullback(
                native, response_values, branch_frame, backend=backend, weights=weights)
            total_response_candidate, frames, checks, certificates = _work_vjp(
                native, candidate_frame, omega, tuple(value[None, ...] for value in response_covectors),
                family_shapes, backend=backend, policy=policy,
                check_status=checks, certificate_status=certificates,
                basis_kind='functional', basis_indices=(0,), frame_owner=frame_owner)
            error_frames.extend(frames)
            branch_covectors = tuple(value[None, ...] for value in branch_covectors)
            branch_cotangents, branch_errors = _branch_pullbacks(
                native, branch_frame, branch_covectors, backend=backend,
                owner=frame_owner, basis_kind='functional', basis_indices=(0,))
            error_frames.extend(branch_errors)
            branch_candidate = (None if branch_cotangents is None
                                else branch_cotangents[0])
            if branch_candidate is None:
                candidate_cotangents = total_response_candidate
            elif total_response_candidate is None:
                candidate_cotangents = branch_candidate
            else:
                candidate_cotangents = native.core.add_candidate_cotangents(
                    total_response_candidate, branch_candidate)
            gradient_bank = _candidate_gradient(native, candidate_frame,
                candidate_cotangents,
                backend=backend, frame_owner=frame_owner, basis_kind='functional', basis_indices=(0,))
            gradient = gradient_bank[0]
            raw = _raw_result(native, candidate_frame, branch_frame, flat, projection,
                              branch_frame.outcome.check_status,
                              branch_frame.outcome.certificate_status)
            return project_native_derivative_outcome(native, (raw, gradient), branch_error,
                output_covectors=output_covectors, error_frames=tuple(error_frames), compact_evidence=compact_evidence)

        raise TypeError(f'unsupported Sensitivity derivative {derivative_kind!r}')


def objective_value_and_gradient(native, arguments, loss, *, active_keys, backend,
                                 coefficients, program_owner, execution_policy,
                                 frame_owner, compact_evidence=False, completed_branches=None, completed_branch_errors=None):
    """Differentiate the exact shared Objective loss through bounded chunks."""
    import jax
    import jax.numpy as jnp
    from ..numerics.evidence import project_native_derivative_outcome

    theta, omega, seeds = arguments
    owner = program_owner
    borrowed = nullcontext() if owner is None else owner.hold_exact_core(native.core)
    with borrowed, jax.enable_x64(True), jax.default_device(backend.device):
        candidate_frame = _candidate_frame(native, theta, backend=backend, owner=owner)
        candidate = candidate_frame.candidate
        from ..numerics.jax_core import first
        initial_checks = first(candidate.physical_checks.status,
            candidate.partition_status != 0, candidate.partition_status)
        response_values, family_shapes, error_frames, checks, certificates = _value_responses(
            native, candidate, omega, backend=backend, owner=owner,
            policy=execution_policy, initial_checks=initial_checks,
            initial_certificates=jnp.int32(0))
        branch_error, branch_frame = _branch_frame(native, candidate, omega, seeds,
            checks, certificates, backend=backend, owner=owner,
                completed_branches=completed_branches,
                completed_branch_errors=completed_branch_errors)
        flat, projection = _project(native, response_values, branch_frame)
        payload, output_covectors = _objective_loss_payload_and_covectors(
            native, loss, projection.output_values, coefficients,
            backend=backend, owner=program_owner,
        )
        output_covectors = native.output_covector_rows(output_covectors)
        response_ct, branch_ct, _, _, _ = _projection_pullback(
            native, response_values, branch_frame, backend=backend,
            weights=output_covectors,
        )
        total_candidate, derivative_frames, checks, certificates = _work_vjp(
            native, candidate_frame, omega,
            tuple(value[None, ...] for value in response_ct), family_shapes,
            backend=backend, policy=execution_policy,
            check_status=checks, certificate_status=certificates,
            basis_kind='functional', basis_indices=(0,), frame_owner=frame_owner)
        error_frames.extend(derivative_frames)
        branch_ct = tuple(value[None, ...] for value in branch_ct)
        branch_candidate, branch_errors = _branch_pullbacks(
            native, branch_frame, branch_ct, backend=backend, owner=frame_owner,
            basis_kind='functional', basis_indices=(0,))
        error_frames.extend(branch_errors)
        branch_candidate = None if branch_candidate is None else branch_candidate[0]
        if branch_candidate is not None:
            total_candidate = (branch_candidate if total_candidate is None else
                native.core.add_candidate_cotangents(total_candidate, branch_candidate))
        gradient_bank = _candidate_gradient(native, candidate_frame,
            total_candidate,
            backend=backend, frame_owner=frame_owner, basis_kind='functional', basis_indices=(0,))
        gradient = gradient_bank[0]
        raw_aux = _raw_result(native, candidate_frame, branch_frame, flat, projection,
                              branch_frame.outcome.check_status,
                              branch_frame.outcome.certificate_status)
        derivative = project_native_derivative_outcome(
            native, (raw_aux, gradient), branch_error,
            output_covectors=output_covectors, error_frames=tuple(error_frames),
            compact_evidence=compact_evidence,
        )
        return derivative, payload
