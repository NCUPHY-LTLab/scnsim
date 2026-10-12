"""Encode numerical diagnostics without acquiring persistence or runtime ownership."""
from __future__ import annotations
import math
import numpy as np
from ..numeric_encoding import array_record, record_bytes, record_document
from .. import errors
from .models import NumericalFailure

def evidence_bytes(value: object) -> bytes:
    """Represent numerical diagnostics, including unresolved nonfinite values."""
    def encode(item: object) -> object:
        if isinstance(item, dict):
            return {str(k): encode(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [encode(v) for v in item]
        if isinstance(item, np.ndarray):
            return array_record(item)
        if isinstance(item, np.generic):
            return encode(item.item())
        if isinstance(item, complex):
            return {"real": encode(item.real), "imag": encode(item.imag)}
        if isinstance(item, float):
            if not math.isfinite(item):
                return "nan" if math.isnan(item) else ("+inf" if item > 0 else "-inf")
            return item
        return item
    return record_bytes(encode(value))



def numerical_error(failure: NumericalFailure):
    """Restore the existing public failure type from a numerical handoff."""
    classes = (errors.DirectResponseFormationError, errors.PortRealizabilityError,
               errors.EliminatedBlockSolveFailure, errors.RootSlopeUnresolved,
               errors.NumericalResolutionUnresolved, errors.InvalidCandidatePhysicalParameter,
               errors.CompilerInvariantError, errors.UnsupportedSingularCapacitanceForDiagonalRootV1)
    error_type = {cls.kind: cls for cls in classes}[failure.kind]
    return error_type(failure.detail, stage=failure.stage, evidence=record_document(failure.evidence_bytes))


def require_primal(outcome):
    """Restore the established numerical failure for one checked program value."""
    from .jax_core import FAILURES
    primal = getattr(outcome, 'primal', outcome)
    aux = primal.aux
    status = int(aux.status)
    if status:
        kind, stage = FAILURES[status]
        for _, result in (*aux.branch_results, *aux.quantity_results):
            if status >= 11 and result.failure is not None and result.failure.kind == kind:
                raise numerical_error(result.failure)
        from dataclasses import fields, is_dataclass
        def certificate_record(certificate):
            if is_dataclass(certificate):
                return {field.name: getattr(certificate, field.name) for field in fields(certificate)}
            return dict(zip(certificate._fields, certificate))
        failure = NumericalFailure(kind, stage,
                                   'numerical operation did not satisfy its existing solve/root contract',
                                   evidence_bytes({'status': status, 'failed_site': int(aux.failed_site),
                                                   'certificates': tuple(certificate_record(c) for c in aux.certificates),
                                                   'realized_capabilities': aux.realized_capabilities}))
        raise numerical_error(failure)
    return primal


def native_primal_exception(error):
    """Restore only an owned archived check's original host exception type.

    Scalar consumers retain their existing SparseEquationCheck classification
    and throw all other errors through the original Error. A BatchedError stays
    batched: this projection does not authorize scalar callers to accept lanes.
    The Error, its archived certificate payloads and namespace are untouched.
    """
    from .native_sparse import ArchivedPrimalCheck
    exception = error.get_exception()
    while isinstance(exception, ArchivedPrimalCheck):
        exception = exception.original_tree.unflatten(exception.payload[:-2])
    return exception


def require_native_candidate(candidate):
    """Restore original physical/domain failures from synchronized candidate aux.

    This is a host projection of the traced authority, not another field/model
    evaluation. A changed finite route partition cannot silently use its old
    Port realization. Numerical selected-dtype failures remain equation-owned.
    """
    status = int(candidate.physical_checks.status)
    if status:
        stage = 'affine_support' if status == 1 else 'physical_validation'
        raise errors.InvalidCandidatePhysicalParameter(
            'affine input is outside declared support' if status == 1 else
            'physical parameter validation failed', stage=stage,
            evidence={'status': status, 'failed_field': int(candidate.physical_checks.failed_field)})
    if int(candidate.partition_status):
        raise errors.PortRealizabilityError(
            'actual point does not belong to the prepared physical route partition',
            stage='selected_network', evidence={'realized_capabilities':
                tuple(bool(value) for value in candidate.realized_capabilities)})
    return candidate


def require_native_frequency(*, check_status, certificate_status,
                             check_position=-1, certificate_position=-1,
                             certificates=(), capabilities=()):
    """Project the completed TWO ordered reducers to the existing typed error.

    The operation must fold all actually evaluated check occurrences before
    selecting certificate failure. Raising from an earlier chunk's certificate
    before later check classification would change the established precedence.
    Certificate payloads are the actual performed equations, never relabelled
    derivative certificates as primal evidence.
    """
    from .jax_core import FAILURES
    from dataclasses import fields, is_dataclass
    def certificate_record(certificate):
        if hasattr(certificate, '_fields'):
            return dict(zip(certificate._fields, certificate))
        if is_dataclass(certificate):
            return {field.name: getattr(certificate, field.name) for field in fields(certificate)}
        return certificate
    checks, equations = int(check_status), int(certificate_status)
    status = checks if checks else equations
    if status:
        kind, stage = FAILURES[status]
        failure = NumericalFailure(kind, stage,
            'numerical operation did not satisfy its existing solve contract',
            evidence_bytes({'status': status, 'check_status': checks,
                'certificate_status': equations, 'check_position': int(check_position),
                'certificate_position': int(certificate_position),
                'certificates': tuple(certificate_record(c) for c in certificates),
                'realized_capabilities': tuple(bool(value) for value in capabilities)}))
        raise numerical_error(failure)


def project_native_root(declaration, packet):
    """Detach one actual root packet for continuation and typed publication.

    This projection neither solves nor differentiates. Failure retains the
    packet's original numerical status/stage and actual closure/equation facts;
    successful roots keep the already selected mode vector where applicable.
    """
    from .jax_core import FAILURES
    from .native_root import ROOT_STAGES
    from .models import EvaluationResult
    def detached(value):
        if hasattr(value, '_fields'):
            return {key: detached(item) for key, item in zip(value._fields, value)}
        if isinstance(value, tuple):
            return tuple(detached(item) for item in value)
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            return np.asarray(value)
        return value
    facts = detached(packet)
    facts['derivative_equation'] = ({
        'Y': 'reduced_Y_entry_local_implicit',
        'Z': 'selected_admittance_cofactor_local_implicit',
        'S': 'scattering_bordered_determinant_local_implicit',
    }[declaration.family] if declaration.kind == 'transfer_zero' else
        'selected_determinant_local_implicit' if declaration.kind == 'hybridized_pole' else
        'selected_operator_entry_local_implicit')
    facts['derivative_limit'] = ('physical local implicit derivative at a machine-closed root; '
        'not finite-Newton differentiation or an exact-zero/forward-accuracy guarantee')
    status = int(packet.status)
    if status:
        kind, stage = FAILURES[status]
        root_stage = int(packet.failure_stage)
        # Only root-owned failures use the root closure-stage vocabulary.
        # Assembly/solve failures retain their governing numerical stage.
        if declaration.kind == 'transfer_zero' and int(packet.source_stage):
            facts['source_stage'] = FAILURES[int(packet.source_stage)][1]
            stage = 'transfer_denominator'
        elif status in (11, 12, 13, 14):
            if declaration.kind == 'transfer_zero':
                from .native_transfer import TRANSFER_STAGES
                stage = TRANSFER_STAGES[root_stage]
            else:
                stage = ROOT_STAGES[root_stage]
        failure = NumericalFailure(kind, stage,
            'root did not satisfy its existing local Newton/closure contract', evidence_bytes(facts))
        return EvaluationResult(declaration.quantity_key, failure=failure)
    body = dict(root_omega_rad_s=np.asarray(packet.omega).item(),
                root_slope=np.asarray(packet.slope).item(), evidence_bytes=evidence_bytes(facts))
    if declaration.kind == 'transfer_zero':
        body['numerator_slope'] = np.asarray(packet.numerator_slope).item()
        body['denominator'] = np.asarray(packet.denominator).item()
    if declaration.kind == 'hybridized_pole':
        body['null_vector'] = np.asarray(packet.right_vector)
    return EvaluationResult(declaration.quantity_key, **body)


def _detach_native(value):
    """Copy synchronized array facts; retain no device frame or callback owner."""
    from dataclasses import fields, is_dataclass, replace
    if is_dataclass(value):
        return replace(value, **{field.name: _detach_native(getattr(value, field.name))
                                 for field in fields(value)})
    if hasattr(value, '_fields'):
        return type(value)(*(_detach_native(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_detach_native(item) for item in value)
    if isinstance(value, dict):
        return {key: _detach_native(item) for key, item in value.items()}
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        array = np.asarray(value)
        return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)
    return value


def project_native_coupling(quantity_key, packet):
    """Publish the actual finished residue packet without locating its roots."""
    from .jax_core import FAILURES
    from .native_coupling import COUPLING_STAGES
    from .models import EvaluationResult
    detached = _detach_native(packet)
    facts = dict(zip(detached._fields, detached))
    facts['derivative_equation'] = 'selected_residue_slope_with_checked_frequency_jet_and_local_null_gauge'
    status = int(detached.status)
    if status:
        kind, stage = FAILURES[status]
        if status == 13:
            stage = COUPLING_STAGES[int(detached.failure_stage)]
        elif status == 12:
            stage = 'newton_certificate'
        return EvaluationResult(quantity_key, failure=NumericalFailure(kind, stage,
            'residue coupling did not satisfy its existing slope/rank/reciprocity contract',
            evidence_bytes(facts)))
    scalar = lambda value: np.asarray(value).item()
    return EvaluationResult(quantity_key,
        coupling_rad_s=scalar(detached.coupling_rad_s),
        residue_a=scalar(detached.residue_a), residue_b=scalar(detached.residue_b),
        branch_roots_rad_s=detached.branch_roots_rad_s,
        evaluation_omega_rad_s=scalar(detached.evaluation_omega_rad_s),
        evidence_bytes=evidence_bytes(facts))



from typing import NamedTuple, Any


class MappedNativeErrorFrame(NamedTuple):
    error: Any
    work_index: int | None
    frequency_source_index: int | None
    basis_kind: str | None = None
    basis_indices: tuple = ()


def response_ad_error_frames(outcome, *, work_indices, frequency_source_indices,
                             basis_kind, basis_indices):
    """Detach ordered [B] and [B,D] Error payloads with BOTH axis locators.

    The JAX transforms have already executed and certified their equations.
    This host projection slices only compact Error data, never numerical state.
    Runtime work/frequency labels are evidence, not primitive static keys.
    """
    import jax
    work_indices, sources, basis = tuple(work_indices), tuple(frequency_source_indices), tuple(basis_indices)
    if len(work_indices) != len(sources):
        raise ValueError('response Error work/frequency locator lengths differ')
    if any(value is None for value in (*work_indices, *sources)):
        raise ValueError('response Errors require their actual work/frequency locators')
    if basis_kind not in ('parameter', 'output_row', 'direction', 'functional'):
        raise ValueError('response derivative Errors require their actual basis kind')
    # JVP/VJP have one declared direction/functional; Jacobians retain exact
    # parameter-coordinate or authoritative output-row ordinals.
    certificate_basis = basis_kind
    def lane(error, index, directions):
        for code in error._code.values():
            if np.ndim(code) == 0 or np.shape(code)[0] != len(work_indices):
                raise ValueError('response Error leading axis differs from its work locators')
            if directions and (np.ndim(code) < 2 or np.shape(code)[1] != len(basis)):
                raise ValueError('response derivative Error direction axis differs from its basis')
        return jax.tree.map(lambda value: np.asarray(value)[index], error)
    frames = []
    for index, (work, source) in enumerate(zip(work_indices, sources, strict=True)):
        frames.append(MappedNativeErrorFrame(lane(outcome.primal_errors, index, False), work, source))
        frames.append(MappedNativeErrorFrame(lane(outcome.derivative_errors, index, True), work, source,
                                            certificate_basis, basis))
    return tuple(frames)


def _native_error_certificate_frames(error, error_frames, *, derivatives):
    from .native_sparse import sparse_certificates
    rows = [(_detach_native(sparse_certificates(error)), error)]
    for frame in error_frames:
        if not derivatives and frame.basis_kind is not None:
            continue
        rows.append((_detach_native(sparse_certificates(frame.error,
            basis_kind=frame.basis_kind, basis_indices=frame.basis_indices,
            work_index=frame.work_index, frequency_source_index=frame.frequency_source_index)), frame.error))
    return tuple(rows)


def _raise_native_unexpected(error, *, derivative_frame):
    from .native_sparse import SparseCertificateEffect, DerivativeAvailabilityEffect
    projected_effects = (SparseCertificateEffect, DerivativeAvailabilityEffect) if derivative_frame else (SparseCertificateEffect,)
    fatal = error._replace(_pred={effect: np.zeros_like(predicate, dtype=bool)
        if isinstance(effect, projected_effects) else predicate
        for effect, predicate in error._pred.items()})
    if fatal.get_exception() is not None:
        fatal.throw()

def project_native_outcome(native, result, checked_error, *, _derivative_frame=False, error_frames=()):
    """Detach a SAME-frame native value into the existing typed handoff.

    This host boundary performs no physical evaluation, projection, solve or
    differentiation. Numerical check/certificate reducers preserve their
    separate precedence. Transformed certificate roles remain explicit; they
    are not relabelled as primal certificates by this value projection.
    """
    from .native_sparse import sparse_certificates
    from .program_models import ProgramOutcome, ProgramAux
    _, aux = result
    require_native_candidate(aux)
    certificate_frames = _native_error_certificate_frames(checked_error, error_frames, derivatives=False)
    certificates = tuple(certificate for rows, _ in certificate_frames for certificate in rows)
    for _, error in certificate_frames:
        _raise_native_unexpected(error, derivative_frame=_derivative_frame)
    branches = tuple((index, project_native_root(
        next(row['declaration'] for row in native.root_layout if row['start_index'] == index), packet))
        for index, packet in aux.branch_packets)
    quantities = tuple((index, project_native_coupling(
        native.occurrence_layout[index]['quantity_key'], packet))
        for index, packet in aux.quantity_packets)
    available = tuple(bool(value) for value in aux.derivative_available)
    reasons = tuple(() if value else ('declared output derivative is unavailable at the actual point',)
                    for value in available)
    checks, equations = int(aux.check_status), int(aux.certificate_status)
    projected = ProgramAux(np.int32(checks if checks else equations), np.int32(-1),
        certificates, np.bool_(all(available)), tuple(reason for row in reasons for reason in row),
        tuple(bool(value) for value in aux.realized_capabilities), available, reasons,
        branches, quantities, _detach_native(aux.observable_values),
        _detach_native(aux.observable_raw_values))
    return ProgramOutcome(_detach_native(aux.output_values), projected)


def project_native_derivative_outcome(native, result, checked_error, *, output_covectors,
                                      _directional_errors=None, _basis_kind=None, _basis_indices=(), error_frames=()):
    """Project one checked transformed frame, retaining actual equation roles.

    ``result`` is ``(native_primal_result, derivative)``. Covectors use the
    authoritative flattened real rows; None means every requested output.
    Zero-weight unavailable leaves do not invalidate an aggregate functional.
    This boundary performs no differentiation, solve or physical projection.
    """
    from .program_models import DerivativeOutcome
    from .native_sparse import (derivative_availability, sparse_certificates,
        SparseCertificateEffect, DerivativeAvailabilityEffect)
    primal_result, derivative = result
    primal = project_native_outcome(native, primal_result, checked_error,
                                   _derivative_frame=True, error_frames=error_frames)
    require_primal(primal)
    blocks = native.output_layout['blocks']
    if output_covectors is None:
        active = tuple(True for _ in blocks)
    else:
        weights = np.asarray(output_covectors).reshape(-1)
        row_count = sum(math.prod(block['shape']) * len(block['components']) for block in blocks)
        if weights.size != row_count:
            raise ValueError('output covector shape differs from authoritative real output rows')
        active = tuple(bool(np.any(weights[block['offset']:block['offset'] +
            math.prod(block['shape']) * len(block['components'])] != 0)) for block in blocks)
    available = list(primal.aux.observable_available)
    reasons = [list(row) for row in primal.aux.observable_reasons]
    directional_frames = ((checked_error, None, ()),)
    if _directional_errors is not None:
        basis = tuple(_basis_indices)
        if _basis_kind not in ('parameter', 'output_row'):
            raise ValueError('mapped derivative errors require their actual basis kind')
        # All mapped payloads have the exact same leading direction axis. A
        # direction Error is data returned by the already checked JAX transform.
        for code in _directional_errors._code.values():
            if np.ndim(code) == 0 or np.shape(code)[0] != len(basis):
                raise ValueError('mapped derivative Error axis disagrees with its ordered basis')
        fatal = _directional_errors._replace(_pred={effect: np.zeros_like(predicate, dtype=bool)
            if isinstance(effect, (SparseCertificateEffect, DerivativeAvailabilityEffect)) else predicate
            for effect, predicate in _directional_errors._pred.items()})
        if fatal.get_exception() is not None:
            fatal.throw()
        directional_frames += ((_directional_errors, _basis_kind, basis),)
    elif _basis_kind is not None or _basis_indices:
        raise ValueError('derivative basis metadata requires mapped errors')

    for frame in error_frames:
        if frame.basis_kind is not None:
            _raise_native_unexpected(frame.error, derivative_frame=True)
        directional_frames += ((frame.error, frame.basis_kind, frame.basis_indices),)

    def affected_basis(mask, kind, basis):
        if kind != 'output_row':
            return None
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 0 or mask.shape[0] != len(basis):
            raise ValueError('mapped derivative availability axis disagrees with its ordered basis')
        lanes = mask.reshape((len(basis), -1)).any(axis=1) if len(basis) else np.empty(0, dtype=bool)
        rows = tuple(row for row, failed in zip(basis, lanes, strict=True) if failed)
        return frozenset(index for index, block in enumerate(blocks)
            if any(block['offset'] <= row < block['offset'] +
                math.prod(block['shape']) * len(block['components']) for row in rows))

    for frame, kind, basis in directional_frames:
        for fact in derivative_availability(frame):
            mask = np.asarray(fact.present) & ~np.asarray(fact.available)
            if bool(np.any(mask)):
                selected_blocks = affected_basis(mask, kind, basis)
                for index, dependencies in enumerate(native.output_dependencies):
                    if fact.dependency in dependencies and (selected_blocks is None or index in selected_blocks):
                        available[index] = False
                        reasons[index].append(fact.reason)
    transformed_certificates = primal.aux.certificates
    for frame in error_frames:
        if frame.basis_kind is not None:
            transformed_certificates += _detach_native(sparse_certificates(frame.error,
                basis_kind=frame.basis_kind, basis_indices=frame.basis_indices,
                work_index=frame.work_index, frequency_source_index=frame.frequency_source_index))
    if _directional_errors is not None:
        transformed_certificates += _detach_native(sparse_certificates(_directional_errors,
            basis_kind=_basis_kind, basis_indices=tuple(_basis_indices)))
    primal = primal._replace(aux=primal.aux._replace(certificates=tuple(
        certificate for certificate in primal.aux.certificates
        if certificate.role not in ('tangent', 'mixed', 'adjoint'))))
    failures = tuple(certificate for certificate in transformed_certificates
        if certificate.role in ('tangent', 'mixed', 'adjoint') and bool(np.any(
            np.asarray(certificate.present) & np.asarray(certificate.performed) &
            (np.asarray(certificate.status) != 0))))
    for certificate in failures:
        dependency = certificate.dependency
        if not dependency:
            affected = tuple(range(len(available)))
        elif dependency[0] == 'responses':
            affected = tuple(index for index, dependencies in enumerate(native.output_dependencies)
                if any(('response', dependency[1], family) in dependencies for family in dependency[2]))
        else:
            affected = tuple(index for index, dependencies in enumerate(native.output_dependencies)
                             if dependency in dependencies)
        selected_blocks = affected_basis(np.asarray(certificate.present) & np.asarray(certificate.performed) &
            (np.asarray(certificate.status) != 0), certificate.basis_kind, certificate.basis_indices)
        for index in affected:
            if selected_blocks is None or index in selected_blocks:
                available[index] = False
                reasons[index].append('checked derivative equation failed at ' + repr(certificate.site))
    complete = all(ok or not selected for ok, selected in zip(available, active, strict=True))
    aux = primal.aux._replace(certificates=transformed_certificates,
        derivative_available=np.bool_(complete),
        derivative_reasons=tuple(reason for index, row in enumerate(reasons)
            if active[index] for reason in row),
        observable_available=tuple(available), observable_reasons=tuple(tuple(row) for row in reasons))
    return DerivativeOutcome(primal, _detach_native(derivative) if complete else None, aux)


def project_native_jvp_outcome(native, result, checked_error, *, error_frames=()):
    """Preserve each independent available physical output directional block."""
    from .program_models import DerivativeOutcome
    aggregate = project_native_derivative_outcome(native, result, checked_error,
                                                 output_covectors=None, error_frames=error_frames)
    flat = np.asarray(result[1])
    blocks = native.output_layout['blocks']
    values = []
    for block, available in zip(blocks, aggregate.aux.observable_available, strict=True):
        count = math.prod(block['shape']) * len(block['components'])
        rows = flat[block['offset']:block['offset'] + count]
        if not available:
            values.append(None)
        elif len(block['components']) == 2:
            parts = rows.reshape((*block['shape'], 2))
            values.append(_detach_native(parts[..., 0] + 1j * parts[..., 1]))
        else:
            values.append(_detach_native(rows.reshape(block['shape'])))
    return DerivativeOutcome(aggregate.primal, tuple(values), aggregate.aux)


def project_native_jacobian_outcome(native, result, checked_error, *, output_indices=None,
                                    derivative_errors=None, basis_kind=None, basis_indices=(), error_frames=()):
    """Return only available authoritative rows, retaining requested row order."""
    from .program_models import DerivativeOutcome, JacobianPayload
    matrix = np.asarray(result[1])
    blocks = native.output_layout['blocks']
    expected = sum(math.prod(block['shape']) * len(block['components']) for block in blocks)
    if matrix.ndim != 2 or matrix.shape[0] != expected:
        raise ValueError('Jacobian shape differs from authoritative output rows')
    selected = tuple(range(expected)) if output_indices is None else tuple(output_indices)
    if any(row < 0 or row >= expected for row in selected):
        raise IndexError('requested Jacobian row differs from authoritative output layout')
    weights = None
    if output_indices is not None:
        weights = np.zeros(expected, dtype=np.bool_)
        weights[np.asarray(selected, dtype=np.intp)] = True
    aggregate = project_native_derivative_outcome(native, result, checked_error,
                                                 output_covectors=weights, _directional_errors=derivative_errors,
        _basis_kind=basis_kind, _basis_indices=basis_indices, error_frames=error_frames)
    intervals = tuple((block['offset'], block['offset'] +
        math.prod(block['shape']) * len(block['components']))
        for block, available in zip(blocks, aggregate.aux.observable_available, strict=True)
        if available)
    rows = tuple(row for row in selected if any(start <= row < stop for start, stop in intervals))
    payload = JacobianPayload(_detach_native(matrix[np.asarray(rows, dtype=np.intp)]), rows, matrix.shape)
    return DerivativeOutcome(aggregate.primal, payload, aggregate.aux)
