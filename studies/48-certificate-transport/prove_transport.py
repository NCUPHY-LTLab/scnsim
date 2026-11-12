"""Proof on small cases: the candidate transport (certificate_transport.py, public
JAX interfaces only) returns the same certificates as today's checked_native /
sparse_certificates for the same call.

Usage (from the SCNSim worktree, with this directory on PYTHONPATH):
    uv run --locked python prove_transport.py

The harness may read private JAX state to obtain the CURRENT mechanism's
registrations and results. The candidate module itself does not.
Timings are diagnostics only; they set no threshold.
"""
import json, sys, time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import checkify

from scnsim.compilation.program import SparsePattern
from scnsim.numerics import native_sparse as ns
from scnsim.numerics import native_pair  # noqa: F401  registers the pair primitives
from scnsim.numerics import native_equation_jet as jet
from scnsim.numerics.sparse_solve import SparseEquation

import certificate_transport as ct

# -- harness only: copy today's primitive registrations into the candidate ---
from jax._src import checkify as _private_checkify
for _primitive in ns._certificate_primitives:
    _keywords = _private_checkify.error_checks[_primitive].keywords
    ct.register_equation(_primitive, value_count=_keywords['value_count'],
                         certificate_groups=_keywords['certificate_groups'])

RESULTS = []
ARRAYS = ('status', 'eta', 'residual_numerator', 'residual_denominator', 'performed', 'present')
STATIC = ('site', 'role', 'stage', 'derivative_order', 'dependency', 'basis_kind', 'basis_indices',
          'work_index', 'frequency_source_index')
OWNED = (ns.SparseCertificateEffect, ns.DerivativeAvailabilityEffect)


class without_private_registrations:
    """Harness only. The product registers its primitives in JAX's private
    checkify table at import. The redesigned product would not, so the
    candidate is traced with those entries absent."""

    def __enter__(self):
        self.saved = {primitive: _private_checkify.error_checks.pop(primitive)
                      for primitive in ns._certificate_primitives}

    def __exit__(self, *exc):
        _private_checkify.error_checks.update(self.saved)


def bits(value):
    value = np.asarray(value)
    return value.dtype.str, value.shape, value.tobytes()


def same_tree(a, b):
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    return ta == tb and all(bits(x) == bits(y) for x, y in zip(la, lb))


def compare_rows(current, candidate):
    problems = []
    if len(current) != len(candidate):
        return [f'row count {len(current)} != {len(candidate)}']
    for index, (old, new) in enumerate(zip(current, candidate)):
        for name in STATIC:
            if getattr(old, name) != getattr(new, name):
                problems.append(f'row {index} {name}: {getattr(old, name)!r} != {getattr(new, name)!r}')
        for name in ARRAYS:
            if bits(getattr(old, name)) != bits(getattr(new, name)):
                problems.append(f'row {index} {name}: {np.asarray(getattr(old, name))!r} != '
                                f'{np.asarray(getattr(new, name))!r}')
    return problems


def owned_failed(error):
    return bool(any(np.any(np.asarray(pred)) for effect, pred in error._pred.items()
                    if isinstance(effect, OWNED)))


def foreign_message(error):
    """Message of a failure SCNSim does not own, as today's routing would throw it."""
    kept = error._replace(_pred={effect: np.zeros_like(np.asarray(pred), dtype=bool)
                                 if isinstance(effect, OWNED) else pred
                                 for effect, pred in error._pred.items()})
    return kept.get()


def timed(function, args, repeats):
    start = time.perf_counter()
    result = jax.block_until_ready(function(*args))
    first = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(function(*args))
        samples.append(time.perf_counter() - start)
    return result, first, float(np.median(samples)) if samples else float('nan')


def case(name, function, args, *, wrap=lambda f: f, expect=None, x64=True,
         candidate_function=None, repeats=0):
    """Run both mechanisms on the same call and compare everything observable."""
    with jax.enable_x64(x64):
        before = bool(jax.config.jax_enable_x64)
        current = jax.jit(wrap(ns.checked_native(function)))
        candidate = jax.jit(wrap(ct.certified(candidate_function or function)))
        (error, old_values), old_first, old_run = timed(current, args, repeats)
        with without_private_registrations():
            (evidence, new_values), new_first, new_run = timed(candidate, args, repeats)
        after = bool(jax.config.jax_enable_x64)
    old_rows = ns.sparse_certificates(error)
    new_rows = ct.ledger_certificates(evidence.ledger)
    problems = compare_rows(old_rows, new_rows)
    if not same_tree(old_values, new_values):
        problems.append('values differ')
    old_failed, new_failed = owned_failed(error), bool(np.any(np.asarray(evidence.ledger.any_failed())))
    if old_failed != new_failed:
        problems.append(f'owned failure {old_failed} != {new_failed}')
    old_foreign, new_foreign = foreign_message(error), evidence.foreign.get()
    if old_foreign != new_foreign:
        problems.append(f'foreign failure {old_foreign!r} != {new_foreign!r}')
    old_facts, new_facts = ns.derivative_availability(error), ct.ledger_availability(evidence.ledger)
    if len(old_facts) != len(new_facts):
        problems.append(f'availability count {len(old_facts)} != {len(new_facts)}')
    for old, new in zip(old_facts, new_facts):
        for field in ('site', 'dependency', 'reason'):
            if getattr(old, field) != getattr(new, field):
                problems.append(f'availability {field}: {getattr(old, field)!r} != {getattr(new, field)!r}')
        for field in ('available', 'present'):
            if bits(getattr(old, field)) != bits(getattr(new, field)):
                problems.append(f'availability {field}: {np.asarray(getattr(old, field))} != '
                                f'{np.asarray(getattr(new, field))}')
    if before != after:
        problems.append('global x64 flag changed')
    summary = dict(case=name, rows=len(old_rows), availability=len(old_facts), owned_failure=old_failed,
                   foreign=old_foreign,
                   roles=sorted({row.role for row in old_rows}),
                   sites=[repr(row.site) for row in old_rows],
                   statuses=sorted({int(v) for row in old_rows for v in np.asarray(row.status).reshape(-1)}),
                   not_present=int(sum(int(np.sum(~np.asarray(row.present))) for row in old_rows)),
                   not_performed=int(sum(int(np.sum(np.asarray(row.present) & ~np.asarray(row.performed)))
                                         for row in old_rows)),
                   unavailable=int(sum(int(np.sum(np.asarray(f.present) & ~np.asarray(f.available)))
                                       for f in old_facts)),
                   payload_dtypes=sorted({np.asarray(row.eta).dtype.name for row in old_rows}),
                   max_eta=float(max((np.nanmax(np.asarray(row.eta, dtype=np.float64), initial=0.0)
                                      for row in old_rows), default=0.0)),
                   first_call_s=(round(old_first, 3), round(new_first, 3)),
                   run_ms=(round(1e3 * old_run, 4), round(1e3 * new_run, 4)),
                   identical=not problems, problems=problems[:6])
    for key, value in (expect or {}).items():
        if summary[key] != value:
            summary['identical'] = False
            summary['problems'].append(f'expected {key}={value!r}, current mechanism gave {summary[key]!r}')
    RESULTS.append(summary)
    flag = 'identical' if summary['identical'] else 'DIFFERENT'
    print(f"{flag:9} {name:50} rows={summary['rows']:2d} roles={','.join(summary['roles'])} "
          f"status={summary['statuses']} absent={summary['not_present']} skipped={summary['not_performed']} "
          f"avail={summary['availability']}/{summary['unavailable']} failed={old_failed} "
          f"{'/'.join(summary['payload_dtypes'])} first={old_first:.2f}/{new_first:.2f}s"
          + (f" run={1e3 * old_run:.3f}/{1e3 * new_run:.3f}ms" if repeats else ''))
    for problem in summary['problems']:
        print('          ', problem)


# ---------------------------------------------------------------------------
# A small circuit-like complex system: 4 nodes, banded sparse pattern.
# ---------------------------------------------------------------------------
N = 4
_mask = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) != 2
PATTERN, _ = SparsePattern.from_coo((N, N), *np.nonzero(_mask))
EQUATION = SparseEquation(PATTERN)
ROWS, COLS = np.asarray(PATTERN.rows), np.asarray(PATTERN.cols)
rng = np.random.default_rng(7)
MATRIX = np.where(_mask, rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)), 0) + 4 * np.eye(N)
VALUES = MATRIX[ROWS, COLS]
RHS = rng.normal(size=N) + 1j * rng.normal(size=N)
DIRECTION = rng.normal(size=VALUES.shape) + 0j
DENSE_PATTERN, _ = SparsePattern.from_coo((N, N), *np.nonzero(np.ones((N, N))))
DENSE_EQUATION = SparseEquation(DENSE_PATTERN)


def solve(values, rhs, **options):
    return ns.checked_sparse_solve(EQUATION, values, rhs, **options)


def direct(values, rhs):
    outcome = solve(values, rhs)
    return outcome.value, outcome.aux


def loss(values, rhs):
    outcome = solve(values, rhs)
    return jnp.sum(jnp.abs(outcome.value) ** 2), outcome.aux.status


def real_grad(function, **options):
    """Gradient with respect to the real part of the complex coefficients."""
    return lambda values, *rest: jax.value_and_grad(
        lambda real: function(real + 1j * jnp.imag(values), *rest), **options)(jnp.real(values))


def two_stage(values, rhs):
    first = solve(values, rhs)
    second = solve(values, first.value, prior_status=first.aux.status, stage=7, dependency=('second',))
    return jnp.sum(jnp.abs(second.value) ** 2), (first.aux.status, second.aux.status)


def branchy(flag, values, rhs):
    return lax.cond(flag, lambda: solve(values, rhs).value,
                    lambda: solve(2 * values, rhs, stage=7, dependency=('other',)).value)


def branchy_loss(values, flag, rhs):
    return jnp.sum(jnp.abs(branchy(flag, values, rhs)) ** 2)


def refine(values, rhs, steps, poison_at):
    """A Newton-like loop: one solve per iteration, data-dependent trip count."""
    def body(state):
        index, x, status = state
        used = jnp.where(index == poison_at, jnp.zeros_like(values), values)
        outcome = solve(used, rhs - 0.5 * x, dependency=('loop',))
        return index + 1, outcome.value, jnp.where(status == 0, outcome.aux.status, status)
    start = (jnp.int32(0), jnp.zeros_like(rhs), jnp.int32(0))
    _, x, status = lax.while_loop(lambda state: state[0] < steps, body, start)
    return status, x


def scanned(values, rhs_rows):
    def step(carry, rhs):
        outcome = solve(values, rhs + carry, dependency=('scan',))
        return outcome.value, outcome.aux.status
    return lax.scan(step, jnp.zeros_like(rhs_rows[0]), rhs_rows)


def jet_case(values, rhs):
    result = jet.checked_sparse_frequency_jet(EQUATION, values, rhs, 0.1j * values, 0.2 * rhs,
                                              dependency=('jet',))
    return result.primal.value, result.frequency.value


def dense_loss(matrix, rhs):
    outcome = ns.checked_dense_solve(DENSE_EQUATION, matrix, rhs, stage=13, dependency=('dense',))
    return jnp.sum(jnp.abs(outcome.value) ** 2)


def with_foreign_check(values, rhs, bound):
    outcome = solve(values, rhs)
    checkify.check(jnp.real(outcome.value[0]) < bound, 'foreign invariant broke at {x}',
                   x=jnp.real(outcome.value[0]))
    return outcome.value


def implicit_root(candidate):
    """A custom-JVP function whose rule states a derivative-domain predicate and
    solves its own tangent equation, like SCNSim's root and null-vector rules."""
    dependency = ('root_occurrence', 0, 0)

    @jax.custom_jvp
    def root(values, rhs, threshold):
        return solve(values, rhs, dependency=dependency).value

    @root.defjvp
    def rule(primals, tangents):
        values, rhs, threshold = primals
        values_dot, rhs_dot, _ = tangents
        out = root(values, rhs, threshold)
        available = jnp.abs(out[0]) > threshold
        if candidate:
            out = ct.derivative_gate(out, available, dependency=dependency, reason='toy slope vanished')
        else:
            ns.derivative_available(available, dependency=dependency, reason='toy slope vanished')
        residual = rhs_dot - ns._matmul(EQUATION, values_dot, out, 'N')
        tangent = solve(values, residual, stage=13, role='tangent', derivative_order=1,
                        dependency=dependency)
        return out, tangent.value

    return lambda values, rhs, threshold, direction: jax.jvp(
        lambda x: root(x, rhs, threshold), (values,), (direction,))


def sequential(candidate, body=None):
    wrapper = ct.certified_sequential_primal if candidate else ns.checked_sequential_primal
    return wrapper(body or refine)


def refine_with_invariant(values, rhs, steps, poison_at):
    """The Newton-like loop with a foreign debug invariant inside its body,
    as native_transfer.py has (checkify.debug_check)."""
    def body(state):
        index, x, status = state
        outcome = solve(values, rhs - 0.5 * x, dependency=('loop',))
        checkify.debug_check(index != poison_at, 'foreign loop invariant broke at step {i}', i=index)
        return index + 1, outcome.value, jnp.where(status == 0, outcome.aux.status, status)
    start = (jnp.int32(0), jnp.zeros_like(rhs), jnp.int32(0))
    _, x, status = lax.while_loop(lambda state: state[0] < steps, body, start)
    return status, x


vmapped = lambda f: jax.vmap(f)


def cases64():
    V, B, dV = jnp.asarray(VALUES), jnp.asarray(RHS), jnp.asarray(DIRECTION)
    S = jnp.zeros_like(V)                # all-zero matrix: the host reports status = stage
    lanes_v = jnp.stack([V, S, 2 * V, V, 3 * V])
    lanes_b = jnp.stack([B, B, 2 * B, -B, 1j * B])
    flags = jnp.asarray([True, False, True, False, False])
    M = jnp.asarray(MATRIX)

    print('== 1. one Direct solve   (first=current/candidate first call incl. compile; run=median of repeats)')
    case('direct solve, success', direct, (V, B), expect=dict(statuses=[0], owned_failure=False), repeats=300)
    case('direct solve, singular matrix', direct, (S, B), expect=dict(statuses=[3], owned_failure=True))
    case('prior failure skips the second equation', two_stage, (S, B),
         expect=dict(not_performed=1, owned_failure=True))
    case('frequency jet (two certificate groups)', jet_case, (V, B))

    print('== 2. batched solve')
    case('vmap of certified, 5 lanes, lane 1 singular', direct, (lanes_v, lanes_b), wrap=vmapped,
         expect=dict(statuses=[0, 3], owned_failure=True), repeats=300)
    case('certified of vmap (batch primitive)', jax.vmap(direct), (lanes_v, lanes_b),
         expect=dict(statuses=[0, 3], owned_failure=True))
    case('vmap of certified, inner vmap over 2 right-hand sides',
         lambda v, b: jax.vmap(lambda one: direct(v, one))(b),
         (lanes_v, jnp.stack([lanes_b, 2 * lanes_b], axis=1)), wrap=vmapped)

    print('== 3. reverse mode')
    case('value_and_grad: primal + adjoint', real_grad(loss, has_aux=True), (V, B),
         expect=dict(roles=['adjoint', 'primal'], owned_failure=False), repeats=300)
    case('value_and_grad at a singular matrix', real_grad(loss, has_aux=True), (S, B),
         expect=dict(owned_failure=True, not_performed=1))
    case('grad through two chained solves', real_grad(two_stage, has_aux=True), (V, B))
    case('vmap of certified value_and_grad, lane 1 singular', real_grad(loss, has_aux=True),
         (lanes_v, lanes_b), wrap=vmapped, expect=dict(owned_failure=True))
    case('grad through cond (adjoint inside a branch)', real_grad(branchy_loss), (V, jnp.bool_(True), B))
    case('vmap of grad through cond, lanes differ', real_grad(branchy_loss), (lanes_v, flags, lanes_b),
         wrap=vmapped)
    case('dense backend under grad', real_grad(dense_loss), (M, B))

    print('== 4. forward mode')
    case('jvp: primal + tangent', lambda v, b, dv: jax.jvp(lambda x: loss(x, b)[0], (v,), (dv,)),
         (V, B, dV), expect=dict(roles=['primal', 'tangent']))
    case('jvp of jvp: mixed equation', lambda v, b, dv: jax.jvp(
        lambda y: jax.jvp(lambda x: solve(x, b).value, (y,), (dv,))[1], (v,), (dv,)), (V, B, dV))
    case('linearize, 3 directions mapped', lambda v, b, dvs: (lambda out, push: (out, lax.map(push, dvs)))(
        *jax.linearize(lambda x: solve(x, b).value, v)), (V, B, jnp.stack([dV, 2 * dV, 1j * dV])))
    case('forward over reverse (Hessian-vector)', lambda v, b, dv: jax.jvp(
        jax.grad(lambda real: loss(real + 1j * jnp.imag(v), b)[0]), (jnp.real(v),), (jnp.real(dv),)),
        (V, B, dV))

    print('== 5. control flow')
    case('cond, taken branch only', branchy, (jnp.bool_(True), V, B), expect=dict(not_present=1))
    case('cond, other branch', branchy, (jnp.bool_(False), V, B), expect=dict(not_present=1))
    case('vmap of certified cond, lanes differ', branchy, (flags, lanes_v, lanes_b), wrap=vmapped)
    case('while_loop, 3 iterations', refine, (V, B, jnp.int32(3), jnp.int32(-1)), repeats=300)
    case('while_loop, zero iterations (site not executed)', refine, (V, B, jnp.int32(0), jnp.int32(-1)),
         expect=dict(not_present=1))
    case('while_loop, failure in iteration 1 of 4 is kept', refine, (V, B, jnp.int32(4), jnp.int32(1)),
         expect=dict(statuses=[3], owned_failure=True))
    case('scan over 3 right-hand sides', scanned, (V, jnp.stack([B, 2 * B, 3 * B])))
    case('lax.map (batch 2 of 5) over an inner certified body', lambda v, bs: lax.map(
        lambda b: ns.checked_native(direct)(v, b)[1], bs, batch_size=2),
        (V, jnp.stack([B, 2 * B, 3 * B, 4 * B, 5 * B])))

    print('== 6. derivative availability inside a custom JVP rule')
    case('availability holds', implicit_root(False), (V, B, jnp.float64(1e-6), dV),
         candidate_function=implicit_root(True), expect=dict(availability=1, unavailable=0))
    case('availability fails', implicit_root(False), (V, B, jnp.float64(1e6), dV),
         candidate_function=implicit_root(True), expect=dict(availability=1, unavailable=1, owned_failure=True))
    case('availability, vmap of certified', implicit_root(False),
         (lanes_v, lanes_b, jnp.asarray([1e-6, 1e-6, 1e6, 1e-6, 1e6]), jnp.stack([dV] * 5)),
         candidate_function=implicit_root(True), wrap=vmapped)
    case('availability under reverse mode', lambda v, b, t: jax.grad(
        lambda real: jnp.sum(jnp.abs(implicit_root(False)(real + 0j, b, t, dV)[0]) ** 2))(jnp.real(v)),
        (V, B, jnp.float64(1e6)),
        candidate_function=lambda v, b, t: jax.grad(
            lambda real: jnp.sum(jnp.abs(implicit_root(True)(real + 0j, b, t, dV)[0]) ** 2))(jnp.real(v)))

    print('== 7. sequentially mapped Newton body (early exit kept per lane)')
    steps = jnp.asarray([3, 0, 2, 4, 1], dtype=jnp.int32)
    poison = jnp.asarray([-1, -1, -1, 1, -1], dtype=jnp.int32)
    case('certified of vmap of sequential primal', jax.vmap(sequential(False)), (lanes_v, lanes_b, steps, poison),
         candidate_function=jax.vmap(sequential(True)), repeats=100)
    case('vmap of certified of sequential primal', sequential(False), (lanes_v, lanes_b, steps, poison),
         candidate_function=sequential(True), wrap=vmapped)
    case('sequential primal, scalar call', sequential(False), (V, B, jnp.int32(2), jnp.int32(-1)),
         candidate_function=sequential(True))
    case('foreign debug_check in a while body, scalar', refine_with_invariant,
         (V, B, jnp.int32(3), jnp.int32(1)))
    case('foreign debug_check inside the sequential body, scalar', sequential(False, refine_with_invariant),
         (V, B, jnp.int32(3), jnp.int32(1)), candidate_function=sequential(True, refine_with_invariant))
    case('foreign debug_check inside the sequential body, lanes',
         jax.vmap(sequential(False, refine_with_invariant)), (lanes_v, lanes_b, steps, poison),
         candidate_function=jax.vmap(sequential(True, refine_with_invariant)))

    print('== 8. foreign checks and dtypes')
    case('foreign checkify.check, passing', with_foreign_check, (V, B, jnp.float64(1e30)))
    case('foreign checkify.check, failing', with_foreign_check, (V, B, jnp.float64(-1e30)))
    case('foreign check failing in one of 5 lanes', with_foreign_check,
         (lanes_v, lanes_b, jnp.asarray([1e30, 1e30, -1e30, 1e30, 1e30])), wrap=vmapped)


def cases32():
    V32, B32 = jnp.asarray(VALUES.astype(np.complex64)), jnp.asarray(RHS.astype(np.complex64))
    case('complex64 operands, ambient x64 off', direct, (V32, B32), x64=False,
         expect=dict(payload_dtypes=['float32']))
    case('complex64 value_and_grad, ambient x64 off', real_grad(loss, has_aux=True), (V32, B32), x64=False,
         expect=dict(payload_dtypes=['float32']))
    case('complex64 lanes, vmap of certified cond, ambient x64 off', branchy,
         (jnp.asarray([True, False]), jnp.stack([V32, 2 * V32]), jnp.stack([B32, B32])), wrap=vmapped,
         x64=False, expect=dict(payload_dtypes=['float32']))


def fail_closed():
    """A site inside a higher-order primitive without a rule must stop the trace."""
    print('== 9. fail closed')
    with jax.enable_x64(True):
        V, B = jnp.asarray(VALUES), jnp.asarray(RHS)
        hidden = lambda v, b: jax.checkpoint(lambda x: solve(x, b).value)(v)
        with without_private_registrations():
            try:
                ct.certified(hidden)(V, B)
                outcome = 'NOT RAISED'
            except ct.CertificateTransportError as failure:
                outcome = f'raised CertificateTransportError: {failure}'
    print('          site inside jax.checkpoint (no rule in the candidate):', outcome)
    RESULTS.append(dict(case='fail closed: site inside jax.checkpoint', identical='NOT RAISED' not in outcome,
                        outcome=outcome, rows=0, sites=[], foreign=None, max_eta=0.0,
                        first_call_s=(0, 0), run_ms=(0, 0), problems=[]))


def main():
    with jax.enable_x64(True):
        cases64()
    assert not jax.config.jax_enable_x64
    cases32()
    fail_closed()
    different = [row['case'] for row in RESULTS if not row['identical']]
    print(f'\n{len(RESULTS) - len(different)} of {len(RESULTS)} cases identical; different: {different}')
    json.dump(RESULTS, open(sys.argv[0].rsplit('/', 1)[0] + '/proof_results.json', 'w'), indent=1, default=str)
    return not different


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
