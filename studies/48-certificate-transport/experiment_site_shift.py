"""Diagnostic: do equation paths ("site") move when the transport equations of a
sequentially mapped body change (archive primitive -> attach primitive)?

A custom-JVP root whose primal is a sequentially mapped Newton-like loop, and
whose JVP rule then solves a tangent equation, as SCNSim's roots do.
Usage: uv run --locked python experiment_site_shift.py   (prove_transport.py's directory on PYTHONPATH)
"""
import numpy as np
import jax
import jax.numpy as jnp

import prove_transport as p
from scnsim.numerics import native_sparse as ns
import certificate_transport as ct


def rooted(candidate):
    primal = p.sequential(candidate)
    dependency = ('root_occurrence', 0, 0)

    @jax.custom_jvp
    def root(values, rhs, steps):
        return primal(values, rhs, steps, jnp.int32(-1))

    @root.defjvp
    def rule(primals, tangents):
        values, rhs, steps = primals
        values_dot, rhs_dot, _ = tangents
        status, x = root(values, rhs, steps)
        if candidate:
            x = ct.derivative_gate(x, status == 0, dependency=dependency, reason='toy root unresolved')
        else:
            ns.derivative_available(status == 0, dependency=dependency, reason='toy root unresolved')
        tangent = p.solve(values, rhs_dot - ns._matmul(p.EQUATION, values_dot, x, 'N'), stage=13,
                          role='tangent', derivative_order=1, dependency=dependency, prior_status=status)
        return (status, x), (np.zeros((), dtype=jax.dtypes.float0), tangent.value)

    return lambda values, rhs, steps, direction: jax.jvp(
        lambda v: root(v, rhs, steps)[1], (values,), (direction,))


with jax.enable_x64(True):
    V, B, dV = jnp.asarray(p.VALUES), jnp.asarray(p.RHS), jnp.asarray(p.DIRECTION)
    args = (V, B, jnp.int32(3), dV)
    error, old = jax.jit(ns.checked_native(rooted(False)))(*args)
    with p.without_private_registrations():
        evidence, new = jax.jit(ct.certified(rooted(True)))(*args)
    old_rows, new_rows = ns.sparse_certificates(error), ct.ledger_certificates(evidence.ledger)
    print('values identical:', p.same_tree(old, new))
    for old_row, new_row in zip(old_rows, new_rows, strict=True):
        arrays = all(p.bits(getattr(old_row, name)) == p.bits(getattr(new_row, name)) for name in p.ARRAYS)
        print(f'{old_row.role:8} stage={old_row.stage:2d} arrays identical={arrays} '
              f'site current={old_row.site} candidate={new_row.site}')
    for old_fact, new_fact in zip(ns.derivative_availability(error), ct.ledger_availability(evidence.ledger), strict=True):
        print(f'availability site current={old_fact.site} candidate={new_fact.site} '
              f'available {bool(old_fact.available)}=={bool(new_fact.available)}')

    # A frame that consumes a completed branch: today the branch's Error is a
    # runtime operand and is re-archived in the program before the frame's own
    # equations (archive_completed_primal_errors). With a ledger the branch
    # evidence can be attached (or merged on the host).
    branch_error, _ = jax.jit(ns.checked_native(p.sequential(False)))(V, B, jnp.int32(2), jnp.int32(-1))
    with p.without_private_registrations():
        branch_evidence, _ = jax.jit(ct.certified(p.sequential(True)))(V, B, jnp.int32(2), jnp.int32(-1))

    def consuming_current(values, rhs, errors):
        ns.archive_completed_primal_errors(errors)
        return p.solve(values, rhs, stage=7, dependency=('consumer',)).value

    def consuming_candidate(values, rhs, ledgers):
        outcome = p.solve(values, rhs, stage=7, dependency=('consumer',))
        status = outcome.aux.status
        for ledger in ledgers:
            status = ct.attach_ledger(status, ledger)
        return outcome.value, status

    error, _ = jax.jit(ns.checked_native(consuming_current))(V, B, (branch_error,))
    with p.without_private_registrations():
        evidence, _ = jax.jit(ct.certified(consuming_candidate))(V, B, (branch_evidence.ledger,))
    print('consuming frame, current  :', [(row.role, row.stage, row.site) for row in ns.sparse_certificates(error)])
    print('consuming frame, candidate:', [(row.role, row.stage, row.site) for row in ct.ledger_certificates(evidence.ledger)])
    same = all(p.bits(getattr(a, n)) == p.bits(getattr(b, n)) for a, b in
               zip(ns.sparse_certificates(error), ct.ledger_certificates(evidence.ledger), strict=True) for n in p.ARRAYS)
    print('consuming frame, certificate arrays identical:', same)
