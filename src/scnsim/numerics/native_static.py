"""Static capacitance leaves of the shared candidate/implicit-equation graph.

The View's routed C slots are the original physical stamps after its actual
runtime congruences. No frequency operator, Port boundary or root is formed.
Only explicit zero-charge coordinates own an elimination equation; its native
factor is call-local and its checked primitive owns primal and AD evidence.
"""
from __future__ import annotations

from .sparse_structure import _block, _dense
from .sparse_solve import SparseEquation
from .native_sparse import checked_sparse_solve


def compile_static_capacitance(view, descriptor):
    pattern = view.patterns[0]
    retained, eliminated = descriptor.retained_indices, descriptor.zero_charge_indices
    rr, rr_slots = _block(pattern, retained, retained)
    if eliminated:
        ee, ee_slots = _block(pattern, eliminated, eliminated)
        er, er_slots = _block(pattern, eliminated, retained)
        re, re_slots = _block(pattern, retained, eliminated)
        equation = SparseEquation(ee)

    def value(candidate, prior_status=0):
        import jax.numpy as jnp
        values = candidate.views[descriptor.view_index].model.C
        matrix = _dense(rr, values[jnp.asarray(rr_slots, dtype=jnp.int32)])
        status = jnp.asarray(prior_status, dtype=jnp.int32)
        if eliminated:
            rhs = _dense(er, values[jnp.asarray(er_slots, dtype=jnp.int32)])
            solved = checked_sparse_solve(equation,
                values[jnp.asarray(ee_slots, dtype=jnp.int32)], rhs,
                prior_status=status, dependency=('static_capacitance', descriptor.view_index,
                                                tuple(eliminated), tuple(retained)))
            matrix = matrix - _dense(re, values[jnp.asarray(re_slots, dtype=jnp.int32)]) @ solved.value
            status = solved.aux.status
        result = matrix if descriptor.kind == 'static_capacitance' else matrix[descriptor.indices]
        return result, status
    return value
