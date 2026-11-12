"""Shadow comparison on real SCNSim programs (diagnostic, study only).

Every outermost ``checked_native`` frame the product builds while the named
unittest targets run is ALSO built with the candidate transport, in the same
traced program. The product keeps using the current mechanism's result. A
host callback compares, at run time and for every execution (every lane):

- every equation certificate row (status, eta, numerator, denominator,
  performed, present) bit for bit,
- the values the frame returns,
- whether a foreign (not SCNSim-owned) check failed,

and, at trace time, the static row metadata and order (site, role, stage,
derivative order, dependency).

Rows that reach the frame through the product's archive primitive, and
derivative-availability facts, are outside this comparison: they need the
product's call sites to change (attach / gate). prove_transport.py covers
them on small cases.

Usage: python shadow_product.py <out.json> <unittest target> [...]
"""
import atexit, collections, json, sys, threading, unittest

out_path, targets = sys.argv[1], sys.argv[2:]

import tests  # noqa: F401  configures the runtime like the suite
import numpy as np
import jax
import jax.numpy as jnp

from scnsim.numerics import native_sparse as ns
from scnsim.numerics import native_pair  # noqa: F401
import certificate_transport as ct
from jax._src import checkify as _private_checkify   # harness only

for _primitive in ns._certificate_primitives:
    _keywords = _private_checkify.error_checks[_primitive].keywords
    ct.register_equation(_primitive, value_count=_keywords['value_count'],
                         certificate_groups=_keywords['certificate_groups'])


class without_private_registrations:
    def __enter__(self):
        self.saved = {primitive: _private_checkify.error_checks.pop(primitive)
                      for primitive in ns._certificate_primitives}

    def __exit__(self, *exc):
        _private_checkify.error_checks.update(self.saved)


ct._foreign_check_scope = without_private_registrations

ARRAYS = ('status', 'eta', 'residual_numerator', 'residual_denominator', 'performed', 'present')
STATIC = ('site', 'role', 'stage', 'derivative_order', 'dependency')
counts = collections.Counter()
roles = collections.Counter()
nests = collections.Counter()
problems = []
lock = threading.Lock()
state = threading.local()
original = ns.checked_native


def note(text):
    with lock:
        if len(problems) < 40:
            problems.append(text)


def host_compare(tag, current, candidate):
    with lock:
        counts['executions'] += 1
    for name, old, new in zip(tag_names[int(tag)], current, candidate):
        old, new = np.asarray(old), np.asarray(new)
        with lock:
            counts['arrays_compared'] += 1
            counts['elements_compared'] += old.size
        if old.dtype != new.dtype or old.shape != new.shape or old.tobytes() != new.tobytes():
            with lock:
                counts['array_mismatches'] += 1
            note(f'program {int(tag)} {name}: {old!r} != {new!r}')


tag_names = []


def nesting(site):
    return '/'.join(str(part) for part in site if isinstance(part, str)) or '<top>'


import os
INNER = os.environ.get('SHADOW_SEQUENTIAL_BODIES') == '1'
original_sequential = ns.checked_sequential_primal


def shadow_sequential_primal(function):
    """With SHADOW_SEQUENTIAL_BODIES=1 the scalar Newton body that the product
    certifies before sequential mapping is compared too. Its second copy stays
    hidden inside the mapped call, so enclosing frames are not affected."""
    state.force = INNER
    try:
        return original_sequential(function)
    finally:
        state.force = False


ns.checked_sequential_primal = shadow_sequential_primal


def shadow_checked_native(function):
    current = original(function)
    candidate = ct.certified(function)
    forced = getattr(state, 'force', False)
    state.force = False

    def checked(*args):
        if getattr(state, 'depth', 0) and not (forced and not getattr(state, 'inner', False)):
            return current(*args)
        if forced:
            with lock:
                counts['sequential_body_frames_traced'] += 1
            state.inner = True
        saved_depth = getattr(state, 'depth', 0)
        state.depth = 1
        try:
            error, values = current(*args)
            try:
                evidence, shadow_values = candidate(*args)
            except Exception as failure:  # the product must keep running
                with lock:
                    counts['candidate_trace_failures'] += 1
                note(f'candidate trace failed: {type(failure).__name__}: {str(failure)[:300]}')
                return error, values
        finally:
            state.depth = saved_depth
            if forced:
                state.inner = False
        with lock:
            counts['frames_traced'] += 1
        with jax.enable_x64(True):
            effects = [effect for effect in sorted(error._payload)
                       if isinstance(effect, ns.SparseCertificateEffect)]
            old_rows = [row for effect, row in zip(effects, ns.sparse_certificates(error), strict=True)
                        if not isinstance(effect, ns.ArchivedSparseEffect)]
            with lock:
                counts['archived_rows_not_compared'] += len(effects) - len(old_rows)
                counts['availability_facts_not_compared'] += sum(
                    isinstance(effect, ns.DerivativeAvailabilityEffect) for effect in error._payload)
            new_rows = list(ct.ledger_certificates(evidence.ledger))
            if len(old_rows) != len(new_rows):
                with lock:
                    counts['row_count_mismatches'] += 1
                note(f'row count {len(old_rows)} != {len(new_rows)}')
                return error, values
            names, old_arrays, new_arrays = [], [], []
            for index, (old, new) in enumerate(zip(old_rows, new_rows)):
                with lock:
                    counts['rows_traced'] += 1
                    roles[old.role] += 1
                    nests[nesting(old.site)] += 1
                for field in STATIC:
                    if getattr(old, field) != getattr(new, field):
                        with lock:
                            counts['static_mismatches'] += 1
                        note(f'row {index} {field}: {getattr(old, field)!r} != {getattr(new, field)!r}')
                for field in ARRAYS:
                    names.append(f'row {index} {old.role} {old.site} {field}')
                    old_arrays.append(getattr(old, field))
                    new_arrays.append(getattr(new, field))
            old_leaves, old_tree = jax.tree.flatten(values)
            new_leaves, new_tree = jax.tree.flatten(shadow_values)
            if old_tree != new_tree:
                with lock:
                    counts['static_mismatches'] += 1
                note('value tree differs')
            else:
                names += [f'value leaf {index}' for index in range(len(old_leaves))]
                old_arrays += old_leaves
                new_arrays += new_leaves
            owned = (ns.SparseCertificateEffect, ns.DerivativeAvailabilityEffect)
            foreign_old = jnp.bool_(False)
            for effect, predicate in error._pred.items():
                if not isinstance(effect, owned):
                    foreign_old = foreign_old | jnp.any(predicate)
            foreign_new = jnp.bool_(False)
            for predicate in evidence.foreign._pred.values():
                foreign_new = foreign_new | jnp.any(predicate)
            owned_old = jnp.bool_(False)
            for effect, predicate in error._pred.items():
                if isinstance(effect, ns.SparseCertificateEffect) and not isinstance(effect, ns.ArchivedSparseEffect):
                    owned_old = owned_old | jnp.any(predicate)
            names += ['foreign failure', 'owned equation failure']
            old_arrays += [foreign_old, owned_old]
            new_arrays += [foreign_new, evidence.ledger.any_failed()]
            with lock:
                tag_names.append(names)
                tag = len(tag_names) - 1
            jax.debug.callback(host_compare, jnp.int32(tag), tuple(old_arrays), tuple(new_arrays))
        return error, values
    return checked


ns.checked_native = shadow_checked_native


@atexit.register
def dump():
    json.dump(dict(counts=dict(counts), roles=dict(roles), nesting=dict(nests), problems=problems),
              open(out_path, 'w'), indent=1)


suite = unittest.defaultTestLoader.loadTestsFromNames(targets)
result = unittest.TextTestRunner(verbosity=1).run(suite)
print('ran', result.testsRun, 'failures', len(result.failures), 'errors', len(result.errors))
for label, entries in (('FAIL', result.failures), ('ERROR', result.errors)):
    for test, text in entries:
        print(label, test, text[-1500:])
print(json.dumps(dict(counts), indent=1))
