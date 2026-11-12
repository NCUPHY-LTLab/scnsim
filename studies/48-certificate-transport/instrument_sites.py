"""Diagnostic only: which JAX higher-order primitives enclose SCNSim equation
sites in the programs that today's checked_native functionalizes.

Usage: python instrument_sites.py <out.json> <unittest target> [...]
Runs the named unittest targets with native_sparse._annotate_sparse_sites wrapped.
"""
import atexit, json, sys, unittest, collections

out_path, targets = sys.argv[1], sys.argv[2:]

import tests  # noqa: F401  (configures the runtime exactly like the suite)
from jax.extend.core import Jaxpr
from scnsim.numerics import native_sparse as ns
from jax._src import checkify as _ck   # harness only: recognise check_p

stats = collections.Counter()
param_keys = collections.defaultdict(set)
programs = collections.Counter()
original = ns._annotate_sparse_sites


def sub_jaxprs(value):
    if isinstance(value, Jaxpr):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from sub_jaxprs(item)


def walk(jaxpr, enclosing):
    sites = 0
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if eqn.primitive in ns._certificate_primitives:
            sites += 1
            stats[('site', name, enclosing)] += 1
        elif eqn.primitive is ns._primal_archive_p:
            stats[('archive', name, enclosing)] += 1
        elif eqn.primitive is _ck.check_p:
            stats[('check_p', 'debug' if eqn.params.get('debug') else 'check', enclosing)] += 1
        nested = [j for v in eqn.params.values() for j in sub_jaxprs(v)]
        if nested:
            param_keys[name] |= set(eqn.params)
            inner = sum(walk(j, enclosing + (name,)) for j in nested)
            stats[('hop', name, 'with_sites' if inner else 'no_sites')] += 1
            sites += inner
    return sites


def wrapped(jaxpr, path=()):
    if path == ():
        programs['programs'] += 1
        programs['sites_total'] += walk(jaxpr, ())
    return original(jaxpr, path)


ns._annotate_sparse_sites = wrapped


@atexit.register
def dump():
    rows = [dict(kind=k[0], name=k[1], where=k[2] if isinstance(k[2], str) else '/'.join(k[2]) or '<top>', count=v)
            for k, v in sorted(stats.items(), key=str)]
    json.dump(dict(programs=dict(programs), rows=rows,
                   param_keys={k: sorted(v) for k, v in param_keys.items()}), open(out_path, 'w'), indent=1)


suite = unittest.defaultTestLoader.loadTestsFromNames(targets)
result = unittest.TextTestRunner(verbosity=1).run(suite)
print('ran', result.testsRun, 'failures', len(result.failures), 'errors', len(result.errors))
