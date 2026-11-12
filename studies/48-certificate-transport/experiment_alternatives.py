"""Measurements behind the comparison of designs B (host-side record) and
C (public checkify only). Public JAX interfaces only. Diagnostics, no product code.

Usage: uv run --locked python experiment_alternatives.py
"""
import threading
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import checkify

LOG = []


def host(tag, x):
    LOG.append((int(np.asarray(tag).reshape(-1)[0]), np.asarray(x).shape, threading.get_ident()))
    return np.asarray(x) * 2


def recorded(tag, x):
    return jax.pure_callback(host, jax.ShapeDtypeStruct(x.shape, x.dtype), jnp.int32(tag), x,
                             vmap_method='sequential')


def run(label, function, *args):
    LOG.clear()
    out = jax.block_until_ready(jax.jit(function)(*args))
    tags = [tag for tag, _, _ in LOG]
    threads = {thread for _, _, thread in LOG}
    print(f'{label:64} calls={len(LOG):2d} tags={tags} caller_thread={threading.get_ident() in threads}')
    return out


x = jnp.arange(5.0)
print('== B. what a host-side record can and cannot see')
run('B1 solve result used', lambda v: recorded(1, v), x[0])
run('B2 solve result NOT used (elided: no host call, no record)',
    lambda v: (recorded(1, v), v + 1)[1], x[0])
run('B3 scalar cond: only the taken branch calls the host',
    lambda p, v: lax.cond(p, lambda: recorded(1, v), lambda: recorded(2, v)), jnp.bool_(True), x[0])
run('B4 vmap of that cond, lanes T,F,T,F,F: BOTH branches, ALL lanes',
    jax.vmap(lambda p, v: lax.cond(p, lambda: recorded(1, v), lambda: recorded(2, v))),
    jnp.asarray([True, False, True, False, False]), x)
run('B5 vmap of while_loop, trip counts 0..4: finished lanes still call',
    jax.vmap(lambda n, v: lax.while_loop(lambda s: s[0] < n, lambda s: (s[0] + 1, recorded(1, s[1])),
                                         (jnp.int32(0), v))[1]),
    jnp.arange(5, dtype=jnp.int32), x)
run('B6 lax.map batch_size=2 over 5: lane identity is call order only',
    lambda v: lax.map(lambda one: recorded(1, one), v, batch_size=2), x)
run('B7 two independent solves: order is not part of the program',
    lambda v: recorded(1, v) + recorded(2, v), x[0])

print('== C. what the public checkify surface returns')


def three_sites(values):
    for index in range(3):
        eta = values[index]
        checkify.check(eta <= 1.0, 'site ' + str(index) + ' failed with eta={eta}', eta=eta)
    return jnp.sum(values)


for label, values in (('all three sites pass', [0.1, 0.2, 0.3]),
                      ('sites 1 and 2 fail', [0.1, 2.0, 3.0])):
    error, _ = jax.jit(checkify.checkify(three_sites))(jnp.asarray(values))
    public = sorted(name for name in dir(error) if not name.startswith('_'))
    print(f'C  {label:22} get() -> {error.get()!r}')
print('C  public attributes of Error:', public)
print('C  leaves of the Error pytree (pred, code, one payload slot for all three sites):',
      len(jax.tree.leaves(error)))
