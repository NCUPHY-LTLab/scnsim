"""Timing diagnostic (no threshold): current mechanism against the candidate on
the small cases, interleaved so that drift affects both alike.

Usage: uv run --locked python time_transport.py   (this directory on PYTHONPATH)
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

import prove_transport as p
from scnsim.numerics import native_sparse as ns
import certificate_transport as ct


def build(function, candidate_function, wrap):
    current = jax.jit(wrap(ns.checked_native(function)))
    candidate = jax.jit(wrap(ct.certified(candidate_function or function)))
    return current, candidate


def measure(name, function, args, *, candidate_function=None, wrap=lambda f: f, rounds=7, calls=200):
    current, candidate = build(function, candidate_function, wrap)
    start = time.perf_counter()
    jax.block_until_ready(current(*args))
    first_current = time.perf_counter() - start
    with p.without_private_registrations():
        start = time.perf_counter()
        jax.block_until_ready(candidate(*args))
        first_candidate = time.perf_counter() - start
    medians = {'current': [], 'candidate': []}
    for _ in range(rounds):
        for label, compiled in (('current', current), ('candidate', candidate)):
            samples = []
            for _ in range(calls):
                start = time.perf_counter()
                jax.block_until_ready(compiled(*args))
                samples.append(time.perf_counter() - start)
            medians[label].append(1e3 * float(np.median(samples)))
    low = lambda label: min(medians[label])
    high = lambda label: max(medians[label])
    print(f'{name:34} run ms current {low("current"):.3f}..{high("current"):.3f}  '
          f'candidate {low("candidate"):.3f}..{high("candidate"):.3f}   '
          f'first call s {first_current:.3f} / {first_candidate:.3f}')


with jax.enable_x64(True):
    V, B, dV = jnp.asarray(p.VALUES), jnp.asarray(p.RHS), jnp.asarray(p.DIRECTION)
    lanes_v = jnp.stack([V, 2 * V, 3 * V, V, 3 * V])
    lanes_b = jnp.stack([B, B, 2 * B, -B, 1j * B])
    steps = jnp.asarray([3, 0, 2, 4, 1], dtype=jnp.int32)
    poison = jnp.full((5,), -1, dtype=jnp.int32)
    print('median of 200 calls, range over 7 interleaved rounds')
    measure('direct solve', p.direct, (V, B))
    measure('vmap, 5 lanes', p.direct, (lanes_v, lanes_b), wrap=jax.vmap)
    measure('value and gradient', p.real_grad(p.loss, has_aux=True), (V, B))
    measure('jvp of jvp (4 equations)', lambda v, b, dv: jax.jvp(
        lambda y: jax.jvp(lambda x: p.solve(x, b).value, (y,), (dv,))[1], (v,), (dv,)), (V, B, dV))
    measure('while_loop, 3 trips', p.refine, (V, B, jnp.int32(3), jnp.int32(-1)))
    measure('sequential map, 5 lanes', jax.vmap(p.sequential(False)), (lanes_v, lanes_b, steps, poison),
            candidate_function=jax.vmap(p.sequential(True)))
