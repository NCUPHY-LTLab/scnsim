"""Sparse pattern and retained-block helpers, without execution ownership.

CSC ordering and submatrix slot mapping follow the compiler's SparsePattern
unchanged. No Run, Workspace, backend, solver or AD interpreter is retained.
"""
from __future__ import annotations

import numpy as np
from ..compilation.program import SparsePattern, _indices


def _dense(pattern, values):
    import jax.numpy as jnp
    return jnp.zeros(pattern.shape, dtype=values.dtype).at[
        jnp.asarray(pattern.rows), jnp.asarray(pattern.cols)].add(values)



def _block(pattern, rows, columns):
    rmap, cmap = {value: index for index, value in enumerate(rows)}, {value: index for index, value in enumerate(columns)}
    sources, rr, cc = [], [], []
    for index, (r, c) in enumerate(zip(pattern.rows, pattern.cols, strict=True)):
        if int(r) in rmap and int(c) in cmap:
            sources.append(index); rr.append(rmap[int(r)]); cc.append(cmap[int(c)])
    result, slots = SparsePattern.from_coo((len(rows), len(columns)), rr, cc)
    # Parent CSC traversal gives submatrix CSC traversal with preserved row order.
    order = np.argsort(slots)
    return result, _indices(np.asarray(sources)[order])



def _full_pattern(size):
    return SparsePattern.from_coo((size, size), np.tile(np.arange(size), size),
                                  np.repeat(np.arange(size), size))[0]

