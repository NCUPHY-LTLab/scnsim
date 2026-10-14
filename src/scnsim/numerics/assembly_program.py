"""Static sparse routing with runtime parameter/View/frequency operands.

One pattern serves all values, tangents and mixed jets of a program. Numerical
coefficients never enter static keys. This adapts the existing assembly kernel,
including its selected-dtype compensation and local series-RL certificates.
Packed material banks and incidence gathers avoid per-section runtime stacks;
static section ordering and section-wise absolute bounds remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
import numpy as np
from ..compilation.program import SparsePattern, _indices
from ..compilation.model_program import _immutable


@dataclass(frozen=True, slots=True, eq=False)
class SeriesGroup:
    indices: tuple[int, ...]
    counts: tuple[int, ...]
    columns: Any
    pair_map: Any
    global_map: Any
    pure: tuple[bool, ...]
    series_group: int
    weight_sources: Any
    weight_mask: Any


class AssemblyPayload(NamedTuple):
    values: Any
    frequency_values: Any
    bound: Any
    status: Any
    low_values: Any
    frequency_low_values: Any


@dataclass(frozen=True, slots=True, eq=False)
class AssemblyProgram:
    view_program: Any
    pattern: SparsePattern
    base_slots: tuple[Any, ...]
    load_columns: Any
    load_slots: Any
    groups: tuple[SeriesGroup, ...]

    def operands(self, view):
        import jax.numpy as jnp
        from .jax_core import inductive_coefficients
        model = view.model
        base = tuple(item for slots, values in zip(self.base_slots, (model.C, model.K, model.G), strict=True)
                     for item in (jnp.asarray(slots, dtype=jnp.int32), values))
        dtype = model.C.dtype
        load = (model.R, model.M, model.B, jnp.asarray(self.load_columns, dtype=jnp.int32),
                jnp.asarray(self.load_slots, dtype=jnp.int32))
        groups = []
        for group in self.groups:
            packed = model.series[group.series_group]
            R, L = packed.R[packed.material_indices], packed.L[packed.material_indices]
            weights = packed.weights[jnp.asarray(group.weight_sources, dtype=jnp.int32)]
            weights = jnp.where(jnp.asarray(group.weight_mask), weights, jnp.zeros_like(weights))
            pure = jnp.asarray(group.pure, dtype=jnp.bool_)
            seed = jnp.zeros_like(L)
            codes = jnp.zeros((len(group.indices),), dtype=jnp.int32)
            groups.append((R, L, weights, jnp.asarray(group.columns, dtype=jnp.int32),
                           jnp.asarray(group.pair_map, dtype=jnp.int32),
                           jnp.asarray(group.global_map, dtype=jnp.int32), pure, seed, codes))
        # The constant-inductor value uses the existing range-safe scalar/2x2
        # implementation, while its derivative is the inverse equation. frexp
        # bit operations are not treated as the physical derivative authority.
        constants = _prepared_inverse(tuple(groups))
        prepared = tuple(group[:7] + (inverse, code) for group, (inverse, code) in zip(groups, constants, strict=True))
        return jnp.arange(len(self.pattern.rows), dtype=jnp.int32), base, load, prepared

    def polynomial_coefficients(self, operands, *, view_status, dtype):
        """Use the same fixed sparse stamps for an admitted pure-inductive pencil."""
        from .jax_core import polynomial_coefficients, first
        from ..errors import RuntimePreparationError
        if any(not all(group.pure) for group in self.groups):
            raise RuntimePreparationError(
                'Polynomial transfer-zero discovery requires declared pure-inductive series sections',
                stage='root_search_prepare')
        K, C, G, status = polynomial_coefficients(operands, loaded=True, dtype=dtype)
        return K, C, G, first(view_status, status != 0, status)

    def __call__(self, theta, omega, *, loaded, compensated=False, port_realizable=None):
        import jax.numpy as jnp
        from .jax_core import assemble, first
        view = self.view_program(theta, port_realizable=port_realizable)
        return self.assemble_prepared(self.operands(view), omega, view_status=view.status,
                                      loaded=loaded, compensated=compensated), view

    def assemble_prepared(self, operands, omega, *, view_status, loaded, compensated=False):
        """Frequency arithmetic over candidate-owned prepared stamp operands.

        Local inverse coefficients and View weights are prepared once by the
        caller, not once per frequency. The existing stamp kernel remains the
        sole arithmetic authority, including ordered compensated coalescing.
        """
        import jax.numpy as jnp
        from .jax_core import assemble, first
        result = assemble(operands, omega, loaded=loaded, derivative=True, compensated=compensated)
        status = first(view_status, result[3] != 0, result[3])
        low, prime_low = result[4:] if compensated else (jnp.zeros_like(result[0]), jnp.zeros_like(result[1]))
        return AssemblyPayload(result[0], result[1], result[2], status, low, prime_low)


    def assemble_prepared_pair(self, operands, omega, *, view_status, loaded):
        """Same ordered stamp equation with finite physical pair AD rules."""
        from .jax_core import _assemble_compensated, first
        from . import native_compensated
        result = _assemble_compensated(operands, omega, loaded=loaded,
                                      derivative=True, arithmetic=native_compensated)
        status = first(view_status, result[3] != 0, result[3])
        return AssemblyPayload(result[0], result[1], result[2], status, result[4], result[5])


_PREPARED_INVERSE = None


def _prepared_inverse(groups):
    global _PREPARED_INVERSE
    if _PREPARED_INVERSE is None:
        import jax
        import jax.numpy as jnp
        from .jax_core import inductive_coefficients
        @jax.custom_jvp
        def inverse(arguments):
            return inductive_coefficients(arguments)
        @inverse.defjvp
        def directional(primals, tangents):
            (arguments,), (directions,) = primals, tangents
            result = inverse(arguments)
            jets = []
            for argument, direction, (value, code) in zip(arguments, directions, result, strict=True):
                delta_L = direction[1]
                derivative = -value @ delta_L @ value
                derivative = jnp.where(argument[6][:, None, None], derivative, jnp.zeros_like(derivative))
                jets.append((derivative, jnp.zeros(code.shape, dtype=jax.dtypes.float0)))
            return result, tuple(jets)
        _PREPARED_INVERSE = inverse
    return _PREPARED_INVERSE(groups)


def compile_assembly(view_program):
    """Bind canonical CSC slots; zero-weight incidence padding adds no nodes."""
    patterns, blocks = view_program.patterns, view_program.series_patterns
    n = len(view_program.node_ids)
    row_parts = [pattern.rows for pattern in patterns[:3]]
    col_parts = [pattern.cols for pattern in patterns[:3]]
    for block in blocks:
        row_parts.append(np.repeat(block.rows, len(block.rows)))
        col_parts.append(np.tile(block.rows, len(block.rows)))
    B = patterns[3]
    row_parts.append(np.repeat(B.rows, len(B.rows)))
    col_parts.append(np.tile(B.rows, len(B.rows)))
    pattern, _ = SparsePattern.from_coo((n, n), np.concatenate(row_parts), np.concatenate(col_parts))
    locations = {(int(row), int(col)): index for index, (row, col) in enumerate(zip(pattern.rows, pattern.cols, strict=True))}
    def slots(rows, columns):
        return _indices([locations[int(row), int(column)] for row, column in zip(rows, columns, strict=True)])
    base = tuple(slots(p.rows, p.cols) for p in patterns[:3])
    load_slots = slots(np.repeat(B.rows, len(B.rows)), np.tile(B.rows, len(B.rows))).reshape((len(B.rows), len(B.rows)))
    pure = tuple(line.declared_lossless for line in view_program.model_program.lines for _ in line.blocks)
    grouped = {}
    for index, block in enumerate(blocks):
        grouped.setdefault(block.shape[1], []).append(index)
    groups = []
    for series_group, (width, indices) in enumerate(grouped.items()):
        counts = tuple(len(blocks[index].rows) for index in indices)
        count = max(counts)
        columns = np.zeros((len(indices), count), dtype=np.int64)
        weight_sources = np.zeros((len(indices), count), dtype=np.int64)
        weight_mask = np.zeros((len(indices), count), dtype=np.bool_)
        weight_offset = 0
        global_slots, pairs, offset = [], [], 0
        for position, index in enumerate(indices):
            block = blocks[index]
            columns[position, :len(block.cols)] = block.cols
            weight_sources[position, :len(block.rows)] = weight_offset + np.arange(len(block.rows))
            weight_mask[position, :len(block.rows)] = True
            weight_offset += len(block.rows)
            # Padded entries have zero weights. Their row is an existing row,
            # and no extra structural/global slot or matrix dimension is added.
            rows = np.pad(block.rows, (0, count - len(block.rows)), constant_values=int(block.rows[0]))
            pair = slots(np.repeat(rows, count), np.tile(rows, count))
            unique = np.unique(pair)
            global_slots.extend(unique)
            pairs.append(offset + np.searchsorted(unique, pair))
            offset += len(unique)
        # Preserve section-wise aggregation BEFORE absolute values. Distinct
        # sections can stamp the same global slot and still own distinct bound
        # terms, exactly as the existing aggregate-norm certificate requires.
        global_map = np.asarray(global_slots, dtype=np.int64)
        pair_map = np.asarray(pairs).reshape((len(indices), count, count))
        groups.append(SeriesGroup(tuple(indices), counts, _indices(columns), _indices(pair_map),
                                  _indices(global_map), tuple(pure[index] for index in indices),
                                  series_group, _indices(weight_sources), _immutable(weight_mask)))
    return AssemblyProgram(view_program, pattern, base, _indices(B.cols), _indices(load_slots), tuple(groups))
