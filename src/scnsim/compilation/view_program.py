"""Immutable View routing with traced pair weights and load normalization.

Coordinate names, retained/eliminated slots and sparse expansion are static.
Capacitance-cut weights, transformed coefficients and generalized Port maps are
runtime values of the same physical parameter point. No candidate is frozen as
an independent View or reconstructed from a saved numerical result. Packed
series namespaces preserve each section's original expansion order without
creating one runtime tuple per section.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
import numpy as np

from .program import SparsePattern, _indices
from .model_program import ModelPayload, ModelProgram, PackedSeries, _immutable


class ViewPayload(NamedTuple):
    model: ModelPayload
    selected_map: Any
    Rk: Any
    Dk: Any
    Go: Any
    status: Any
    failed_site: Any
    coordinate_transform: Any = ()


@dataclass(frozen=True, slots=True, eq=False)
class Expansion:
    pattern: SparsePattern
    sources: Any
    slots: Any
    kinds: Any

    def apply(self, values, alpha, beta, *, xp=None):
        if xp is None:
            import jax.numpy as xp
        jnp = xp
        factors = jnp.stack((jnp.asarray(1, dtype=values.dtype), beta, -alpha))
        kinds = jnp.asarray(self.kinds)
        weights = factors[kinds[:, 0]] * factors[kinds[:, 1]]
        result = jnp.zeros((len(self.pattern.rows),), dtype=values.dtype)
        update = values[jnp.asarray(self.sources)] * weights
        if xp is np:
            np.add.at(result, self.slots, update)
            return result
        return result.at[jnp.asarray(self.slots)].add(update)


@dataclass(frozen=True, slots=True, eq=False)
class PackedExpansion:
    """Independent block namespaces packed in original section order."""
    size: int
    sources: Any
    slots: Any
    kinds: Any

    def apply(self, values, alpha, beta):
        import jax.numpy as jnp
        factors = jnp.stack((jnp.asarray(1, dtype=values.dtype), beta, -alpha))
        kinds = jnp.asarray(self.kinds)
        weights = factors[kinds[:, 0]] * factors[kinds[:, 1]]
        update = values[jnp.asarray(self.sources)] * weights
        # Offset namespaces never mix different sections. Within each section
        # this is the identical existing Expansion scatter, not an EFT sum.
        return jnp.zeros((self.size,), dtype=values.dtype).at[jnp.asarray(self.slots)].add(update)


@dataclass(frozen=True, slots=True, eq=False)
class PairRoute:
    left: int
    right: int
    cut_slots: tuple[tuple[int, ...], tuple[int, ...]]
    expansions: tuple[Expansion, ...]
    series_expansions: tuple[Expansion, ...]
    map_left: int
    map_right: int


@dataclass(frozen=True, slots=True, eq=False)
class ViewProgram:
    model_program: ModelProgram
    node_ids: tuple[str, ...]
    terminal_ids: tuple[str, ...]
    selected: tuple[int, ...]
    eliminated: tuple[int, ...]
    routes: tuple[PairRoute, ...]
    patterns: tuple[SparsePattern, ...]
    series_patterns: tuple[SparsePattern, ...]
    source_map: Any
    terminal_map_rows: tuple[int, ...]
    retained: bool
    port_realizable: bool
    port_mask: Any
    declaration_bytes: bytes

    def lineage(self, fields):
        """Detached actual-point lineage from the compiled physical routing."""
        from ..errors import InvalidCandidatePhysicalParameter
        C = self.model_program.physical_capacitance(fields)
        maps = np.array(self.source_map, copy=True)
        import json
        declaration = json.loads(self.declaration_bytes)
        transforms = []
        for route, declared in zip(self.routes, declaration.get("transforms", ()), strict=True):
            cl = sum(C[slot] for slot in route.cut_slots[0])
            cr = sum(C[slot] for slot in route.cut_slots[1])
            if not np.isfinite(cl) or not np.isfinite(cr) or cl < 0 or cr < 0 or cl + cr <= 0:
                raise InvalidCandidatePhysicalParameter("pair transform capacitance cut is invalid", stage="physical_validation")
            alpha, beta = cl / (cl + cr), cr / (cl + cr)
            common, differential = declared.get("output_coordinates") or (declared["common_id"], declared["differential_id"])
            transforms.append(dict(input_coordinates=list(declared["input_coordinates"]),
                                   output_coordinates=[common, differential], weights=[float(alpha), float(beta)]))
            C = route.expansions[0].apply(C, alpha, beta, xp=np)
            keep = [i for i in range(len(maps)) if i not in (route.map_left, route.map_right)]
            maps = np.concatenate((maps[keep], (alpha * maps[route.map_left] + beta * maps[route.map_right])[None],
                                   (maps[route.map_left] - maps[route.map_right])[None]))
        selected = maps[list(self.terminal_map_rows)] if self.retained else np.eye(len(self.model_program.port_ids))
        realizable = bool(self.port_realizable and np.linalg.matrix_rank(selected) == len(self.terminal_ids))
        return dict(declaration=declaration, transforms=transforms, port_realizable=realizable)

    def capability(self, fields):
        """Actual binary64 physical rank, independent of numerical dtype."""
        return self.lineage(fields)["port_realizable"]

    def capability_array(self, physical_model, physical_theta):
        """Existing matrix-rank rule on ORIGINAL binary64 runtime weights.

        This is a route/domain predicate, not a differentiated matrix-rank
        observable. The fixed prepared partition reports a switch explicitly.
        """
        import jax.numpy as jnp
        if not self.port_realizable:
            return jnp.bool_(False)
        point = self.apply_model(physical_model, physical_theta, port_realizable=False)
        singular = jnp.linalg.svd(point.selected_map, compute_uv=False)
        tolerance = jnp.max(singular, initial=0) * max(point.selected_map.shape) * jnp.finfo(physical_theta.dtype).eps
        rank = jnp.sum(singular > tolerance)
        return rank == len(self.terminal_ids)

    def __call__(self, theta, *, port_realizable=None) -> ViewPayload:
        return self.apply_model(self.model_program(theta), theta, port_realizable=port_realizable)

    def apply_model(self, payload: ModelPayload, theta, *, port_realizable=None, static_coordinates=False) -> ViewPayload:
        """Route the SAME candidate fields without rebuilding its physical model.

        The caller owns the array payload. This program retains only immutable
        routing; parameter-dependent transform weights remain runtime operands.
        """
        import jax.numpy as jnp
        matrices = (payload.C, payload.K, payload.G, payload.B)
        series = payload.series
        maps = jnp.asarray(self.source_map, dtype=theta.dtype)
        transform = ()
        status, site = payload.status, payload.failed_site
        for index, route in enumerate(self.routes):
            cl = sum(matrices[0][slot] for slot in route.cut_slots[0])
            cr = sum(matrices[0][slot] for slot in route.cut_slots[1])
            bad = (~jnp.isfinite(cl)) | (~jnp.isfinite(cr)) | (cl < 0) | (cr < 0) | (cl + cr <= 0)
            first = (status == 0) & bad
            status = jnp.where(first, jnp.int32(3), status)
            site = jnp.where(first, jnp.int32(index), site)
            alpha, beta = cl / (cl + cr), cr / (cl + cr)
            if static_coordinates:
                # Compact same-call weights and ordered declarations define T;
                # no dense evidence-only transform or frozen baseline is used.
                transform = (*transform, (alpha, beta))
            matrices = tuple(expansion.apply(value, alpha, beta)
                             for expansion, value in zip(route.expansions, matrices, strict=True))
            series = tuple(PackedSeries(group.R, group.L, group.material_indices,
                                         expansion.apply(group.weights, alpha, beta))
                           for expansion, group in zip(route.series_expansions, series, strict=True))
            keep = tuple(i for i in range(maps.shape[0]) if i not in (route.map_left, route.map_right))
            maps = jnp.concatenate((maps[jnp.asarray(keep, dtype=jnp.int32)],
                                    (alpha * maps[route.map_left] + beta * maps[route.map_right])[None],
                                    (maps[route.map_left] - maps[route.map_right])[None]), axis=0)
        mask = jnp.asarray(self.port_mask, dtype=theta.dtype)
        model = ModelPayload(*matrices, payload.R, mask, series, status, site)
        p = len(self.model_program.port_ids)
        selected_map = maps[jnp.asarray(self.terminal_map_rows)] if self.retained else jnp.eye(p, dtype=theta.dtype)
        def normalized(incoming_status):
            # Principal SPD root is a matrix function; its directional derivative
            # is the Sylvester solve, not an arbitrary eigenvector derivative.
            from ..numerics.matrix_functions import spd_root_checked
            Dp, dp_status = spd_root_checked(payload.R)
            Rk = selected_map @ payload.R @ selected_map.T
            Dk, dk_status = spd_root_checked(Rk)
            first = (incoming_status == 0) & ((dp_status != 0) | (dk_status != 0))
            branch_status = jnp.where(first, jnp.int32(3), incoming_status)
            Qk = jnp.linalg.solve(Dk, selected_map @ Dp)
            Po = jnp.eye(p, dtype=theta.dtype) - Qk.T @ Qk
            inverse = jnp.linalg.solve(Dp, jnp.eye(p, dtype=theta.dtype))
            Go = inverse @ Po @ jnp.diag(mask) @ Po @ inverse
            return Rk, Dk, Go, branch_status
        def unnormalized(incoming_status):
            q = len(self.terminal_ids)
            Rk = Dk = jnp.zeros((q, q), dtype=theta.dtype)
            Go = jnp.zeros((p, p), dtype=theta.dtype)
            return Rk, Dk, Go, incoming_status
        capability = self.port_realizable if port_realizable is None else port_realizable
        if isinstance(capability, (bool, np.bool_)):
            Rk, Dk, Go, status = normalized(status) if capability else unnormalized(status)
        else:
            import jax
            Rk, Dk, Go, status = jax.lax.cond(capability, normalized, unnormalized, status)
        return ViewPayload(model, selected_map, Rk, Dk, Go, status, site, transform)


def _expansion(pattern, mapping, size, *, congruence):
    rows, cols, sources, kinds = [], [], [], []
    for index, (row, col) in enumerate(zip(pattern.rows, pattern.cols, strict=True)):
        for target, kind in mapping[row]:
            right = mapping[col] if congruence else ((col, 0),)
            for other, other_kind in right:
                rows.append(target); cols.append(other); sources.append(index); kinds.append((kind, other_kind))
    shape = (size, size if congruence else pattern.shape[1])
    result, slots = SparsePattern.from_coo(shape, rows, cols)
    return Expansion(result, _indices(sources), slots, _indices(np.asarray(kinds).reshape((-1, 2))))



def _packed_series_expansions(patterns, expansions):
    grouped = {}
    for index, pattern in enumerate(patterns):
        grouped.setdefault(pattern.shape[1], []).append(index)
    result = []
    for indices in grouped.values():
        sources, slots, kinds = [], [], []
        source_offset = target_offset = 0
        for index in indices:
            expansion = expansions[index]
            sources.extend(source_offset + expansion.sources)
            slots.extend(target_offset + expansion.slots)
            kinds.extend(expansion.kinds)
            source_offset += len(patterns[index].rows)
            target_offset += len(expansion.pattern.rows)
        result.append(PackedExpansion(target_offset, _indices(sources), _indices(slots),
                                       _indices(np.asarray(kinds).reshape((-1, 2)))))
    return tuple(result)


def compile_view_program(model_program: ModelProgram, declaration: dict) -> ViewProgram:
    from ..canonical import canonical_json_bytes
    nodes, patterns = model_program.node_ids, model_program.patterns
    original_count = len(nodes)
    ports = model_program.port_ids
    Bpattern = patterns[3]
    B = np.zeros((original_count, len(ports)))
    B[Bpattern.rows, Bpattern.cols] = model_program.B_values
    original_names = tuple(__import__('json').loads(model_program.evidence_bytes)['original_coordinates'])
    maps = np.asarray([B[nodes.index(name)] for name in original_names])
    names = original_names
    series = tuple(SparsePattern.from_coo(incidence[3], incidence[0], incidence[1])[0]
                   for line in model_program.lines for incidence in line.blocks)
    # Incidence payloads were canonical row-major in the old compiler. Reorder
    # them to the declared CSC slots at the model/program boundary below.
    routes = []
    for transform in declaration.get('transforms', ()):
        left, right = transform['input_coordinates']
        common, differential = transform.get('output_coordinates') or (transform['common_id'], transform['differential_id'])
        i, j = nodes.index(left), nodes.index(right)
        lookup = {(int(r), int(c)): index for index, (r, c) in enumerate(zip(patterns[0].rows, patterns[0].cols, strict=True))}
        cuts = (tuple(lookup[key] for key in ((i, i), (i, j)) if key in lookup),
                tuple(lookup[key] for key in ((j, j), (i, j)) if key in lookup))
        next_nodes = tuple(name for name in nodes if name not in (left, right)) + (common, differential)
        indices = {name: index for index, name in enumerate(next_nodes)}
        mapping = tuple(((indices[common], 0), (indices[differential], 1)) if row == i else
                        ((indices[common], 0), (indices[differential], 2)) if row == j else
                        ((indices[name], 0),) for row, name in enumerate(nodes))
        expanded = tuple(_expansion(p, mapping, len(next_nodes), congruence=k < 3) for k, p in enumerate(patterns))
        expanded_series = tuple(_expansion(p, mapping, len(next_nodes), congruence=False) for p in series)
        ml, mr = names.index(left), names.index(right)
        routes.append(PairRoute(i, j, cuts, expanded,
                                _packed_series_expansions(series, expanded_series), ml, mr))
        names = tuple(name for name in names if name not in (left, right)) + (common, differential)
        nodes, patterns = next_nodes, tuple(item.pattern for item in expanded)
        series = tuple(item.pattern for item in expanded_series)
        # Only dependency routing is needed here; numerical weights are traced.
        maps = np.vstack([maps[k] for k in range(len(maps)) if k not in (ml, mr)] +
                         [maps[ml] + maps[mr], maps[ml] - maps[mr]])
    retained = bool(declaration.get('retain'))
    terminals = tuple(declaration['retain']['retained_coordinates']) if retained else ports
    rows = tuple(names.index(name) for name in terminals) if retained else tuple(range(len(ports)))
    selected = tuple(nodes.index(name) for name in terminals) if retained else tuple(
        int(Bpattern.rows[Bpattern.cols == index][0]) for index in range(len(ports))) if not routes else ()
    # Declaration-only potential; actual physical rank is decided per point.
    realizable = bool(terminals) and len(terminals) <= len(ports)
    mask = np.ones(len(ports))
    for port in (declaration.get('ptc') or {}).get('selected_ports', ()):
        mask[ports.index(port)] = 0
    return ViewProgram(model_program, nodes, terminals, selected,
                       tuple(index for index in range(len(nodes)) if index not in set(selected)),
                       tuple(routes), tuple(patterns), series,
                       _immutable(np.asarray([B[model_program.node_ids.index(name)] for name in original_names])),
                       rows, retained, realizable, _immutable(mask), canonical_json_bytes(declaration))
