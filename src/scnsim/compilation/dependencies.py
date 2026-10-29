"""Declaration-only dependency projection of the physical program's bindings.

No mesh realization, JAX/device initialization, Workspace or numerical solve is
needed. Rows describe declared consumers, not a claim of nonzero sensitivity;
physical cancellation and disconnected-output sparsity are numerical questions.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from ..canonical import canonical_json_bytes
from .compiler import parameter_key


@dataclass(frozen=True, slots=True)
class DependencyRow:
    parameter_key: tuple[str, str]
    parameter_bytes: bytes
    field_paths: tuple[tuple[tuple[str, ...], str], ...]
    view_paths: tuple[tuple[str, ...], ...]
    quantity_paths: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class DependencyProjection:
    declaration_sha256: str
    rows: tuple[DependencyRow, ...]


def _sources(binding):
    kind = binding['kind']
    if kind == 'ref':
        yield binding['parameter']
    elif kind == 'affine':
        yield binding['input']
    elif kind == 'scalar_rlgc':
        for term in binding['terms']:
            yield from _sources(term['source'])
    elif kind != 'constant':
        raise TypeError(f'unsupported physical binding {kind}')


def _named_paths(value, path=()):
    if isinstance(value, Mapping):
        if 'view' in value:
            yield 'view', path + ('view',)
        if 'type' in value and value['type'] in (
                'direct_solve', 'operator', 'diagonal_root', 'operator_element_root',
                'hybridized_pole', 'transfer_zero', 'residue_normalized_coupling',
                'response_element'):
            yield 'quantity', path
        for key, item in value.items():
            yield from _named_paths(item, path + (str(key),))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _named_paths(item, path + (str(index),))


def describe_dependencies(plan_document, quantity_declarations, parameter_set):
    """Immutable ordered ref/field/View/Quantity declaration inventory.

    Every field's declared ref is retained even at a zero derivative point.
    Quantity/View paths identify consumers of the shared network program; this
    static inventory does not prune a ref because one observed scalar is zero.
    """
    consumers = {}
    for leaf in plan_document['physical_leaves']:
        for field in leaf['fields']:
            for ref in _sources(field['binding']):
                key = parameter_key(ref)
                consumers.setdefault(key, []).append((tuple(leaf['path']), field['id']))
    paths = tuple(_named_paths(quantity_declarations))
    views = tuple(dict.fromkeys(path for kind, path in paths if kind == 'view'))
    quantities = tuple(dict.fromkeys(path for kind, path in paths if kind == 'quantity'))
    rows = []
    for binding in parameter_set['bindings']:
        ref = binding['parameter']
        key = parameter_key(ref)
        if key in consumers:
            rows.append(DependencyRow(key, canonical_json_bytes(ref),
                                      tuple(dict.fromkeys(consumers[key])), views, quantities))
    # Point values and numerical/mesh/executable identities are deliberately
    # absent: this identity describes declarations, not a candidate evaluation.
    document = dict(plan=plan_document, quantities=quantity_declarations,
                    parameter_refs=[binding['parameter'] for binding in parameter_set['bindings']])
    return DependencyProjection(sha256(canonical_json_bytes(document)).hexdigest(), tuple(rows))
