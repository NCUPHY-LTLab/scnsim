"""Factorized immutable output axes, with lazily projected scalar row metadata.

Compilation declares blocks once. Row access creates only selected metadata;
compact identity/storage encoding never walks every frequency/matrix element.
Numerical arrays, status, units conversion and execution remain other owners.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence, Mapping
from dataclasses import dataclass
from math import prod
from operator import index as integer_index
from typing import Literal

from .program import OutputAxis


@dataclass(frozen=True, slots=True)
class OutputRow:
    row: int
    observable_index: int
    quantity_key: str
    index: tuple[int, ...]
    component: Literal['real', 'imag']
    si_unit: str


@dataclass(frozen=True, slots=True)
class OutputBlock:
    observable_index: int
    quantity_key: str
    shape: tuple[int, ...]
    components: tuple[Literal['real', 'imag'], ...]
    si_unit: str
    offset: int

    def __post_init__(self):
        object.__setattr__(self, 'shape', tuple(self.shape))
        object.__setattr__(self, 'components', tuple(self.components))
        if self.components not in (('real',), ('real', 'imag')):
            raise ValueError('components must be real or real/imag in that order')
        if any(integer_index(size) < 0 for size in self.shape):
            raise ValueError('output shape dimensions must be nonnegative')

    @property
    def row_count(self):
        return prod(self.shape) * len(self.components)

    def compact_record(self):
        return {'observable_index': self.observable_index, 'quantity_key': self.quantity_key,
                'shape': self.shape, 'components': self.components,
                'si_unit': self.si_unit, 'offset': self.offset}


def _physical_index(flat, shape):
    coordinates = []
    for size in reversed(shape):
        flat, coordinate = divmod(flat, size)
        coordinates.append(coordinate)
    return tuple(reversed(coordinates))


@dataclass(frozen=True, slots=True, init=False)
class FactorizedRows(Sequence[OutputRow]):
    """Readonly row Sequence backed by one record per physical output block."""

    blocks: tuple[OutputBlock, ...]
    _starts: tuple[int, ...]
    _length: int

    def __init__(self, blocks):
        frozen = tuple(OutputBlock(**block) if isinstance(block, Mapping) else block
                       for block in blocks)
        length = 0
        for block in frozen:
            if block.offset != length:
                raise ValueError('output block offsets must be contiguous and ordered')
            length += block.row_count
        object.__setattr__(self, 'blocks', frozen)
        object.__setattr__(self, '_starts', tuple(block.offset for block in frozen))
        object.__setattr__(self, '_length', length)

    def __len__(self):
        return self._length

    def __getitem__(self, selected):
        if isinstance(selected, slice):
            return tuple(self[position] for position in range(*selected.indices(self._length)))
        position = integer_index(selected)
        if position < 0:
            position += self._length
        if position < 0 or position >= self._length:
            raise IndexError('output row index out of range')
        block = self.blocks[bisect_right(self._starts, position) - 1]
        physical, component = divmod(position - block.offset, len(block.components))
        return OutputRow(position, block.observable_index, block.quantity_key,
                         _physical_index(physical, block.shape),
                         block.components[component], block.si_unit)

    def row_index(self, quantity_key, index=(), component='real', *, observable_index=None):
        """Locate one row directly from its block, without expanding row metadata."""
        for block in self.blocks:
            if block.quantity_key != quantity_key or (observable_index is not None and
                                                       block.observable_index != observable_index):
                continue
            if len(index) != len(block.shape):
                raise IndexError('output coordinate dimensionality differs from its block')
            flat = 0
            for coordinate, size in zip(index, block.shape, strict=True):
                coordinate = integer_index(coordinate)
                if coordinate < 0 or coordinate >= size:
                    raise IndexError('output physical coordinate out of range')
                flat = flat * size + coordinate
            return block.offset + flat * len(block.components) + block.components.index(component)
        raise KeyError((quantity_key, observable_index))

    def compact_record(self):
        return tuple(block.compact_record() for block in self.blocks)


@dataclass(frozen=True, slots=True)
class OutputLayout:
    axes: tuple[OutputAxis, ...]
    rows: FactorizedRows
    flatten_order: str = 'physical_C_order_then_component_real_imag'

    def __post_init__(self):
        object.__setattr__(self, 'axes', tuple(self.axes))
        if not isinstance(self.rows, FactorizedRows):
            raise TypeError('rows must be immutable FactorizedRows')

    @classmethod
    def from_blocks(cls, axes, blocks):
        return cls(tuple(axes), FactorizedRows(blocks))

    @property
    def blocks(self):
        return self.rows.blocks

    def row_index(self, quantity_key, index=(), component='real', *, observable_index=None):
        return self.rows.row_index(quantity_key, index, component,
                                   observable_index=observable_index)

    def compact_record(self):
        """Factorized identity/layout codec: no access or iteration of rows."""
        return {'axes': tuple({'quantity_key': axis.quantity_key, 'component': axis.component,
                              'si_unit': axis.si_unit, 'coordinates': axis.coordinates,
                              'shape': axis.shape} for axis in self.axes),
                'blocks': self.rows.compact_record(), 'flatten_order': self.flatten_order}
