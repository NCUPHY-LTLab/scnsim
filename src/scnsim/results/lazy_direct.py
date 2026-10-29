"""Lazy fixed-reader projections for direct matrix Results."""

from __future__ import annotations

from collections.abc import Iterator
from types import MappingProxyType
from typing import Any

import numpy as np

from .. import units
from ..errors import EvidenceIntegrityError
from ..value_storage import immutable_array, immutable_quantity
from .base import ResultIdentity, _LazyResultMapping
from .derived import TraceResult
from .factory import _verified_result


class DirectTraceMapping(_LazyResultMapping):
    """Read one declared S-parameter trace per mapping access, without a cache."""

    __slots__ = ("_reader", "_channels", "_declarations", "_frequencies", "_identity", "_presentation")

    def __init__(
        self,
        *,
        reader: object,
        channels: tuple[tuple[str, tuple[int, ...]], ...],
        declarations: tuple[dict[str, Any], ...],
        frequencies: np.ndarray,
        identity: ResultIdentity,
        presentation: dict[str, object],
    ) -> None:
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_channels", tuple(channels))
        object.__setattr__(self, "_declarations", tuple(
            MappingProxyType(dict(item)) for item in declarations
        ))
        object.__setattr__(self, "_frequencies", immutable_array(frequencies))
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_presentation", MappingProxyType(dict(presentation)))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("direct trace mapping is immutable")

    def __getitem__(self, key: str) -> TraceResult:
        declaration = next(
            (item for item in self._declarations if item.get("id") == key), None
        )
        if declaration is None:
            raise KeyError(key)
        output_index = tuple(channel[0] for channel in self._channels).index(
            declaration["output_port"]
        )
        input_index = tuple(channel[0] for channel in self._channels).index(
            declaration["input_port"]
        )
        selected = self._reader.project_matrix(
            "S",
            self._channels[input_index],
            self._channels[output_index],
            slice(0, self._frequencies.size),
        )
        frequencies = np.asarray(selected["frequencies"], dtype=np.float64)
        values = np.asarray(selected["values"])
        if frequencies.shape != self._frequencies.shape or values.shape != self._frequencies.shape:
            raise EvidenceIntegrityError(
                "Selected direct trace differs from its fixed frequency axis",
                stage="result_decode",
                evidence={"trace_id": key},
            )
        if not np.array_equal(frequencies, self._frequencies):
            raise EvidenceIntegrityError(
                "Selected direct trace frequency values differ from its fixed axis",
                stage="result_decode",
                evidence={"trace_id": key},
            )
        return _verified_result(
            TraceResult,
            frequencies=units.registry.Quantity(self._frequencies, "hertz"),
            value=immutable_quantity(units.registry.Quantity(values, "dimensionless")),
            _parent_identity=self._identity,
            _presentation=dict(
                self._presentation,
                id=key,
                family="S",
                input_channel={"coordinate": self._channels[input_index][0], "mode": []},
                output_channel={"coordinate": self._channels[output_index][0], "mode": []},
            ),
        )

    def __iter__(self) -> Iterator[str]:
        return iter(tuple(item["id"] for item in self._declarations))

    def __len__(self) -> int:
        return len(self._declarations)

    def __repr__(self) -> str:
        return f"DirectTraceMapping(ids={tuple(iter(self))!r})"
