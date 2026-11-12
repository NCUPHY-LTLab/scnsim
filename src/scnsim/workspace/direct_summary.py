"""Columnar layout of the work evidence in one Direct chunk summary.

Every work row of a Direct chunk runs the same checked equations. The summary
therefore keeps each work counter and each certificate field once, as one
array whose leading axis is the chunk's ordered work rows. How many arrays a
summary holds depends on the equations, never on how many frequency points
the chunk has.

Only the arrangement is defined here. Each value keeps the dtype, shape and
bits of the per-row record it was gathered from, so the row form is recovered
exactly. No solver runs while gathering or reading.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..numeric_encoding import array_from_record, array_record
from .validation.common import _integrity

WORK_COUNTERS = ("work_index", "source_index", "check_status", "certificate_status",
                 "check_position", "certificate_position")
CERTIFICATE_IDENTITY = ("site", "role", "stage", "derivative_order")
CERTIFICATE_FIELDS = ("status", "eta", "residual_numerator", "residual_denominator",
                      "performed", "present")
_ROW_KEYS = frozenset((*WORK_COUNTERS, "certificates"))
_CERTIFICATE_KEYS = frozenset((*CERTIFICATE_IDENTITY, *CERTIFICATE_FIELDS))


def work_evidence_columns(rows):
    """Gather the ordered work rows of one chunk into one array per field."""
    rows = tuple(rows)
    if not rows:
        raise ValueError("a Direct chunk summary needs at least one work row")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
            raise TypeError("Direct work row does not have the summary's closed fields")
        for certificate in row["certificates"]:
            if not isinstance(certificate, Mapping) or set(certificate) != _CERTIFICATE_KEYS:
                raise TypeError("Direct certificate row does not have the summary's closed fields")
    layout = _certificate_identities(rows[0])
    if any(_certificate_identities(row) != layout for row in rows[1:]):
        raise RuntimeError("Direct work rows of one chunk do not share one certificate layout")
    columns = {name: _counter_column(tuple(row[name] for row in rows)) for name in WORK_COUNTERS}
    columns["certificates"] = [
        {**dict(zip(CERTIFICATE_IDENTITY, identity)),
         **{name: _stacked(tuple(row["certificates"][position][name] for row in rows))
            for name in CERTIFICATE_FIELDS}}
        for position, identity in enumerate(layout)
    ]
    return columns


def work_evidence_rows(columns):
    """Return the per-row work evidence that a stored columnar summary holds."""
    if not isinstance(columns, Mapping) or set(columns) != _ROW_KEYS:
        raise _integrity("Direct chunk summary work evidence is not the closed columnar layout.")
    counters = {name: _stored_array(columns[name]) for name in WORK_COUNTERS}
    shape = counters["work_index"].shape
    if len(shape) != 1 or any(column.dtype != np.dtype("<i8") or column.shape != shape
                              for column in counters.values()):
        raise _integrity("Direct chunk summary work counters are not one integer per work row.")
    count = shape[0]
    certificates = columns["certificates"]
    if not isinstance(certificates, (list, tuple)):
        raise _integrity("Direct chunk summary certificates are malformed.")
    fields = []
    for certificate in certificates:
        if not isinstance(certificate, Mapping) or set(certificate) != _CERTIFICATE_KEYS:
            raise _integrity("Direct chunk summary certificate is not the closed columnar layout.")
        arrays = {name: _stored_array(certificate[name]) for name in CERTIFICATE_FIELDS}
        if any(array.shape[:1] != (count,) for array in arrays.values()):
            raise _integrity("Direct chunk summary certificate field does not cover its work rows.")
        fields.append((certificate, arrays))
    return [
        {**{name: int(column[position]) for name, column in counters.items()},
         "certificates": [
             {**{name: _plain(certificate[name]) for name in CERTIFICATE_IDENTITY},
              **{name: array_record(array[position], exact_dtype=True)
                 for name, array in arrays.items()}}
             for certificate, arrays in fields
         ]}
        for position in range(count)
    ]


def _certificate_identities(row):
    return tuple(tuple(_plain(certificate[name]) for name in CERTIFICATE_IDENTITY)
                 for certificate in row["certificates"])


def _plain(value):
    # A site is an ordered tuple before encoding and a list after decoding;
    # each returned row also owns its copy, as the per-row records did.
    return [_plain(item) for item in value] if isinstance(value, (tuple, list)) else value


def _counter_column(values):
    if any(type(value) is not int for value in values):
        raise TypeError("Direct work counters must be plain integers")
    return np.asarray(values, dtype="<i8")


def _stacked(records):
    arrays = tuple(array_from_record(record) for record in records)
    first = arrays[0]
    if any(array.dtype != first.dtype or array.shape != first.shape for array in arrays[1:]):
        raise RuntimeError("Direct work rows of one chunk do not share one certificate layout")
    return np.stack(arrays)


def _stored_array(record):
    if not isinstance(record, Mapping) or set(record) != {"dtype", "shape", "data_hex"}:
        raise _integrity("Direct chunk summary column is not a typed array.")
    try:
        return array_from_record(record)
    except (KeyError, TypeError, ValueError) as error:
        raise _integrity("Direct chunk summary column bytes differ from their declared type.") from error
