"""Stored form of the work evidence that one parameter-sweep record carries.

A sweep point of Direct, of the operator or of a response element carries the
work evidence of every frequency it solved: one row per work frequency, in
work order. The producer hands over that row form and a reader returns it.

Stored, the rows are gathered into runs. A run is a stretch of consecutive
work rows that ran the same checked equations. It holds each work counter and
each certificate field once, as one array over its rows, in the columnar
layout of ``direct_summary``. The rows of one sweep point normally run one
frequency function and form one run, so the number of stored arrays depends
on the checked equations and never on the number of frequencies.

Only the arrangement is defined here. Each value keeps the dtype, shape and
bits of the row it was gathered from, so the row form is recovered exactly.
No solver runs while gathering or reading.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import groupby

import numpy as np

from .direct_summary import (CERTIFICATE_FIELDS, CERTIFICATE_IDENTITY, WORK_COUNTERS,
                             work_evidence_columns, work_evidence_rows)
from .validation.common import _integrity

# Rows of the operator and of a response element also name their view and the
# matrix families they asked for. A run stores them once.
RUN_CONSTANTS = ("view_index", "demand")
_COLUMN_KEYS = frozenset((*WORK_COUNTERS, "certificates"))


def stored_sweep_evidence(evidence):
    """Return a sweep record's evidence with its work rows gathered into runs.

    Evidence without work rows, and evidence that already holds runs, is
    returned as it is. A repeated sweep point reuses the gathered evidence of
    its first occurrence, so gathering must not be applied twice.
    """
    if not isinstance(evidence, Mapping):
        raise TypeError("Sweep evidence must be a mapping")
    if "work_evidence" not in evidence or _holds_runs(evidence["work_evidence"]):
        return evidence
    return {**evidence, "work_evidence": work_evidence_runs(evidence["work_evidence"])}


def read_sweep_evidence(evidence):
    """Return stored sweep evidence with its runs expanded into work rows again."""
    if not isinstance(evidence, Mapping):
        raise _integrity("Sweep evidence body is malformed.")
    if "work_evidence" not in evidence:
        return evidence
    return {**evidence, "work_evidence": work_evidence_run_rows(evidence["work_evidence"])}


def work_evidence_runs(rows):
    """Gather ordered work rows into runs of consecutive rows that share one layout."""
    runs = []
    for (constants, _), members in groupby(rows, key=_row_layout):
        columns = work_evidence_columns(
            {name: value for name, value in row.items() if name not in RUN_CONSTANTS}
            for row in members)
        runs.append({**constants, **columns})
    return runs


def work_evidence_run_rows(runs):
    """Return the ordered work rows that stored runs hold."""
    if not isinstance(runs, (list, tuple)):
        raise _integrity("Sweep work evidence is not an ordered list of runs.")
    rows = []
    for run in runs:
        if not isinstance(run, Mapping) or set(run) - set(RUN_CONSTANTS) != _COLUMN_KEYS:
            raise _integrity("Sweep work evidence run is not the closed columnar layout.")
        constants = {name: run[name] for name in RUN_CONSTANTS if name in run}
        if constants and not _valid_constants(constants):
            raise _integrity("Sweep work evidence run names a malformed view or demand.")
        members = work_evidence_rows({name: run[name] for name in _COLUMN_KEYS})
        if not members:
            raise _integrity("Sweep work evidence run holds no work row.")
        # Each returned row owns its copy of the run's view and demand.
        rows.extend({**row, **{name: _plain(value) for name, value in constants.items()}}
                    for row in members)
    return rows


def _holds_runs(work):
    if (not isinstance(work, (list, tuple))
            or any(not isinstance(item, Mapping) for item in work)):
        raise TypeError("Sweep work evidence must be an ordered sequence of mappings")
    # A work row counts with plain integers; a gathered run holds one array per counter.
    gathered = set()
    for item in work:
        counter = item.get("work_index")
        if type(counter) is not int and not isinstance(counter, np.ndarray):
            raise TypeError("Sweep work evidence entry is neither a work row nor a gathered run")
        gathered.add(isinstance(counter, np.ndarray))
    if len(gathered) > 1:
        raise TypeError("Sweep work evidence mixes work rows and gathered runs")
    return gathered != {False}


def _row_layout(row):
    """Name what one work row is, apart from its counters and certificate values."""
    if not isinstance(row, Mapping):
        raise TypeError("Sweep work row must be a mapping")
    constants = {name: _plain(row[name]) for name in RUN_CONSTANTS if name in row}
    if constants and not _valid_constants(constants):
        raise TypeError("Sweep work row names a malformed view or demand")
    return constants, _certificate_layout(row)


def _valid_constants(constants):
    if set(constants) != set(RUN_CONSTANTS):
        return False
    view, demand = constants["view_index"], constants["demand"]
    return (type(view) is int and view >= 0 and isinstance(demand, (list, tuple))
            and all(isinstance(family, str) for family in demand))


def _certificate_layout(row):
    """Name what one row's certificates are, apart from their values."""
    certificates = row.get("certificates")
    if not isinstance(certificates, (list, tuple)):
        raise TypeError("Sweep work row does not have the closed fields")
    layout = []
    for certificate in certificates:
        if not isinstance(certificate, Mapping):
            raise TypeError("Sweep certificate row does not have the closed fields")
        try:
            identity = [_plain(certificate[name]) for name in CERTIFICATE_IDENTITY]
            fields = [(certificate[name]["dtype"], list(certificate[name]["shape"]))
                      for name in CERTIFICATE_FIELDS]
        except (KeyError, TypeError) as error:
            raise TypeError("Sweep certificate row does not have the closed fields") from error
        layout.append((identity, fields))
    return layout


def _plain(value):
    return [_plain(item) for item in value] if isinstance(value, (tuple, list)) else value
