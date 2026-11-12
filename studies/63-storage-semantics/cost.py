"""Measure what `selected_state` and `null_vector` cost per root, by circuit size.

Run from the repository root:

    uv run --locked python cost.py <output.json> <kind> <size> [<kind> <size> ...]

kind = ladder: a chain of <size> grounded LC tanks joined by 2 fF capacitors, built
       like the ring of benchmarks/circuits.py (`_tank`, `_couple`); two detuned
       tanks at one end are retained.
kind = line:   the ordered two-conductor line of benchmarks/circuits.py (`_c4`) with
       <size> sections; two of its Port coordinates are retained.

For a diagonal root and a hybridized pole on the two retained coordinates, the
script records the bytes of each key of the public Workspace evidence, the bytes
of the compact in-memory record (`result.evaluation.evidence_bytes`) and, read
directly from the SQLite file (not a public call), the stored bytes by role.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scnsim as s
from lib import array_bytes, failure_row, json_bytes, keys_of, open_run, u

sys.path.insert(0, os.getcwd())
from benchmarks.circuits import _c4, _couple, _tank


def ladder(size):
    inputs = s.ParameterDefinitions(id="study_ladder")
    cap = inputs.parameter(id="mode0_capacitance", baseline=110.0 * u.fF, spec=s.ParameterSpec(unit=u.fF))
    plan = s.CircuitPlan(id="study_ladder_%d" % size)
    nodes = []
    for i in range(size):
        if i == 0:
            nodes.append(_tank(s, plan, "mode0", cap, 5.8))
        elif i == 1:
            nodes.append(_tank(s, plan, "mode1", 105.0 * u.fF, 5.8))
        else:
            # The rest of the chain sits near 4.6 GHz, away from the two retained tanks.
            nodes.append(_tank(s, plan, "mode" + str(i), (150.0 + 0.01 * i) * u.fF, 8.0))
    for i in range(size - 1):
        _couple(s, plan, "link" + str(i), nodes[i], nodes[i + 1], 2.0)
    return plan, (nodes[0].node, nodes[1].node), cap, size


def line(sections):
    plan, length, ports, _, declared = _c4(s, sections)
    return plan, (ports[0], ports[1]), length, None


def _selected_states(value):
    """Every `selected_state` mapping inside one decoded stored object."""
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "selected_state":
                yield child
            else:
                yield from _selected_states(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _selected_states(child)


def stored_bytes(directory):
    """Stored objects by role, read from the SQLite file with the product's block decoder."""
    from scnsim.workspace.binary_blocks import record_document
    paths = list(Path(directory).rglob("operations.sqlite3"))
    connection = sqlite3.connect(paths[0].absolute().as_uri() + "?mode=ro", uri=True)
    roles = {}
    try:
        for role, payload in connection.execute(
                "SELECT r.role, o.payload FROM objects o JOIN object_roles r USING(sha256)"):
            payload = bytes(payload)
            row = roles.setdefault(role, {"objects": 0, "bytes": 0, "selected_state_count": 0,
                                          "selected_state_array_bytes": 0})
            row["objects"] += 1
            row["bytes"] += len(payload)
            try:
                document = record_document(payload)
            except Exception:
                continue
            for state in _selected_states(document):
                row["selected_state_count"] += 1
                row["selected_state_array_bytes"] += array_bytes(state)
    finally:
        connection.close()
    return roles


def evidence_cost(evidence):
    per_key = {str(key): array_bytes(value) for key, value in evidence.items()}
    return {"key_count": len(evidence), "array_bytes": sum(per_key.values()), "json_bytes": json_bytes(evidence),
            "selected_state_array_bytes": per_key.get("selected_state"),
            "selected_state_json_bytes": json_bytes(evidence["selected_state"]) if "selected_state" in evidence else None,
            "selected_state_fields": ({str(k): array_bytes(v) for k, v in evidence["selected_state"].items()}
                                      if "selected_state" in evidence else None),
            "null_vector_array_bytes": per_key.get("null_vector"),
            "other_array_bytes": sum(v for k, v in per_key.items() if k not in ("selected_state", "null_vector"))}


def one(kind, size, quantity):
    builder = ladder if kind == "ladder" else line
    row = {"kind": kind, "size": size, "quantity": quantity}
    for storage in ("memory", "workspace"):
        with TemporaryDirectory() as directory:
            plan, coordinates, parameter, _ = builder(size)
            started = time.perf_counter()
            run = open_run(plan, storage, directory)
            try:
                view = run.original.reduce(s.ReductionPipeline().retain(*coordinates))
                anchor = 6.2 if kind == "ladder" else 30.0
                spec = (s.DiagonalRootSpec(coordinate=coordinates[0], root_hint=anchor * u.GHz)
                        if quantity == "diagonal_root" else
                        s.HybridizedPoleSpec(coordinates=coordinates, anchor=anchor * u.GHz))
                try:
                    result = run.evaluate(view, spec)
                    outcome, failure = "returned", getattr(result, "failure", None)
                except s.SCNSimError as error:
                    result, outcome, failure = None, "raised", error
            finally:
                run.close()
            entry = {"outcome": outcome, "seconds": round(time.perf_counter() - started, 1)}
            if failure is not None:
                entry["failure"] = {key: value for key, value in failure_row(failure, with_tree=False).items()
                                    if key in ("type", "kind", "stage", "evidence_key_count")}
                entry["failure_evidence"] = evidence_cost(failure.evidence)
            if result is not None and storage == "workspace":
                entry["public_evidence"] = evidence_cost(result.evidence)
                entry["root_hz"] = float(result.frequency.to("hertz").magnitude)
            if result is not None and storage == "memory":
                entry["public_evidence_key_count"] = len(result.evidence)
                document = json.loads(result.evaluation.evidence_bytes)
                entry["evaluation_evidence_bytes_len"] = len(result.evaluation.evidence_bytes)
                entry["evaluation_evidence"] = evidence_cost(document)
                entry["evaluation_null_vector_bytes"] = (None if result.evaluation.null_vector is None
                                                         else int(result.evaluation.null_vector.nbytes))
                if result.failure is None:
                    entry["root_hz"] = float(result.frequency.to("hertz").magnitude)
            if storage == "workspace":
                entry["stored"] = stored_bytes(directory)
                entry["stored_total_bytes"] = sum(item["bytes"] for item in entry["stored"].values())
            row[storage] = entry
    return row


def main():
    output = Path(sys.argv[1])
    pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
    rows = json.loads(output.read_text()) if output.exists() else []
    for kind, size in pairs:
        for quantity in ("diagonal_root", "hybridized_pole"):
            row = one(kind, int(size), quantity)
            rows.append(row)
            workspace = row["workspace"]
            cost = workspace.get("public_evidence") or workspace.get("failure_evidence")
            print(kind, size, quantity, workspace["outcome"], workspace["seconds"], "s",
                  "selected_state", cost and cost["selected_state_array_bytes"],
                  "null_vector", cost and cost["null_vector_array_bytes"],
                  "other", cost and cost["other_array_bytes"], flush=True)
            output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
