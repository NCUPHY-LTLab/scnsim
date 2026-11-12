"""Shared helpers for the Memory/Workspace public-surface study.

Run every script from the repository root with `uv run --locked python <script>`.
Only public SCNSim calls are used to produce Results; the helpers below only
describe the public attributes of what those calls return.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np

from scnsim import (CircuitPlan, CircuitRun, ParameterDefinitions, ParameterSpec, components,
                    units as u)
from scnsim.numeric_encoding import array_from_record


def is_array_record(value):
    return isinstance(value, Mapping) and set(value) == {"dtype", "shape", "data_hex"}


def array_of(value):
    """Return the ndarray behind an array-like public value, or None."""
    if is_array_record(value):
        return array_from_record(dict(value))
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        return np.asarray(value.magnitude)
    return None


def tree(value):
    """Exact, JSON-able projection of a public value (bits of every number)."""
    array = array_of(value)
    if array is not None:
        row = {"dtype": array.dtype.str, "shape": list(array.shape), "bytes": array.tobytes().hex()}
        if hasattr(value, "units"):
            row["unit"] = str(value.units)
        return row
    if isinstance(value, Mapping):
        return {str(key): tree(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)) and not hasattr(value, "_fields"):
        return [tree(child) for child in value]
    if isinstance(value, (bytes, bytearray)):
        return {"raw_bytes": bytes(value).hex()}
    if isinstance(value, (float, np.floating)):
        return {"f64": struct.pack(">d", float(value)).hex()}
    if isinstance(value, complex):
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, np.generic):
        return tree(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if is_dataclass(value):
        return {"dataclass": type(value).__name__,
                "fields": {f.name: tree(getattr(value, f.name)) for f in fields(value)
                           if not f.name.startswith("_")}}
    if hasattr(value, "_fields"):
        return {"namedtuple": type(value).__name__,
                "fields": {name: tree(getattr(value, name)) for name in value._fields}}
    return {"repr": type(value).__name__}


def array_bytes(value):
    """Total bytes of every numeric array reachable in a public value."""
    array = array_of(value)
    if array is not None:
        return int(array.nbytes)
    if isinstance(value, Mapping):
        return sum(array_bytes(child) for child in value.values())
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if hasattr(value, "_fields"):
        return sum(array_bytes(getattr(value, name)) for name in value._fields)
    if isinstance(value, (tuple, list)):
        return sum(array_bytes(child) for child in value)
    if is_dataclass(value):
        return sum(array_bytes(getattr(value, f.name)) for f in fields(value) if not f.name.startswith("_"))
    return 0


def json_bytes(value):
    """Length of the exact JSON projection: a storage-independent size measure."""
    return len(json.dumps(tree(value), sort_keys=True, separators=(",", ":")))


def keys_of(value):
    return sorted(str(key) for key in value) if isinstance(value, Mapping) else None


def describe_mapping(value):
    if not isinstance(value, Mapping):
        return {"type": type(value).__name__}
    return {"key_count": len(value), "keys": keys_of(value),
            "array_bytes": array_bytes(value), "json_bytes": json_bytes(value)}


def failure_row(error, *, with_tree=True):
    """Public facts of a typed failure, raised or returned."""
    if error is None:
        return None
    evidence = getattr(error, "evidence", None)
    row = {"type": type(error).__module__ + "." + type(error).__qualname__,
           "kind": getattr(error, "kind", None), "category": getattr(error, "category", None),
           "stage": getattr(error, "stage", None), "message": str(error)}
    if isinstance(evidence, Mapping):
        row["evidence_keys"] = keys_of(evidence)
        row["evidence_key_count"] = len(evidence)
        row["evidence_array_bytes"] = array_bytes(evidence)
        row["evidence_json_bytes"] = json_bytes(evidence)
        if with_tree:
            row["evidence_tree"] = tree(evidence)
    else:
        row["evidence_keys"] = None
    return row


def public_names(value):
    """Public attribute names of a Result object: dataclass fields and properties."""
    names = []
    if is_dataclass(value):
        names += [f.name for f in fields(value) if not f.name.startswith("_")]
    for klass in type(value).__mro__:
        for name, member in vars(klass).items():
            if isinstance(member, property) and not name.startswith("_") and name not in names:
                names.append(name)
    return names


# ---------------------------------------------------------------- fixtures
def primitive_plan(*, plan_id="primitive_full_v1", capacitance=110.0):
    """The two-node fixture of tests/test_workspace_readback.py (_primitive_plan)."""
    plan = CircuitPlan(id=plan_id)
    inputs = ParameterDefinitions(id="primitive_inputs")
    capacitance_ref = inputs.parameter(id="capacitance", baseline=capacitance * u.fF,
                                       spec=ParameterSpec(unit=u.fF))
    capacitor = plan.add(components.capacitor(id="resonator_cap", capacitance=capacitance_ref))
    inductor = plan.add(components.inductor(id="resonator_ind", inductance=5.8 * u.nH))
    node = plan.bus(id="resonator_node")
    plan.parallel(id="parallel_lc", start=node, branches=((capacitor,), (inductor,)), end=plan.ground)
    boundary = plan.bus(id="boundary")
    coupling = plan.add(components.capacitor(id="coupling_cap", capacitance=6.0 * u.fF))
    plan.series(id="coupler", start=boundary, elements=(coupling,), end=node)
    plan.add_port(id="signal_in", at=boundary, role="terminated", reference_impedance=50.0 * u.ohm)
    return plan, node.node, capacitance_ref


def coupled_plan(*, capacitance=80.0):
    """Two LC tanks joined by a small capacitor (tests/test_evidence_v4.py, _coupled_plan)."""
    inputs = ParameterDefinitions(id="coupled_inputs")
    capacitance_ref = inputs.parameter(id="capacitance", baseline=capacitance * u.fF,
                                       spec=ParameterSpec(unit=u.fF))
    plan = CircuitPlan(id="coupled_tanks")
    a, b = plan.bus(id="a"), plan.bus(id="b")
    ca = plan.add(components.capacitor(id="c_a", capacitance=capacitance_ref))
    la = plan.add(components.inductor(id="l_a", inductance=8.0 * u.nH))
    plan.parallel(id="lc_a", start=a, branches=((ca,), (la,)), end=plan.ground)
    cb = plan.add(components.capacitor(id="c_b", capacitance=82.0 * u.fF))
    lb = plan.add(components.inductor(id="l_b", inductance=8.0 * u.nH))
    plan.parallel(id="lc_b", start=b, branches=((cb,), (lb,)), end=plan.ground)
    cross = plan.add(components.capacitor(id="cross", capacitance=2.0 * u.fF))
    plan.series(id="cross_branch", start=a, elements=(cross,), end=b)
    return plan, (a.node, b.node), capacitance_ref


def open_run(plan, storage, directory=None):
    if storage == "memory":
        return CircuitRun(plan=plan, storage="memory", precision="float64")
    return CircuitRun(plan=plan, storage="workspace", workspace=Path(directory), precision="float64")
