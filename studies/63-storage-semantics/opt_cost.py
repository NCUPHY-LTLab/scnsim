"""How many root bodies an optimization of a root-type objective keeps, and their bytes.

Run from the repository root:

    uv run --locked python opt_cost.py <output.json> <objective> [<objective> ...]

objective = root | pole | coupling. The optimizer is the CMA-ES workload of
benchmarks/circuits.py (seed 7, population 8, 64 evaluations). Memory figures come
from the public Result; Workspace stored bytes are read directly from the SQLite
file (not a public call) with the product's block decoder.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scnsim as s
from cost import stored_bytes
from lib import array_bytes, coupled_plan, json_bytes, open_run, primitive_plan, u


def request(objective, run, coordinates, parameter):
    if objective == "root":
        view = run.original.reduce(s.ReductionPipeline().retain(coordinates))
        quantity = s.DiagonalRootSpec(coordinate=coordinates, root_hint=6.0 * u.GHz).frequency
        target, bounds = 6.2 * u.GHz, (80.0, 140.0)
    elif objective == "pole":
        view = run.original.reduce(s.ReductionPipeline().retain(*coordinates))
        quantity = s.HybridizedPoleSpec(coordinates=coordinates, anchor=6.1 * u.GHz).frequency
        target, bounds = 6.2 * u.GHz, (70.0, 90.0)
    else:
        view = run.original.reduce(s.ReductionPipeline().retain(*coordinates))
        quantity = s.ResidueNormalizedCouplingSpec(
            branch_a=s.DiagonalRootSpec(coordinate=coordinates[0], root_hint=6.2 * u.GHz),
            branch_b=s.DiagonalRootSpec(coordinate=coordinates[1], root_hint=6.2 * u.GHz),
            frequency=5.5 * u.GHz).magnitude
        target, bounds = 2.0e8 * u.rad / u.s, (70.0, 90.0)
    spec = s.OptimizationSpec(
        variables=(s.OptimizationVariable(parameter=parameter, bounds=(bounds[0] * u.fF, bounds[1] * u.fF)),),
        objectives=(s.CostObjective(id="objective", quantity=quantity, target=target,
                                    weight=1.0 * u.dimensionless),),
        optimizer=s.CMAESSpec(seed=7, population_size=8, max_evaluations=64))
    return view, spec


def one(objective):
    row = {"objective": objective}
    fixture = primitive_plan if objective == "root" else coupled_plan
    for storage in ("memory", "workspace"):
        with TemporaryDirectory() as directory:
            plan, coordinates, parameter = fixture()
            run = open_run(plan, storage, directory)
            started = time.perf_counter()
            try:
                view, spec = request(objective, run, coordinates, parameter)
                result = run.optimize(view, spec, progress=False)
            finally:
                run.close()
            entry = {"seconds": round(time.perf_counter() - started, 1), "class": type(result).__name__}
            if storage == "memory":
                history = result.history
                bodies = [body for trial in history for body in trial.bodies.values()]
                kinds = Counter()
                for body in bodies:
                    document = json.loads(body.evidence_bytes)
                    kinds["root packet" if "omega" in document and "steps" in document else
                          "coupling packet" if "branches" in document or "residue" in json.dumps(sorted(document)) else
                          "other"] += 1
                entry.update(
                    history_rows=len(history), trials=len(result.trials),
                    bodies=len(bodies), bodies_per_row=sorted(Counter(len(trial.bodies) for trial in history).items()),
                    body_kinds=dict(kinds),
                    evidence_bytes_total=sum(len(body.evidence_bytes) for body in bodies),
                    evidence_bytes_per_body=sorted(Counter(len(body.evidence_bytes) for body in bodies).items()),
                    null_vector_bytes_total=sum(int(body.null_vector.nbytes) for body in bodies
                                                if body.null_vector is not None),
                    bodies_with_selected_state=sum(1 for body in bodies
                                                   if "selected_state" in json.loads(body.evidence_bytes)),
                    certificate_records=sum(len(trial.certificates) for trial in history))
            else:
                history = result.stage().history
                rows = list(history)
                with_dependencies = [item for item in rows if "dependencies" in item]
                bodies = [body for item in with_dependencies for body in item["dependencies"].values()]
                entry.update(
                    history_rows=len(rows), rows_with_dependencies=len(with_dependencies),
                    public_bodies=len(bodies),
                    public_bodies_with_selected_state=sum(
                        1 for body in bodies if "selected_state" in body.get("evidence", {})),
                    public_body_array_bytes=sum(array_bytes(body) for body in bodies),
                    public_body_json_bytes=sum(json_bytes(body) for body in bodies),
                    stored=stored_bytes(directory))
                entry["stored_total_bytes"] = sum(item["bytes"] for item in entry["stored"].values())
                entry["stored_selected_state_count"] = sum(
                    item["selected_state_count"] for item in entry["stored"].values())
                entry["stored_selected_state_array_bytes"] = sum(
                    item["selected_state_array_bytes"] for item in entry["stored"].values())
            row[storage] = entry
    return row


def main():
    output = Path(sys.argv[1])
    rows = []
    for objective in sys.argv[2:]:
        row = one(objective)
        rows.append(row)
        memory, workspace = row["memory"], row["workspace"]
        print(objective, "memory: rows", memory["history_rows"], "bodies", memory["bodies"],
              "evidence bytes", memory["evidence_bytes_total"], "with selected_state",
              memory["bodies_with_selected_state"], "| workspace: stored", workspace["stored_total_bytes"],
              "selected_state", workspace["stored_selected_state_count"],
              workspace["stored_selected_state_array_bytes"], flush=True)
        output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
