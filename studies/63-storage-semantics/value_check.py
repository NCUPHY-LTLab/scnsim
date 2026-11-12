"""Count the cases whose public numbers are bit-identical between Memory and Workspace: python3 value_check.py"""
import json

from compare import kind_of, same

report = json.load(open("measure-all.json"))
both_returned = same_values = stable_readings = 0
different = []
for name, case in sorted(report.items()):
    memory, workspace = case["memory"], case["workspace"]
    if memory["outcome"] != "returned" or workspace["outcome"] != "returned":
        continue
    both_returned += 1
    m, w = memory["result"], workspace["result"]
    kind = kind_of(m)
    if kind == "analysis":
        equal = m["values_sha"] == w["values_sha"]
    elif kind == "sweep":
        equal = len(m["points"]) == len(w["points"]) and all(
            a["succeeded"] == b["succeeded"] and (
                a["result"]["values_sha"] == b["result"]["values_sha"] if a["succeeded"] else
                all(a["failure"][key] == b["failure"][key] for key in ("type", "kind", "stage", "message")))
            for a, b in zip(m["points"], w["points"]))
    elif kind == "sensitivity":
        equal = m["values_sha"] == w["values_sha"]
    else:
        equal = (m["best_cost"] == w["best_cost"] and m["best_ordinal"] == w["best_ordinal"]
                 and [row["status"] for row in m["history"]] == [row["status"] for row in w["history"]]
                 and [row["cost"].get("f64") for row in m["history"]] == [row["cost_f64"] for row in w["history"]])
    same_values += equal
    if not equal:
        different.append(name)
    readings = [case[key]["result"] for key in ("workspace", "workspace_after_close", "workspace_reopened",
                                                "workspace_fresh_process")]
    stable = all(same(readings[0], other) for other in readings[1:]) and same(m, case["memory_after_close"]["result"])
    stable_readings += stable
print("cases measured:", len(report))
print("cases where both storages returned a Result:", both_returned)
print("  public numbers (and sweep failure identities) bit-identical between Memory and Workspace:", same_values)
print("  every reading of each storage bit-identical (after close, reopened, fresh process):", stable_readings)
print("  different:", different)
