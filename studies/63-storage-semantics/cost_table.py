"""Print the cost tables: python3 cost_table.py <cost*.json ...> [--opt opt-cost.json]"""
import json
import sys

paths = [p for p in sys.argv[1:] if not p.startswith("--")]
opt = None
if "--opt" in sys.argv:
    opt = sys.argv[sys.argv.index("--opt") + 1]
    paths.remove(opt)

rows = []
for path in paths:
    rows.extend(json.load(open(path)))

print("| Circuit | Quantity | Workspace outcome | Public `evidence`: `selected_state` | `null_vector` | all other keys | "
      "Stored `selected_state` (count, array bytes) | Stored bytes, all objects | Memory `evaluation.evidence_bytes` | "
      "Memory `evaluation.null_vector` |")
print("|---|---|---|---:|---:|---:|---|---:|---:|---:|")
for row in rows:
    w, m = row["workspace"], row["memory"]
    public = w.get("public_evidence") or w.get("failure_evidence") or {}
    stored = w["stored"]
    count = sum(item["selected_state_count"] for item in stored.values())
    state_bytes = sum(item["selected_state_array_bytes"] for item in stored.values())
    where = "failure evidence" if "failure_evidence" in w and "public_evidence" not in w else "public evidence"
    outcome = w["outcome"] + ("" if "failure" not in w else " " + w["failure"]["kind"] + "/" + str(w["failure"]["stage"]))
    memory_bytes = m.get("evaluation_evidence_bytes_len")
    if memory_bytes is None and "failure_evidence" in m:
        memory_bytes = "failure: %d B arrays" % m["failure_evidence"]["array_bytes"]
    print("| %s %s | %s | %s | %s | %s | %s | %d, %d B | %d | %s | %s |" % (
        row["kind"], row["size"], row["quantity"], outcome,
        public.get("selected_state_array_bytes") if public.get("selected_state_array_bytes") is not None else "not exposed",
        public.get("null_vector_array_bytes") if public.get("null_vector_array_bytes") is not None else "-",
        public.get("other_array_bytes", "-") if public.get("key_count") else "`{}`",
        count, state_bytes, w["stored_total_bytes"], memory_bytes, m.get("evaluation_null_vector_bytes")))
print()
for row in rows:
    w = row["workspace"]
    public = w.get("public_evidence") or w.get("failure_evidence") or {}
    if public.get("selected_state_fields"):
        print(row["kind"], row["size"], row["quantity"], "selected_state fields:", public["selected_state_fields"])
    print(row["kind"], row["size"], row["quantity"], "stored by role:",
          {role: (item["objects"], item["bytes"], item["selected_state_count"], item["selected_state_array_bytes"])
           for role, item in w["stored"].items()}, "seconds m/w", row["memory"]["seconds"], w["seconds"])

if opt:
    print()
    print("| Objective | Memory history rows | Bodies kept | Bodies per row | `evidence_bytes` total | with `selected_state` | "
          "certificate records | Workspace public bodies | Stored bytes | Stored `selected_state` (count, array bytes) |")
    print("|---|---:|---:|---|---:|---:|---:|---|---:|---|")
    for row in json.load(open(opt)):
        m, w = row["memory"], row["workspace"]
        print("| %s | %d | %d | %s | %d | %d | %d | %d of %d rows, %d with `selected_state` | %d | %d, %d B |" % (
            row["objective"], m["history_rows"], m["bodies"], m["bodies_per_row"], m["evidence_bytes_total"],
            m["bodies_with_selected_state"], m["certificate_records"], w["rows_with_dependencies"], w["history_rows"],
            w["public_bodies_with_selected_state"], w["stored_total_bytes"], w["stored_selected_state_count"],
            w["stored_selected_state_array_bytes"]))
        print("   stored by role:", {role: (item["objects"], item["bytes"], item["selected_state_count"],
                                            item["selected_state_array_bytes"]) for role, item in w["stored"].items()})
        print("   memory evidence_bytes per body:", m["evidence_bytes_per_body"], "null_vector total", m["null_vector_bytes_total"])
