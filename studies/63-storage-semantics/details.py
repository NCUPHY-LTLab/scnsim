"""Per-key sizes and a few cross-checks from measure-all.json: python3 details.py"""
import json

report = json.load(open("measure-all.json"))

print("| Case | Workspace `evidence` keys | array bytes | JSON bytes | `selected_state` | `null_vector` | rest | "
      "Memory compact record (JSON bytes) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for name in ("operator_element_root/single", "pole/single", "transfer_zero/single", "coupling/single",
             "coupling/single-pole-branches", "coupling/sweep"):
    workspace = report[name]["workspace"]["result"]
    memory = report[name]["memory"]["result"]
    if "points" in workspace:
        workspace, memory = workspace["points"][0]["result"], memory["points"][0]["result"]
    evidence = workspace["evidence"]
    per_key = evidence["per_key_array_bytes"]
    state = per_key.get("selected_state", 0)
    vector = per_key.get("null_vector", 0)
    print("| %s | %d | %d | %d | %d | %d | %d | %d |" % (
        name, evidence["key_count"], evidence["array_bytes"], evidence["json_bytes"], state, vector,
        evidence["array_bytes"] - state - vector, memory["evaluation"]["evidence_bytes_len"]))

print()
pole_m = report["pole/single"]["memory"]["result"]["evaluation"]["null_vector"]
pole_w = report["pole/single"]["workspace"]["result"]["evidence"]["tree"]["null_vector"]
print("pole null vector: memory evaluation.null_vector", pole_m, "| workspace evidence['null_vector']", pole_w,
      "| same bits:", pole_m["bytes"] == pole_w["bytes"])

print()
print("Workspace pole evidence keys:", report["pole/single"]["workspace"]["result"]["evidence"]["keys"])
print("Workspace zero evidence keys:", report["transfer_zero/single"]["workspace"]["result"]["evidence"]["keys"])
print("Workspace coupling evidence keys:", report["coupling/single"]["workspace"]["result"]["evidence"]["keys"])
print("Workspace coupling branches[0] keys:",
      report["coupling/single"]["workspace"]["result"]["evidence"]["nested_keys"]["branches"])
tree = report["coupling/single"]["workspace"]["result"]["evidence"]["tree"]["branches"][0]
print("Workspace coupling branches[0].certificates keys:", sorted(tree["certificates"]),
      "| has selected_state:", "selected_state" in tree["certificates"])
tree = report["coupling/sweep"]["workspace"]["result"]["points"][0]["result"]["evidence"]["tree"]["branches"][0]
print("Workspace coupling sweep branches[0].certificates has selected_state:", "selected_state" in tree["certificates"])
print("selected_state fields (pole):",
      sorted(report["pole/single"]["workspace"]["result"]["evidence"]["tree"]["selected_state"]))

print()
print("Memory public attributes with evidence, by family (single point):")
for name in ("direct/single", "operator/single", "response/single", "static/matrix-retained",
             "diagonal_root/single", "operator_element_root/single", "pole/single", "transfer_zero/single",
             "coupling/single"):
    memory = report[name]["memory"]["result"]
    workspace = report[name]["workspace"]["result"]
    evaluation = memory.get("evaluation")
    print(" ", name, "| memory class", memory["class"], "| evidence keys", memory["evidence"]["keys"][:12],
          "| evaluation.evidence_bytes keys", None if evaluation is None else evaluation["evidence_bytes"]["keys"],
          "| dependencies", len(memory.get("dependencies", [])), "| context", memory.get("context_keys"),
          "| metadata", memory.get("metadata_keys"), "|| workspace class", workspace["class"],
          "| public", [item for item in workspace["public"] if item in ("evidence", "identity", "discretization")])

print()
print("Memory sweep-only public attributes (diagonal root sweep):")
sweep = report["diagonal_root/sweep"]["memory"]["result"]
print("  sweep public:", sweep["public"])
print("  baseline:", None if sweep["baseline"] is None else
      {"succeeded": sweep["baseline"]["succeeded"],
       "evaluation.evidence_bytes keys": len(sweep["baseline"]["analysis"]["evaluation"]["evidence_bytes"]["keys"])})
print("  workspace sweep public:", report["diagonal_root/sweep"]["workspace"]["result"]["public"])
