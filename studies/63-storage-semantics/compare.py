"""Build the comparison tables from measurement files.

    python3 compare.py <tables.md> <measure.json> [<more.json> ...]

No SCNSim import: this only reads the JSON that measure.py wrote.
"""
import json
import sys

EXTRA_PUBLIC_KEYS = {"selected_state", "null_vector", "family", "input", "output", "row", "column", "branches"}


def load(paths):
    report = {}
    for path in paths:
        report.update(json.load(open(path)))
    return report


def strip(value):
    """Drop timings and identifiers that differ by execution, keep everything else."""
    if isinstance(value, dict):
        return {key: strip(item) for key, item in value.items() if key not in ("seconds",)}
    if isinstance(value, list):
        return [strip(item) for item in value]
    return value


def same(a, b):
    return json.dumps(strip(a), sort_keys=True) == json.dumps(strip(b), sort_keys=True)


def first_difference(a, b, path=""):
    if type(a) != type(b):
        return path + ": type"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key == "seconds":
                continue
            if key not in a or key not in b:
                return path + "/" + key + ": only on one side"
            found = first_difference(a[key], b[key], path + "/" + key)
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return path + ": length %d vs %d" % (len(a), len(b))
        for index, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, path + "[%d]" % index)
            if found:
                return found
        return None
    return None if a == b else path + ": %r vs %r" % (a, b)


def evidence_text(row):
    if row is None or not row.get("present"):
        return "absent"
    if row["key_count"] == 0:
        return "`{}`"
    return "%d keys, %d B arrays" % (row["key_count"], row["array_bytes"])


def kind_of(result):
    name = result["class"]
    if "Optimization" in name:
        return "optimization"
    if "Sensitivity" in name:
        return "sensitivity"
    if "Sweep" in name:
        return "sweep"
    return "analysis"


def points_of(result):
    """Analysis rows of a single Result or of every succeeded sweep point."""
    if kind_of(result) == "sweep":
        return [point.get("result") for point in result["points"]]
    if kind_of(result) == "analysis":
        return [result]
    return []


def facts_difference(memory_point, workspace_point):
    """Compare the in-memory compact record with the Workspace public evidence."""
    if memory_point is None or workspace_point is None or "evaluation" not in memory_point:
        return None
    compact = memory_point["evaluation"]["evidence_bytes"]["tree"]
    public = workspace_point["evidence"]["tree"]
    if not public:
        return None
    extra = sorted(set(public) - set(compact))
    missing = sorted(set(compact) - set(public))
    differing = sorted(key for key in set(public) & set(compact)
                       if json.dumps(public[key], sort_keys=True) != json.dumps(compact[key], sort_keys=True))
    return {"workspace_only": extra, "memory_only": missing, "differing_common_keys": differing}


def failure_text(row):
    if row is None:
        return "-"
    keys = row.get("evidence_keys") or []
    return "%s, kind `%s`, stage `%s`, %d evidence keys%s" % (
        row["type"].rsplit(".", 1)[-1], row["kind"], row["stage"], len(keys),
        ", `selected_state` included" if "selected_state" in keys else "")


def failure_evidence_difference(a, b):
    if a is None or b is None or a.get("evidence_tree") is None or b.get("evidence_tree") is None:
        return None
    left, right = a["evidence_tree"], b["evidence_tree"]
    identity = [name + ": `%s` vs `%s`" % (a[name].rsplit(".", 1)[-1], b[name].rsplit(".", 1)[-1])
                for name in ("type", "kind", "stage", "message") if a[name] != b[name]]
    return {"memory_only": sorted(set(left) - set(right)), "workspace_only": sorted(set(right) - set(left)),
            "differing_common_keys": sorted(
                key for key in set(left) & set(right)
                if json.dumps(left[key], sort_keys=True) != json.dumps(right[key], sort_keys=True)),
            "identity_differences": identity,
            "same_type_kind_stage_message": not identity}


def outcome_text(reading):
    if reading is None:
        return "not run"
    if reading["outcome"] == "raised":
        return "RAISED " + failure_text(reading["failure"])
    result = reading["result"]
    if kind_of(result) == "analysis" and result.get("status") == "failure":
        return "RETURNED failed Result (`status == \"failure\"`): " + failure_text(result["returned_failure"])
    if kind_of(result) == "sweep":
        flags = "".join("T" if point["succeeded"] else "F" for point in result["points"])
        failed = [point for point in result["points"] if not point["succeeded"]]
        text = "RETURNED sweep, points " + flags
        if failed:
            text += "; failed point: " + failure_text(failed[0]["failure"]) + ", `point.result` " + failed[0]["result_access"]
        return text
    if kind_of(result) == "sensitivity":
        blocks = [block["status"] for block in result["primal"]]
        operations = {kind: (op["status"], [block["status"] for block in op["blocks"]])
                      for kind, op in result["operations"].items()}
        return "RETURNED %s, status `%s`, primal %s, operations %s" % (result["class"], result["status"], blocks, operations)
    if kind_of(result) == "optimization":
        statuses = "".join("S" if row["status"] == "success" else "F" for row in result["history"])
        return "RETURNED %s, history %s" % (result["class"], statuses)
    return "RETURNED " + result["class"]


def main():
    output, paths = sys.argv[1], sys.argv[2:]
    report = load(paths)
    lines = []
    out = lines.append
    differences = []

    # ------------------------------------------------------------ readings of one storage agree
    out("## T1. Readings of the same storage\n")
    out("Each Workspace Result is summarised four times: as returned, after `run.close()`, through `run.resolve` "
        "on a reopened Workspace, and through `run.resolve` in a fresh process. Each Memory Result is summarised "
        "as returned and after `run.close()`. `same` means every summarised public value, evidence tree included, "
        "is bit-identical.\n")
    out("| Case | Memory after close | Workspace after close | Workspace reopened | Workspace fresh process |")
    out("|---|---|---|---|---|")
    for name, case in report.items():
        if "script_error" in case:
            out("| %s | script error: %s | | | |" % (name, case["script_error"][:80]))
            continue
        def cell(base, other):
            if other not in case:
                return "n/a (call raised)" if case[base]["outcome"] == "raised" else "missing"
            if case[base]["outcome"] == "raised" or case[other]["outcome"] == "raised":
                if case[base]["outcome"] == case[other]["outcome"] == "raised":
                    left, right = case[base]["failure"], case[other]["failure"]
                    return "raised: " + ("same failure" if same(left, right) else
                                         right["type"].rsplit(".", 1)[-1] + " (`" + str(right["kind"]) + "`)")
                return case[other]["outcome"] + " vs " + case[base]["outcome"]
            if same(case[base]["result"], case[other]["result"]):
                return "same"
            return "DIFFERENT at " + str(first_difference(strip(case[base]["result"]), strip(case[other]["result"])))[:140]
        out("| %s | %s | %s | %s | %s |" % (name, cell("memory", "memory_after_close"),
                                           cell("workspace", "workspace_after_close"),
                                           cell("workspace", "workspace_reopened"),
                                           cell("workspace", "workspace_fresh_process")))
    out("")

    # ------------------------------------------------------------ success evidence
    out("## T2. Public evidence of a successful call\n")
    out("`evidence` is `result.evidence` (for a sweep: of each succeeded point, listed once when all points agree). "
        "The Memory column also lists the other public attributes of the in-memory Result that carry numerical "
        "evidence. `values` compares every public number of the Result bit for bit.\n")
    out("| Case | Memory `evidence` | Memory `evaluation.evidence_bytes` | Memory `dependencies` | Workspace `evidence` | Values | `evidence` equal |")
    out("|---|---|---|---|---|---|---|")
    for name, case in report.items():
        if "script_error" in case or "fail" in name or "refused" in name:
            continue
        memory, workspace = case["memory"], case["workspace"]
        if memory["outcome"] != "returned" or workspace["outcome"] != "returned":
            continue
        m, w = memory["result"], workspace["result"]
        if kind_of(m) not in ("analysis", "sweep"):
            continue
        mp, wp = points_of(m), points_of(w)
        def uniq(rows):
            seen = []
            for row in rows:
                if row not in seen:
                    seen.append(row)
            return "; ".join(seen)
        m_ev = uniq([evidence_text(p["evidence"]) for p in mp if p])
        w_ev = uniq([evidence_text(p["evidence"]) for p in wp if p])
        m_bytes = uniq(["%d keys, %d B JSON" % (p["evaluation"]["evidence_bytes"]["key_count"],
                                               p["evaluation"]["evidence_bytes_len"])
                        if "evaluation" in p else "no `evaluation`" for p in mp if p])
        m_deps = uniq([str(len(p["dependencies"])) if "dependencies" in p else "n/a" for p in mp if p])
        values = all(a["values_sha"] == b["values_sha"] for a, b in zip(mp, wp) if a and b)
        equal = all(a["evidence"]["sha"] == b["evidence"]["sha"] for a, b in zip(mp, wp) if a and b)
        out("| %s | %s | %s | %s | %s | %s | %s |" % (name, m_ev, m_bytes, m_deps, w_ev,
                                                    "same" if values else "DIFFERENT", "yes" if equal else "**no**"))
        if not equal:
            facts = [facts_difference(a, b) for a, b in zip(mp, wp)]
            differences.append({"case": name, "kind": "success_evidence", "memory": m_ev, "workspace": w_ev,
                                "facts": facts[0]})
        if not values:
            differences.append({"case": name, "kind": "values"})
    out("")

    # ------------------------------------------------------------ compact facts vs public evidence
    out("## T3. Is the Workspace evidence the in-memory record plus extra keys?\n")
    out("For each successful root-type call: the keys of the Workspace `result.evidence` that the in-memory record "
        "(`result.evaluation.evidence_bytes`, decoded as JSON) does not have, the keys only the in-memory record "
        "has, and the common keys whose bits differ.\n")
    out("| Case | Workspace-only keys | Memory-only keys | Common keys that differ |")
    out("|---|---|---|---|")
    for name, case in report.items():
        if "script_error" in case or "fail" in name:
            continue
        if case["memory"]["outcome"] != "returned" or case["workspace"]["outcome"] != "returned":
            continue
        m, w = case["memory"]["result"], case["workspace"]["result"]
        if kind_of(m) not in ("analysis", "sweep"):
            continue
        rows = [facts_difference(a, b) for a, b in zip(points_of(m), points_of(w))]
        rows = [row for row in rows if row]
        if not rows:
            continue
        row = rows[0]
        out("| %s | %s | %s | %s |" % (name, ", ".join("`%s`" % k for k in row["workspace_only"]) or "none",
                                     ", ".join("`%s`" % k for k in row["memory_only"]) or "none",
                                     ", ".join("`%s`" % k for k in row["differing_common_keys"]) or "none"))
    out("")

    # ------------------------------------------------------------ failures
    out("## T4. Failing calls\n")
    out("| Case | Memory | Workspace | Workspace `resolve` after reopening | Same surface | Evidence difference |")
    out("|---|---|---|---|---|---|")
    for name, case in report.items():
        if "script_error" in case or not ("fail" in name or "refused" in name):
            continue
        memory, workspace = case["memory"], case["workspace"]
        reopened = case.get("workspace_reopened")
        m_text, w_text = outcome_text(memory), outcome_text(workspace)
        r_text = outcome_text(reopened)
        def failure_of(reading):
            if reading["outcome"] == "raised":
                return reading["failure"]
            result = reading["result"]
            if kind_of(result) == "analysis":
                return result.get("returned_failure")
            if kind_of(result) == "sweep":
                failed = [point for point in result["points"] if not point["succeeded"]]
                return failed[0]["failure"] if failed else None
            return None
        mf, wf = failure_of(memory), failure_of(workspace)
        same_surface = (memory["outcome"] == workspace["outcome"]
                        and (memory["outcome"] == "raised" or not (
                            kind_of(memory["result"]) == "analysis"
                            and (memory["result"].get("status") == "failure") != (workspace["result"].get("status") == "failure"))))
        diff = failure_evidence_difference(mf, wf)
        if diff is None:
            diff_text = "n/a"
        else:
            parts = []
            if diff["workspace_only"]:
                parts.append("Workspace only: " + ", ".join("`%s`" % k for k in diff["workspace_only"]))
            if diff["memory_only"]:
                parts.append("Memory only: " + ", ".join("`%s`" % k for k in diff["memory_only"]))
            if diff["differing_common_keys"]:
                parts.append("common keys differ: " + ", ".join("`%s`" % k for k in diff["differing_common_keys"]))
            if not diff["same_type_kind_stage_message"]:
                parts.append("failure identity differs (" + "; ".join(diff["identity_differences"]) + ")")
            diff_text = "; ".join(parts) or "none"
        if mf is None and wf is None and memory["outcome"] == workspace["outcome"] == "returned":
            out("| %s | no failure with this input | no failure with this input | - | yes | n/a |" % name)
            continue
        out("| %s | %s | %s | %s | %s | %s |" % (name, m_text, w_text, r_text,
                                               "yes" if same_surface else "**no**", diff_text))
        if not same_surface:
            differences.append({"case": name, "kind": "failure_surface", "memory": m_text, "workspace": w_text})
        if diff and (diff["workspace_only"] or diff["memory_only"] or diff["differing_common_keys"]
                     or not diff["same_type_kind_stage_message"]):
            differences.append({"case": name, "kind": "failure_evidence", "detail": diff_text})
    out("")

    # ------------------------------------------------------------ sensitivity
    out("## T5. Sensitivity\n")
    out("| Case | Memory | Workspace | Values and block failures equal | Memory-only public attributes |")
    out("|---|---|---|---|---|")
    for name, case in report.items():
        if "script_error" in case or not name.startswith("sensitivity/"):
            continue
        memory, workspace = case["memory"], case["workspace"]
        m_text, w_text = outcome_text(memory), outcome_text(workspace)
        if memory["outcome"] == workspace["outcome"] == "returned":
            m, w = memory["result"], workspace["result"]
            equal = m["values_sha"] == w["values_sha"]
            extra = sorted(set(m["public"]) - set(w["public"]))
            certificates = m.get("certificates", {})
            extra_text = ", ".join("`%s`" % k for k in extra) + "; `certificates`: %d records, %d B arrays" % (
                certificates.get("count", 0), certificates.get("array_bytes", 0))
            if not equal:
                differences.append({"case": name, "kind": "sensitivity_values",
                                    "detail": first_difference(m["primal"], w["primal"]) or
                                    first_difference(m["operations"], w["operations"])})
        else:
            equal, extra_text = None, "n/a"
            if memory["outcome"] != workspace["outcome"]:
                differences.append({"case": name, "kind": "failure_surface", "memory": m_text, "workspace": w_text})
        out("| %s | %s | %s | %s | %s |" % (name, m_text, w_text,
                                          {True: "yes", False: "**no**", None: "n/a"}[equal], extra_text))
    out("")

    # ------------------------------------------------------------ optimization
    out("## T6. Optimization\n")
    out("| Case | Memory | Workspace | Memory: bodies kept | Workspace: evidence reachable from public history |")
    out("|---|---|---|---|---|")
    for name, case in report.items():
        if "script_error" in case or not name.startswith("optimization/"):
            continue
        memory, workspace = case["memory"], case["workspace"]
        m_text, w_text = outcome_text(memory), outcome_text(workspace)
        bodies_text = ws_text = "n/a"
        if memory["outcome"] == "returned":
            history = memory["result"]["history"]
            body_count = sum(len(row["bodies"]) for row in history)
            body_bytes = sum(body["evidence_bytes_len"] for row in history for body in row["bodies"].values())
            with_state = sum(1 for row in history for body in row["bodies"].values()
                             if "selected_state" in body["evidence_bytes"]["keys"])
            certificates = sum(row["certificates"]["count"] for row in history)
            failed = [row for row in history if row["status"] == "failure"]
            bodies_text = "%d history rows, %d bodies, %d B of `evidence_bytes`, %d with `selected_state`, %d certificate records" % (
                len(history), body_count, body_bytes, with_state, certificates)
            if failed:
                bodies_text += "; failed trial `failure` keys: " + ", ".join(failed[0]["failure"]["keys"])
                if "evidence" in failed[0]["failure"]:
                    bodies_text += " (evidence keys: %d%s)" % (
                        failed[0]["failure"]["evidence"]["key_count"],
                        ", `selected_state` included" if "selected_state" in failed[0]["failure"]["evidence"]["keys"] else "")
        if workspace["outcome"] == "returned":
            history = workspace["result"]["history"]
            with_dependencies = [row for row in history if "dependencies" in row]
            bodies = [body for row in with_dependencies for body in row["dependencies"].values()]
            with_state = sum(1 for body in bodies if body["evidence"]["present"] and "selected_state" in body["evidence"]["keys"])
            term_evidence = [term for row in history for term in row["terms"] if term["evidence"]]
            coupling = sum(1 for row in history for term in row["terms"] if term["residue_coupling_evidence"])
            failed = [row for row in history if row["status"] == "failure"]
            ws_text = ("%d history rows; %d rows carry `dependencies` (%d bodies, %d B arrays, %d with `selected_state`); "
                       "%d terms carry `evidence`; %d terms carry `residue_coupling_evidence`; `result.ledger` length %d") % (
                len(history), len(with_dependencies), len(bodies), sum(body["array_bytes"] for body in bodies),
                with_state, len(term_evidence), coupling, workspace["result"]["ledger_length"])
            if failed:
                ws_text += "; failed row `failure` keys: " + ", ".join(failed[0]["failure"]["keys"])
                if "evidence" in failed[0]["failure"]:
                    ws_text += " (evidence keys: %d%s)" % (
                        failed[0]["failure"]["evidence"]["key_count"],
                        ", `selected_state` included" if "selected_state" in failed[0]["failure"]["evidence"]["keys"] else "")
        if memory["outcome"] != workspace["outcome"]:
            differences.append({"case": name, "kind": "failure_surface", "memory": m_text, "workspace": w_text})
        out("| %s | %s | %s | %s | %s |" % (name, m_text, w_text, bodies_text, ws_text))
    out("")

    out("## T7. Every Memory/Workspace difference found by this script\n")
    out("```json")
    out(json.dumps(differences, indent=1))
    out("```")
    open(output, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
