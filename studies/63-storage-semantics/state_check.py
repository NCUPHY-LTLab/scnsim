"""What is inside `selected_state`, and are its certificate records also at the top level? python3 state_check.py"""
import json

report = json.load(open("measure-all.json"))


def dump(value):
    return json.dumps(value, sort_keys=True)


def nested_states(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "selected_state":
                yield path + "/selected_state", child
            yield from nested_states(child, path + "/" + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from nested_states(child, path + "[%d]" % index)


def show(label, tree):
    top = [dump(item) for item in tree.get("certificates", [])]
    for path, state in nested_states(tree):
        if not isinstance(state, dict):
            print("%-44s %-40s %s of %d entries" % (label, path, type(state).__name__, len(state)))
            continue
        certificates = [dump(item) for item in state.get("certificates", [])]
        print("%-44s %-40s fields %s" % (label, path, sorted(state)))
        print("%-44s    certificate records in state: %d; also at top level: %d of %d (top level has %d)" % (
            "", len(certificates), sum(1 for item in certificates if item in top), len(certificates), len(top)))


show("operator_element_root/single", report["operator_element_root/single"]["workspace"]["result"]["evidence"]["tree"])
show("pole/single", report["pole/single"]["workspace"]["result"]["evidence"]["tree"])
show("transfer_zero/single", report["transfer_zero/single"]["workspace"]["result"]["evidence"]["tree"])
show("coupling/single", report["coupling/single"]["workspace"]["result"]["evidence"]["tree"])
failed = [point for point in report["diagonal_root/sweep-fail-point"]["workspace"]["result"]["points"]
          if not point["succeeded"]][0]
show("diagonal_root failure (sweep point)", failed["failure"]["evidence_tree"])
