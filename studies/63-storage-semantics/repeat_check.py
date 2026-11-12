"""Compare the first full run (measure.json) with the batch runs (measure-all.json): python3 repeat_check.py"""
import json

from compare import first_difference, same

first = json.load(open("measure.json"))
second = json.load(open("measure-all.json"))
common = sorted(set(first) & set(second))
agree, differ = 0, []
for name in common:
    ok = True
    for reading in ("memory", "workspace", "workspace_reopened", "workspace_fresh_process"):
        if reading not in first[name] or reading not in second[name]:
            continue
        if not same(first[name][reading], second[name][reading]):
            ok = False
            differ.append((name, reading, first_difference(first[name][reading], second[name][reading])))
    agree += ok
print("cases in both runs:", len(common), "| identical summaries in every reading:", agree)
for row in differ:
    print("  differs:", row[0], row[1], str(row[2])[:200])
