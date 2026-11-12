"""Merge the batch files into one measurement file and report coverage: python3 merge.py"""
import glob
import json
import re

merged = {}
for path in sorted(glob.glob("measure-b*.json"), key=lambda p: int(re.findall(r"\d+", p)[0])):
    data = json.load(open(path))
    merged.update(data)
    errors = [name for name, row in data.items() if "script_error" in row]
    missing = [name for name, row in data.items() if "script_error" not in row and "workspace_fresh_process" not in row]
    print(path, len(data), "cases; script errors:", errors, "; no fresh-process reading:", missing)
source = open("measure.py").read()
declared = set(re.findall(r'@case\("([^"]+)"', source))
for form in ("jacobian", "jvp", "vjp"):
    declared |= {"sensitivity/%s-%s" % (quantity, form) for quantity in ("response", "root", "pole")}
declared = {name for name in declared if not name.endswith("-")}
print("measured", len(merged), "declared", len(declared), "not measured:", sorted(declared - set(merged)))
json.dump(merged, open("measure-all.json", "w"), sort_keys=True, indent=1)
print("wrote measure-all.json")
