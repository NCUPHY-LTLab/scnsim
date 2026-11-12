"""Print selected fields of proof_results.json. Usage: python show_proof.py <json>"""
import json, sys

for row in json.load(open(sys.argv[1])):
    print(f"{row['case'][:52]:52} sites={row['sites']} foreign={row['foreign']!r} "
          f"max_eta={row['max_eta']:.3g} first={row['first_call_s']} run_ms={row['run_ms']}")
