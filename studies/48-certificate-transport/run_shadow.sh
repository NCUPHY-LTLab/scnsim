#!/usr/bin/env bash
# Run from the SCNSim worktree: bash run_shadow.sh <name> <unittest target> [...]
# Writes <name>.json and <name>.log next to this script.
here="$(dirname "$(readlink -f "$0")")"
name="$1"; shift
PYTHONPATH=".:$here" uv run --locked python "$here/shadow_product.py" "$here/$name.json" "$@" > "$here/$name.log" 2>&1
grep -v "^H2 \|^AD_RULE \|NVIDIA\|UserWarning\|to_dtype" "$here/$name.log" | tail -40
