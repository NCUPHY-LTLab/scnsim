#!/usr/bin/env bash
# Regenerate the JAX internals comparison table (text comparison only).
here="$(dirname "$(readlink -f "$0")")"
python3 -I "$here/compare_jax_internals.py" "$here/jaxsrc" > "$here/jax_internals_table.txt" 2>&1
wc -l "$here/jax_internals_table.txt"
grep -n "^class ErrorEffect\|^ErrorEffect\|^class Error\|^_INACTIVE" "$here"/jaxsrc/0.10.2/jax/_src/checkify.py "$here"/jaxsrc/0.11.0/jax/_src/checkify.py "$here"/jaxsrc/0.11.2/jax/_src/checkify.py "$here"/jaxsrc/0.11.0/jax/_src/effects.py | sed "s|$here/jaxsrc/||"
grep -n "Error class body sha\|Error field order\|ErrorEffect body sha\|ErrorEffect defined in" "$here/jax_internals_table.txt"
sed -n '/^scan_p params have num_carry/,$p' "$here/jax_internals_table.txt"
