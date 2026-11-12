#!/usr/bin/env bash
# Fetch a few JAX source files at release tags for a text comparison only.
# The files are data: nothing here is imported or executed.
set -u
dest="$1"
tags="0.4.30 0.5.3 0.6.2 0.7.2 0.8.3 0.9.2 0.10.2 0.11.0 0.11.2 0.12.0.1"
files="jax/_src/checkify.py jax/_src/effects.py jax/_src/core.py jax/_src/xla_bridge.py jax/_src/custom_batching.py jax/_src/interpreters/partial_eval.py jax/_src/lax/control_flow/loops.py jax/_src/lax/control_flow/conditionals.py jax/extend/core/__init__.py jax/extend/core/primitives.py jax/core.py jax/experimental/checkify.py jax/interpreters/partial_eval.py jax/extend/backend.py"
for tag in $tags; do
  for file in $files; do
    out="$dest/$tag/$file"
    mkdir -p "$(dirname "$out")"
    if [ ! -s "$out" ]; then
      curl -sfL --max-time 60 "https://raw.githubusercontent.com/jax-ml/jax/jax-v$tag/$file" -o "$out" || echo "missing $tag $file"
    fi
  done
done
find "$dest" -type f -size +0 | wc -l
