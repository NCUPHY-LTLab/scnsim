# Certificate transport design study (Issue 48)

This directory records a design study for
[Issue 48](https://github.com/NCUPHY-LTLab/scnsim/issues/48): how SCNSim can carry
equation certificates out of compiled JAX programs with public JAX interfaces only.
Start with `design.md`.

The files are study material, not product code. This branch is a reference for the
implementation and is not meant to be merged. The study ran on `develop` at `cdfe0d7`
with JAX 0.11.2.

Left out of the record: the JAX source files fetched for the text comparison
(`fetch_jax_sources.sh` fetches them again into a `jaxsrc/` directory) and the logs of
the shadow runs (their summaries are the `shadow_*.json` files).
