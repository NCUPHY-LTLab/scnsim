# Memory against Workspace: what a Result exposes and how a failure surfaces (Issue 63)

This directory records a read-and-measure study for
[Issue 63](https://github.com/NCUPHY-LTLab/scnsim/issues/63). Start with `semantics.md`;
`tables.md` holds the full comparison tables.

The files are study material, not product code. This branch is a reference for the decision
and for the implementation that follows it, and is not meant to be merged. The study ran on
`develop` at `7db7c53` with JAX 0.11.2, float64, on CPU, with public calls only.

Left out of the record: the raw outputs of the measurement (`measure-*.json`), the run logs,
the cost and optimization outputs in JSON, and the exploratory scripts with their trial
outputs. `measure.py` and the other scripts produce them again from the repository root.
