---
title: "Circuit inputs and components"
lang: en
sidebar: testing
---

[English](circuit-data.md) · [繁體中文](zh-TW/circuit-data.md)

These tests begin with explicit input data and declarations. They check that SCNSim keeps conductor order and source identity, and that a small component network is assembled from the declared branches.

## What the RLGC/Q2D fixture contains

Ansys Q2D is a quasi-static two-dimensional cross-section extraction tool that can export the per-unit-length parameters of a distributed transmission line. This test uses a synthetic CSV only to exercise that file format and its conductor mapping; it does not read a CAD layout or run an electromagnetic solver. The 5 GHz field is extraction-frequency metadata, not a frequency-sweep result.

The CSV names two signal conductors, `trace_a` and `trace_b`, and maps them one-to-one, in order, to `readout` and `filter`. `ground` is the reference conductor and is implicit in the two signal rows and columns. The fields are resistance `R` (series conductor loss, Ω/m), inductance `L` (self and mutual magnetic storage, H/m), conductance `G` (shunt leakage, S/m), and capacitance `C` (nodal electric storage, F/m); together these per-length matrices are the RLGC parameters:

```text
R = [[0.18,  0.00], [ 0.00, 0.22]] ohm/m
L = [[420,   75 ], [ 75, 395 ]] nH/m
G = [[0,      0 ], [  0,   0  ]] S/m
C = [[175,  -22 ], [-22, 168 ]] pF/m
```

The negative off-diagonal entries of `C` encode coupling under the matrix convention; they are not negative physical capacitors. The fixture contains no line length, conductor width, gap, thickness, substrate, or dielectric stack, so it cannot determine a unique physical geometry. These values describe a loader contract, not extracted geometry or a fabricated cross-section.

![Synthetic Q2D CSV topology with two signal conductors and an implicit ground reference; the lines are schematic and have no CAD dimensions.](figures/rlgc-inputs.svg){#fig-rlgc-inputs}

{{< askr-image-view target="fig-rlgc-inputs" mode="inline" >}}

[Open the interactive Plotly diagram](figures/rlgc-inputs.html).

## How the input checks work

1. Construct an `RLGC` object, then check the ordered signal names, reference conductor, and all four matrices. Attempt to mutate returned arrays and assign attributes; the values must remain unchanged and attribute assignment must be rejected.
2. Load the synthetic CSV and check its content digest and one-to-one ordered mapping. Change the mapping or file contents and confirm that an incomplete mapping, changed header, or invalid text fails explicitly.
3. Supply invalid coefficients and duplicate conductor names. The tests reject a nonsymmetric resistance matrix, non-positive-definite inductance or capacitance matrices, positive off-diagonal conductance, and duplicate names with `SCNSimValidationError`. These are RLGC coefficient checks, not checks of a separately constructed admittance matrix. Source: [`test_rlgc.py`](test_rlgc.py).

## How catalog components are checked

The component fixture declares two symmetric SQUID components in parallel with ground. Each Josephson junction has 8 nH and 2 fF; the two loops have 4 nH and 9 nH, the mutual-coupling coefficient is 0.25, and the model includes a 50-ohm port.

1. Read each branch incidence from the declared component expansion and independently assemble the three-by-three stiffness matrix `K` using `M = k sqrt(La Lb)`.
2. Compare the assembled matrix with the compiled component stamp under the test's declared numerical condition bound.
3. Replace the mutual-inductance graph with three 1 nH inductors and pairwise coefficients 0.9, 0.9, and -0.9. Because the complete inductance matrix is not positive definite, the component declaration must raise `SCNSimError`.

These component-factory and branch-expansion tests do not run a nonlinear or harmonic-balance simulation. Source: [`test_component_models.py`](test_component_models.py).
