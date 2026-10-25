---
title: "Parameters and optimization"
lang: en
sidebar: testing
---


These cases distinguish an ordinary parameter change from an affine model that has explicit authorization to evaluate candidates inside its declared support. They also check how a physical coordinate is mapped to an optimizer coordinate before a bounded search begins.

## Affine support and authorization

An affine declaration maps the physical length `l` to capacitances using fixed coefficients over a trusted support interval. This fixture has baseline 1 m and support `[0.99, 1.01] m`. Its parallel capacitances are `C1 = (20 fF/m) l + 10 fF` and `C2 = (30 fF/m) l + 5 fF`, with an 8 nH branch to ground and a 50-ohm port. A requested point at 1.05 m lies outside the support interval; the fixture also contains the repeated grid `[1.05, 1.00, 1.05] m` so source order and repeated points can be checked.

1. Request 1.05 m through an ordinary Operator declaration without authorization. Authoring must raise `SCNSimValidationError` before creating a request or starting execution.
2. With explicit candidate-policy authorization, run Direct, Operator, and admittance fan-out through Memory and Workspace. Check that ordered support edges and each authorization source remain attached to their original rows.
3. Run the CMA case with bounds `[0.9, 1.1] m`, seed 23, and five evaluations. An unsupported candidate must be recorded as `invalid_candidate_physical_parameter` at `affine_support` with infinite cost; it does not become a public best candidate or add authorization history.

![Affine fixture: baseline length 1 m, support [0.99, 1.01] m, and the explicitly authorized out-of-support input at 1.05 m; the figure is not an optimization trajectory.](figures/affine-domain.svg){#fig-affine-domain}

{{< askr-image-view target="fig-affine-domain" mode="inline" >}}

[Open the interactive Plotly diagram](figures/affine-domain.html).

Source: [`test_affine_parameters.py`](test_affine_parameters.py).

## Local-coordinate scales and the C1 case

The bounded optimizer fixture uses a 110 fF capacitor with a 5.8 nH resonator, 6 fF coupling capacitor, and 50-ohm port. Its frequency hint is 6 GHz and target is 6.2 GHz. The parameter bounds initially span 80–140 fF, then an override resolves them to 90–130 fF. A local-coordinate scale converts a physical parameter displacement into the coordinates used by the optimizer. Scale resolution prefers an explicit coordinate scale, then a declared variable scale, then the width of the resolved bounds; this fixture reaches the last choice and records `scale_source` as `bounds_width`.

1. Resolve the parameter bounds and check which scale source supplies the optimizer coordinates.
2. Run C1 with both L-BFGS-B, a bounded gradient-based method, and Projected L-BFGS, which enforces bounds by projection, using the existing controls.
3. Require a converged result with at least one iteration and an independently calculated analytic loss below its baseline. The test reports final frequency and target distance; it does not set a target-distance tolerance.

Source: [`test_local_coordinate_defaults.py`](test_local_coordinate_defaults.py).

## CMA selectors

A separate CMA evolution-strategy fixture samples candidates within bounds 79.5–80.5 fF and checks seven declared selectors: Hybrid frequency and linewidth, TransferZero frequency, coupling magnitude, and response magnitude, real part, and imaginary part. A seed-29 run has five evaluations; a matched-RLC zero case uses seed 31 and five evaluations.

1. Check that each selector is declared and its corresponding branch executes.
2. Check the zero-cost baseline in each CMA case. Candidate trajectories need not match across backends.

Source: [`test_optimization.py`](test_optimization.py).
