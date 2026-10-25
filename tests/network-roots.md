---
title: "Network equations and roots"
lang: en
sidebar: testing
---

[English](network-roots.md) · [繁體中文](zh-TW/network-roots.md)

## Direct networks and quantities

Direct-analysis cases preserve declaration order and retain matrix-valued multiconductor and n-port networks through selected reductions, transforms, and probes. Their operators, responses, poles, residues, couplings, and transfer zeros are compared with independent fixture calculations, catching scalarization, ordering, and equation-construction errors.

Source: [test_direct_analysis.py](test_direct_analysis.py).

## Admittance and derivatives

Admittance cases compare dissipative networks with an independent oracle and NumPy Schur reduction, and vary reference impedance to ensure it does not create device loss. Derivative and matched-scattering-zero cases check the requested response and analytic frequency jet, catching loss-accounting and retained-coordinate errors.

Source: [test_direct_admittance.py](test_direct_admittance.py).

## Root and coupling certificates

Root cases compare high-Q poles with a global quadratic eigenvalue calculation under both coordinate orders. A no-loss pole is evaluated with five anchor choices—the real anchor and positive/negative imaginary perturbations at two scales—and must retain identical frequency bits; repeated modes, zero slopes, or unusable residues must not receive simple-root certificates.

These cases catch sign-rounding and invalid-certification errors. Source: [test_root_certificates.py](test_root_certificates.py).
