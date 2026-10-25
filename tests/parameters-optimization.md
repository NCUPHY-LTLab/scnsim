---
title: "Parameters and optimization"
lang: en
sidebar: testing
---

[English](parameters-optimization.md) · [繁體中文](zh-TW/parameters-optimization.md)

## Affine authorization

Affine fixtures submit ordinary unauthorized points and authorized fan-out through Memory and Workspace. Unauthorized extrapolation must fail before request creation; authorized rows retain the support edge and authorization source, repeated points preserve their original row, and CMA records rejected and authorized candidates, catching missing or misattributed evidence.

Source: [test_affine_parameters.py](test_affine_parameters.py).

## Local-coordinate scales

Bounded local-coordinate fixtures exercise explicit scales, parameter-domain scales, resolved bound widths, and the existing signed-log mapping. The C1 case runs both local optimizers, requires iterations and a converged result, and reports improved independent analytic loss and remaining target distance without imposing a target-distance tolerance; these checks catch scale-precedence and optimizer-wiring errors.

Source: [test_local_coordinate_defaults.py](test_local_coordinate_defaults.py).

## CMA selectors

CMA fixtures check that all seven declared selectors appear in JAX explanations and that selector branches, including a zero-cost baseline, execute. They do not require matching candidate trajectories across backends.

Source: [test_optimization.py](test_optimization.py).
