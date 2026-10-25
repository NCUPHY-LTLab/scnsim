---
title: "Failure handling and diagnostics"
lang: en
sidebar: testing
---

[English](diagnostics.md) · [繁體中文](zh-TW/diagnostics.md)

These tests show two separate responsibilities: route an actual owned native error without hiding it, and describe the solution that a linear solve actually returned. The routing fixtures transport errors; they do not solve a physical circuit.

![Identity-matrix diagnostic fixture: one right-hand side has a missing second component, producing componentwise residual 1 but aggregate residual 5 × 10⁻²¹.](figures/residual-metrics.svg){#fig-residual-metrics}

{{< askr-image-view target="fig-residual-metrics" mode="inline" >}}

[Open the interactive Plotly diagram](figures/residual-metrics.html).

## Native failure routing

The routing cases construct scalar, batched, and archived equation errors with a real checkify error payload. For an error owned by the declared operation, the wrapper must preserve the original status, failing batch indices, source site, response context, and payload through the wrapper layers. A sparse equation fixture uses status 7; the test checks that the original status and payload bits survive.

1. Route a sparse equation error owned by the declared operation and confirm its status, batch indices, source site, response context, and payload survive.
2. Pass an unknown or mixed error and confirm the original runtime error and cause are re-raised. Invalid wrapper combinations are rejected, and inactive checks do not become thrown failures.

These cases catch swallowed native errors and wrappers that claim an error they cannot interpret. Source: [`test_native_failure_routing.py`](test_native_failure_routing.py).

## Solve diagnostics

A small identity-matrix system makes the aggregate metric differ from a componentwise diagnostic. With `A = I₂`, one right-hand side containing two components, `b = [1, 1e-20]`, and approximate solution `x = [1, 0], the aggregate normalized residual is `5e-21`, while the second equation component has residual 1. The aggregate metric satisfies the original solver threshold even though the per-RHS diagnostic exposes that component. A separate two-column case checks per-RHS values `[1, 0]` when the second solution column is exact.

The dense case compares diagnostics with the actual original solve result. Sparse and compensated paths also retain the original solution and their four auxiliary scalar values. No new threshold is introduced: the test checks that diagnostic fields describe the solve that ran and do not hide a failing right-hand side behind aggregation. Source: [`test_solve_diagnostics.py`](test_solve_diagnostics.py).
