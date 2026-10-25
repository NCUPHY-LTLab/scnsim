---
title: "Failure handling and diagnostics"
lang: en
sidebar: testing
---

[English](diagnostics.md) · [繁體中文](zh-TW/diagnostics.md)

## Native failure routing

Failure-routing cases pass owned scalar, batched, and archived equation errors through their wrappers. The original payload must survive, while unknown or mixed errors are re-raised and invalid wrappers rejected; this catches hidden native failures and unsupported classifications.

Source: [test_native_failure_routing.py](test_native_failure_routing.py).

## Solve diagnostics

Solve diagnostics compare per-right-hand-side componentwise metrics with aggregate metrics and exercise dense, sparse, and compensated solve paths. The diagnostics must describe the actual solve and retain its auxiliary data, catching aggregation that hides a failing right-hand side or diagnostics detached from the solve.

Source: [test_solve_diagnostics.py](test_solve_diagnostics.py).
