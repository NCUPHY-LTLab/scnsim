---
title: "Evidence and readback"
lang: en
sidebar: testing
---

[English](evidence-readback.md) · [繁體中文](zh-TW/evidence-readback.md)

## Evidence v3

Evidence-v3 cases preserve literal binary64 values and detect corrupted bytes. Short, missing, or interrupted writes must not publish success; unsupported evidence must be rejected before Workspace mutation; partial sweeps retain failures and source order; and Direct, Operator, root, sensitivity, and optimization results resolve to exact saved bits after close and from a fresh process.

Source: [test_evidence_v3.py](test_evidence_v3.py).

## Workspace identities and resolution

Workspace cases cover version upgrades, forbidden downgrades, shared-plan identities, warm reuse, and guarded fresh-process resolution. Optimization histories survive close and resolve; repeated optimize/resolve calls add no native work after the original run, catching stale identities, unintended recomputation, and lost results.

Sources: [test_workspace_readback.py](test_workspace_readback.py) and [test_workspace_optimization_readback.py](test_workspace_optimization_readback.py).
