---
title: "Evidence and readback"
lang: en
sidebar: testing
---


Memory keeps results in the running process. Workspace persists plans and results in a SQLite store. The evidence tests follow small calculations through encoding, completion, close, and readback; they check that saved values and failures are read from storage rather than recalculated. The SCNSim software version identifies a package release, while evidence format version `v3` identifies the encoded array layout.

## What evidence v3 stores

The `v3` evidence format encodes typed array columns with exact binary64 values. Its fixtures include negative zero (`0x8000000000000000`) and a NaN with a chosen payload (`0x7ff8000000000042`); decoding must retain those bits.

1. Encode and decode the typed columns, then compare their exact binary64 representation.
2. Flip bytes in the header or payload. The reader must raise `EvidenceIntegrityError` rather than return a plausible-looking value.
3. Truncate a block, omit an object, or interrupt the success-pointer transaction. The incomplete write must not be published as success, while an earlier completed block remains readable. An unsupported evidence format is rejected before Workspace changes.

Source: [`test_evidence_v3.py`](test_evidence_v3.py).

## Partial sweeps keep order and failure

A three-point sweep requests `[110, 1e308, 110]` fF for the scalar impedance response of a resonator at 6 GHz.

1. Evaluate the grid in Memory and Workspace. The middle row fails during impedance conversion; the first and last rows succeed.
2. Check that the saved row order is success/failure/success with source indices `(0,)`, `(1,)`, `(2,)`. The repeated endpoint values remain separate rows.
3. Close and reopen Workspace, then read the partial result in a fresh process. Its saved values and failure must match the original rows.

Source: [`test_evidence_v3.py`](test_evidence_v3.py).

## Workspace identity and fresh-process readback

1. Upgrade the package version while keeping the existing plan leaf; change the plan and check that it receives a new leaf; request a forbidden downgrade and confirm rejection before mutation.
2. Reuse a shared plan identity and resolve a completed result after closing and reopening the SQLite store.
3. For optimization readback, run CMA or L-BFGS-B once, exercise warm-cache reuse, close Workspace, and resolve the saved history in a fresh process. Cached values and identities must match exactly, and later `optimize` or `resolve` calls must add no native work.

These steps catch stale identities, lost optimization history, or readback that silently computes again. Sources: [`test_workspace_readback.py`](test_workspace_readback.py), [`test_workspace_optimization_readback.py`](test_workspace_optimization_readback.py), and [`test_evidence_v3.py`](test_evidence_v3.py).
