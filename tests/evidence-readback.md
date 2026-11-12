---
title: "Evidence and readback"
lang: en
sidebar: testing
---


Memory keeps results in the running process. Workspace persists plans and results in a SQLite store. The evidence tests follow small calculations through encoding, completion, close, and readback; they check that saved values and failures are read from storage rather than recalculated. The SCNSim software version identifies a package release, while evidence format version `v4` identifies the encoded array layout.

## What evidence v4 stores

The `v4` evidence format encodes typed array columns with exact binary64 values. Its fixtures include negative zero (`0x8000000000000000`) and a NaN with a chosen payload (`0x7ff8000000000042`); decoding must retain those bits.

1. Encode and decode the typed columns, then compare their exact binary64 representation.
2. Flip bytes in the header or payload. The reader must raise `EvidenceIntegrityError` with the stage `evidence_v4` rather than return a plausible-looking value.
3. Truncate a block, omit an object, or interrupt the success-pointer transaction. The incomplete write must not be published as success, while an earlier completed block remains readable. An unsupported evidence format is rejected before Workspace changes.
4. Take a complete Workspace with one saved Direct result and declare evidence version 3 in its root envelope, its leaf envelope or its database. Opening it must raise `UnsupportedEvidenceVersionError` without changing a byte. A version 3 block, header or manifest is not read either.

Source: [`test_evidence_v4.py`](test_evidence_v4.py).

## Direct summaries and what a Direct call leaves behind

A Direct result keeps its S, Y and Z arrays in data chunks and the equation certificates of the same call in chunk summaries. A summary stores each certificate field once, as one array over the work rows of its chunk; a work row is one solved frequency.

1. Gather work rows that carry negative zero, a NaN payload, infinity and a vector-valued field into columns, store them and read the rows back. Every bit must return, and rows with different certificates are refused.
2. Solve eight frequencies with chunk sizes 3 and 8. The summaries over 3, 2 and 8 work rows must store the same columns: one per work counter and one per certificate field.
3. Compare every stored column, sliced at each work row, with the certificate values the solver returned in the same call. The stored values are read again after close and in a fresh process, and the solver's values must agree between Memory and Workspace.
4. Run a Direct call that completes, one that fails at its last frequency, one that is interrupted while solving and one that is interrupted while its result is written. The operation database must then hold no staging object and no result object of an unfinished call. The only objects that nothing refers to are earlier states of the operation row and of the attempt, which the store replaces behind a pointer.

Source: [`test_evidence_v4.py`](test_evidence_v4.py).

## Sweep points and what a sweep leaves behind

A sweep keeps each point's own arrays in a sweep chunk and the point's evidence in a separately referenced object. A point of a Direct, operator or response-element sweep stores its work evidence as runs: for the work rows that ran the same checked equations, each certificate field once, as one array over those rows.

1. Gather work rows that carry negative zero, a NaN payload, infinity and a vector-valued field into runs, store them and read the rows back. A change of view, demand or certificates starts a new run. Every bit and the row order must return; rows mixed with runs and damaged runs are refused.
2. Run a Direct sweep and an operator sweep at 3 and at 17 frequencies. The evidence object of every point must store the same columns at both counts: one per work counter and one per certificate field. The chunk must hold a locator to that object and never the evidence itself.
3. Sweep Direct, the operator, a response element, a root, a hybridized pole, a transfer zero and a residue-normalised coupling over three points; the second is the plan baseline and the third repeats the first. One more Direct sweep takes a length outside its declared support, so two of its points carry public affine evidence, and one more root sweep has a fourth point whose root fails. Every public value must be bit-identical in Memory and Workspace, after close, after reopening and in a fresh process that may not construct a numerical backend. The work rows that the Workspace reader returns must equal the rows the solver produced in the same call.
   Two kinds of evidence are compared between Workspace readings only, not with Memory: the public evidence of a pole, a zero and a coupling, which Memory does not expose, and the `selected_state` arrays in the evidence of a failed root, which only a Workspace keeps. Single points differ in the same way; which form both storages should expose is open on [Issue 58](https://github.com/NCUPHY-LTLab/scnsim/issues/58#issuecomment-6825469879).
4. Run sweeps that complete (a Direct sweep, and sweeps of a root, of a root with a failing point, of a hybridized pole, of a transfer zero and of a coupling), a sweep with a failing point, a sweep whose root anchor fails, a sweep interrupted while solving its second point and a sweep interrupted while its chunks are written. The operation database must then hold no staging object, each point body once behind its locator, and no result object of an unfinished sweep. The evidence of each coupling point names its two branch roots, and those are exactly the point's dependency bodies. The only objects that nothing refers to are earlier states of the operation row and of the attempt.

Source: [`test_evidence_v4.py`](test_evidence_v4.py).

## Partial sweeps keep order and failure

A three-point sweep requests `[110, 1e308, 110]` fF for the scalar impedance response of a resonator at 6 GHz.

1. Evaluate the grid in Memory and Workspace. The middle row fails during impedance conversion; the first and last rows succeed.
2. Check that the saved row order is success/failure/success with source indices `(0,)`, `(1,)`, `(2,)`. The repeated endpoint values remain separate rows.
3. Close and reopen Workspace, then read the partial result in a fresh process. Its saved values and failure must match the original rows.

Source: [`test_evidence_v4.py`](test_evidence_v4.py).

## Workspace identity and fresh-process readback

1. Upgrade the package version while keeping the existing plan leaf; change the plan and check that it receives a new leaf; request a forbidden downgrade and confirm rejection before mutation.
2. Reuse a shared plan identity and resolve a completed result after closing and reopening the SQLite store.
3. For optimization readback, run CMA or L-BFGS-B once, exercise warm-cache reuse, close Workspace, and resolve the saved history in a fresh process. Cached values and identities must match exactly, and later `optimize` or `resolve` calls must add no native work.

These steps catch stale identities, lost optimization history, or readback that silently computes again. Sources: [`test_workspace_readback.py`](test_workspace_readback.py), [`test_workspace_optimization_readback.py`](test_workspace_optimization_readback.py), and [`test_evidence_v4.py`](test_evidence_v4.py).
