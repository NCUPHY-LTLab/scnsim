---
title: "Failure handling and diagnostics"
lang: en
sidebar: testing
---


These tests show three separate responsibilities: route an actual owned native error without hiding it, return the certificate of every equation from a compiled program, and describe the solution that a linear solve actually returned. The routing fixtures transport errors; they do not solve a physical circuit.

![Identity-matrix diagnostic fixture: one right-hand side has a missing second component, producing componentwise residual 1 but aggregate residual 5 × 10⁻²¹.](figures/residual-metrics.svg){#fig-residual-metrics}

{{< askr-image-view target="fig-residual-metrics" mode="inline" >}}

[Open the interactive Plotly diagram](figures/residual-metrics.html).

## Native failure routing

The routing cases construct scalar, batched, and archived equation errors with a real checkify error payload. For an error owned by the declared operation, the wrapper must preserve the original status, failing batch indices, source site, response context, and payload through the wrapper layers. A sparse equation fixture uses status 7; the test checks that the original status and payload bits survive.

1. Route a sparse equation error owned by the declared operation and confirm its status, batch indices, source site, response context, and payload survive.
2. Pass an unknown or mixed error and confirm the original runtime error and cause are re-raised. Invalid wrapper combinations are rejected, and inactive checks do not become thrown failures.

These cases catch swallowed native errors and wrappers that claim an error they cannot interpret. Source: [`test_native_failure_routing.py`](test_native_failure_routing.py).

## Equation certificates from a compiled program

Every equation SCNSim solves returns a certificate: a status, the backward residual `eta` with its numerator and denominator, and whether the solve was performed. These cases solve one banded four-node complex system inside a compiled JAX program and read the certificates back as rows. Each row is identified by its role, stage, derivative order and dependency.

| Case | What the rows must show |
| --- | --- |
| Solved, failed, skipped and not run | An equation that did not run is marked absent, with status 0 and NaN residual values. A solve skipped after an earlier failure carries that failure's status and is marked as not performed. A singular matrix reports the stage of its equation as status, with infinite residual values. |
| `cond`, `while_loop` and `scan` | An equation keeps one row. A branch that was not taken and a loop without a trip leave it absent. A loop reports its first failure, otherwise its latest trip. |
| `vmap` | A row has one entry per lane. A lane is absent in the branch it did not take. |
| Derivative equations | Reverse mode adds an adjoint row, forward mode a tangent row, a second forward derivative a mixed row, and a frequency jet a frequency-tangent row. A failed primal equation skips its derivative equation. |
| Derivative availability | The null-vector chart records whether its derivative is available only where its derivative rule ran, with its dependency and reason, separately for each lane, and also under reverse mode. |
| Sequentially mapped Newton body | Lanes that stop after different trip counts, including a lane without a trip and a failing lane, return their rows with a lane axis. |
| Single precision | Complex64 operands keep float32 certificate values, and the global x64 setting is unchanged. |

No tolerance is asserted. The cases check status codes, the performed and present flags, and whether the residual values are finite, infinite, NaN or exactly zero. A zero right-hand side gives an exactly zero residual, which shows which trip a loop reports.

One separate test records the `site` label of each row and the order in which rows are returned. `site` is the position of an equation in the traced program, counting JAX's own equations, so no other case relies on it. Source: [`test_certificate_transport.py`](test_certificate_transport.py).

## The certificate ledger

SCNSim is moving these certificates from JAX's internal check mechanism to a ledger of its own: an ordinary output of the compiled program with one entry for every equation. Both ways of returning certificates exist until the move is complete, and analyses still use the first. Two groups of cases cover the ledger.

The first group compiles every call of the previous section both ways and requires the same result: the same rows in the same order with the same labels, all six arrays of every row equal bit for bit, the same returned values, and the same derivative-availability facts. Further calls add a sequentially mapped Newton body under a custom derivative rule, inside a branch, under reverse mode and inside another such body. In two places the results agree in everything except `site`, because each mechanism adds a different number of its own equations to the traced program; those cases record both labels. Source: [`test_certificate_transport_parity.py`](test_certificate_transport_parity.py).

The second group covers what a comparison cannot show.

| Case | What must hold |
| --- | --- |
| Public interfaces | The ledger's module imports only public JAX modules and reads no private attribute. |
| An equation out of reach | An equation inside a JAX primitive that the ledger has no rule for, or in the predicate of a loop, stops compilation with a typed error. Its certificate cannot be lost silently. |
| One entry per equation | An entry keeps its first failure, otherwise its latest execution. |
| A program that did not run | Its ledger is built from shapes alone and marks every entry absent. A derivative-availability fact that is absent reads as available, so only a fact that is present can make a derivative unavailable. |
| Single precision | Entries of complex64 equations hold float32 values, present or absent. |
| The ledger as data | A ledger passes through a later compiled call unchanged, and the slice of one lane equals the ledger of that lane computed alone. |
| Checks SCNSim does not own | Such a check returns with its own message. An equation failure is never returned as one, and the current mechanism keeps recording the programs it certifies. |
| Facts that travel with a value | The two operations that attach a fact to a value return the value unchanged outside a certified program, and their derivative rules return the tangent and the cotangent unchanged. |

Source: [`test_certificate_ledger.py`](test_certificate_ledger.py); the derivative rules are in [`test_differentiation_rules.py`](test_differentiation_rules.py).

## Solve diagnostics

A small identity-matrix system makes the aggregate metric differ from a componentwise diagnostic. With `A = I₂`, one right-hand side containing two components, `b = [1, 1e-20]`, and approximate solution `x = [1, 0], the aggregate normalized residual is `5e-21`, while the second equation component has residual 1. The aggregate metric satisfies the original solver threshold even though the per-RHS diagnostic exposes that component. A separate two-column case checks per-RHS values `[1, 0]` when the second solution column is exact.

The dense case compares diagnostics with the actual original solve result. Sparse and compensated paths also retain the original solution and their four auxiliary scalar values. No new threshold is introduced: the test checks that diagnostic fields describe the solve that ran and do not hide a failing right-hand side behind aggregation. Source: [`test_solve_diagnostics.py`](test_solve_diagnostics.py).
