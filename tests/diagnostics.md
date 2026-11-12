---
title: "Failure handling and diagnostics"
lang: en
sidebar: testing
---


These tests show three separate responsibilities: raise a native failure that SCNSim does not own while an equation failure stays evidence, return the certificate of every equation from a compiled program, and describe the solution that a linear solve actually returned. The routing and certificate cases solve one small banded system inside a compiled program; they do not solve a physical circuit.

![Identity-matrix diagnostic fixture: one right-hand side has a missing second component, producing componentwise residual 1 but aggregate residual 5 × 10⁻²¹.](figures/residual-metrics.svg){#fig-residual-metrics}

{{< askr-image-view target="fig-residual-metrics" mode="inline" >}}

[Open the interactive Plotly diagram](figures/residual-metrics.html).

## Native failure routing

A compiled program returns two kinds of failure beside its values. A failed equation is evidence: the analysis reports it as a typed failure from the status values, so the host must not raise it and must leave its certificate as it is. A failed check that SCNSim does not own means that an internal invariant broke, so the host must raise it with its original message and cause.

The routing cases run compiled programs on the banded four-node system of the next section. A singular matrix makes an equation fail, and a check written with JAX's `checkify.check` stands for one that SCNSim does not own. The equation belongs to a response and fails with status 7.

1. Let the equation fail in one program, in some lanes of a batched program, and inside a sequentially mapped Newton body. Nothing is raised, every array of the returned evidence is unchanged bit for bit, and the certificate row still names the response and holds status 7 in exactly the lanes that failed.
2. Let the foreign check fail alone, in one lane, in another lane than an equation failure, and inside a sequentially mapped body. It is raised as JAX's runtime error with the message and cause that the program itself raises. A program in which nothing failed raises nothing.
3. Hand over failures in their wrappers. JAX wraps the failures of a batched program, and SCNSim wraps those of a sequentially mapped body. A wrapper counts as an equation failure only when everything inside it is one; an empty, mislabelled or truncated wrapper is refused.

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

## Evidence as the analyses use it

The analyses do not look inside the evidence a compiled program returns, whichever of the two ways returned it. They use a small set of operations: cut out one lane, build the evidence of a program that did not run, ask inside a program whether anything failed, keep the entries of one root, read rows and facts, and raise a check that SCNSim does not own. These cases cover what only those operations show, on the same four-node system. A frame here is one compiled program together with the evidence it returned.

| Case | What must hold |
| --- | --- |
| Ordered chunk of frequencies | After the first failing point the remaining points of the chunk do not run, and their rows are absent. The same points computed as independent lanes are all present. |
| A frame that did not run | Its evidence is built from shapes alone: every row and every derivative-availability fact is absent, nothing has failed and nothing is raised. |
| A failure seen inside a program | A failed equation in any lane counts, and so does an equation skipped after an earlier failure. |
| A completed root | The rows of one root are separated from a frame that solved several. When a later frame continues from a completed root, the rows of that root, successful or failed, appear in its evidence bit for bit beside its own. |
| Evidence as data | Evidence passes through a later compiled program unchanged. One lane of batched evidence equals the evidence of that lane computed alone, also with two lane axes. |
| Order of frames | A sensitivity over three frequencies in chunks of two lists the rows of its frames in the order they ran: first the frames without a derivative direction, then those of the direction, each in work order. |
| Message of a foreign check | The raised text is the text of the check with the values it was given. A check inside a sequentially mapped body with several lanes lists the values of every lane. |
| Equation failure and foreign check together | The reader of a result always raises the foreign check. A reader that takes evidence straight from a call follows the failure that came first, so an earlier equation failure hides the check. The certificate ledger keeps foreign checks apart, which ends this difference. |
| Who reads the evidence | A scan of the package sources finds the inside of the evidence read in two modules only: the one that returns certificates and the one that holds these operations. |

Source: [`test_frame_evidence.py`](test_frame_evidence.py).

## Solve diagnostics

A small identity-matrix system makes the aggregate metric differ from a componentwise diagnostic. With `A = I₂`, one right-hand side containing two components, `b = [1, 1e-20]`, and approximate solution `x = [1, 0], the aggregate normalized residual is `5e-21`, while the second equation component has residual 1. The aggregate metric satisfies the original solver threshold even though the per-RHS diagnostic exposes that component. A separate two-column case checks per-RHS values `[1, 0]` when the second solution column is exact.

The dense case compares diagnostics with the actual original solve result. Sparse and compensated paths also retain the original solution and their four auxiliary scalar values. No new threshold is introduced: the test checks that diagnostic fields describe the solve that ran and do not hide a failing right-hand side behind aggregation. Source: [`test_solve_diagnostics.py`](test_solve_diagnostics.py).
