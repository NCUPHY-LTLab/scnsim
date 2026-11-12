# Carrying equation certificates out of compiled JAX programs with public interfaces only

A design study for SCNSim Issue 48. Source read: `origin/develop` at `cdfe0d7`,
JAX 0.11.2. Nothing in the product was changed.

## The answer

SCNSim can drop every `jax._src` import of the certificate mechanism without
changing a certificate value. The recommended design keeps what already works
(the equation primitives return their certificate as ordinary outputs) and
replaces only the part that rides on JAX's private check machinery:

- **A pass that SCNSim owns** walks the finished program (after `jvp`, `vjp`
  or `grad` have run), collects the certificate outputs of every equation
  site, and re-emits `cond`, `while_loop` and `scan` through the public
  `jax.lax` functions with the certificates as extra branch outputs or loop
  carry. The result is `(evidence, values)`, where `evidence` is SCNSim's own
  pytree of arrays.
- **Two small pure primitives** keep facts on the data path where today a
  private check effect is used: a *gate* for "derivative unavailable" and an
  *attach* for the certificates of a sequentially mapped Newton body.
- **Public `checkify`** stays only for checks SCNSim does not own
  (`checkify.debug_check` in `native_transfer.py`), read back through its
  public `Error.get()` / `Error.throw()`.

On JAX 0.11.2 the scratch implementation
(`certificate_transport.py`, public imports only) reproduced the current
mechanism's certificates bit for bit:

| Evidence | Result |
|---|---|
| 42 small cases (Direct, batched, reverse, forward, second order, `cond`, `while_loop`, `scan`, sequential map, availability, foreign checks, 32-bit) | 41 identical in everything compared: row count, order, `site`, role, stage, derivative order, dependency, all six arrays, returned values, owned-failure flag, foreign message. 1 identical except the text of a foreign check's message (it prints one lane's value where today's prints all lanes) |
| Shadow run inside 44 product tests (every outermost certified frame built with both mechanisms, compared on every execution) | 154 frames, 453 certificate rows, 386 executions, 29 686 arrays compared bit for bit: 0 differences in values, 0 in `site`, role, stage, order or dependency. The 44 tests still pass |
| Shadow run that also compares the Newton bodies under the sequential map (3 tests) | 18 frames, 49 rows (24 inside `while_loop` bodies), 7 902 arrays: 0 differences |
| Run time, small cases, current / candidate (medians) | Direct 0.17 / 0.17 ms; 5 lanes 0.40 / 0.39 ms; value and gradient 0.40 / 0.33 ms; `while_loop` 0.56 / 0.55 ms; sequential map 1.5 / 1.7 ms |

Two things do change and need the Human's decision before the redesign Issue
is written:

1. **`site` values in frames that consume a completed root branch.** `site`
   is the path of the equation inside the traced program. In those frames the
   transport equations that precede the frame's own equations change, so the
   path changes (measured: `(12,)` becomes `(0,)`), and with it the row order
   (rows sort by the text of `site`). Every certificate number, role, stage
   and dependency stays the same. `site` is in stored evidence and in one
   reason string.
2. **`site` is not stable across JAX versions under any design that keeps the
   current definition**, because it counts JAX's own equations. A release that
   supports a JAX range needs a decision on what `site` means (section
   "What changes").

The rest of this document gives the requirements with their sources, the
comparison of designs, the proof, the fragility evidence, and the staging plan.

Each statement below is marked where it matters:
**[measured]** I ran it; **[read]** I read the code or the upstream source;
**[helper]** a read-only helper located it and I did not re-read every line;
**[inference]** reasoning that was not run.

---

## 1. What the mechanism does today

`src/scnsim/numerics/native_sparse.py` [read]:

- Every equation primitive (`_solve_p` and its batch, residual, pair and jet
  relatives) returns `value..., status, eta, numerator, denominator, performed`
  as ordinary outputs (`_callback`, line 513; `checked_sparse_solve`, 859).
  The program builder already threads the primal ones into packets and into
  the int32 status reducers.
- What ordinary outputs cannot reach are the equations JAX itself creates: the
  tangent solve inside `_jvp` (643) and the adjoint solve inside `_transpose`
  (663). Their certificate outputs are dropped by the rule, and they may sit
  inside a `cond` branch that JAX differentiated.
- `checked_native` (271) therefore traces the complete transformed function,
  labels each equation with its path (`_annotate_sparse_sites`, 241) and runs
  JAX's `checkify` over it. `_error_check` (806), registered in the private
  table `checkify.error_checks`, turns every equation's certificate into the
  payload of a private `Error` slot keyed by a subclass of the private
  `ErrorEffect`. `sparse_certificates` (109) reads `error._payload` and
  `error._code` back.
- Three further jobs ride on the same machinery: derivative-availability
  facts (`derivative_available`, 184, builds a private `Error` by hand), the
  archive that brings certificates out of a sequentially mapped Newton body
  (`checked_sequential_primal`, 294; `_primal_archive_p`, 321;
  `_archive_layout`, 354; `_archive_error_check`, 476), and foreign checks.

One fact shapes the whole study: **the first failure, its kind and its stage
never come from the `Error` object.** They come from SCNSim's own int32
reducers, which are ordinary traced data: `first()` in
`numerics/jax_core.py:61` and the table `FAILURES` at `jax_core.py:12-28`
[read]. The `Error` carries evidence (certificates, availability) and decides
whether a foreign check throws.

---

## 2. Requirements

"Pinned by" names the test or contract line that would fail if the behaviour
changed. Where it says "none", the behaviour exists only in the code.

| # | Requirement | Established by | Pinned by |
|---|---|---|---|
| R1 | A certificate row exists for every equation site, successful ones included. | `_error_check` asserts for every registered bind, `native_sparse.py:806-830`; `sparse_certificates:109-148` ("This includes successful equations") [read] | `tests/test_evidence_v3.py:311-335, 989`: every executed lane and every site is stored with exact bits [helper] |
| R2 | A row tells executed, not executed and skipped apart: `present` false for a site that did not run (status 0, `eta`/numerator/denominator NaN, `performed` false); `performed` false when a prior failure skipped the solve. | `native_sparse.py:133-147`; `performed = prior == 0` in `_callback:525` [read]; contract `docs/v1-runtime-contract.qmd:269-270, 354-357` [helper] | Fields are required in stored rows, `test_evidence_v3.py:82-84`. No test builds an absent or a skipped site [helper] |
| R3 | Static identity of a row: `site`, role, stage, derivative order, dependency. Grouped primitives use `site + ('equation', ordinal)`. `site` is the equation's path in the traced program. | `_annotate_sparse_sites:241-268`, `_error_check:806-830` [read] | `site`, role, stage, derivative order are stored, `test_evidence_v3.py:83` [helper]; `site` text is in a stored reason, `evidence.py:508` [read] |
| R4 | Row order inside a frame is the sort order of `_owned_error_effect_key`: exception class text, payload shapes and dtypes, effect class, `repr(site)`, `repr(dependency)`, role, stage, order, reason. Frames are appended in call order. | `native_sparse.py:40-49, 126`; `evidence.py:346-355, 477-485` [read] | Only through bit-exact stored rows (`test_evidence_v3.py:989`) [helper] |
| R5 | Typed failure, kind, stage and first-failure order come from the status reducers: checks before certificates, program order of stages, lanes in work order. | `jax_core.py:12-28, 61`; `native_program.py:218-226, 252-257` [read]; `evidence.py:159-189, 394`, `physical_analysis.py:1259-1304` [helper] | `test_evidence_v3.py:892-928` (kind `direct_response_formation`, stage `y_to_z`, lane order); `tests/test_root_certificates.py:146-199`; contract `:2997-2999` [helper] |
| R6 | An owned equation failure is evidence and is not thrown; a foreign check failure must still throw with its original message. A program may stop its ordered tail early on any failure. | `physical_analysis.py:56-106`; `native_program.py:222-226` [read]; `evidence.py:358-365` [helper]. Exceptions that throw owned failures: `native_ad.py:52-55` [read], `physical_analysis.py:146` [helper] | `tests/test_native_failure_routing.py:48-118` (built on private `checkify.Error` envelopes) [helper] |
| R7 | The evidence object is a pytree of arrays: an output of `jit`, an *input* of later `jit` calls, sliced per lane on the host; its structure is part of compile keys. | `physical_analysis.py:1029-1073`, `objective_program.py:82-96, 139-143`, `evidence.py:312-343` [helper]; `native_program.py:144-178` [read] | none directly |
| R8 | `vmap` of a certified frame gives per-lane rows; an inner batch gives a further axis; skipped tail lanes are absent. | `native_program.py:236-259`; `_batch:688`, `_nested_batch:700`; `_empty_chunk_outcome:144-178` [read] | Real lanes in `test_evidence_v3.py:323-334` [helper] |
| R9 | Inside `cond`, `while_loop` and `scan` a site keeps one slot: the first failing execution, otherwise the latest; an untaken branch or a zero-trip loop leaves it absent. | JAX's `update_error` and placeholder rule [read]; mirrored in `_archive_error_check:476-504` [read] | none |
| R10 | A Newton body keeps its scalar early exit under `vmap` (`sequential_vmap`); its certificates reach the enclosing frame with a lane axis. No per-iteration history is kept. | `checked_sequential_primal:294-318`, `_archive_layout:354-402`, `_archive_batch:456` [read]; `native_root.py:345-360, 393-394` [helper]; contract `:358-366` | Root family end to end, `test_evidence_v3.py:1048` [helper] |
| R11 | The certificates of a completed root branch belong to the frame that consumes it. | `archive_completed_primal_errors:411-418`, `root_error_frame:421-437` [read]; contract `:2090-2096` [helper] | Root and optimization families, `test_evidence_v3.py:1002-1015` [helper] |
| R12 | Tangent, mixed, adjoint and frequency-tangent equations carry their own certificates: stage 4 (13 for the root's implicit equation), role, derivative order; a successful primal does not certify a failed derivative equation. The functionalizing step wraps the complete transformed computation, pullback included. | `_jvp:643-656`, `_transpose:663-681`, module docstring [read]; `native_program.py:371-441`, `evidence.py:403-515` [read/helper]; contract `:301-304` | Sensitivity family `test_evidence_v3.py:997-1000`; `tests/test_differentiation_rules.py:542-559` (status only). No test names a derivative role [helper] |
| R13 | "Derivative unavailable" is recorded only where a JVP rule ran, with dependency and reason, and marks the dependent outputs unavailable. | `derivative_available:184-196`; callers `native_root.py:408-411`, `native_transfer.py:431-433`, `native_coupling.py:119`; consumer `evidence.py:468-476` [read] | none [helper] |
| R14 | An optimisation step certifies value and gradient in one frame; each iteration keeps only projected certificates. | `objective_program.py:147` and `physical_optimization.py:338-392` [helper] | Optimization family `test_evidence_v3.py:1054`; `tests/test_affine_parameters.py:286-388` [helper] |
| R15 | Payloads keep the selected dtype (32-bit stays 32-bit); no global configuration changes. | local `jax.enable_x64(True)` scopes, `native_sparse.py:119, 283, 304, 478, 611, 811` and their comments [read] | none; every test runs float64 [helper] |
| R16 | Evidence is identical between a memory run and a Workspace run. | Transport is deterministic and storage-independent by construction | `test_evidence_v3.py:930-1055` [helper] |
| R17 | Every registered primitive rule is listed in the rule inventory. | - | `tests/test_differentiation_rules.py:154-167` names `_primal_archive_p` [helper] |

Where equation sites actually sit, measured by instrumenting
`_annotate_sparse_sites` during product tests (152 certified programs,
308 site occurrences; `sites_differentiation.json`, `sites_evidence.json`)
[measured]:

| Enclosing primitives | Site occurrences |
|---|---|
| top level | 185 |
| `cond` (also nested `cond`) | 33 |
| `while` | 33 |
| `scan` | 9 |
| `custom_jvp_call` → `custom_vmap_call` (→ `while`): the sequential Newton body | 24 |
| `cond` → `custom_jvp_call` → `custom_vmap_call` (→ `while`) | 24 |

`jit` (116 047 equations), `custom_linear_solve` and scatter equations never
enclosed a site. `checkify.debug_check` sits inside `cond` and `while`. The
archive primitive sits at top level, inside `custom_jvp_call`, and inside
`cond` → `custom_jvp_call`.

Two corrections to the brief: `tests/test_root_certificates.py` and
`tests/analytic_conditions.py` test root and passivity behaviour, not sparse
equation certificates [helper]. And most of the mechanism (R2, R4, R9, R13,
R15) is pinned by no test at all.

---

## 3. Designs compared

### A. Explicit outputs threaded by the program builder

The builder returns every certificate as an ordinary output and threads it
through its own `cond` / `while_loop` / `scan` calls. This is what the primal
equations already do.

It cannot reach the equations that JAX's AD creates. A JVP rule returns
`(primal_out, tangent_out)` for the same outputs and a transpose rule returns
cotangents; neither has a place for the certificate of the tangent or adjoint
solve. Under `jax.linearize` and `jax.vjp` a primal output may not depend on
a tangent. So pure A meets R12 only if SCNSim stops letting JAX differentiate
through a solve and stages every adjoint itself, which means re-implementing
reverse mode over the frequency program. Rejected as a whole, kept as the
foundation.

### B. A host-side record

The native solve runs on the host and knows its certificate; it writes it into
a per-call collector. Measured with `experiment_alternatives.py` on 0.11.2
[measured]:

| Probe | Host calls seen | Consequence |
|---|---|---|
| Result of the callback unused | 0 | JAX elides the call; no record |
| `vmap` of a `cond` with a solve in each branch, 5 lanes | 10 (both branches, all lanes) | The host cannot know which lane took which branch: `present` (R2, R9) is lost |
| `vmap` of a `while_loop` with trip counts 0..4 | 20 instead of 10 | Lanes that finished still call; per-lane history is wrong |
| `lax.map(batch_size=2)` over 5 | 5 calls, no index | Lane identity is call order only |
| Two independent solves | called in order `[2, 1]` | Call order is not program order |

Further gaps [read/inference]: the dense backend (`_dense_equation`, used for
the stage 13 implicit equation and its adjoint) and the availability
predicates run on the device, so each would need a new host call; a jaxpr that
JAX lowers once and calls from two places shares one callback, so two sites
merge; `pure_callback` is documented as free to be elided, duplicated or
reordered. Every one of these can be patched by passing lane, branch and
iteration identity as runtime operands, but then the structure travels in the
graph anyway and the host record adds nothing. B cannot meet R2, R8, R9, R10
without relying on execution order, which is the kind of undocumented
behaviour the redesign is meant to remove.

### C. Public `checkify` only

`checkify.check(pred, msg, **payload)` accepts arrays and public `checkify`
threads them through control flow. The public read-back is `Error.get()` and
`Error.throw()`. Measured [measured]: with three sites that pass, `get()`
returns `None`; with two that fail it returns one message, the first. All
sites with the same payload type share one slot (the `Error` pytree had 3
leaves for 3 sites). So C returns nothing for a successful equation (R1), one
failure per frame (R8), no presence (R2). Reading the payload dictionary is
exactly the private access in use today. Rejected as the transport; kept for
foreign checks, which is what it is for.

### D. Recommended: explicit outputs, brought to the boundary by SCNSim's own pass

Section 4. It is A made complete: the certificates stay ordinary outputs, and
a pass over the *finished* program does the threading that the builder cannot
do for AD-created equations.

### E. Considered and rejected: AD's own channels

A tangent-equation certificate could ride as the tangent of a dedicated
output, and an adjoint certificate as the cotangent of a dummy input. Both use
only public rules, but both make a "linear" map non-linear, which JAX's
contract for JVP and transpose rules forbids, and every site would need new
plumbing. Not built [inference].

### Comparison

| | A builder only | B host record | C public checkify | D own pass |
|---|---|---|---|---|
| Reverse-mode adjoint solves (R12) | Not reachable | Recorded for sparse; dense adjoint needs a new host call | Failure only | Collected: an adjoint is an ordinary equation in the finished program [measured] |
| Batched lanes (R8) | Yes | Position inside one batch call only; across calls by order | First failing lane only | Yes, lane axes are array axes [measured] |
| `cond` / loops (R9, R10) | Yes for primal | Wrong under `vmap` (table above) | Failure only | Yes; same slot rule as today [measured] |
| Absent vs executed (R2) | Yes | No under batched `cond` | No | Yes, an explicit `present` array [measured] |
| Failure order (R5) | Unchanged | Unchanged | Unchanged | Unchanged: status reducers are untouched |
| Elision, duplication, reordering by the compiler | Not affected | Affected (documented caveat of `pure_callback`) | Not affected | Not affected: certificates are program outputs, so their equations are live |
| Determinism | Yes | Depends on execution order | Yes | Yes |
| Run time | Same | Extra host calls for dense and availability | Same | Equal or faster on small cases, except the sequential map at +12 % [measured] |
| JAX surface | Public | Public | Public write, private read | Public imports; relies on the parameter names of `cond`/`while`/`scan` (section 6) |
| Cannot meet | R12, R13 | R2, R8, R9, R10 | R1, R2, R8, R12 certificates | none found |

---

## 4. The recommended design

Scratch implementation: `certificate_transport.py` in this directory (about
640 lines with comments). It is a proof, not product code.

### 4.1 Evidence is SCNSim's own value

```
Evidence(ledger, foreign)
  ledger : Ledger   # pytree; static keys as aux data, array slots as leaves
  foreign: jax.experimental.checkify.Error   # opaque; only .get() / .throw()

EntryKey(kind, archived, site, role, stage, derivative_order, dependency, reason, descriptors)
Slot(failed, present, values)
  values = (status, eta, numerator, denominator, performed)   for an equation
         = (available,)                                       for availability
```

`present` is an explicit boolean array. It replaces the private "inactive
code" (`_INACTIVE_CODE`, which exists only since JAX 0.11.2). `failed` is the
scalar "this slot already holds a failure" that the slot rule needs.

Because `Ledger` is an ordinary registered pytree, everything the consumers do
to `Error` today works with public operations: output and input of `jit`,
`jax.tree.map(lambda v: v[i], evidence)` for a lane, `jax.eval_shape` plus
zeros for the skipped tail lane, `jax.tree.structure` in a compile key.

### 4.2 The pass (`certified`, the successor of `checked_native`)

1. `jax.make_jaxpr(function, return_shape=True)` on the complete transformed
   function, in the same local `enable_x64` scope as today. This is the same
   trace `checked_native` takes, so equation paths are the same.
2. Evaluate the program equation by equation (as the public
   `jax.core.eval_jaxpr` does):
   - an equation primitive is bound unchanged and its certificate outputs are
     written to the ledger under `(site, role, stage, order, dependency)`,
     with the group rule of `_error_check` (role `tangent`, `mixed` or
     `frequency_tangent`, `site + ('equation', ordinal)`);
   - `cond` with sites inside: each branch is probed once with
     `jax.eval_shape` to learn its keys and slot shapes, slots a branch does
     not write get an absent placeholder, and the equation is emitted again
     with `lax.switch(index, branches, slots, *operands)`;
   - `while` with sites in its body: the slots join the carry of
     `lax.while_loop`; a site in the loop predicate is refused;
   - `scan`: the slots join the carry of `lax.scan`. The split into constants,
     carry and mapped inputs is derived from shapes (a carry keeps its type, a
     mapped input and a stacked output have one more leading axis), because
     the bookkeeping parameters of `scan_p` changed in JAX 0.10 and 0.11;
   - `jit` and `custom_jvp_call` with sites inside are evaluated inline;
   - `custom_vmap_call` (the sequential map) is left untouched: its body was
     certified before it was hidden (4.4);
   - any other primitive that encloses a site raises
     `CertificateTransportError`. The pass fails closed [measured: a site
     inside `jax.checkpoint` raises].
   - every other equation is bound unchanged.
3. Slot rule, identical to today's: `failed' = failed | new_failed`; values
   and `present` keep the old slot if it had failed, otherwise take the new
   one.
4. Wrap the result in public `checkify.checkify(..., errors=user_checks)` so
   that foreign checks are functionalized by JAX itself.

The pass must be applied last with respect to differentiation, exactly as
`checked_native` is today ("Checkify must wrap the complete transformed
computation"). `vmap`, `lax.map` and `scan` over a certified function work
[measured]. Because `custom_jvp_call` is evaluated inline, differentiating a
certified function afterwards would lose the custom rule; the product never
does this [inference from the call sites], and the implementation can refuse
it by wrapping the result in a `jax.custom_jvp` whose rule raises.

### 4.3 Derivative availability: a gate on the data path

Today `derivative_available(flag, ...)` emits a private check effect. A custom
effect type cannot be used instead: an effect must be registered in JAX's
private allow-lists before it may appear inside control flow or a custom
derivative (`effects.control_flow_allowed_effects` and relatives) [read].

`value = derivative_gate(value, available, dependency=..., reason=...)` is a
pure identity primitive on `value` with the predicate as a second operand. The
pass records `(available, present)` and removes the gate. All six call sites
have a value at hand that the rule returns or feeds into its tangent equation
[read]: the implicit denominator (`native_root.py:408, 410`,
`native_transfer.py:431, 433`), the null vector (`native_root.py:492`), the
square root (`native_coupling.py:119`). The gate emits the same number of equations as the
current call (`not` and one primitive), so equation paths do not move
[measured: availability `site` `(4,)` and the following tangent `site`
`(20,)` are the same in both mechanisms].

### 4.4 Sequentially mapped bodies: attach

`certified_sequential_primal` certifies the scalar body, runs it under
`jax.custom_batching.sequential_vmap`, and receives `(evidence, packet)` as
ordinary outputs, as today. Then:

- the foreign `Error` goes back into the enclosing frame through the public
  `checkify.check_error`;
- the ledger is attached to one leaf of the packet (the status) with
  `attach_ledger`, a pure identity primitive whose batching rule moves the
  lane axis to the front. The enclosing pass reads it, marks the entries
  `archived`, and keeps the per-lane `present`.

Closed-over constants are made explicit with the public
`jax.core.eval_jaxpr(program, consts, *operands)`, which replaces the private
`partial_eval.separate_consts`.

For a completed branch (R11) the consuming frame can attach the branch ledger
the same way, or the host can merge it after the call. The host merge is
simpler (the ledger no longer has to be a `jit` input and compile keys stop
depending on its structure) and is the better end state; the attach form is
the smaller first step.

### 4.5 Host projection and failure routing

- `sparse_certificates` and `derivative_availability` become loops over the
  ledger with the same masking as today (status 0 and NaN where absent).
- Row order: `entry_order` reproduces the current sort key from SCNSim-owned
  static metadata, so stored rows keep their order.
- Foreign checks: `evidence.foreign.throw()`. `BatchedError` is no longer
  inspected, because owned failures no longer travel as exceptions.
- In-program gating of the ordered tail (`native_program.py:222-226`):
  `ledger.any_failed()` replaces the loop over `error._pred`. A foreign check
  failure no longer stops the tail inside the program; the host still throws
  it after the call. Only the work done before the throw differs.
- The two places that throw owned failures (`native_ad.py:52-55`,
  `physical_analysis.py:146`) raise from the ledger. To keep type and message,
  SCNSim can raise the public `checkify.JaxRuntimeError` with the same text
  (`sparse <role> equation status <n> at site <site>`); a typed SCNSim error
  would be a public change.

### 4.6 A guard against silent loss

The pass can mark every equation it records (the existing static parameter
`certificate_site` serves). The lowering rule of the equation primitives,
which is SCNSim's own, can then refuse to lower an unmarked equation inside
product execution. A JAX release that hides a body from the pass (JAX 0.12
represents `custom_jvp` as an object, section 6) then fails loudly in CI
instead of losing a certificate. Not built in the scratch code [inference].

---

## 5. The proof

Run from the worktree, with this directory on `PYTHONPATH`:

```
uv run --locked python prove_transport.py        # 42 comparisons + 1 fail-closed check
bash run_shadow.sh <name> <unittest targets>     # shadow run inside product tests
uv run --locked python experiment_site_shift.py  # where `site` moves
uv run --locked python experiment_alternatives.py
```

The harness reads private JAX state only to obtain the *current* mechanism's
registrations and results, and to hide the product's private registrations
while the candidate is traced (a redesigned product has none). The candidate
module imports `jax`, `jax.numpy`, `jax.lax`, `jax.experimental.checkify`,
`jax.extend.core`, `jax.interpreters` and nothing else from JAX.

### 5.1 Small cases [measured]

System: 4 nodes, banded complex sparse pattern, real SCNSim primitives and
host solver. Each case runs the same function through
`jit(checked_native(f))` and `jit(certified(f))` and compares row count, row
order, every static field including `site`, the six arrays bit for bit (dtype,
shape, bytes), the returned values, the owned-failure flag, the foreign
message and the availability facts. 41 of 42 are identical in all of these.
The one difference: a foreign `debug_check` that fails inside a sequentially
mapped body, called with lanes, raises the same exception type with a
different text. Today's archive prints the payload of every lane
(`... at step [2 1 1 1 0]`); the public `check_error` prints the failing
lane's (`... at step 1`). Certificates and values are identical in that case
too. A 43rd check confirms that the pass fails closed.

| Group | Cases | What they show |
|---|---|---|
| Direct | success; singular matrix (status 3, `eta` inf); prior failure skips the second equation (`performed` false); frequency jet (two groups) | R1, R2, R3 |
| Batched | `vmap` of certified, 5 lanes with lane 1 singular; certified of `vmap` (batch primitive); inner `vmap` over 2 right-hand sides | R8 |
| Reverse | value and gradient (primal + adjoint); at a singular matrix (adjoint skipped); two chained solves (4 rows); lanes; adjoint inside a `cond` branch; lanes taking different branches (10 absent entries); dense backend | R12, R9 |
| Forward | `jvp` (tangent); `jvp` of `jvp` (mixed); `linearize` with 3 mapped directions; forward over reverse (adjoint, mixed, primal, tangent) | R12 |
| Control flow | `cond` either branch; `vmap` of `cond`; `while_loop` 3 trips, 0 trips (absent), failure in trip 1 of 4 kept; `scan`; `lax.map(batch_size=2)` over an inner certified body | R9 |
| Availability | holds; fails; lanes; under reverse mode | R13 |
| Sequential map | certified of `vmap`; `vmap` of certified; scalar; trip counts 3, 0, 2, 4, 1 with one failing lane | R10 |
| Foreign and dtypes | passing and failing `checkify.check`, one failing lane of 5 (same message text); `debug_check` inside a `while` body and inside a sequential body; complex64 with ambient x64 off, `eta` stays float32 and the global flag is unchanged | R6, R15 |

Row order is the legacy one, for example `['(22,)', '(7,)']` for adjoint and
primal: the sort is on the text of `site`.

Timing (`time_transport.py`): median of 200 calls, range over 7 rounds that
alternate the two mechanisms; 4-node system, so the host call dominates.
These are diagnostics and set no threshold.

| Case | Current, ms | Candidate, ms | First call incl. compile, s |
|---|---|---|---|
| Direct solve | 0.172 to 0.184 | 0.165 to 0.180 | 0.275 / 0.034 (the first includes process warm-up) |
| `vmap`, 5 lanes | 0.397 to 0.399 | 0.390 to 0.393 | 0.043 / 0.036 |
| Value and gradient | 0.399 to 0.410 | 0.328 to 0.342 | 0.059 / 0.048 |
| `jvp` of `jvp`, 4 equations | 0.866 to 0.879 | 0.588 to 0.598 | 0.074 / 0.065 |
| `while_loop`, 3 trips | 0.557 to 0.566 | 0.552 to 0.559 | 0.051 / 0.054 |
| Sequential map, 5 lanes | 1.524 to 1.555 | 1.715 to 1.794 | 0.094 / 0.077 |

The candidate is equal or faster except in the sequential map, where it is
about 12 % slower (about 0.04 ms per lane). I did not find the cause of either
difference; both should be profiled on real programs when the backend is
switched.

### 5.2 Shadow run inside product tests [measured]

`shadow_product.py` replaces `checked_native` for the duration of a test run
with a wrapper that builds every outermost frame with both mechanisms in the
same traced program, lets the product use the current result, and compares on
the host at every execution. Static metadata and row order are compared at
trace time.

| Run (unittest targets) | Tests | Frames | Rows | Executions | Arrays compared | Differences |
|---|---|---|---|---|---|---|
| `test_evidence_v3`, `test_optimization`, `test_direct_analysis` | 20, pass | 45 | 110 | 162 | 13 784 | 0 |
| `test_workspace_optimization_readback`, `test_affine_parameters`, `test_network_invariants` | 13, pass | 57 | 303 | 119 | 8 758 | 0 |
| `test_differentiation_rules.PublicRuleTests`, `test_root_certificates` | 11, pass | 52 | 40 | 105 | 7 144 | 0 |
| Total | 44 | 154 | 453 | 386 | 29 686 | 0 |

"Rows" are certificate rows with their static metadata compared at trace
time; "arrays" are the six certificate arrays of each row, every leaf of the
frame's returned values (the status reducers' outputs among them), the
foreign-failure flag and the owned-failure flag, compared on the host at each
execution. No candidate trace failed.

The 453 rows by role: 421 primal, 13 frequency tangent, 11 tangent, 8 adjoint.
By position: 391 at top level, 48 inside `cond` (up to three deep), 14 inside
`scan` bodies.

A fourth run (`SHADOW_SEQUENTIAL_BODIES=1`; the root family of
`test_evidence_v3`, `test_diagonal_root`, `test_transfer_zero`; 3 tests, pass)
also compared the scalar Newton body that the product certifies before the
sequential map: 18 frames, 6 of them Newton bodies; 49 rows, 24 of them inside
`while_loop` bodies; 68 executions; 7 902 arrays; 0 differences.

Not covered by the shadow run, because they need the product's call sites to
change: 150 rows that arrive through the product's archive primitive and 8
availability facts. The small cases cover both.

Cost: the third run took 1041 s with the shadow and 645 s without (one run
each). Building, compiling and running every outermost frame a second time,
plus the comparison, therefore cost 61 % of the whole unshadowed test time.
That bounds the candidate's cost for those frames; it does not compare it
with the current mechanism's own share of the 645 s, which I did not isolate.

### 5.3 Where `site` moves [measured]

`experiment_site_shift.py`:

- A custom-JVP root over a sequentially mapped body, followed by a gate and a
  tangent solve: all `site` values equal in both mechanisms. The transport
  equations are inside the `custom_jvp_call`, as in `native_root.py:396-433`.
- A frame that consumes a completed branch: today
  `archive_completed_primal_errors` emits 12 equations before the frame's own
  solve, whose `site` is `(12,)`; with attach it is `(0,)`. Certificate arrays
  identical.

---

## 6. How fragile the current dependence is

Text comparison of `jax/_src/checkify.py` and neighbours at ten release tags
from the public repository (`compare_jax_internals.py`,
`jax_internals_table.txt`) [read].

| Internal SCNSim uses | 0.4.30 to 0.11.0 | 0.11.2 | 0.12.0.1 |
|---|---|---|---|
| `checkify._INACTIVE_CODE` | does not exist; placeholder code is the literal `-1` | `-2` | `np.int32(-2)` |
| Rule signature in `error_checks`, `assert_func` | `(error, enabled_errors, ...)`, `assert_func(error, pred, new_error)` | `(ctx, error, enabled_errors, ...)`, `assert_func(ctx, ...)`; `CheckifyContext` is new | same as 0.11.2 |
| Order in which placeholder effects are admitted | iteration order of a Python set | `sorted(effects)` | same |
| `check` discharge | no code remapping | remaps codes into the context's namespace (`_archive_error_check` mirrors it) | same; codes are now `np.int32` |
| `ErrorEffect` | defined in `checkify.py` | moved to `effects.py` | same |
| `Error` class text | changed at 0.6.2, 0.9.2, 0.10.2 | changed | same |
| `Error.get_exception` text | changed at 0.9.2, 0.10.2 | same as 0.10.2 | same |
| Placeholder payload | ones | ones | ones |
| `partial_eval.separate_consts` | absent up to 0.7.2, present from 0.8.3 | present | present |
| `scan` parameters seen by a rule | `num_consts, num_carry, linear, _split_transpose` up to 0.9.2; `linear`, `_split_transpose` dropped in 0.10.2; `ft_in, ft_out` from 0.11.0 | `ft_in, ft_out` | same |
| `Jaxpr` and `ClosedJaxpr` | separate classes | one class since 0.11.0 | `jax.extend.core.ClosedJaxpr` no longer exported |
| `custom_jvp` in `checkify` | `custom_jvp_call` rule | same | additionally handled as a `CustomJVPTraced` object |
| `xla_bridge.backends_are_initialized` | present | present | present |

325 lines of `checkify.py` differ between 0.11.0 and 0.11.2. The current code
therefore works on 0.11.2 and nowhere before it: neither `CheckifyContext`
(the `ctx` argument of `_error_check`) nor `_INACTIVE_CODE` exists in 0.11.0
or 0.11.1 [read; not run]. The changes landed inside one minor series. Whether the current code runs on 0.12 was not
tried; the code dtype change touches `derivative_available`, which builds a
code with `jnp.asarray(0)`.

### What the recommended design still relies on

Stated plainly, because "public imports" is not the same as "documented":

| Touch point | Kind | Across 0.5.3 to 0.12.0.1 [read] |
|---|---|---|
| `jax.make_jaxpr`, `jax.eval_shape`, `lax.switch/while_loop/scan`, `jax.tree`, `checkify.checkify/check_error/user_checks`, `Error.get/throw`, `jax.core.eval_jaxpr`, `sequential_vmap` | documented API | present throughout |
| `jax.extend.core`: `Primitive`, `Literal`, `jaxpr_as_fun`, `primitives.cond_p/while_p/scan_p/custom_jvp_call_p` | public, less stable namespace | present throughout; the nested-`jit` primitive is exported as `jit_p` from 0.7.2, as `pjit_p` in 0.6.2, not at all in 0.5.3 (no product program needed it: no `jit` equation enclosed a site) |
| `jaxpr.eqns/invars/outvars/constvars`, `eqn.primitive/params/invars/outvars`, `eqn.ctx.manager` | attributes of public classes | present throughout; consts are attached to the `Jaxpr` from 0.11.0, before that a `ClosedJaxpr` must be opened |
| `Primitive.get_bind_params` | method of a public class | returns `(subfuns, params)` up to 0.8.3, `params` from 0.9.2 (the scratch code accepts both) |
| Parameter names `branches`; `cond_nconsts, cond_jaxpr, body_nconsts, body_jaxpr`; `jaxpr, length, reverse, unroll` | undocumented, visible in every printed jaxpr | unchanged throughout |
| Operand order of `scan` (constants, carry, mapped) | undocumented | unchanged [read in 0.11.2 only] |
| The name `custom_vmap_call` and its parameter `call` | undocumented | unchanged throughout |

This is far less than today (no private class is subclassed, no private table
is written, no private field is read, no internal ordering or placeholder
value matters), but it is not zero. The answer to the remainder is the guard
of 4.6 plus a CI matrix over the supported JAX range, with the pass's own
tests as the first thing that runs. The candidate was run on 0.11.2 only.

---

## 7. What the redesign touches

### 7.1 Files and functions

| File | Change |
|---|---|
| `numerics/native_sparse.py` | Remove: `from jax._src import checkify`, the four effect and exception subclasses, `ArchivedPrimalCheck`, `_availability_tree`, `_annotate_sparse_sites`, `_primal_archive_p` and its rules, `_archive_layout`, `_archive_error_check`, `_error_check`, the `separate_consts` import. Replace: `checked_native`, `checked_sequential_primal`, `sparse_certificates`, `derivative_availability`, `derivative_available` (becomes the gate and returns a value), `archive_completed_primal_errors`, `root_error_frame`, `register_equation_certificate` (schema registry only). Unchanged: every equation primitive, its lowering, JVP, transpose and batching rules, `checked_sparse_solve`, `checked_dense_solve`, `SparseCertificate`. The ledger and the pass are better placed in a new module of their own. |
| `numerics/native_pair.py`, `numerics/native_equation_jet.py` | None beyond the registry call staying the same. |
| `numerics/native_root.py` (408-413, 474-492), `numerics/native_transfer.py` (429-435), `numerics/native_coupling.py` (114-119) | `derivative_available(flag, ...)` becomes `value = derivative_gate(value, flag, ...)`. |
| `numerics/native_program.py` | `_empty_chunk_outcome` (144-178): placeholders from `eval_shape` and zeros, no `_INACTIVE_CODE`. Ordered tail (222-226): `ledger.any_failed()`. `archive_completed_primal_errors` call (1007-1009). |
| `numerics/evidence.py` | `native_primal_exception`, `native_error_may_raise` (109-134); axis checks on `error._code` (332, 439); `_raise_native_unexpected` (358-365) and the same pattern at 442-446 become `foreign.throw()`; the two projection functions read the ledger. |
| `execution/physical_analysis.py` | 56-106 (`BatchedError` routing becomes `foreign.throw()`), 124-146, 742-763, 989-1073 (compile key from the evidence structure), 1181-1316, 1719-1756. |
| `execution/native_ad.py` | `_checked_failure` (52-55), lane slicing, variant keys. |
| `execution/warmup.py`, `compilation/objective_program.py`, `execution/physical_optimization.py` | Pass the new value through; keys from its structure. |
| `execution/config.py` | Section 9. |
| `tests/test_native_failure_routing.py` | Rewritten: it builds private `checkify.Error` envelopes. Meaning kept: owned failures are not thrown, foreign and mixed ones are, payload bits unchanged. |
| `tests/test_differentiation_rules.py` | Inventory: `_primal_archive_p` rules leave, gate and attach rules enter. |
| `docs/v1-runtime-contract.qmd` | Wording at `:358-366` and `:2090-2096` ("pinned JAX checkify boundary", "Error archive", "owned Error PyTree metadata"). |

### 7.2 Steps, each leaving the suite green

1. **Pin today's behaviour first.** Add tests for what no test pins (R2, R4,
   R9, R12 roles, R13, R15), written against `checked_native`,
   `sparse_certificates` and `derivative_availability`, not against `Error`
   internals. `prove_transport.py` is the seed. They pass on the current
   mechanism and are the acceptance tests of every later step.
2. **`config.py`** (independent of the rest; section 9).
3. **Add the ledger module and the pass beside the current mechanism.** No
   caller changes. Tests compare both mechanisms on the step 1 cases.
4. **Put a narrow interface between consumers and the evidence object**
   (lane slice, empty value, in-program "any failure", throw foreign, filter
   by dependency, project rows), implemented on today's `Error`. Move
   `evidence.py`, `native_program.py`, `physical_analysis.py`, `native_ad.py`,
   `objective_program.py`, `warmup.py`, `physical_optimization.py` onto it.
   Rewrite `test_native_failure_routing.py` against it. No behaviour change;
   `BatchedError` and `_INACTIVE_CODE` now live in one place.
5. **Move the call sites to the gate and attach forms**, still backed by the
   current mechanism.
6. **Switch the backend** of `checked_native` to the pass. Run the comparison
   the Human assigns for this milestone. This is the step where `site` in
   consuming frames changes. The shadow harness of this study
   (`shadow_product.py`) can be rerun before the switch as a local check on
   any test target.
7. **Delete the old mechanism**, add the lowering guard (4.6), update the
   inventory test and the contract wording. `git grep "jax\._src" -- src` is
   now empty.
8. **Widen the JAX range**: CI matrix over the lowest, the pinned and the
   newest JAX; add the version shims the matrix asks for; then the dependency
   change itself, which is the Human's.

### 7.3 Tests

Unchanged in meaning: `test_evidence_v3.py` (all five families, exact bits,
memory against Workspace), `test_solve_diagnostics.py`,
`test_root_certificates.py`, `test_differentiation_rules.py` public rule
tests, `test_affine_parameters.py`, the readback tests. `test_evidence_v3.py`
reaches into private SCNSim names (`packet.errors`, `_execute_native_chunk`
and others, lines 499-542) [helper]; keeping those names keeps the test
untouched.

New: the step 1 characterization tests; the pass's unit tests (each rule, the
fail-closed error, the slot rule, placeholders, 32-bit, nested certification);
a test that an unmarked equation cannot be lowered; a test that the candidate
module's sources contain no `jax._src`.

---

## 8. What changes

| Subject | Changes? |
|---|---|
| Certificate numbers (status, `eta`, numerator, denominator, `performed`, `present`), their dtypes | No [measured] |
| Role, stage, derivative order, dependency of each row | No [measured] |
| Failure kind, stage, first-failure order | No: they come from the status reducers, which the redesign does not touch [read; values measured equal in the shadow run] |
| `site` and row order in frames with no transport equations before their sites, and in root JVP rules | No [measured] |
| `site`, and through it row order and the reason text `checked derivative equation failed at <site>`, in frames that consume a completed root branch | **Yes** [measured on a small case]. Decision needed. |
| Exception type and message of a foreign check | No, with one exception: inside a sequentially mapped body called with lanes the type is the same and the text names the failing lane's value instead of all lanes' [measured] |
| Exception type of the two owned throws | No, if SCNSim raises `JaxRuntimeError` with the same text; a decision if a typed error is preferred |
| Precedence when an owned failure and a foreign check fail in the same frame | Today JAX's internal code order decides at `physical_analysis.py:65-67`; the redesign needs an explicit rule. Suggested: foreign first, since it means an internal invariant broke. Decision needed. |
| Work done after a foreign check fails in an ordered tail | The tail no longer stops inside the program; the host throws as before |
| Compile keys | Their text changes (structure of the new value); internal |
| Public API | None |

On `site`: three options, in order of preference.

1. **Count only equation sites.** `site` becomes the ordinal path of SCNSim's
   equations (for example `(2,)` for the third equation of a frame,
   `('branches', 1, 0)` inside a branch), not of all JAX equations. Stored
   values change once; afterwards they no longer depend on how a JAX version
   lowers ordinary arithmetic, nor on transport equations. Smallest change
   that makes `site` meaningful across a JAX range [inference].
2. **Keep the path definition** and accept that `site` differs between JAX
   versions and changed once in consuming frames. Document `site` as a label
   that is comparable only within one build.
3. **Names given by the builder.** Most stable, largest change.

---

## 9. The other private uses

| Use | Public replacement |
|---|---|
| `checkify._INACTIVE_CODE`, `native_program.py:148` | Gone: an absent lane is a ledger whose `present` arrays are false, built from `jax.eval_shape` and zeros. |
| `checkify.BatchedError`, `physical_analysis.py:76` | Gone: owned failures are never exceptions; a foreign `Error` is thrown with its public `throw()`. |
| `partial_eval.separate_consts`, `native_sparse.py:309` | `jax.core.eval_jaxpr(program, consts, *operands)` with the constants as explicit leading inputs [measured in the sequential cases]. `jax.interpreters.partial_eval.convert_constvars_jaxpr` also exists but its public re-export is missing in 0.10.2 and 0.11.0 [read]. |
| `xla_bridge.backends_are_initialized` through `sys.modules`, `config.py:46-50` | No public equivalent exists: `jax.extend.backend` exports `backends`, `get_backend` and `clear_backends`, all of which initialize or reset [read]. SCNSim already records its own initialization (`mark_jax_initialized`, `config.py:108`). Recommended: use that record alone. What is lost is one refusal: when other code initialized JAX before `configure_runtime(cpu_threads=...)`, the request is accepted although it cannot take effect. State this precondition in the documentation, and ask upstream to export the query from `jax.extend.backend`. This changes when `RuntimePreparationError` is raised, so it is the Human's decision. |

---

## 10. Risks

1. **Row order and `site` are reproduced from static metadata.** The legacy
   order depends on class-name text and payload shapes. Any slip changes
   stored rows. Mitigation: step 1 tests and the milestone comparison.
2. **A JAX release hides a body from the pass.** JAX 0.12 already represents
   `custom_jvp` differently inside `checkify` [read]. Mitigation: the lowering
   guard (4.6), fail-closed rules, CI matrix. The same change would silently
   mislabel sites in the current mechanism.
3. **Parameter names of control-flow primitives** (section 6). Stable for the
   whole range read, undocumented.
4. **Trace and compile time.** The program is interpreted twice (the pass,
   then public `checkify`). Small cases show no difference in first-call
   time, and the shadow run gives only a loose bound (section 5.2); trace and
   compile time on real programs were not measured on their own. The public
   wrap can be skipped when a program holds no foreign check.
5. **"Certify last."** Inline evaluation of `custom_jvp_call` loses the custom
   rule if someone differentiates afterwards. Mitigation: refuse it (4.2).
6. **Coverage of the proof.** The shadow runs met 14 rows inside `scan`
   bodies, 48 inside `cond` and 24 inside Newton `while_loop` bodies; the
   tests use few directions and small circuits. The attach and gate
   primitives were proven on small cases, not inside the product.
7. **The size of step 4.** About ten modules read `Error` internals. The
   narrow interface makes each move mechanical, but it is the largest diff.
8. **A fact follows its value.** The gate and the attach primitive are pure,
   so JAX may drop them when the value they sit on is unused, where today's
   check effect always survives. The values chosen (the implicit denominator
   that feeds the tangent solve, the packet status that feeds the status
   reducers) are used whenever the result is used [inference]. Equation sites
   already behave this way today.
9. **The sequential map was 12 % slower on the small case** (section 5.1),
   cause not found.

---

## 11. Not determined

- Whether the candidate, or the current mechanism, runs on any JAX other than
  0.11.2. Only sources were compared.
- The milestone comparison (38 cases) and the benchmarks were not run, as
  instructed. Whether its golden data pins `site`.
- Trace and compile cost of the pass on real programs, separated from the
  rest; and the cause of the run-time differences on the small cases (faster
  under AD, slower in the sequential map).
- Whether any product frame is mapped with `vmap` over a `cond` that encloses
  a site (the case where design B is wrong and D is right); the design does
  not depend on the answer.
- GPU execution.

---

## 12. Files in this directory

| File | Purpose |
|---|---|
| `certificate_transport.py` | The candidate transport (public JAX imports only) |
| `prove_transport.py`, `proof_results.json` | Small-case comparison and its results |
| `shadow_product.py`, `run_shadow.sh`, `shadow_*.json`, `shadow_*.log` | Shadow comparison inside product tests |
| `time_transport.py` | Interleaved timing of both mechanisms |
| `experiment_site_shift.py` | Where `site` moves |
| `experiment_alternatives.py` | Measurements for designs B and C |
| `instrument_sites.py`, `sites_differentiation.json`, `sites_evidence.json`, `show_sites.py` | Which primitives enclose equation sites |
| `fetch_jax_sources.sh`, `jaxsrc/`, `checkify_0.11.1.txt`, `compare_jax_internals.py`, `compare_pass_surface.py`, `jax_internals_table.txt` | JAX source comparison across releases (fetched text, never imported) |
