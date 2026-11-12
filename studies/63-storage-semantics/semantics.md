# Memory and Workspace: what a Result exposes as evidence, and how a failure reaches the caller

SCNSim runs a calculation in memory (`CircuitRun(plan=..., storage="memory")`) or in a Workspace on disk (`storage="workspace"`). This report measures where the two give a different public Result, quotes what the documents promise, explains why the code behaves as it does, and sets out the choices for the owner of the project.

Every statement carries its basis: **[M]** measured (a script here ran it through public calls), **[R]** read (in the code or a document, with file and line), **[I]** inferred. Measurements were made on `develop` at commit `7db7c53`, JAX 0.11.2, float64, CPU.

## The answer

**The numbers agree; the evidence and the failures do not.** In all 54 cases in which both storages returned a Result, over every operation family, every public number is bit-identical between Memory and Workspace, and every Workspace Result is bit-identical after `run.close()`, after reopening and in a fresh process. **[M]** (`value_check.log`) The differences are all in what is exposed as evidence and in how a failure surfaces:

1. **Evidence of four families.** `result.evidence` of an operator-element root, a hybridized pole, a transfer zero and a residue-normalised coupling is `{}` in Memory and the stored record in a Workspace (25, 24, 31 and 17 or 18 keys). **[M]** The Workspace record is exactly the record Memory already holds elsewhere (`result.evaluation.evidence_bytes`), bit for bit, plus `selected_state` and a few keys the reader adds (`null_vector`, `row`, `column`, `family`, `input`, `output`, `branches`). **[M]** So the only fact a Workspace has and Memory lacks is `selected_state`.
2. **Evidence of a failed root.** A Workspace failure carries `selected_state`; a Memory failure does not. All other keys are bit-identical. **[M]**
3. **Raised or returned.** A single-point `evaluate` whose requested point fails is returned as a failed Result in Memory and raised in a Workspace. This holds for every `evaluate` family, not only roots: the operator, a response element and static capacitance behave the same way. **[M]**
4. **A different failure for the same call.** For a single-point call that fails, a Workspace often raises a different record from Memory: another message, another set of evidence keys and, for a transfer zero whose anchor fails, another `stage` (`dut_voltage_constraint` in a Workspace, `transfer_denominator` in Memory). **[M]**
5. **Other public carriers of evidence exist in one storage only.** Memory Results expose `evaluation`, `dependencies`, `context`, Sensitivity `certificates` and optimization `bodies`; Workspace Results expose none of these. **[M]**

**What the documents say.** The runtime contract says storage selection changes "result ownership and publication, not … failure semantics", and that a started operation raises its typed failure. One guide tells Memory users to read `result.status` and `result.failure` instead. An implementation page says the pole's null vector is in the public evidence in both storages, which is false for Memory. No document defines the keys of `evidence`, says who may rely on it, or mentions `selected_state`. **[R]**

**Why.** Three separate causes. The in-memory Result builder sets `evidence` to `{}` and keeps the record under `evaluation`. Memory producers drop `selected_state` on purpose, because it is the solver's working state; the Workspace stores and returns the whole packet. And a single-point `evaluate` runs through two different producer paths: Memory uses the sweep producer, a Workspace uses the older point producer. **[R] [M]**

**Recommendation.** Make both storages expose the same record without `selected_state`, and stop storing `selected_state` (option 2b below). Treat raising as the specified behaviour for a single-point failure and make Memory raise. Schedule the two producer paths as a defect of its own, because they give different failure stages for one call.

## How it was measured

`measure.py` runs 85 cases. Each case is one public call (`solve`, `evaluate`, `sensitivity` or `optimize`) on a small Plan, run in Memory and in a Workspace. The Workspace Result is summarised four times: as returned, after `run.close()`, through `run.resolve` on a reopened Workspace, and through `run.resolve` in a fresh process. The Memory Result is summarised as returned and after `run.close()`. A summary holds every public number, the keys and exact bits of every public evidence mapping, and for a failure its type, kind, stage, message and evidence. `compare.py` builds the tables in `tables.md`.

The Plans are the two-node fixture of `tests/test_workspace_readback.py` (an LC tank behind a coupling capacitor and a 50 Ω Port) and the two coupled LC tanks of `tests/test_evidence_v4.py`. Failures come from public inputs only: a capacitance of 1e308 fF at the requested point or at the Plan baseline, an isolated capacitor pair in the eliminated complement, two identical uncoupled tanks, a constant cross entry, a root hint of 1e30 Hz, the admittance of a one-port (which has no transfer zero), and residue slopes whose product underflows.

A second, independent run of 72 of the cases gave the same summary in every reading; the only field that differed was a timing value. **[M]** (`repeat_check.log`)

## Table 1. Every difference between Memory and Workspace

All rows measured. "Single" is a call with no parameters or one `ParameterSet`; "sweep" is a call with a `ParameterSpace`.

| # | What differs | Where | Memory | Workspace |
|---|---|---|---|---|
| D1 | `result.evidence`, success | Operator-element root, single and sweep point | `{}` | 25 keys: the root record, `selected_state`, `row`, `column` |
| D2 | `result.evidence`, success | Hybridized pole, single and sweep point | `{}` | 24 keys: the root record, `selected_state`, `null_vector` |
| D3 | `result.evidence`, success | Transfer zero, single and sweep point | `{}` | 31 keys: the root record, `selected_state`, `family`, `input`, `output` |
| D4 | `result.evidence`, success | Residue coupling, single and sweep point | `{}` | 17 keys (single) or 18 keys (sweep point), `branches` included; each branch holds its root record with `selected_state` |
| D5 | One key of the coupling record | Residue coupling, single | The record has `equation_certificates` | The single-point record lacks it; the sweep-point record has it |
| D6 | `failure.evidence` of a failed root | Diagonal root, operator-element root, pole, zero, coupling branch: single raised, sweep point, sweep baseline | Root record, 22 keys (27 for a zero) | The same bits plus `selected_state` |
| D7 | Raised or returned | Single-point `evaluate` whose requested point fails: operator, response element, static capacitance, diagonal root, operator-element root, pole, zero, coupling | Returned: `result.status == "failure"`, `result.failure` holds the error | Raised |
| D8 | Failure message and evidence keys | Single-point operator and response element | "…existing solve contract", 7 keys | "…existing solve/root contract", 4 keys |
| D9 | Failure message, evidence keys and, for the zero, `stage` | Single-point root that fails at the Plan baseline outside its Newton iteration: a diagonal root whose eliminated block cannot be solved, a transfer zero of a response that has none | Root record: 22 or 27 keys; zero: stage `transfer_denominator` | Generic record of 4 keys; zero: stage `dut_voltage_constraint` |
| D10 | Form of `failure.evidence["certificates"]` | Single-point Direct `solve` that fails | 15 fields per record, plain scalars and strings such as `"+inf"` | 10 fields per record, float64 arrays |
| D11 | Public attributes that carry evidence | Every `evaluate` Result; Sensitivity; Optimization | `evaluation` (with `evidence_bytes`, `null_vector`, `failure`), `dependencies`, `context`, `status`, `failure`; Sensitivity `certificates`; trial `bodies`, `certificates` | None of these. Only the baseline row of `stage.history` carries bodies, with `selected_state` |
| D12 | Key name inside a failed optimization trial | Optimization with failing trials | `trial.failure["evidence"]` | `row["failure"]["evidence_bytes"]` (a mapping despite its name), same 4 keys |
| D13 | A root with an explicit search cannot be stored | Diagonal root with `RootSearchSpec(method="finite_seeds")` | Returns the root | Raises `SCNSimValidationError`, stage `canonical_identity`, evidence `{"type": "complex"}` |

D13 is outside the question asked but appeared in the measurement; only this one search configuration was tried.

## Table 2. Where there is no difference

All rows measured.

| Family and form | What was compared and found bit-identical |
|---|---|
| Direct `solve`, single and sweep | S, Y, Z, frequencies, `evidence` (`{}`). A failing sweep point: type, kind, stage, message and evidence. A preflight failure (`PortRealizabilityError`): everything |
| Operator, single and sweep | Operator matrices, `evidence` (`{}`). A failing sweep point: everything |
| Response element, single and sweep | Value, `evidence` (`{}`). A failing sweep point: everything |
| Static capacitance, matrix and element | Matrix or value, coordinate axes, `evidence` (11 keys) |
| Diagonal root, single and sweep | Root, frequency, linewidth, slope, `evidence` (`{}` in both) |
| Operator-element root, pole, zero, coupling | Every public number. Only `evidence` differs (D1 to D5) |
| Sensitivity: Jacobian, JVP and VJP of a response, a root and a pole; Jacobian of a coupling and of a zero | Status, primal blocks, derivative blocks, block failures, `diagnostics`. A failing call is raised in both with the same type, kind, stage, message and evidence |
| Optimization: CMA-ES on a root, a pole, a coupling, a zero and a response; L-BFGS-B on a root | Status of every history row, cost of every row, best cost and ordinal. A failing baseline is raised in both with the same failure |
| A sweep whose root baseline fails | Raised in both with the same type, kind, stage and message (evidence: D6) |
| A sweep with a failing point | Returned in both; `point.succeeded`, `point.failure`; `point.result` raises the failure in both (evidence: D6) |
| A baseline failure of a single-point operator-element root, pole or coupling | Raised in both with the same type, kind, stage and message (evidence: D6; coupling identical) |

Affine evidence (`result.evidence["affine_evidence"]`) was not measured again; the existing tests compare it between the storages. **[R]** (`tests/test_evidence_v4.py:1850-1855`)

## Table 3. Public evidence of a successful call

All cells measured on the two-node fixtures. A single point and each sweep point agree unless stated. "Record" is the JSON document in `result.evaluation.evidence_bytes`; its size is the length of that document.

| Family | Memory `result.evidence` | Memory record and other carriers | Workspace `result.evidence` |
|---|---|---|---|
| Direct solve | `{}` | Single: `metadata`, 22 keys, no certificates. Sweep point: record of 4 keys, 11,363 B, with the work evidence | `{}` |
| Operator | `{}` | Record of 5 keys, 1,644 B, with certificates and work evidence | `{}` |
| Response element | `{}` | Record of 5 keys, 1,463 B | `{}` |
| Static capacitance | 11 keys | The same 11 keys | 11 keys, same bits |
| Diagonal root | `{}` | Record of 22 keys, 2,872 B | `{}`. The stored body holds the packet with `selected_state`, but it is not exposed |
| Operator-element root | `{}` | Record of 22 keys, 2,960 B | 25 keys, 719 B of arrays: `selected_state` 391 B |
| Hybridized pole | `{}` | Record of 22 keys, 1,921 B; `evaluation.null_vector`, 32 B | 24 keys, 585 B of arrays: `selected_state` 304 B, `null_vector` 32 B (same bits as Memory's) |
| Transfer zero | `{}` | Record of 27 keys, 6,387 B | 31 keys, 1,113 B of arrays: `selected_state` 493 B |
| Residue coupling | `{}` | Record of 17 keys, 12,437 B; `dependencies` holds the two branch roots, 22 keys each | Single: 17 keys, 2,668 B of arrays. Sweep point: 18 keys, 2,968 B |
| Sensitivity | No `evidence` attribute; `diagnostics` is `{}` | `certificates`: 1 to 21 records, 30 to 630 B; `summary` | No `evidence` attribute; `diagnostics` is `{}`; no certificates |
| Optimization | Trial objects | Each history row: `bodies` (one record per root), `certificates` | History rows are mappings; only the baseline row has `dependencies` |

For the three root families, the common keys of the Workspace evidence and of the Memory record have identical bits; the Workspace has `selected_state` and the reader's keys in addition, and Memory has nothing the Workspace lacks. For the coupling, the Workspace adds `branches` and the single-point Workspace record lacks `equation_certificates`. **[M]** (`tables.md`, T3)

Every certificate record inside `selected_state` is also present at the top level of the same packet (3 of 3 and 9 of 9 in the four packets checked). What `selected_state` adds is working arrays: `F`, `Fp`, `Q_values`, `Qp_values`, `X`, `Xp`, `bound_values` and `elimination_eta` for a root or a pole; `H_values`, `Hp_values`, `V`, `Vp`, `Y`, `Yp` and a nested `selected_state` for a transfer zero. Its layout depends on the family. **[M]** (`state_check.log`)

## Table 4. Failing calls

All cells measured. "22 keys" is the root record; "+ `selected_state`" means the same bits plus that key.

### A single-point `evaluate` whose requested point fails

| Family, input | Memory | Workspace |
|---|---|---|
| Operator, isolated pair in the complement | Returned. `EliminatedBlockSolveFailure`, stage `eliminated_block`, 7 keys | Raised. Same type, kind and stage; other message; 4 keys |
| Response element, 1e308 fF | Returned. `DirectResponseFormationError`, stage `y_to_z`, 7 keys | Raised. Same type, kind and stage; other message; 4 keys |
| Static capacitance, matrix and element, isolated pair | Returned. `InMemoryStaticCapacitanceResult` with `status == "failure"`; its `evidence` holds the 4 failure keys | Raised. `EliminatedBlockSolveFailure`, same 4 keys, same bits |
| Diagonal root, 1e308 fF | Returned. `EliminatedBlockSolveFailure`, stage `eliminated_block`, 22 keys | Raised. Same failure + `selected_state` |
| Operator-element root, 1e308 fF | Returned. `NumericalResolutionUnresolved`, stage `newton_certificate`, 22 keys | Raised. Same failure + `selected_state` |
| Hybridized pole, 1e308 fF | Returned. `RootSlopeUnresolved`, stage `determinant_scaling`, 22 keys | Raised. Same failure + `selected_state` |
| Transfer zero, 1e308 fF | Returned. `DirectResponseFormationError`, stage `transfer_denominator`, 27 keys | Raised. Same failure + `selected_state` |
| Residue coupling, 1e308 fF | Returned. `NumericalResolutionUnresolved`, stage `newton_certificate`, 22 keys (a branch root) | Raised. Same failure + `selected_state` |

A returned Memory failure stays readable after `run.close()`; its `result.benchmark.status` is `"success"`. After a raised Workspace failure, `run.resolve` raises `ResultUnavailableError` (stage `resolve`) on a reopened Workspace and in a fresh process. **[M]**

### A single-point root that fails at the Plan baseline

Both storages raise.

| Family, input | Memory | Workspace |
|---|---|---|
| Diagonal root, baseline 1e308 fF | `EliminatedBlockSolveFailure`, stage `eliminated_block`, "root did not satisfy its existing local Newton/closure contract", 22 keys | Same type, kind and stage; "numerical operation did not satisfy its existing solve/root contract"; 4 keys (`certificates`, `failed_site`, `realized_capabilities`, `status`) |
| Diagonal root, hint 1e30 Hz | `NumericalResolutionUnresolved`, stage `newton_certificate`, 22 keys | Same failure + `selected_state` |
| Operator-element root, constant cross entry | `RootSlopeUnresolved`, stage `newton`, 22 keys | Same failure + `selected_state` |
| Hybridized pole, repeated mode | `NumericalResolutionUnresolved`, stage `rank_certificate`, 22 keys | Same failure + `selected_state` |
| Transfer zero, admittance of a one-port | `DirectResponseFormationError`, stage `transfer_denominator`, 27 keys | `DirectResponseFormationError`, stage `dut_voltage_constraint`, other message, 4 keys |
| Residue coupling, underflowing slopes | `RootSlopeUnresolved`, stage `residue_coupling`, 16 keys | Identical |

### Other forms

| Form | Memory | Workspace |
|---|---|---|
| Sweep with a failing point (Direct, operator, response) | Returned; failed point with its failure | Identical |
| Sweep with a failing root point (diagonal root, operator-element root, pole, zero, coupling) | Returned; failed point, 22 or 27 keys | Returned; same failure + `selected_state` |
| Sweep whose root baseline fails (diagonal root, zero) | Raised, root record | Raised, same failure + `selected_state` |
| Direct `solve`, single, 1e308 fF or 1e308 Hz | Raised, `DirectResponseFormationError`, stage `dut_voltage_constraint`, 7 keys | Raised, same type, kind, stage, message and keys; `certificates` in another form (D10) |
| Sensitivity, failing point or baseline | Raised | Identical |
| Optimization, failing baseline | Raised | Identical |
| Optimization, failing trials | Returned, rows S F F F F | Returned, same rows (D12) |
| Static capacitance with a `ParameterSpace` | Raised, `RuntimePreparationError`, stage `preflight` | Identical |

## Table 5. What `selected_state` and `null_vector` cost per root

All cells measured by `cost.py`. The ladder is a chain of grounded LC tanks joined by 2 fF capacitors, built with the helpers of `benchmarks/circuits.py`; the line is that file's ordered two-conductor line (`C4`), whose 800-section case is the largest benchmark circuit. Two coordinates are retained. Stored bytes were read from the SQLite file with the product's block decoder, which is not a public call.

| Circuit | `selected_state`, bytes of arrays | `null_vector` | Rest of the root record | Stored evidence object | Memory record today |
|---|---:|---:|---:|---:|---:|
| Two-node fixtures | 304 (pole), 391 (operator-element root), 493 (zero) | 32 | 249 to 620 | 10,854 (pole), 19,657 (diagonal root) | 1,921 to 6,387 |
| Ladder, 50 tanks | 9,223 | 32 | 336 | 28,854 | 2,973 |
| Ladder, 200 tanks | 36,823 | 32 | 336 | 56,484 | 2,973 |
| Ladder, 800 tanks | 147,223 | 32 | 336 | 166,920 | 2,973 |
| Line, 50 sections | 31,111 | not reached | 336 | 50,956 | 2,960 |
| Line, 800 sections | 487,111 | not reached | 336 | 506,997 | 2,960 |

- **`selected_state` grows with the eliminated part of the circuit; the rest does not.** At 800 line sections it is 96% of the stored evidence object. `null_vector` is 16 bytes per retained coordinate. **[M]**
- **A diagonal root stores `selected_state` although its public evidence is `{}`.** Each stored diagonal-root body contains one. **[M]**
- **A failed single-point call stores it five times.** The failed pole on the 800-section line left 2,435,555 B of `selected_state` in the operation, the attempt, two task changes and the value object. **[M]**
- **Memory keeps none.** The Memory record of a root does not grow with the circuit: 2,960 to 2,973 B from 50 to 800 tanks. **[M]**

## Table 6. How many roots an optimization keeps

All cells measured by `opt_cost.py` with the CMA-ES workload of `benchmarks/circuits.py` (seed 7, population 8, 64 evaluations) on the two-node fixtures.

| Objective | History rows | Bodies kept in Memory | Memory record bytes, total | Bodies with `selected_state` in Memory | `selected_state` stored in the Workspace |
|---|---:|---:|---:|---:|---|
| Diagonal-root frequency | 57 | 57 (one per row) | 163,704 | 0 | 57 copies, 18,639 B |
| Pole frequency | 57 | 57 (one per row) | 109,497 | 0 | 60 copies, 18,240 B |
| Coupling magnitude | 57 | 171 (two branch roots and the coupling per row) | 654,588 | 0 | 114 copies, 44,574 B |

A Memory optimization keeps one root record for every root of every evaluated point, for as long as the Result lives. If each carried `selected_state`, the 57 roots of this workload would add 57 × 487,111 B = 27.8 MB on the 800-section line, and the figure grows linearly with the number of evaluations. **[I]** from Tables 5 and 6. In a Workspace only the baseline row of the public `stage.history` carries bodies; the evaluated rows carry values and no evidence. **[M]**

## What is promised

### Memory and Workspace give the same result

- **The runtime contract promises the same producer and the same failure semantics, not bit-identical Results.** "Storage selection changes result ownership and publication, not physical equations, branch certification, units, declaration order or failure semantics. The same prepared request and physical numerical producer feed either a RAM result sink or Workspace publication." **[R]** (`docs/v1-runtime-contract.qmd:62-65`)
- **It describes Memory as holding the evidence of the call.** "`storage="memory"` | Detached values and actual same-call numerical evidence; no Workspace or saved identity". "The RAM sink owns actual synchronized output arrays and typed numerical evidence. Point and Sweep Results retain … actual failure information." **[R]** (`docs/v1-runtime-contract.qmd:16`, `:68-70`)
- **The bit-identity rule is written in the test documentation and in Issue 58.** "Every public value must be bit-identical in Memory and Workspace, after close, after reopening and in a fresh process". The next sentence records today's exception: "Two kinds of evidence are compared between Workspace readings only, not with Memory: the public evidence of a pole, a zero and a coupling, which Memory does not expose, and the `selected_state` arrays in the evidence of a failed root, which only a Workspace keeps." **[R]** (`tests/evidence-readback.md:38-39`)
- **One guide promises the same interface for static capacitance and leaves the roots out.** "Memory and Workspace results expose the same static matrix and element value interfaces. … The dynamic response, root, and residue-normalized-coupling contracts remain separate." **[R]** (`docs/guides/static-capacitance.qmd:172-175`)

### What `evidence` is

- **The contract names public evidence for Direct and Operator only.** "Direct and Operator Results, including each Sweep point, expose readonly same-call evidence in Memory, Workspace and resolve. Conditional affine evidence is absent when unused". **[R]** (`docs/v1-runtime-contract.qmd:31-34`; the same in `docs/contracts/workspace-evidence-v4.qmd:171-174`) In both storages that evidence is `{}` unless affine evidence is present. **[M]**
- **An implementation page describes the evidence of quantity Results.** "Public `DirectQuantityResult.evidence` supplements existing scalar fields with ordered coordinate/channel bindings, null vectors and branch certificates. … `DirectQuantityResult.evidence` defaults to an empty mapping. … Hybridized evidence exposes the actual dtype-preserved dimensionless null-vector Quantity. Coupling evidence exposes ordered a/b branch bindings and their already-certified roots, slopes, optional vectors, residues and certificates." **[R]** (`docs/implementation/direct-poles-zeros-and-operators.qmd:40-52`)
- **The same page says this holds in Memory.** "The hybridized-pole Result retains the actual dtype-preserved dimensionless null-vector Quantity and exact ordered coordinate bindings in its readonly evidence. Workspace serializes those actual values through the v4 storage codec; Memory retains their immutable backing." **[R]** (`:433-437`) In Memory the evidence is `{}`; the vector is at `result.evaluation.null_vector`, as an array and not a Quantity. The Workspace evidence of a pole has no coordinate-binding key either. **[M]**
- **The diagonal-root page says the certificate facts are returned.** "Current JAX projects the actual `RootPacket` certificate and performed equation facts into returned and saved evidence under the existing Result contracts". **[R]** (`docs/implementation/diagonal-root-numerical-procedure.qmd:175-177`) The public `evidence` of a diagonal root is `{}` in both storages. **[M]**
- **Docstrings.** `DirectQuantityResult`: "One verified scalar Direct quantity and its contract-defined evidence." `DiagonalRootResult`: "Loaded root and local slope evidence from one diagonal-root request." `InMemoryAnalysisResult`: "One detached evaluate result with execution-local provenance only." `SCNSimError`: "`evidence` is an immutable recursively frozen mapping suitable for receipt reconstruction." **[R]** (`src/scnsim/results/matrix.py:100`, `:179`; `src/scnsim/results/memory_analysis.py:90`; `src/scnsim/errors.py:21-25`)
- **The stored form is specified.** "A root sweep solves no frequency row; its points keep their root packet with one array per field, as a single-point root call does. The evidence of a residue-normalised coupling point names its two branch roots in `branches` … A single-point coupling names its branches the same way." **[R]** (`docs/contracts/workspace-evidence-v4.qmd:112-116`)

### Raised or returned

- **The contract specifies raising.** "Every typed failure exposes a stable machine `kind`, an exact `stage`, and immutable evidence. A started operation seals its typed failure receipt before raising; … `resolve()` reconstructs the typed failure boundary but, without a verified success, raises `ResultUnavailableError` and never retries." **[R]** (`docs/v1-runtime-contract.qmd:2999-3004`) The sentence is written in terms of receipts, which Memory does not have.
- **A guide says so without qualification.** "A returned ParameterSweepResult may contain both successful and failed points. … Single-point numerical failures still raise typed errors." **[R]** (`docs/guides/benchmark.qmd:305-308`)
- **Another guide documents the returned failure for Memory.** After an example that calls `run.evaluate(view, spec, parameters=point)` on a Memory run: "Read `result.status`, `result.failure`, `result.context`, `result.discretization` and `result.benchmark`; when `result.succeeded`, `result.frequency` is the Hz projection … Typed numerical failure remains visible; unexpected transport/native errors propagate." **[R]** (`docs/guides/roots-and-branch-selection.qmd:175-180`)
- **A sweep point keeps its failure.** "Each immutable `ParameterPointOutcome` exposes … `failure`, discriminated on numerical non-success as the preserved typed point failure. … Reading success data from a failed point raises that preserved failure." **[R]** (`docs/contracts/parameter-system.qmd:311-331`) Both storages do this. **[M]**
- **One sentence contradicts both storages.** "If shared preparation encounters an expected baseline-root numerical failure, that failure is retained at every dependent sweep point". **[R]** (`docs/contracts/parameter-system.qmd:391-394`) A sweep whose root baseline fails is raised in both storages. **[M]**

### `selected_state` and "compact"

- **No product document names the key `selected_state` or a compact evidence form.** Both appear only in the code, in the tests and in the test documentation. **[R]** (search of `docs/`, `examples/`, `README.md`, the agent catalogue)
- **The architecture page describes the selected state as working state of one frame, not as something saved.** "Completed branch packets retain the actual selected state for same-frame consumers; neither packet is a saved Result, factor, pullback tape or Run-cache entry." **[R]** (`docs/implementation/system-architecture.qmd:23-25`) A Workspace stores it with every root. **[M]**
- **The code states the intent.** "Memory histories need actual closure/equation facts, not the working selected-state arrays retained transiently for implicit differentiation." **[R]** (`src/scnsim/numerics/evidence.py:210-211`)
- **The tests record the current state.** "selected_state is required only for a full saved packet, never for compact Memory root histories." "The public evidence of these families differs between the storages for single points and sweeps alike: Memory exposes no evidence, a Workspace exposes the stored record." **[R]** (`tests/test_evidence_v4.py:1989-1991`, `:901-905`)

### Where the documents are silent or disagree

- **Silent.** What `InMemoryAnalysisResult.evidence`, `evaluation`, `dependencies` and `context` contain; the keys of any root evidence; whether `evidence` is a stable interface or a diagnostic; what a single-point failure does in Memory, apart from the one guide; `selected_state`. The agent catalogue page of `InMemoryAnalysisResult` lists methods and properties only, without `status`, `failure`, `evidence` or `evaluation`. **[R]**
- **Disagree with each other.** The contract and the benchmark guide (raise) against the roots guide (read `result.failure`). **[R]**
- **Disagree with the code.** The implementation page on the pole's evidence in Memory; the diagonal-root page on returned certificate facts; the sweep-baseline sentence of the parameter contract; and the contract's "same … numerical producer", since a single-point `evaluate` uses two producer paths. **[R] [M]**

## Why the code is as it is

### The in-memory Result builder exposes no root evidence

`_evaluation_result` sets `"evidence": {}` for every family, then adds `affine_evidence` when present, and fills the whole mapping only for static capacitance. **[R]** (`src/scnsim/results/memory_analysis.py:641`, `:647-649`, `:651-653`) The record stays in `evaluation`, an `EvaluationResult` that is a public field of the Result. The line dates from the commit that added in-memory analysis (`bb82f3e`, 14 October 2026). **[R]** (`git blame`) No comment or test says the empty mapping was intended; the only later statement is the test comment written when Issue 58 found the difference. **[R]**

### The Workspace reader returns the stored record

`_quantity_evidence` returns `dict(record["evidence"])` and adds `row` and `column`, or `family`, `input` and `output`. The pole adds `null_vector`; the coupling adds `branches`. The diagonal root, the response element, the operator and Direct use `_affine_evidence` instead, which returns `{}` unless affine evidence exists. **[R]** (`src/scnsim/results/decode_jax.py:123-138`, `:189`, `:213`, `:219-220`, `:244`, `:249-250`) This reader dates from 8 October 2026 (`910d398`), before in-memory analysis existed; the implementation page sentences on `DirectQuantityResult.evidence` came with it. **[R]** So the Workspace form is the older, documented one, and Memory never caught up.

### Memory producers drop `selected_state` on purpose

`project_native_root` keeps every field of the root packet except `selected_state` when `compact_evidence` is set. **[R]** (`src/scnsim/numerics/evidence.py:192-213`) Memory sets it everywhere: `compact_evidence=self.ephemeral` in the analysis producer, `compact_evidence=True` for Memory Sensitivity, and `compact_evidence=publication_mode == 'memory'` in optimization. **[R]** (`src/scnsim/execution/physical_analysis.py:598`, `:732`, `:1073`, `:1125`, `:1540`; `src/scnsim/execution/memory_analysis.py:368`; `src/scnsim/execution/physical_optimization.py:276-404`) The flag and its comment came with in-memory optimization (`6dae2cf`, 13 October 2026), whose histories keep one body per evaluated point. **[R]** Table 6 shows what that protects.

The Workspace side has no such filter: it serialises the whole packet. No product code reads a stored `selected_state`. A search of `src/scnsim` finds the name only in the solver, where the live packet is used in the same frame, and in the two filters that drop it. The stored evidence keys that product code does read are `branches`, `terminal_ids` and `axes`. **[R]**

### A single-point `evaluate` runs through two producer paths

`InMemoryAnalysisSink` declares `domain = 'sweep'`, so `PhysicalAnalysis.run` sends every Memory `evaluate`, single points included, to `_run_native_sweep`. Static capacitance is the exception: its sink sets `domain = 'point'`. A Workspace sends only sweeps there; for a single-point `evaluate` `_chunked_operation_domain` returns `None` and the older point loop runs. **[R]** (`src/scnsim/execution/memory_analysis.py:200-209`; `src/scnsim/execution/jax_operation.py:1601-1617`; `src/scnsim/execution/physical_analysis.py:1995-2009`) A profile hook that only counts calls confirmed it: a single-point diagonal root, response element, operator and coupling call `_run_native_sweep` in Memory and `evaluate` plus `require_primal` in a Workspace; sweeps call `_run_native_sweep` in both. **[M]** (`trace_producer.log`)

This one fact explains four rows of Table 1:

- **D7, raised or returned.** The sweep producer keeps a failed point as data, and the Memory builder returns that one point as the Result. The point loop ends with `raise numerical_error(first_failure)` for a point source. **[R]** (`src/scnsim/execution/physical_analysis.py:2046-2047`) Memory has used the sweep producer since in-memory analysis was added; the sink's docstring reads "Same ordered Sweep producer with typed public RAM occurrences". **[R]** (`src/scnsim/execution/memory_analysis.py:194-201`, `bb82f3e`) The condition `and not self.ephemeral` on the raising line came a day later with static capacitance (`71b152d`, 15 October 2026), the one Memory family that uses the point loop, together with the comment "Failure facts remain readable; no failed numerical placeholder is usable". **[R]** (`src/scnsim/results/memory_analysis.py:777-779`) It makes Memory static capacitance return as the other Memory families already did.
- **D8 and D9, another record for the same failure.** In the point loop a failed value goes through `require_primal`, which builds a generic record with `failed_site` unless the status is one of four root codes. The sweep producer builds its own record for a frequency row, with the check and certificate positions, and returns the root packet's own failure for a root. **[R]** (`src/scnsim/numerics/evidence.py:84-104`, `:178-190`) Tracebacks of the failing baseline root show the two raise sites: `_run_native_sweep` line 1948 in Memory, `run` line 2017 in a Workspace. **[M]** (`trace_baseline.log`)
- **D5, `equation_certificates`.** Only the sweep producer adds it to the coupling record. **[R]** (`src/scnsim/execution/physical_analysis.py:1811`)

Both storages raise a root baseline failure because both paths raise it explicitly. **[R]** (`:1947-1948`, `:2016-2017`)

### What is still unexplained

Reusing the sweep producer for a Memory single point was deliberate: the contract says "Declared Direct ParameterSpace and checked evaluate point/Sweep consume the same ordinary numerical producer through a RAM sink". **[R]** (`docs/v1-runtime-contract.qmd:110-112`) Whether returning the failed point to the caller was a decision or a side effect of that reuse cannot be told from the code. The Result class has `status`, `failure` and `succeeded`, the report composer handles a failed Memory Result, and a guide documents it, which points to a decision. No test asserts it, the contract sentence written in the same commit says failure semantics do not change, and Direct `solve`, Sensitivity and every baseline failure raise in Memory, which points to a side effect. **[R] [I]**

## Who uses it

- **No documented use.** No tutorial, example, guide, README or agent catalogue entry contains code that reads `.evidence` of a root-type Result or of a failure. The only code reading in the guides is `matrix_result.evidence` for static capacitance, which is identical in both storages. **[R]**
- **No presentation code.** `scalar_show`, `scalar_plot` and the report composer read `_presentation`, the scalar fields and, for Memory, `status` and `failure`. They never read `.evidence`. **[R]** (`src/scnsim/visualization/quantity_presentation.py:191-234`; `src/scnsim/visualization/report.py:140-184`)
- **Benchmark.** `benchmarks/run.py` reads `omega`, `slope`, `status`, `steps` and `closure` from the evidence of a raised transfer-zero failure. All five are in the root record in both storages. `benchmarks/golden.py` captures every public field without naming keys. **[R]** (`benchmarks/run.py:145-157`, `:367-369`)
- **CI script.** `scripts/ci_windows_wheel.py` reads `value.status`, `value.failure` and `value.evaluation` of a Memory diagonal root and raises `value.failure` when it is set. It also stores Memory Sensitivity and trial `certificates`. **[R]** (`:424-429`, `:442`, `:459`)
- **Tests.** `tests/test_evidence_v4.py` asserts today's differences: Memory evidence `== {}` and Workspace evidence non-empty for a pole, a zero and a coupling in sweeps; `selected_state` absent from a Memory failure and present in a Workspace failure. `tests/test_root_certificates.py:216` reads `coupling.evaluation.branch_roots_rad_s` in Memory. No test asserts a returned single-point failure in Memory, the content of `null_vector` or `branches`, or the evidence of a single-point pole, zero or coupling. **[R]** (`tests/test_evidence_v4.py:901-906`, `:996-1006`, `:1856-1863`)
- **Nobody reads `selected_state`.** Outside the solver and the two filters, it appears only in those test assertions and in the test documentation. **[R]**

So: removing `selected_state` from the Workspace breaks nothing documented. Filling Memory `evidence` breaks nothing documented. Making Memory raise changes one guide paragraph and its zh-TW copy. Making a Workspace return a failed Result changes the contract and a guide. Only this repository was searched.

## Options for the evidence

The options concern `result.evidence` and `failure.evidence` (D1 to D6). Two things hold for all of them:

- **D5 has to be settled with any option that fills Memory evidence**, because the single-point coupling record must then match the Workspace's.
- **None of them removes D11 and D12.** Memory and Workspace Results are different classes with different attributes: Memory has `evaluation`, `dependencies`, Sensitivity `certificates` and trial `bodies`; a Workspace has `identity`. The rule as the tests apply it compares the values and the evidence that both classes name, not the shape of the classes. Extending it to every public attribute would mean converging the classes, which is a larger question than this one. **[I]**

### 1. Memory exposes what a Workspace exposes, `selected_state` included

- **What a user sees.** `result.evidence` has 25, 24, 31 and 17 or 18 keys in both storages. A failed root carries `selected_state` in both.
- **Memory.** Every root a Memory Result holds grows by `selected_state`: 304 to 493 B on the two-node fixtures, 147 kB on the 800-tank ladder, 487 kB on the 800-section line. A sweep holds one per point and per root. An optimization holds one per root of every evaluated point: 27.8 MB for the benchmark CMA-ES workload on the 800-section line, growing with the evaluation count, unless trial bodies are exempted. **[M] [I]**
- **Stored bytes.** Unchanged.
- **Contract.** A new sentence that quantity Results expose the root packet in Memory. The layout of `selected_state`, which differs by family, becomes public and stored interface; the comment at `evidence.py:210` becomes false.
- **Tests.** The storage carve-out and the `selected_state` assertions collapse into full equality; the compact branch of the same-call observer (`tests/test_evidence_v4.py:515-516`) goes; single-point evidence comparisons are added.
- **Work.** Medium. Remove the flag at about a dozen call sites, build the mapping in the Memory builder (`null_vector`, bindings, `branches` from `dependencies`), decide what optimization histories keep. **[I]**

### 2. Both expose the same record without `selected_state`

The record is the one Memory already holds, so no numerical code changes. The Memory builder fills `evidence` from `evaluation` and `dependencies`; the Workspace stops returning `selected_state`. Two variants differ in what is stored.

**2a. The Workspace keeps storing the full packet and filters it on read.**

- **What a user sees.** `result.evidence` has 24, 23, 30 and 17 or 18 keys in both storages, the same bits. A failed root has 22 or 27 keys in both, apart from the D9 cases, which come from the producer paths and not from `selected_state`.
- **Memory.** No growth beyond a decoded view of the record it already holds.
- **Stored bytes.** Unchanged. The stored `selected_state` is then read by nothing and reachable only with private readers.
- **Contract.** `workspace-evidence-v4.qmd:112-114` stays true for the stored form and gains a sentence that the public evidence omits the working arrays. The runtime contract sentence on same-call evidence (`:31-34`) extends from Direct and Operator to quantity Results. The implementation page (`:433-437`) becomes true. The exception paragraph of `tests/evidence-readback.md:39` and its two zh-TW copies goes.
- **Tests.** Remove `_STORAGE_SPECIFIC_EVIDENCE` and `_storage_comparable` (`:901-906`, `:996-1006`); replace the assertions at `:1856-1863` with full equality and `assertNotIn("selected_state", …)` on both sides; add single-point evidence comparisons for the four families, which `_public_values` does not make today (`:730-737`).
- **Work.** Small to medium. **[I]**

**2b. The Workspace stops storing `selected_state` as well.**

- **What a user sees.** As 2a.
- **Memory.** As 2a.
- **Stored bytes.** Smaller. Each stored root loses 304 to 493 B on the two-node fixtures, 147 kB on the 800-tank ladder, 487 kB on the 800-section line; a failed single-point call on that line loses 2.4 MB; the benchmark optimization loses 57 to 114 copies. Evidence stays version 4, which has not shipped in a release; version 3 was written by 1.0.0a3. **[M] [R]**
- **Contract.** As 2a, and `workspace-evidence-v4.qmd:112-114` says the root packet is stored without the solver's working arrays.
- **Tests.** As 2a; the same-call observer always drops `selected_state` (`:515-516`); the stored-bytes comparison used for the last two sweep changes shows changed bytes for every root body and has to be regenerated.
- **Work.** Medium. The `compact_evidence` parameter disappears, because every caller would pass the same value. **[I]**

A further choice inside option 2 is whether a diagonal root should expose its record too. Today its `evidence` is `{}` in both storages although the record exists in both, and the diagonal-root page says the facts are returned. Exposing it is consistent with the other three root families and changes a Workspace Result that is `{}` today.

### 3. Both keep today's behaviour and the rule excludes this evidence

- **What a user sees.** As today: `{}` in Memory and the stored packet in a Workspace; `selected_state` in Workspace failures only. Code written against a Workspace (`result.evidence["null_vector"]`) raises `KeyError` in Memory. Memory users reach the same facts through `result.evaluation`, which no document describes.
- **Memory and stored bytes.** Unchanged, including 487 kB per stored root on the 800-section line.
- **Contract.** A sentence in the runtime contract that the public evidence of root, pole, zero and coupling Results and of root failures is specific to the storage; the false Memory sentence of the implementation page (`:433-437`) corrected; `evaluation` documented if Memory users are meant to use it.
- **Tests.** None change; they already encode this.
- **Work.** Documentation only. **[I]**

### Why not a narrower public evidence

A fourth idea is to expose only the named physical extras in both storages (`null_vector`, `branches`) and no certificate facts. It removes from the Workspace the 22 to 27 certificate keys that single-point calls return today, and leaves `result.evaluation.evidence_bytes` as a second, undocumented surface in Memory. It is option 2 with less in it, and the documents already say the certificate facts are returned, so it is not recommended.

## The failure, decided separately

### Which behaviour the documents specify

Raising. The runtime contract says a started operation raises its typed failure and that storage selection does not change failure semantics; the benchmark guide says single-point numerical failures raise. **[R]** Against that stand one guide paragraph and the Memory Result class. On the documents, Memory `evaluate` is the side that deviates.

Memory is also inconsistent with itself: Direct `solve`, Sensitivity, optimization baselines and every root baseline raise in Memory; only a requested point of `evaluate` is returned. **[M]**

### The two ways to make it consistent

| | Memory raises, as a Workspace does | A Workspace returns a failed Result, as Memory does |
|---|---|---|
| What a user sees | One rule: a single call raises its typed failure; a sweep keeps failed points | A single call returns a Result that must be checked with `succeeded` before reading a value |
| Documents | The roots guide paragraph (`:175-180`) and its zh-TW copy change | The contract (`:2999-3004`), the benchmark guide (`:305-308`) and the meaning of `resolve` without a verified success change |
| Code | The Memory builder raises when its single point failed; `status == "failure"` remains for sweep points | Workspace Result classes have no `status` or `failure`; a failed call has no verified result to return or resolve. A new Result state is needed |
| Tests | None found that assert the returned form; `scripts/ci_windows_wheel.py` already raises `value.failure` | None found for a single-point `evaluate`. The tests that assert a raise cover Direct `solve` and a sweep whose root anchor fails (`tests/test_evidence_v4.py:1322-1324`, `:1767-1769`) and change only if those calls follow |
| Work | Small **[I]** | Large **[I]** |

What a user loses when Memory raises is the readable failed Result after the call; the raised error carries the same kind, stage and evidence.

### The defect underneath

Whichever way that goes, D8 and D9 remain: the same single-point call fails with a different message, different evidence keys and, for a transfer zero, a different `stage`, because the two storages run different producer paths. This contradicts "an exact `stage`" and "the same … numerical producer" and needs no decision on semantics. Running a single-point `evaluate` through one path in both storages removes D5, D8 and D9 together, and D7 with them if that path raises. **[I]** D10 is a difference of encoding in the Direct failure record and is a small separate fix.

## Recommendation

**Evidence: option 2b.** Both storages expose the same record without `selected_state`, and the Workspace stops storing it.

- **`selected_state` is not evidence about the result.** The code calls it working state kept for differentiation, and the architecture page says it serves consumers of the same frame and is not saved; its certificate records are duplicated at the top level; its layout differs by family. Making it public in both storages (option 1) would freeze solver internals into the interface and the stored format.
- **Nobody reads it.** No document, example, benchmark, presentation code or product reader uses it. Only the tests that record today's difference name it.
- **It is the expensive part.** It is 96% of a stored root on the 800-section line and is stored five times for a failed call. Memory dropped it for exactly that reason.
- **The rest already agrees.** The record is bit-identical in both storages today, so the change is one of exposure. It also makes the implementation page true and removes the exception from the rule.
- **Now is the time for the stored change.** Evidence version 4 has not shipped.

If the owner wants the full packet kept on disk for later forensic use, option 2a gives the same public behaviour without touching stored bytes, at the price of storing data nothing reads. Option 3 is the cheapest but leaves the rule with a permanent hole exactly where a user is most likely to reach for evidence, the pole's mode vector.

**Failure: Memory raises.** It is what the contract and the benchmark guide say, it is what Memory already does for every other single call (Direct `solve`, Sensitivity, an optimization baseline, a root baseline), and it is a small change. The roots guide paragraph changes with it.

**Producer paths: a defect to schedule.** D5, D8 and D9 should have their own Issue: one producer path for a single-point `evaluate` in both storages. D13 (a searched root cannot be stored) and D10 are separate small defects.

## What could not be determined

- **Readers outside this repository.** Only this repository was searched for readers of root evidence, `selected_state`, `null_vector` and the returned Memory failure.
- **Whether any recovery path needs a stored `selected_state`.** The name search finds no reader, but nothing was run with it removed, and interrupted or resumed optimizations were not exercised.
- **Whether the returned Memory failure was a decision.** The evidence points both ways (see "What is still unexplained").
- **Other configurations.** Only float64 on CPU was run. Listed-point sweeps (`ParameterSpace.points`), versioned Workspaces, `resolve(attempt=…)` on a failed attempt, float32 and affine evidence were not measured.
- **Process memory under option 1.** The figures are array bytes from the stored packets, scaled; peak memory of a Memory run that kept `selected_state` was not measured.
- **A successful pole on the line.** The pole on the benchmark line did not converge at the anchor tried, so its `selected_state` size comes from the failure evidence and from the diagonal root's stored body, which agree (487,111 B).
- **D13 beyond one case.** Only `finite_seeds` on a diagonal root was tried in a Workspace; `polynomial_transfer_zero` was refused before execution on these fixtures in both storages.
- **Work estimates.** The sizes of the work are judgements from reading the code; nothing was implemented.

## Files beside this report

| File | What it is |
|---|---|
| `lib.py` | Fixtures and the helpers that describe a public value |
| `measure.py` | The 85 cases and the summariser; `measure-b1.json` to `measure-b12.json` are its outputs, `measure-all.json` their union (`merge.py`), `measure-b*.log` the run logs (the twelfth batch was run without a log file) |
| `measure.json`, `measure.log` | The first, independent run of 72 cases; `repeat_check.py` and `repeat_check.log` compare it with the union |
| `compare.py`, `tables.md` | The comparison and its full tables (T1 readings, T2 and T3 evidence, T4 failures, T5 Sensitivity, T6 optimization, T7 the differences the script found) |
| `value_check.py`, `value_check.log` | The count of cases whose public numbers are bit-identical between the storages |
| `details.py`, `details.log` | Per-key sizes and the null-vector cross-check |
| `state_check.py`, `state_check.log` | What `selected_state` contains and whether its certificates are duplicated |
| `cost.py`, `cost-ladder-small.json`, `cost-ladder-800.json`, `cost-line.json`, `cost_table.py`, `cost-tables.txt` | Table 5 |
| `opt_cost.py`, `opt-cost.json`, `opt-cost.log` | Table 6 |
| `trace_producer.py`, `trace_producer.log` | Which producer path runs for each storage and form |
| `trace_baseline.py`, `trace_baseline.log` | Where a failing baseline root is raised in each storage |
| `explore1.py`, `explore2.py`, `explore3.py`, `peek.py` and their logs | First looks at the public objects; `explore3.log` shows the failed optimization rows |
| `try-pole.json`, `try-tables*.md` | Trial runs of the scripts, superseded by the files above |
