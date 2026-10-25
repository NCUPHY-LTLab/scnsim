# Test suite

Run the purpose-based suite from the repository root on Python 3.12. JAX tests
run without an opt-in flag. The existing locked `course-generation` group
supplies IPython for the real notebook-display tests.

```sh
uv sync --locked --group course-generation
uv run --locked --group course-generation python -m unittest discover -s tests -t . -v
```

CI also runs the migrated suite against installed platform wheels. Keep the
installed wheel when adding locked display dependencies to those environments.

## Coverage by purpose

| Module | Checks |
| --- | --- |
| `test_rlgc.py` | Ordered immutable RLGC values, defensive isolation, physical matrices and Q2D source loading. |
| `test_component_models.py` | Library provenance/factory-once, JJ/IDC/SQUID expansion, non-SPD rejection and an independent coupled-inductor stiffness stamp. |
| `test_direct_analysis.py` | Ordered multiconductor and selected-network S/Y/Z, PTC, transform/retain, complex anchors and analytic operator/response/pole/coupling/residue/zero comparisons. |
| `test_affine_parameters.py` | Ordinary unauthorized affine points fail in authoring before request creation. Same-call authorized fan-out and rejected/authorized CMA candidate evidence run on JAX in memory and Workspace; source lists distinguish ParameterSet and OptimizationSpec approval. |
| `test_optimization.py` | Seven-selector explanation and actual JAX CMA selector execution with a zero-cost baseline; no cross-backend candidate-trajectory equality. |
| `test_workspace_readback.py` | JAX Workspace version upgrade/downgrade, shared-Plan Direct/root/CMA identities, exact warm reuse and guarded fresh-process resolve. |
| `test_workspace_optimization_readback.py` | Actual CMA and L-BFGS-B completion, repeated optimize/resolve with zero native work, fresh-process guarded resolve, exact saved values and identities. |
| `test_manual_diagrams.py` | Explicit diagram declarations, missing/stale drawings, presentation and actual connectivity-sensitive geometry. |
| `test_evidence_v3.py` | Evidence v3 bit preservation and corruption checks, interrupted/partial publication, unsupported-version rejection before Workspace mutation, partial-sweep failure/order retention, and exact post-close/fresh-process readback across Direct, Operator, roots, sensitivity and optimization. |
| `test_native_failure_routing.py` | Scalar and batched native failures preserve owned error payloads; unknown/mixed errors re-raise, invalid wrappers are rejected, and inactive checks do not throw. |
| `test_path_portability.py` | Native paths/URIs, source provenance and public Memory/Workspace handle lifetime. |
| `test_portable_notebook.py` | Portable notebook cleanup preserves cell and kernel identity, removes only the machine-local kernel path, and leaves path-free bytes unchanged. |
| `test_solve_diagnostics.py` | Per-RHS componentwise diagnostics and dense, sparse and compensated solve paths retain the original solve outcome. |
| `test_engineer_chapter3_generation.py` | Chapter 3 prepares the declared Direct request and exports its explicit schematic certificate without solving. |
| `test_engineer_chapter4_generation.py` | Chapter 4 exports the explicit capstone layout certificate from its authored declarations. |
| `test_engineer_publication_binding.py` | Course helper binding preserves foreign namespaces and import state, and checks owned kernel temporary-directory cleanup without invoking the producer. |
| `test_direct_admittance.py` | Retained coordinate Y/Z and matched-S TransferZero refinement. |
| `test_root_certificates.py` | Five existing Hybrid complex anchors and shared passive-root rejection. |
| `test_local_coordinate_defaults.py` | Resolved local-coordinate defaults and actual C1 optimization. |

The retired version-numbered `dev4`, `dev5`, `dev6` and `full_v1` modules no
longer duplicate these scenarios in discovery.

## Numerical comparisons

Forward comparisons use the Human-assigned bound
`relative_error <= 100 * condition_bound * float64_epsilon`.
`analytic_conditions.py` derives each bound independently from the original
fixture's physical equations. Each `H2` JSON line records the actual/oracle
values, units, condition bound and `error / (condition_bound * epsilon)` ratio
before the assertion. Evaluation-frequency perturbations are recorded explicitly.
The scalar comparisons use their own condition bounds; matrix or pole
conditions are not substituted for unrelated observables.

Operator *formation* differences are forward errors. Actual operator/solve
backward residuals retain `256 * (n + 1) * float64_epsilon`. A zero relative
normalization requires a separately justified scale; the suite adds no epsilon
floor. Shape/order, declared encodings, unit projections and exact saved-result
readback remain structural assertions rather than forward accuracy claims.

## Backend scope

The default unittest suite exercises the JAX path and does not launch Julia.
The retained affine and CMA evidence cases run through JAX; they do not assert
Julia receipt or attempt-directory formats, or exact cross-backend CMA
trajectories. Julia-specific execution records are outside this suite.

## Same-call affine evidence

`AffineEvidenceTests` uses the existing two-consumer affine fixture and authored
support interval. Only actual out-of-support edges add `affine_evidence` rows.
Each row retains the exact binary64 input/support/distance, consumer path/field,
side and `authorization_sources`: `parameter_set`, then `optimization_spec` when
applicable; an unauthorized candidate records an empty list.

Direct and Operator results expose the same-call rows through their read-only
`evidence` mapping. Workspace optimization history mappings and memory trial
objective-component mappings expose the actual candidate rows. A saved cache
read retains the sources of its original execution.
Absent rows add no empty key to non-affine Results.

The test preserves the original CMA fixture, seed and budget. Rejected affine
candidates must retain `invalid_candidate_physical_parameter` / `affine_support`
and infinite cost; unknown preparation errors remain fatal.
