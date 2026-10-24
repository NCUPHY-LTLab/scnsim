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
| `test_workspace_readback.py` | JAX Workspace version upgrade/downgrade, shared-Plan Direct/root/CMA identities, exact warm reuse and guarded fresh-process resolve; retained Julia attempt/inventory cases remain below. |
| `test_workspace_optimization_readback.py` | Actual CMA and L-BFGS-B completion, repeated optimize/resolve with zero native work, fresh-process guarded resolve, exact saved values and identities. |
| `test_julia_replay.py` | Retained Julia CMA interruption/replay and corruption rejection. |
| `test_manual_diagrams.py` | Explicit diagram declarations, missing/stale drawings, presentation and actual connectivity-sensitive geometry. |
| `test_path_portability.py` | Native paths/URIs, source provenance and public Memory/Workspace handle lifetime. |
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

## Retained Julia cases

Install the existing `julia` extra and provision the pinned Julia 1.12.6 runtime
used by CI, then run the explicit opt-in:

```sh
uv sync --locked --extra julia --group course-generation
SCNSIM_RUN_JULIA_TESTS=1 uv run --locked --extra julia --group course-generation \
  python -W error::ResourceWarning -m unittest discover -s tests -t . -v
```

These backend-specific assertions remain under their original Julia gate until
the ordered Julia retirement. Their physical purposes and JAX replacements
are listed below; inventory and attempt-directory behavior are not redefined
as a JAX API.

| Group | Retained assertions |
| --- | --- |
| `JuliaSelectedNetworkEvidenceTests` | Julia audit/stations layout and inventory retire with their features. JAX `SelectedNetworkTests` checks actual `pi_section`, matrix node/order/cross-coupling and `compiled.views[primary_view_id].lineage`. |
| `JuliaComponentEvidenceTests` | Julia mutual-inductance row/derived-M encoding and no-request inventory. The physical stiffness comparison also runs independently on JAX. |
| `JuliaAffineEvidenceTests` | Original receipt layout stays gated. JAX `AffineEvidenceTests` reads recorded fan-out rows from the actual Result/history, including candidate values, consumer edges and ordered authorization sources. Ordinary authoring rejection remains distinct from candidate `affine_support` plus infinite candidate cost. |
| `JuliaOptimizationRetainedTests` | Julia inventory/ledger/seed observations retire with those features. Existing JAX selector and exact zero-baseline assertions remain unchanged; no cross-backend candidate trajectory is required. |
| `JuliaWorkspaceVersioningInventoryTests`, `WorkspaceReadbackTests` | Julia inventory/attempt selection/interruption/retry remain gated until retirement. JAX versioning and `JaxWorkspaceReadbackTests` preserve shared-Plan identities, warm reuse and fresh-process resolve. No new JAX retry promise is introduced. |

Julia replay remains in scope for the retained Julia backend. A reported skip
means that group did not execute; it is not a passing numerical observation.

## Same-call affine evidence

`AffineEvidenceTests` uses the existing two-consumer affine fixture and authored
support interval. Only actual out-of-support edges add `affine_evidence` rows.
Each row retains the exact binary64 input/support/distance, consumer path/field,
side and `authorization_sources`: `parameter_set`, then `optimization_spec` when
applicable; an unauthorized candidate records an empty list.

The ordinary test preserves the original Operator execution/readback, and reads
the rows through the existing scalar Y Result evidence at the same point and
5 GHz. Operator has no new public evidence field. Workspace optimization history
mappings and memory trial objective-component mappings expose the actual
candidate rows. A saved cache read retains the sources of its original execution.
Absent rows add no empty key to non-affine Results.

The test preserves the original CMA fixture, seed and budget. Rejected affine
candidates must retain `invalid_candidate_physical_parameter` / `affine_support`
and infinite cost; unknown preparation errors remain fatal. Julia receipt
helpers are removed only with the later ordered backend retirement.
