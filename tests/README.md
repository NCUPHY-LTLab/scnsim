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
| `test_affine_parameters.py` | Ordinary unauthorized affine points fail in authoring before request creation. Optimization-candidate physical rejection remains a distinct boundary. |
| `test_optimization.py` | Seven-selector explanation and actual JAX CMA selector execution with a zero-cost baseline; no cross-backend candidate-trajectory equality. |
| `test_workspace_readback.py` | JAX Workspace version upgrade/downgrade plus the retained Julia attempt/inventory cases below. |
| `test_workspace_optimization_readback.py` | Actual CMA and L-BFGS-B completion, repeated optimize/resolve with zero native work, fresh-process guarded resolve, exact saved values and identities. |
| `test_harmonic_balance.py` | HB declaration/preflight, typed numerical failures and sealed readback; packaged Julia execution requires the opt-in below. |
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

These assertions remain under their original Julia gate while their evidence
interfaces await migration:

| Group | Retained assertions |
| --- | --- |
| `JuliaSelectedNetworkEvidenceTests` | `transmission_line_audit`/stations, `ref_lineage`, inventory requests. |
| `JuliaComponentEvidenceTests` | Julia mutual-inductance row/derived-M encoding and no-request inventory. The physical stiffness comparison also runs independently on JAX. |
| `JuliaAffineEvidenceTests` | Receipt fan-out, candidate rejection/authorization and its ledger fields. Ordinary evaluate rejection is an authoring error, distinct from candidate `affine_support`. |
| `JuliaOptimizationRetainedTests` | Julia selector/inventory and CMA ledger/branch observations. |
| `JuliaWorkspaceVersioningInventoryTests`, `WorkspaceReadbackTests` | Inventory, exact attempt selection, interruption/retry and fresh-process attempt readback. |
| `JuliaComplexAnchorDecodingTests` | Exact packaged Julia complex-anchor decoding. Public anchor execution lives in the JAX Direct tests. |

HB and Julia replay remain in scope for their retained backend. A reported skip
means that group did not execute; it is not a passing numerical observation.
