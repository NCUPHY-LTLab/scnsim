# Test suite

The maintained suite uses Python's `unittest` discovery. Run it from the
repository root on Python 3.12 after syncing the locked environment:

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -t . -v
```

With the opt-in unset, Python-only cases run and Julia-gated cases report
skipped. Tests that require packaged Julia execution opt in only when
`SCNSIM_RUN_JULIA_TESTS=1`; their skip reports in the package-and-docs job are
not evidence that native tests ran.

For the Julia-backed groups, install the existing `julia` extra and prepare
the pinned Julia 1.12.6 runtime used by CI. Then run the same discovery with
the explicit opt-in and the CI ResourceWarning policy:

```sh
uv sync --locked --extra julia
SCNSIM_RUN_JULIA_TESTS=1 uv run --locked python -W error::ResourceWarning \
  -m unittest discover -s tests -t . -v
```

The CI runtime-preparation matrix installs each Linux or macOS platform wheel
with its `julia` extra, provisions Julia 1.12.6 explicitly, and runs this
opted-in discovery. The package-and-docs job leaves the opt-in disabled. A
successful readback verifies recorded data and identity; it does not by itself
establish numerical accuracy. Numerical assertions are scoped to the authored
fixtures and predicates in each test. Diagram tests exercise explicit
authoring and presentation contracts, not solver correctness.

## Coverage by purpose

| Module | Purpose and boundary |
| --- | --- |
| `test_rlgc.py` | Ordered immutable RLGC values, physical matrix validation, and strict Q2D source loading. |
| `test_component_models.py` | Library provenance and explicit native component expansion; Julia expansion cases require the opt-in. |
| `test_direct_analysis.py` | Selected-network Direct behavior and analytic response, pole, coupling, and transfer-zero quantities; packaged Julia cases require the opt-in. |
| `test_harmonic_balance.py` | HB declaration/preflight behavior, numerical classification, typed failures, and sealed-result readback; runtime cases require the opt-in. |
| `test_workspace_readback.py` | Workspace versioning, append-only attempts, exact identity resolution, and retained Julia execution behavior. |
| `test_julia_replay.py` | Julia CMA interruption/replay and corruption rejection; this is distinct from JAX resume behavior. |
| `test_affine_parameters.py` | Affine support rejection before request creation, authorized fanout evidence in the sealed Julia receipt, and CMA rejection/authorization readout. The retained optimizer case uses seed 23, five evaluations, 0.9–1.1 m bounds, and the authored 1 mS target. |
| `test_optimization.py` | Seven-selector public explanation without an attempt, plus retained Julia CMA selector and matched parallel-LC zero cases. The CMA cases retain seeds 29 and 31, five evaluations, and 79.5–80.5 fF bounds. |
| `test_manual_diagrams.py` | Explicit diagram declarations, geometry ownership, stale/missing declaration failures, and connectivity-sensitive layout behavior without a numerical backend. |

The affine and optimization regressions preserve the original scoped fixtures,
units, objective order, and assertions. They do not assert new authoring
override ownership, JAX staged-history semantics, a general optimizer
accuracy bound, or a cross-backend equivalence rule.
