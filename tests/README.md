# Testing

The suite covers small, declared circuit fixtures and their failure boundaries, while CI also checks packages, installed wheels, and documentation. Its numerical comparisons apply to those fixtures, not arbitrary circuits or parameter ranges.

## When CI runs

The workflow runs on pushes to `develop` and `main`, on pull requests opened, synchronized, reopened, or marked ready for review, and on manual dispatch. A draft pull request runs the change classifier and `ci-status`, while the product and documentation jobs are skipped.

The classifier has a narrow prose-only lane for changes limited to root `README.md` or regular Markdown files under `docs/`, with every changed line classified as prose. Code, protected markup, executable content, an unrecognized or mixed path, or an uncertain classification selects the product lane.

This is not a fast lane for all Markdown: `tests/README.md` is outside the allowlist and selects the product lane. The rules are in `.github/workflows/ci.yml` and `.github/scripts/change_scope.py`.

## What the product lane checks

The product lane has five job families, expanded to seven jobs by the Linux/macOS matrices:

1. **`package-and-docs`** syncs the locked environment, checks version and public API contracts, and discovers the purpose-based unittest suite. It builds and inspects the wheel and source distribution, checks minimum-Python and portable wheel identity, validates notebook hygiene and generated tutorial figures, and renders and inspects the documentation without executing notebooks.
2. **`native-platform-wheels`** builds and audits the CPython 3.12 Linux x86_64 manylinux and macOS 12 Apple ARM wheels, then checks the installed native extension against the packaged C++ source. The macOS job runs on a macOS 15 host with a macOS 12 ARM target.
3. **`windows-native-wheel`** builds the CPython 3.12 Windows AMD64 wheel with strict MSVC floating-point settings and audits the wheel resources.
4. **`windows-jax-wheel-smoke`** installs that actual Windows wheel in a consumer environment, exercises its native kernel and public Memory/Workspace JAX paths, discovers the default suite, and runs the Windows Path Portability Set.
5. **`runtime-preparation-smoke`** runs on Linux and macOS with the platform wheels installed. It exercises Direct and root calculations, sensitivity derivatives, CMA optimization, saved-result resolution, the default suite, and the platform Path Portability Set.

The separate **`documentation`** job serves the prose-only lane. It runs `uv sync --locked --no-install-project --no-build`, then renders with Quarto 1.10.18 without executing code. Its source check confirms that local includes and Markdown/QMD link targets exist and that generated output is not tracked; it does not inspect rendered HTML links.

`ci-status` checks that the jobs required by the selected lane succeeded and that jobs for the other lane were skipped as expected. Failures, cancellations, and unexpected skips do not count as a passing run.

## Documentation Pages

The CI documentation job renders the current source version once. The Pages workflow prepares a versioned site from the presentation source, the live `develop` source, and pinned release sources, then renders each source with Quarto 1.10.18. A non-draft pull request builds the site for review but does not deploy it; draft pull requests skip the build. Pushes and eligible manual dispatches can deploy the assembled Pages site. The rules are in `.github/workflows/pages.yml`.

## What the tests protect

### Circuit construction and imported data

Tests build ordered RLGC matrix inputs and load Q2D source data. Values must remain immutable and ordered, matrices must satisfy physical requirements, and Q2D mappings must be bijective and match their content digest; these checks catch mutation, nonphysical matrices, incomplete mappings, and mismatched source files. Source: [test_rlgc.py](test_rlgc.py).

Component fixtures assemble junctions, capacitors, SQUIDs, resonators, and coupled inductors from catalog declarations. Provenance is captured once, expansions remain explicit, non-positive-definite inductance graphs are rejected, and valid coupling is compared with an independent stiffness-matrix stamp; these checks catch factory reinvocation and expansion errors. Source: [test_component_models.py](test_component_models.py).

### Network equations and frequency roots

Direct-analysis cases preserve declaration order and retain matrix-valued multiconductor and n-port networks through selected reductions, transforms, and probes. Their operators, responses, poles, residues, couplings, and transfer zeros are compared with independent fixture calculations, catching scalarization, ordering, and equation-construction errors. Source: [test_direct_analysis.py](test_direct_analysis.py).

Admittance cases compare dissipative networks with an independent oracle and NumPy Schur reduction, and vary reference impedance to ensure it does not create device loss. Derivative and matched-scattering-zero cases check the requested response and analytic frequency jet, catching loss-accounting and retained-coordinate errors. Source: [test_direct_admittance.py](test_direct_admittance.py).

Root cases compare high-Q poles with a global quadratic eigenvalue calculation under both coordinate orders. A no-loss pole is evaluated with five anchor choices—the real anchor and positive/negative imaginary perturbations at two scales—and must retain identical frequency bits; repeated modes, zero slopes, or unusable residues must not receive simple-root certificates. Together these catch sign-rounding and invalid-certification errors. Source: [test_root_certificates.py](test_root_certificates.py).

### Parameters and optimization

Affine fixtures submit ordinary unauthorized points and authorized fan-out through Memory and Workspace. Unauthorized extrapolation must fail before request creation; authorized rows retain the support edge and authorization source, repeated points preserve their original row, and CMA records rejected and authorized candidates, catching missing or misattributed evidence. Source: [test_affine_parameters.py](test_affine_parameters.py).

Bounded local-coordinate fixtures exercise explicit scales, parameter-domain scales, resolved bound widths, and the existing signed-log mapping. The C1 case runs both local optimizers, requires iterations and a converged result, and reports improved independent analytic loss and remaining target distance without imposing a target-distance tolerance; these checks catch scale-precedence and optimizer-wiring errors. Source: [test_local_coordinate_defaults.py](test_local_coordinate_defaults.py).

CMA fixtures check that all seven declared selectors appear in JAX explanations and that selector branches, including a zero-cost baseline, execute. They do not require matching candidate trajectories across backends. Source: [test_optimization.py](test_optimization.py).

### Saved evidence and readback

Evidence-v3 cases preserve literal binary64 values and detect corrupted bytes. Short, missing, or interrupted writes must not publish success; unsupported evidence must be rejected before Workspace mutation; partial sweeps retain failures and source order; and Direct, Operator, root, sensitivity, and optimization results resolve to exact saved bits after close and from a fresh process. Source: [test_evidence_v3.py](test_evidence_v3.py).

Workspace cases cover version upgrades, forbidden downgrades, shared-plan identities, warm reuse, and guarded fresh-process resolution. Optimization histories survive close and resolve; repeated optimize/resolve calls add no native work after the original run, catching stale identities, unintended recomputation, and lost results. Sources: [test_workspace_readback.py](test_workspace_readback.py) and [test_workspace_optimization_readback.py](test_workspace_optimization_readback.py).

### Failure handling and diagnostics

Failure-routing cases pass owned scalar, batched, and archived equation errors through their wrappers. The original payload must survive, while unknown or mixed errors are re-raised and invalid wrappers rejected; this catches hidden native failures and unsupported classifications. Source: [test_native_failure_routing.py](test_native_failure_routing.py).

Solve diagnostics compare per-right-hand-side componentwise metrics with aggregate metrics and exercise dense, sparse, and compensated solve paths. The diagnostics must describe the actual solve and retain its auxiliary data, catching aggregation that hides a failing right-hand side or diagnostics detached from the solve. Source: [test_solve_diagnostics.py](test_solve_diagnostics.py).

### Paths, diagrams, and teaching outputs

Path cases cover native filesystem paths, file URIs, Windows drive and UNC conversion, package-source provenance, and Memory/Workspace path lifetimes. Unreadable required paths retain their strict failure cause, catching double decoding and platform-specific path interpretation errors. Source: [test_path_portability.py](test_path_portability.py).

Manual drawings are checked against current circuit declarations. Presentation settings must not change connectivity, `show` requires a current declaration, and wrong wire endpoints, MTL pins, missing crossing jumps, or layouts from another plan are rejected; these checks catch drawings that do not represent the declared circuit. Source: [test_manual_diagrams.py](test_manual_diagrams.py).

Notebook and course-helper cases preserve cell and kernel identity while removing a machine-local kernel path, prepare declared chapter requests without solving, and export certificates for authored diagrams. Publication-binding cases preserve foreign namespaces and keep cleanup failures from replacing the primary error; together these checks catch nonportable notebooks, accidental solves during documentation preparation, and unsafe cleanup. Sources: [test_portable_notebook.py](test_portable_notebook.py), [test_engineer_chapter3_generation.py](test_engineer_chapter3_generation.py), [test_engineer_chapter4_generation.py](test_engineer_chapter4_generation.py), and [test_engineer_publication_binding.py](test_engineer_publication_binding.py).

The suite has no dedicated numerical regression for `StaticCapacitance`.

## How numerical comparisons work

For declared forward-comparison fixtures, the bound is `relative_error <= 100 * condition_bound * float64_epsilon`. The tests derive each condition bound independently from the fixture equations and report the actual value, oracle value, units, bound, and normalized error ratio.

Actual operator and solve backward residuals use `256 * (n + 1) * float64_epsilon`. These checks are separate from exact shape, order, encoded-value, unit-projection, binary64-bit, and saved-result comparisons. The suite does not add an epsilon floor to a zero relative normalization.

## Run locally

From the repository root, use the locked course-generation group and discover the default unittest suite:

```sh
uv sync --locked --group course-generation
uv run --locked --group course-generation python -m unittest discover -s tests -t . -v
```

This default suite exercises JAX and does not start Julia. The 120-case Golden comparison and performance runs are separately assigned work; they are not part of default CI.
