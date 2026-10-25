---
title: "CI and Pages"
lang: en
sidebar: testing
---

[English](ci.md) · [繁體中文](zh-TW/ci.md)

CI runs on pushes to `develop` and `main`, on pull requests opened, synchronized, reopened, or marked ready for review, and on manual dispatch. A draft pull request runs the change classifier and `ci-status`, while the product and documentation jobs are skipped.

## Change classification

The classifier has a narrow prose-only lane for changes limited to root `README.md` or regular Markdown files under `docs/`, with every changed line classified as prose. Code, protected markup, executable content, an unrecognized or mixed path, or an uncertain classification selects the product lane. `tests/README.md` is outside the allowlist, so changing this Testing section selects the product lane.

The behavior is defined in `.github/workflows/ci.yml` and `.github/scripts/change_scope.py`.

## Product jobs

The product lane has five job families, expanded to seven jobs by the Linux/macOS matrices:

1. **`package-and-docs`** syncs the locked environment, checks version and public API contracts, and discovers the purpose-based unittest suite. It builds and inspects the wheel and source distribution, checks minimum-Python and portable wheel identity, validates notebook hygiene and generated tutorial figures, and renders and inspects the documentation without executing notebooks.
2. **`native-platform-wheels`** builds and audits the CPython 3.12 Linux x86_64 manylinux and macOS 12 Apple ARM wheels, then checks the installed native extension against the packaged C++ source. The macOS job runs on a macOS 15 host with a macOS 12 ARM target.
3. **`windows-native-wheel`** builds the CPython 3.12 Windows AMD64 wheel with strict MSVC floating-point settings and audits the wheel resources.
4. **`windows-jax-wheel-smoke`** installs that actual Windows wheel in a consumer environment, exercises its native kernel and public Memory/Workspace JAX paths, discovers the default suite, and runs the Windows Path Portability Set.
5. **`runtime-preparation-smoke`** runs on Linux and macOS with the platform wheels installed. It exercises Direct and root calculations, sensitivity derivatives, CMA optimization, saved-result resolution, the default suite, and the platform Path Portability Set.

## Documentation job and status

The separate **`documentation`** job serves the prose-only lane. It runs `uv sync --locked --no-install-project --no-build`, then renders with Quarto 1.10.18 without executing code. Its source check confirms that local includes and Markdown/QMD link targets exist and that generated output is not tracked; it does not inspect rendered HTML links.

`ci-status` checks that the jobs required by the selected lane succeeded and that jobs for the other lane were skipped as expected. Failures, cancellations, and unexpected skips do not count as a passing run.

## Pages

The CI documentation job renders the current source version once. The Pages workflow prepares a versioned site from the presentation source, the live `develop` source, and pinned release sources, then renders each source with Quarto 1.10.18. A non-draft pull request builds the site for review but does not deploy it; draft pull requests skip the build. Pushes and eligible manual dispatches can deploy the assembled Pages site. The rules are defined in `.github/workflows/pages.yml`.
