---
title: "CI and Pages"
lang: en
sidebar: testing
---

[English](ci.md) · [繁體中文](zh-TW/ci.md)

CI checks the declared unittest suite, installed platform wheels, package contents, and documentation. Pushes to `develop` or `main`, pull requests opened, synchronized, reopened, or marked ready for review, and manual dispatch start the workflow; a draft pull request runs only change classification and `ci-status`.

## How a change selects its workflow

The classifier has a narrow prose-only lane. A change must be limited to the root `README.md` or regular Markdown under `docs/`, and every changed line must be classified as prose. Code, QMD or IPYNB files, shared configuration, dependency changes, protected markup, executable content, unknown paths, mixed changes, and uncertain classifications select the product lane. `tests/README.md` is not on the prose allowlist, so edits to the Testing guide use the product lane.

The workflow has seven product jobs grouped into five families. `package-and-docs` syncs the locked environment, checks version and public API contracts, discovers the purpose-based unittest suite, inspects the wheel and source distribution, checks notebook hygiene and generated tutorial figures, and renders documentation without executing notebook cells.

`native-platform-wheels` builds and audits the CPython 3.12 Linux x86_64 manylinux and macOS 12 Apple ARM wheels, then compares the installed native extension with packaged C++ sources. The macOS job uses a macOS 15 host and a macOS 12 ARM target. `windows-native-wheel` builds the CPython 3.12 Windows AMD64 wheel with strict MSVC floating-point settings and audits its resources.

`windows-jax-wheel-smoke` installs the actual Windows wheel in a consumer environment and exercises the native kernel, public Memory/Workspace JAX paths, default unittest discovery, and Windows Path Portability Set. `runtime-preparation-smoke` runs with installed Linux and macOS wheels and covers Direct and root calculations, sensitivity derivatives, CMA optimization, saved-result resolution, default unittest discovery, and the platform Path Portability Set.

A draft pull request skips those product and documentation jobs. The small `ci-status` job still checks that required jobs for the selected lane succeeded and jobs for the other lane were skipped as expected. A failed, cancelled, or unexpectedly skipped required job keeps the workflow from succeeding.

## Documentation builds and Pages publishing

The separate `documentation` job serves the prose-only lane. It syncs locked dependencies without installing or building the project, checks that local includes and Markdown/QMD link targets exist, and renders with Quarto 1.10.18 without executing code. That source check does not inspect links in rendered HTML.

The documentation job renders one current source version. Pages instead assembles a versioned site from the presentation source, live `develop` source, and pinned release sources, then renders each source with Quarto 1.10.18. A non-draft pull request builds a review site without deploying it; a draft pull request skips that build. Pushes and eligible manual dispatches can deploy the assembled Pages site.

Workflow definitions: [CI](../.github/workflows/ci.yml), [Pages](../.github/workflows/pages.yml), and the [change classifier](../.github/scripts/change_scope.py).
