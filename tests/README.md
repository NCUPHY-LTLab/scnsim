---
title: "Testing"
lang: en
sidebar: testing
---

[English](README.md) · [繁體中文](zh-TW/README.md)

SCNSim’s tests use small, declared circuit fixtures to explain what the package computes, how saved results behave, and how invalid inputs are reported. Use the pages below to explore those cases and the checks CI runs.

## Explore the test topics

::::{.askr-card}
::: {.askr-card-body}
### [CI and Pages](ci.md) {.askr-card-title .unlisted}

Learn when the workflows run, what each product job checks, and how documentation builds differ from Pages deployment.
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [Circuit inputs and components](circuit-data.md) {.askr-card-title .unlisted}

Follow the synthetic RLGC/Q2D loader fixture and small catalog-declared component models, including their invalid-input cases.
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [Network equations and roots](network-roots.md) {.askr-card-title .unlisted}

See how Direct network quantities, admittance, transfer zeros, and root certificates are checked against independent fixtures.
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [Parameters and optimization](parameters-optimization.md) {.askr-card-title .unlisted}

Trace affine parameter authorization, local-coordinate scales, and the small optimizer cases without assuming different optimizers follow the same path.
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [Evidence and readback](evidence-readback.md) {.askr-card-title .unlisted}

Follow values from a completed calculation into v3 evidence, then through Memory or Workspace close and readback.
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [Failure handling and diagnostics](diagnostics.md) {.askr-card-title .unlisted}

Understand how owned native failures preserve their payloads and how solve diagnostics distinguish aggregate from per-right-hand-side behavior.
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [Paths, diagrams, and teaching outputs](portability-diagrams.md) {.askr-card-title .unlisted}

Review file-URI conversion, declaration-matched drawings, portable notebooks, and course preparation that stops before solving.
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [Numerical checks and local run](numerical-checks.md) {.askr-card-title .unlisted}

Learn what forward bounds, backward residuals, exact comparisons, and the default local unittest command each cover.
:::
::::
