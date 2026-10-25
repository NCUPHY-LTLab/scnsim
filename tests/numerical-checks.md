---
title: "Numerical checks and local run"
lang: en
sidebar: testing
---

[English](numerical-checks.md) · [繁體中文](zh-TW/numerical-checks.md)

## Forward comparisons

For declared forward-comparison fixtures, the bound is `relative_error <= 100 * condition_bound * float64_epsilon`. The tests derive each condition bound independently from the fixture equations and report the actual value, oracle value, units, bound, and normalized error ratio.

## Backward and exact checks

Actual operator and solve backward residuals use `256 * (n + 1) * float64_epsilon`. These checks are separate from exact shape, order, encoded-value, unit-projection, binary64-bit, and saved-result comparisons. The suite does not add an epsilon floor to a zero relative normalization. The current suite has no dedicated numerical regression for `StaticCapacitance`.

## Run the default suite

From the repository root, use the locked course-generation group and discover the default unittest suite:

```sh
uv sync --locked --group course-generation
uv run --locked --group course-generation python -m unittest discover -s tests -t . -v
```

This default suite exercises JAX and does not start Julia. The 120-case Golden comparison and performance runs are separately assigned work; they are not part of default CI.
