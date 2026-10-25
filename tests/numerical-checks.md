---
title: "Numerical checks and local run"
lang: en
sidebar: testing
---


SCNSim’s numerical tests use different checks for different questions. A forward comparison asks whether the computed observable (`actual`) agrees with an independently calculated reference (`oracle`). A backward residual asks how closely the computed values satisfy the original equation. Exact checks compare representation and saved data; these checks answer different questions.

## Forward comparison

For declared fixtures, the normalized forward error is

```text
relative_error = max(abs(actual - oracle)) / max(abs(oracle))
relative_error <= 100 * condition_bound * float64_epsilon
```

Here `κ` (`condition_bound`) estimates how the fixture’s equations affect error in the selected observable, and `float64_epsilon` is the spacing scale of binary64 arithmetic. The bound is derived from the fixture equations and independent oracle for that observable. Reports include computed and reference values, units, shape, `κ`, and the normalized error ratio. A pole uses its observable-specific condition bound, not a generic linewidth-relative error.

## Backward residual and exact comparisons

Actual operator and solve backward residuals use `256 * (n + 1) * float64_epsilon`. Exact checks separately compare shapes, ordering, encoded values, unit projections, binary64 bits, and saved-result fields. A zero oracle normalization does not receive an added epsilon floor.

These checks apply to the small declared fixtures described on the topic pages. The current suite has no dedicated numerical regression for `StaticCapacitance`; the 120-case Golden comparison and performance runs are not part of default CI.

## Run the default unittest suite

From the repository root, use the locked course-generation dependency group:

```sh
uv sync --locked --group course-generation
uv run --locked --group course-generation python -m unittest discover -s tests -t . -v
```

This default suite uses JAX and does not start Julia. CI also runs the suite in its installed-wheel jobs.
