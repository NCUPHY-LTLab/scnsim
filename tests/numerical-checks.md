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

## Verification families

Five further families check properties that a single analytic fixture cannot. Each forward assertion records its `κ` and how the oracle side estimated it.

| Family | What it checks | Oracle | Source |
| --- | --- | --- | --- |
| Network invariants | Reciprocity, passivity, losslessness, the frequency and impedance scaling laws, and node and port permutation of Direct `Y`, `Z` and `S` for a two-port with one eliminated node | Exact rational nodal solution; `κ` sums the exact first-order change of each entry over every branch value, the angular frequency and the reference impedance | [`test_network_invariants.py`](test_network_invariants.py), [`exact_network.py`](exact_network.py) |
| Differentiation rules | An inventory of every custom JVP and transpose rule in the installed package, each tested rule with a Taylor test and the JVP/VJP inner-product identity | Exact or 60-digit primal and derivative values; the Taylor remainder and the identity use bounds derived from the forward bounds of their parts | [`test_differentiation_rules.py`](test_differentiation_rules.py) |
| Global pole oracle | Every pole of a three-mode lossy network, from the descriptor pencil `A = [[0, I], [-K, -G]]`, `B = diag(I, C)`, matched one-to-one by an anchored hybridized pole; and the impedance zero from `b` to `a` of the View that retains `a` and `b`. Separately, the choice made after the last Newton correction of a transfer zero, tested directly with constructed packets: a corrected point that closes is returned, one that does not close falls back to the accepted point, and a failed evaluation there is returned as that failure | Generalized eigenvalues polished in 60-digit Decimal arithmetic; the zero polished from its anchor in the same arithmetic on two formulations, the bordered Rosenbrock system matrix and the cofactor of `Q`; componentwise root `κ` from the adjugate | [`test_pole_oracle.py`](test_pole_oracle.py) |
| Telegrapher line | The π ladder of a lossy line against a chain-matrix solution of the same ladder (up to 16 sections), and, separately, the strict decrease of its error against the exact telegrapher solution as the sections double | 60-digit chain matrices and hyperbolic functions | [`test_telegrapher_line.py`](test_telegrapher_line.py) |
| Error-free transformations | TwoSum and TwoProduct of the platform kernel in each wheel, binary32 and binary64, in the normal and the subnormal range; the JAX compensated arithmetic in the normal range | Exact `Fraction` arithmetic, so no tolerance is involved | [`test_probe_eft.py`](test_probe_eft.py) |

The inventory test fails when a new differentiation rule is added without a test or a recorded reason. The convergence check asserts no rate: it only requires the error to fall, after the oracle shows that the exact ladder's improvement exceeds both forward bounds.

## Run the default unittest suite

From the repository root, use the locked course-generation dependency group:

```sh
uv sync --locked --group course-generation
uv run --locked --group course-generation python -m unittest discover -s tests -t . -v
```

This default suite uses JAX and does not start Julia. CI also runs the suite in its installed-wheel jobs.
