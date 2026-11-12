"""Global pole oracle through the linearized descriptor pencil.

The network operator is Q(omega) = K - omega**2 C - i omega G. With
lambda = -i omega, Q(omega) x = 0 is the generalized eigenproblem
A z = lambda B z of A = [[0, I], [-K, -G]], B = diag(I, C), z = [x, lambda x].
All 2n eigenvalues are computed at once, so every pole is found without an
anchor. Each eigenvalue is polished by Newton's method on det Q in 60-digit
Decimal arithmetic (oracle precision, not a tolerance). The condition bound of
a root w of det Q(w, p) = 0 is the componentwise triangle bound over the
relatively perturbed primitives p:
kappa = sum_p |tr(adj Q  p dQ/dp)| / (|w| |tr(adj Q  dQ/dw)|).
The product's anchored poles are compared with assert_forward, and the pole
set must be matched one-to-one.

The impedance zero of the retained View (a, b), with m eliminated, is a root of
the bordered Rosenbrock system matrix [[Q, e_b], [e_a^T, 0]], whose determinant
is -adj(Q)[a, b], and independently of the cofactor that deletes row b and
column a of Q. Both are polished from the anchor in the same 60-digit
arithmetic, with the same componentwise condition bound, and the product's zero
is compared with each.
"""
from __future__ import annotations

from decimal import localcontext
import math
import unittest

import numpy as np
from scipy import linalg

from scnsim import (CircuitPlan, CircuitRun, HybridizedPoleSpec, ReductionPipeline, TransferZeroSpec,
                    components, units as u)
from tests.analytic_conditions import assert_forward
from tests.decimal_complex import DComplex as _D


PRECISION = 60  # Oracle arithmetic precision, not a comparison tolerance.

# Three moderately lossy modes coupled capacitively, inductively and resistively.
BRANCHES = (("C", "a", None, 100e-15), ("L", "a", None, 10e-9),
            ("C", "m", None, 250e-15), ("L", "m", None, 3e-9), ("R", "m", None, 5e4),
            ("C", "b", None, 90e-15), ("L", "b", None, 7e-9),
            ("C", "a", "m", 4e-15), ("L", "m", "b", 300e-9),
            ("C", "a", "b", 1e-15), ("R", "a", "b", 2e5))
NODES = ("a", "m", "b")
# Transfer impedance from b to a of the retained View (a, b), near 5.906 GHz.
ZERO_INPUT, ZERO_OUTPUT, ZERO_ANCHOR_HZ = "b", "a", 5.906e9


def _plan():
    plan = CircuitPlan(id="pole_oracle")
    buses = {name: plan.bus(id=name) for name in NODES}
    for index, (kind, a, b, value) in enumerate(BRANCHES):
        element = {"C": lambda: components.capacitor(id=f"e{index}", capacitance=value * u.F),
                   "L": lambda: components.inductor(id=f"e{index}", inductance=value * u.H),
                   "R": lambda: components.resistor(id=f"e{index}", resistance=value * u.ohm)}[kind]()
        plan.series(id=f"s{index}", start=buses[a], elements=(plan.add(element),),
                    end=buses[b] if b is not None else plan.ground)
    return plan, buses


def _stamps():
    """Per-branch nodal stamps of C, G and K (unit incidence products)."""
    index = {name: position for position, name in enumerate(NODES)}
    stamps = []
    for kind, a, b, value in BRANCHES:
        vector = np.zeros(len(NODES))
        if a is not None:
            vector[index[a]] += 1
        if b is not None:
            vector[index[b]] -= 1
        stamps.append((kind, value, np.outer(vector, vector)))
    return stamps


def _matrices():
    c, g, k = (np.zeros((3, 3)) for _ in range(3))
    for kind, value, stamp in _stamps():
        target = {"C": c, "R": g, "L": k}[kind]
        target += stamp * (value if kind == "C" else 1 / value)
    return c, g, k


ZERO, ONE, MINUS_I = _D(0), _D(1), _D(0, -1)


def _det(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    total = ZERO
    for column in range(n):
        minor = [row[:column] + row[column + 1:] for row in matrix[1:]]
        term = matrix[0][column] * _det(minor)
        total = total + term if column % 2 == 0 else total - term
    return total


def _adjugate_trace(matrix, direction):
    """tr(adj(M) D) = sum_ij cofactor_ij D_ij, without inverting M."""
    n = len(matrix)
    total = ZERO
    for i in range(n):
        for j in range(n):
            if complex(direction[i][j]) == 0:
                continue
            minor = [row[:j] + row[j + 1:] for r, row in enumerate(matrix) if r != i]
            cofactor = _det(minor) if minor else ONE
            term = cofactor * direction[i][j]
            total = total + term if (i + j) % 2 == 0 else total - term
    return total


class _Equation:
    """det Q(omega) = 0 for the fixture's nodal operator."""

    def system(self, q, constant):
        """The matrix whose determinant is the equation; Q itself for poles."""
        return q

    def operator(self, omega):
        q = [[ZERO] * 3 for _ in range(3)]
        for kind, value, stamp in _stamps():
            local = self.branch_term(kind, value, omega, relative=False)
            q = [[q[i][j] + local * _D.of(stamp[i, j]) for j in range(3)] for i in range(3)]
        return q

    @staticmethod
    def branch_term(kind, value, omega, *, relative):
        """Branch term of Q, or with relative=True its p d/dp."""
        v = _D.of(value)
        if kind == "C":
            return ZERO - omega * omega * v  # Linear in C: p dQ/dp equals the term.
        if kind == "L":
            term = ONE / v
            return ZERO - term if relative else term
        term = MINUS_I * omega / v  # -i omega G with G = 1/R
        return ZERO - term if relative else term

    def frequency_derivative(self, omega):
        q = [[ZERO] * 3 for _ in range(3)]
        for kind, value, stamp in _stamps():
            local = {"C": _D(-2) * omega * _D.of(value), "L": ZERO,
                     "R": MINUS_I / _D.of(value)}[kind]
            q = [[q[i][j] + local * _D.of(stamp[i, j]) for j in range(3)] for i in range(3)]
        return q

    def polish(self, initial):
        with localcontext() as context:
            context.prec = PRECISION
            omega = _D.of(initial)
            for _ in range(60):
                operator = self.system(self.operator(omega), ONE)
                step = _det(operator) / _adjugate_trace(
                    operator, self.system(self.frequency_derivative(omega), ZERO))
                updated = omega - step
                if updated.re == omega.re and updated.im == omega.im:
                    break
                omega = updated
            return complex(omega), self._condition(omega)

    def _condition(self, omega):
        matrix = self.system(self.operator(omega), ONE)
        slope = _adjugate_trace(matrix, self.system(self.frequency_derivative(omega), ZERO))
        total = 0.0
        for kind, value, stamp in _stamps():
            local = self.branch_term(kind, value, omega, relative=True)
            direction = self.system([[local * _D.of(stamp[i, j]) for j in range(3)] for i in range(3)], ZERO)
            total += abs(complex(_adjugate_trace(matrix, direction)))
        return total / (abs(complex(omega)) * abs(complex(slope)))


class _ZeroEquation(_Equation):
    """A zero of Z[output, input] of the retained View, in one of two forms.

    Eliminating m leaves Z = (Y_view)^-1 with the same nonzero-frequency zeros
    as adj(Q)[output, input]. The constant border of the Rosenbrock matrix does
    not depend on frequency or on any branch value, so its derivatives are zero.
    """

    def __init__(self, formulation):
        self.formulation = formulation

    def system(self, q, constant):
        output, input_ = NODES.index(ZERO_OUTPUT), NODES.index(ZERO_INPUT)
        n = len(NODES)
        if self.formulation == "rosenbrock":
            return ([row + [constant if r == input_ else ZERO] for r, row in enumerate(q)]
                    + [[constant if c == output else ZERO for c in range(n)] + [ZERO]])
        return [[q[r][c] for c in range(n) if c != output] for r in range(n) if r != input_]


def _pencil():
    """Physical (Re omega > 0) finite eigenvalues, as starting points only."""
    c, g, k = _matrices()
    n = len(NODES)
    # lambda = scale * mu keeps the binary64 pencil well scaled; dividing the
    # second block row by max|C| does not change the eigenvalues.
    scale, row_scale = 2 * math.pi * 1e9, np.max(np.abs(c))
    a = np.block([[np.zeros((n, n)), np.eye(n)], [-k / scale**2 / row_scale, -g / scale / row_scale]])
    b = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), c / row_scale]])
    eigenvalues = linalg.eigvals(a, b)
    omegas = 1j * scale * eigenvalues[np.isfinite(eigenvalues)]
    return [complex(value) for value in omegas if value.real > 0]


class PoleOracleTests(unittest.TestCase):
    def test_every_pencil_pole_is_matched_by_one_hybridized_pole(self):
        candidates = _pencil()
        self.assertEqual(len(candidates), len(NODES))  # One physical pole per mode.
        equation = _Equation()
        oracle = sorted((equation.polish(value) for value in candidates), key=lambda item: item[0].real)
        plan, buses = _plan()
        run = CircuitRun(plan=plan, storage="memory", precision="float64")
        try:
            coordinates = tuple(buses[name].node for name in NODES)
            view = run.original.reduce(ReductionPipeline().retain(*coordinates))
            found = []
            for expected, condition in oracle:
                result = run.evaluate(view, HybridizedPoleSpec(
                    coordinates=coordinates, anchor=expected.real / (2 * math.pi) * u.Hz))
                actual = complex(result.root.to("radian / second").magnitude)
                assert_forward(self, f"pencil_pole_{expected.real:.6e}", actual, expected,
                               condition, "radian / second")
                found.append(actual)
            # One-to-one: each oracle pole is nearest to its own product pole.
            for position, expected in enumerate(item[0] for item in oracle):
                nearest = min(range(len(found)), key=lambda k: abs(found[k] - expected))
                self.assertEqual(nearest, position)
        finally:
            run.close()


class TransferZeroOracleTests(unittest.TestCase):
    def test_retained_view_impedance_zero_matches_both_formulations(self):
        plan, buses = _plan()
        run = CircuitRun(plan=plan, storage="memory", precision="float64")
        try:
            view = run.original.reduce(ReductionPipeline().retain(buses["a"].node, buses["b"].node))
            result = run.evaluate(view, TransferZeroSpec(
                family="Z", input_coordinate=buses[ZERO_INPUT].node,
                output_coordinate=buses[ZERO_OUTPUT].node, anchor=ZERO_ANCHOR_HZ * u.Hz))
            actual = complex(result.zero.to("radian / second").magnitude)
        finally:
            run.close()
        for formulation in ("rosenbrock", "cofactor"):
            expected, condition = _ZeroEquation(formulation).polish(2 * math.pi * ZERO_ANCHOR_HZ)
            assert_forward(self, f"retained_view_z_zero_{formulation}", actual, expected,
                           condition, "radian / second")


if __name__ == "__main__":
    unittest.main()
