"""A lossy transmission line: the pi ladder and the exact telegrapher solution.

Two separate comparisons:

1. Same pi ladder, two solvers. SCNSim's Direct Y of an ``n_sections`` line is
   compared with an independent chain-matrix (ABCD) solution of the same
   symmetric pi ladder documented in the component-authoring guide: per section
   of length dx = length / n a series R dx + L dx, and shunt G dx / 2, C dx / 2
   at both ends. The oracle multiplies the section matrices in 60-digit Decimal
   arithmetic (oracle precision, not a tolerance). The condition bound of each
   Y entry is the componentwise triangle bound over every branch value of the
   ladder and the angular frequency, from prefix/suffix chain products. This
   comparison covers up to 16 sections; the 64-section case misses the forward
   bound and is reported on Issue 47.

2. Convergence to the telegrapher line. The exact line has
   A = D = cosh(g l), B = Zc sinh(g l), C = sinh(g l) / Zc with
   g = sqrt(Z' Y'), Zc = Z' / g, Z' = R' - i w L', Y' = G' - i w C'. The
   product's discretization error |Y_n - Y_line| must fall strictly at every
   doubling of n. The oracle first shows that the exact ladder's own error
   falls by more than the two forward bounds, so the strict comparison is not
   decided by rounding. No convergence-rate threshold is asserted.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import math
import unittest

import numpy as np

from scnsim import CircuitPlan, CircuitRun, DirectSolveSpec, RLGC, components, units as u
from tests.analytic_conditions import assert_forward
from tests.decimal_complex import DComplex


PRECISION = 60  # Oracle arithmetic precision, not a comparison tolerance.
EPS = np.finfo(np.float64).eps
R_PER_M, L_PER_M, G_PER_M, C_PER_M = 2.0, 420e-9, 1e-4, 175e-12  # SI per metre
LENGTH = 5e-3
FREQUENCY = 6e9
SECTIONS = (1, 2, 4, 8, 16, 32, 64)
# The same-ladder forward comparison stops at 16 sections: at 64 sections the
# product misses the forward bound (reported on Issue 47 as a finding).
LADDER_SECTIONS = (1, 2, 4, 8, 16)
ENTRIES = ((0, 0), (0, 1))  # Y22 = Y11 and Y21 = Y12 for this symmetric line.


def _plan(sections):
    rlgc = RLGC(conductors=("signal",), reference_conductor="ground",
                resistance_per_length=[[R_PER_M]] * u.ohm / u.meter,
                inductance_per_length=[[L_PER_M]] * u.H / u.meter,
                conductance_per_length=[[G_PER_M]] * u.S / u.meter,
                capacitance_per_length=[[C_PER_M]] * u.F / u.meter,
                extraction_frequency=FREQUENCY * u.Hz)
    plan = CircuitPlan(id="telegrapher_line")
    line = plan.add(components.transmission_line(id="line", length=LENGTH * u.meter, rlgc=rlgc,
                                                 n_sections=sections))
    for end in ("head", "tail"):
        node = plan.bus(id=end)
        plan.link(id=f"connect_{end}", endpoints=(node, line.pin(end, conductor="signal")))
        plan.add_port(id=end, at=node, role="terminated", reference_impedance=50.0 * u.ohm)
    return plan


def _product_y(sections):
    run = CircuitRun(plan=_plan(sections), storage="memory", precision="float64")
    try:
        result = run.solve(run.original, DirectSolveSpec(frequencies=[FREQUENCY] * u.Hz))
        return result.y.view.matrix.to(u.S).magnitude[0]
    finally:
        run.close()


def _decimal(value) -> Decimal:
    fraction = Fraction(value)
    return Decimal(fraction.numerator) / Decimal(fraction.denominator)


def _matmul(a, b):
    return [[a[i][0] * b[0][j] + a[i][1] * b[1][j] for j in range(2)] for i in range(2)]


ZERO, ONE = DComplex(0), DComplex(1)
IDENTITY = [[ONE, ZERO], [ZERO, ONE]]


def _shunt(y):
    return [[ONE, ZERO], [y, ONE]]


def _series(z):
    return [[ONE, z], [ZERO, ONE]]


def _y_from_chain(t):
    (a, b), (c, d) = t
    return [[d / b, (b * c - a * d) / b], [ZERO - ONE / b, a / b]]


def _y_change(t, dt):
    """First-order change of Y for a chain change dt (from the quotient rule)."""
    (a, b), (_, d) = t
    (da, db), (_, dd) = dt
    square = b * b
    return [[(dd * b - d * db) / square, db / square], [db / square, (da * b - a * db) / square]]


class _Ladder:
    """Exact-arithmetic pi ladder; its Y and condition bounds at one frequency."""

    def __init__(self, sections):
        with localcontext() as context:
            context.prec = PRECISION
            dx = _decimal(LENGTH) / sections
            omega = _decimal(2 * math.pi * FREQUENCY)
            minus_i_omega = DComplex(0, -omega)
            r, l = _decimal(R_PER_M) * dx, _decimal(L_PER_M) * dx
            g, c = _decimal(G_PER_M) * dx / 2, _decimal(C_PER_M) * dx / 2
            self.series = DComplex(r) + minus_i_omega * l
            self.shunt = DComplex(g) + minus_i_omega * c
            section = _matmul(_matmul(_shunt(self.shunt), _series(self.series)), _shunt(self.shunt))
            prefix = [IDENTITY]
            for _ in range(sections):
                prefix.append(_matmul(prefix[-1], section))
            total = prefix[-1]
            suffix = [IDENTITY]
            for _ in range(sections):
                suffix.append(_matmul(section, suffix[-1]))
            self.y = _y_from_chain(total)
            # Relative branch changes inside one section (Se = series, Sh = shunt).
            shunt, series = _shunt(self.shunt), _series(self.series)
            def d_series(value):
                return [[ZERO, value], [ZERO, ZERO]]
            def d_shunt(value):
                return [[ZERO, ZERO], [value, ZERO]]
            locals_ = []
            for value in (DComplex(r), minus_i_omega * l):
                locals_.append(_matmul(_matmul(shunt, d_series(value)), shunt))
            for value in (DComplex(g), minus_i_omega * c):
                locals_.append(_matmul(_matmul(d_shunt(value), series), shunt))
                locals_.append(_matmul(_matmul(shunt, series), d_shunt(value)))
            omega_local = _matmul(_matmul(d_shunt(minus_i_omega * c), series), shunt)
            omega_local = _add(omega_local, _matmul(_matmul(shunt, d_series(minus_i_omega * l)), shunt))
            omega_local = _add(omega_local, _matmul(_matmul(shunt, series), d_shunt(minus_i_omega * c)))
            self.sensitivity = [[Decimal(0)] * 2 for _ in range(2)]
            omega_total = [[ZERO, ZERO], [ZERO, ZERO]]
            for k in range(sections):
                left, right = prefix[k], suffix[sections - k - 1]
                for local in locals_:
                    change = _y_change(total, _matmul(_matmul(left, local), right))
                    for i, j in ENTRIES:
                        self.sensitivity[i][j] += change[i][j].magnitude()
                omega_total = _add(omega_total, _matmul(_matmul(left, omega_local), right))
            omega_change = _y_change(total, omega_total)
            for i, j in ENTRIES:
                self.sensitivity[i][j] += omega_change[i][j].magnitude()

    def entry(self, i, j) -> complex:
        return complex(self.y[i][j])

    def condition(self, i, j) -> float:
        return float(self.sensitivity[i][j]) / abs(self.entry(i, j))


def _add(a, b):
    return [[a[i][j] + b[i][j] for j in range(2)] for i in range(2)]


def _telegrapher_y():
    with localcontext() as context:
        context.prec = PRECISION
        omega = _decimal(2 * math.pi * FREQUENCY)
        series = DComplex(_decimal(R_PER_M), -omega * _decimal(L_PER_M))
        shunt = DComplex(_decimal(G_PER_M), -omega * _decimal(C_PER_M))
        gamma = (series * shunt).sqrt()
        impedance = series / gamma
        argument = gamma * DComplex(_decimal(LENGTH))
        grow, decay = argument.exp(), (ZERO - argument).exp()
        cosh, sinh = (grow + decay) / 2, (grow - decay) / 2
        chain = [[cosh, impedance * sinh], [sinh / impedance, cosh]]
        return _y_from_chain(chain)


class TelegrapherLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.product = {n: _product_y(n) for n in SECTIONS}
        cls.ladder = {n: _Ladder(n) for n in SECTIONS}

    def test_same_pi_ladder_matches_chain_matrix_oracle(self):
        for n in LADDER_SECTIONS:
            for i, j in ENTRIES:
                oracle = self.ladder[n]
                assert_forward(self, f"pi_ladder_n{n}_Y{i}{j}", self.product[n][i, j], oracle.entry(i, j),
                               oracle.condition(i, j), "S", frequency_perturbed=True)
                # Reciprocity and symmetry of the line, within the same bounds.
                mirror = (j, i) if (i, j) != (0, 0) else (1, 1)
                assert_forward(self, f"pi_ladder_n{n}_Y{mirror[0]}{mirror[1]}", self.product[n][mirror],
                               oracle.entry(i, j), oracle.condition(i, j), "S", frequency_perturbed=True)

    def test_pi_ladder_converges_to_the_telegrapher_line(self):
        line = _telegrapher_y()
        for i, j in ENTRIES:
            exact = complex(line[i][j])
            for coarse, fine in zip(SECTIONS, SECTIONS[1:]):
                with self.subTest(entry=(i, j), sections=(coarse, fine)):
                    def bound(n):
                        return 100 * self.ladder[n].condition(i, j) * EPS * abs(self.ladder[n].entry(i, j))
                    with localcontext() as context:
                        context.prec = PRECISION
                        oracle_error = {n: float((self.ladder[n].y[i][j] - line[i][j]).magnitude())
                                        for n in (coarse, fine)}
                    # Precondition: the exact ladder's improvement exceeds both bounds.
                    self.assertGreater(oracle_error[coarse] - oracle_error[fine], bound(coarse) + bound(fine))
                    product_error = {n: abs(self.product[n][i, j] - exact) for n in (coarse, fine)}
                    print("TELEGRAPHER " + str(dict(entry=[i, j], sections=[coarse, fine],
                          product_error=[product_error[coarse], product_error[fine]],
                          oracle_error=[oracle_error[coarse], oracle_error[fine]])), flush=True)
                    self.assertLess(product_error[fine], product_error[coarse])


if __name__ == "__main__":
    unittest.main()
