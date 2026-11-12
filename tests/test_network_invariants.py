"""Network invariants of Direct Y, Z and S against an exact rational oracle.

Each family (reciprocity, passivity, losslessness, the scaling laws and node
permutation) first checks every entry with ``assert_forward``. Its condition
bound comes from ``tests.exact_network``: the sum over independently and
relatively perturbed primitives (branch values, angular frequency and, for S,
the reference impedance) of the exact first-order change, divided by the exact
entry. The invariant is then checked on the product's matrices with a bound
derived only from those per-entry forward bounds.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import unittest

import numpy as np

from scnsim import CircuitPlan, CircuitRun, DirectSolveSpec, components, units as u
from tests.analytic_conditions import assert_forward
from tests.exact_network import Branch, Network, ONE, ZERO, matmul, transpose


EPS = np.finfo(np.float64).eps
FREQUENCIES = (4.1e9, 5.3e9, 7.9e9)
REFERENCE = 50.0

# Port node a and b, internal node m. Two resonators share a series coupler.
LOSSY = (("C", "a", None, 120e-15), ("L", "a", "m", 3.3e-9), ("C", "m", None, 250e-15),
         ("R", "m", None, 2.0e3), ("C", "m", "b", 15e-15), ("L", "b", None, 6.8e-9),
         ("C", "b", None, 90e-15), ("R", "a", "b", 7.5e3))
LOSSLESS = tuple(branch for branch in LOSSY if branch[0] != "R")
UNITS = {"Y": u.S, "Z": u.ohm, "S": u.dimensionless}


def _element(identifier, kind, value):
    if kind == "C":
        return components.capacitor(id=identifier, capacitance=value * u.F)
    if kind == "L":
        return components.inductor(id=identifier, inductance=value * u.H)
    return components.resistor(id=identifier, resistance=value * u.ohm)


def _plan(branches, *, bus_order=("a", "m", "b"), branch_order=None, port_order=("a", "b"),
          reference=REFERENCE):
    plan = CircuitPlan(id="invariant_two_port")
    buses = {name: plan.bus(id=name) for name in bus_order}
    order = range(len(branches)) if branch_order is None else branch_order
    for position in order:
        kind, a, b, value = branches[position]
        element = plan.add(_element(f"e{position}", kind, value))
        plan.series(id=f"s{position}", start=buses[a], elements=(element,),
                    end=buses[b] if b is not None else plan.ground)
    for name in port_order:
        plan.add_port(id=f"p{name}", at=buses[name], role="terminated",
                      reference_impedance=reference * u.ohm)
    return plan


def _network(branches, ports=("a", "b")):
    return Network(tuple(Branch(*branch) for branch in branches), ports)


@contextmanager
def _solved(plan, frequencies):
    run = CircuitRun(plan=plan, storage="memory", precision="float64")
    try:
        result = run.solve(run.original, DirectSolveSpec(frequencies=list(frequencies) * u.Hz))
        yield {family: getattr(result, family.lower()).view.matrix.to(unit).magnitude
               for family, unit in UNITS.items()}
    finally:
        run.close()


def _assert_entries(test, label, actual, oracle, family, *, permutation=(0, 1)):
    """Forward-check every entry; return the per-entry absolute bounds."""
    size = len(permutation)
    bounds = np.zeros((size, size))
    for row in range(size):
        for column in range(size):
            source_row, source_column = permutation[row], permutation[column]
            condition = oracle.condition(family, source_row, source_column)
            expected = oracle.entry(family, source_row, source_column)
            assert_forward(test, f"{label}_{family}{row}{column}", actual[row, column], expected,
                           condition, str(UNITS[family]), frequency_perturbed=True)
            bounds[row, column] = 100 * condition * EPS * abs(expected)
    return bounds


def _gram_bound(matrix, bounds):
    """Bound of |(M^H M)_ij - (M0^H M0)_ij| from entrywise bounds of M."""
    magnitude = np.abs(matrix) + bounds
    return magnitude.T @ magnitude - np.abs(matrix).T @ np.abs(matrix)


class ReciprocityTests(unittest.TestCase):
    def test_lossy_two_port_y_z_s_are_symmetric(self):
        network = _network(LOSSY)
        with _solved(_plan(LOSSY), FREQUENCIES) as families:
            for index, frequency in enumerate(FREQUENCIES):
                oracle = network.response(frequency, REFERENCE)
                for family in ("Y", "Z", "S"):
                    exact = oracle.matrix(family)
                    self.assertEqual(exact[0][1], exact[1][0])  # Oracle reciprocity is exact.
                    actual = families[family][index]
                    bounds = _assert_entries(self, f"reciprocity_{frequency:.3e}", actual, oracle, family)
                    # Both entries are within their forward bound of one exact value.
                    self.assertLessEqual(abs(actual[0, 1] - actual[1, 0]), bounds[0, 1] + bounds[1, 0])


class PassivityTests(unittest.TestCase):
    def test_lossy_two_port_hermitian_part_and_scattering_contraction(self):
        network = _network(LOSSY)
        with _solved(_plan(LOSSY), FREQUENCIES) as families:
            for index, frequency in enumerate(FREQUENCIES):
                oracle = network.response(frequency, REFERENCE)
                with self.subTest(frequency=frequency):
                    y = families["Y"][index]
                    y_bounds = _assert_entries(self, f"passivity_{frequency:.3e}", y, oracle, "Y")
                    hermitian = (y + y.conj().T) / 2
                    exact = np.array([[complex(value) for value in row] for row in oracle.y])
                    exact_hermitian = (exact + exact.conj().T) / 2
                    h_bounds = (y_bounds + y_bounds.T) / 2
                    self._assert_positive_definite(hermitian, exact_hermitian, h_bounds)

                    s = families["S"][index]
                    s_bounds = _assert_entries(self, f"contraction_{frequency:.3e}", s, oracle, "S")
                    exact_s = np.array([[complex(value) for value in row] for row in oracle.s])
                    defect = np.eye(2) - s.conj().T @ s
                    exact_defect = np.eye(2) - exact_s.conj().T @ exact_s
                    self._assert_positive_definite(defect, exact_defect, _gram_bound(exact_s, s_bounds))

    def _assert_positive_definite(self, actual, exact, bounds):
        # A 2x2 Hermitian matrix is positive definite iff both diagonal entries
        # and the determinant are positive. The precondition shows that the
        # entry bounds cannot change any of these signs for this fixture.
        determinant_bound = (abs(exact[0, 0]) * bounds[1, 1] + abs(exact[1, 1]) * bounds[0, 0]
                             + bounds[0, 0] * bounds[1, 1] + 2 * abs(exact[0, 1]) * bounds[0, 1]
                             + bounds[0, 1] ** 2)
        exact_determinant = (exact[0, 0] * exact[1, 1] - abs(exact[0, 1]) ** 2).real
        self.assertGreater(exact[0, 0].real, bounds[0, 0])
        self.assertGreater(exact[1, 1].real, bounds[1, 1])
        self.assertGreater(exact_determinant, determinant_bound)
        self.assertGreater(actual[0, 0].real, 0)
        self.assertGreater(actual[1, 1].real, 0)
        self.assertGreater((actual[0, 0] * actual[1, 1] - abs(actual[0, 1]) ** 2).real, 0)


class LosslessnessTests(unittest.TestCase):
    def test_lossless_two_port_y_is_imaginary_and_s_is_unitary(self):
        network = _network(LOSSLESS)
        with _solved(_plan(LOSSLESS), FREQUENCIES) as families:
            for index, frequency in enumerate(FREQUENCIES):
                oracle = network.response(frequency, REFERENCE)
                with self.subTest(frequency=frequency):
                    # The oracle is exactly lossless: Re Y = 0 and S^H S = I.
                    self.assertTrue(all(value.re == 0 for row in oracle.y for value in row))
                    gram = matmul([[value.conjugate() for value in row] for row in transpose(oracle.s)],
                                  oracle.s)
                    self.assertEqual(gram, [[ONE, ZERO], [ZERO, ONE]])

                    y = families["Y"][index]
                    y_bounds = _assert_entries(self, f"lossless_{frequency:.3e}", y, oracle, "Y")
                    self.assertTrue(np.all(np.abs(y.real) <= y_bounds), y.real.tolist())
                    s = families["S"][index]
                    s_bounds = _assert_entries(self, f"unitary_{frequency:.3e}", s, oracle, "S")
                    exact_s = np.array([[complex(value) for value in row] for row in oracle.s])
                    defect = np.abs(s.conj().T @ s - np.eye(2))
                    self.assertTrue(np.all(defect <= _gram_bound(exact_s, s_bounds)), defect.tolist())


class ScalingLawTests(unittest.TestCase):
    # Powers of two keep every scaled primitive exact, so the scaled exact
    # oracle equals the unscaled one exactly (asserted below).
    def test_frequency_scaling_keeps_y_z_s(self):
        alpha = 2.0 ** 10
        scaled = tuple((kind, a, b, value * alpha if kind in "LC" else value)
                       for kind, a, b, value in LOSSY)
        frequencies = tuple(frequency / alpha for frequency in FREQUENCIES)
        network = _network(LOSSY)
        with _solved(_plan(scaled), frequencies) as families:
            for index, frequency in enumerate(FREQUENCIES):
                oracle = network.response(frequency, REFERENCE)
                scaled_oracle = _network(scaled).response(frequency / alpha, REFERENCE)
                for family in ("Y", "Z", "S"):
                    self.assertEqual(scaled_oracle.matrix(family), oracle.matrix(family))
                    _assert_entries(self, f"frequency_scaling_{frequency:.3e}",
                                    families[family][index], oracle, family)

    def test_impedance_scaling_scales_y_and_z_and_keeps_s(self):
        beta = 2.0 ** -3
        def scale(kind, value):
            return value / beta if kind == "C" else value * beta
        scaled = tuple((kind, a, b, scale(kind, value)) for kind, a, b, value in LOSSY)
        network = _network(LOSSY)
        factors = {"Y": beta, "Z": 1 / beta, "S": 1.0}
        with _solved(_plan(scaled, reference=REFERENCE * beta), FREQUENCIES) as families:
            for index, frequency in enumerate(FREQUENCIES):
                oracle = network.response(frequency, REFERENCE)
                scaled_oracle = _network(scaled).response(frequency, REFERENCE * beta)
                for family, factor in factors.items():
                    self.assertEqual([[complex(value) * factor for value in row]
                                      for row in scaled_oracle.matrix(family)],
                                     [[complex(value) for value in row] for row in oracle.matrix(family)])
                    # Multiplying by a power of two is exact in binary64.
                    _assert_entries(self, f"impedance_scaling_{frequency:.3e}",
                                    families[family][index] * factor, oracle, family)


class NodePermutationTests(unittest.TestCase):
    def test_bus_and_branch_declaration_order_do_not_change_ports(self):
        network = _network(LOSSY)
        reordered = _plan(LOSSY, bus_order=("b", "m", "a"), branch_order=(7, 3, 5, 0, 6, 1, 4, 2))
        with _solved(reordered, FREQUENCIES) as families:
            for index, frequency in enumerate(FREQUENCIES):
                oracle = network.response(frequency, REFERENCE)
                for family in ("Y", "Z", "S"):
                    _assert_entries(self, f"bus_order_{frequency:.3e}", families[family][index],
                                    oracle, family)

    def test_port_declaration_order_permutes_rows_and_columns(self):
        network = _network(LOSSY)
        with _solved(_plan(LOSSY, port_order=("b", "a")), FREQUENCIES) as families:
            for index, frequency in enumerate(FREQUENCIES):
                oracle = network.response(frequency, REFERENCE)
                for family in ("Y", "Z", "S"):
                    _assert_entries(self, f"port_order_{frequency:.3e}", families[family][index],
                                    oracle, family, permutation=(1, 0))


if __name__ == "__main__":
    unittest.main()
