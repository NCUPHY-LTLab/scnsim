"""Inventory of SCNSim's custom differentiation rules, with Taylor and JVP/VJP tests.

The inventory test reads the installed package sources and requires every
registered rule (``jax.custom_jvp``/``custom_vjp`` functions and JAX primitive
JVP/transpose rules) to appear in ``RULES``. Each entry names the tests that
exercise it.

Taylor test. For a scalar functional phi, a step h and a direction v, the
remainder r(h) = phi(x + h v) - phi(x) - h * (J v) is formed exactly (Fraction)
from the product's own binary64 primal values and JVP, and compared with the
same remainder formed from the exact oracle at the same binary64 inputs.
Its condition bound follows from the triangle inequality:
kappa = (S(phi(x + h v)) + S(phi(x)) + |h| S(J v)) / |r_oracle|, where S is the
oracle's absolute first-order sensitivity sum over independently and relatively
perturbed primitives.

JVP/VJP identity. <w, J v> from the JVP and <J^T w, v> from the VJP are both
formed exactly from the product's values. Both are the same bilinear function
of the primitives, v and w, so each is compared with the exact oracle value
using the sensitivity S of that bilinear form, and their difference is bounded
by the sum of the two forward bounds.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import unittest

import numpy as np

import scnsim
from scnsim import (CircuitPlan, CircuitRun, DiagonalRootSpec, JVP, VJP, OutputCoordinate,
                    ParameterDefinitions, ParameterSet, ParameterSpec, ReductionPipeline,
                    ResponseElementSpec, SensitivitySpec, TransferZeroSpec, components, units as u)
from tests.analytic_conditions import assert_forward, matched_zero_oracle
from tests.exact_network import Branch, Exact, HyperDual, Network, ONE, ZERO


EPS = np.finfo(np.float64).eps
# Oracle arithmetic precision for irrational values, not a comparison tolerance.
PRECISION = 60
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")

# Every rule registered in the installed package, keyed by (module, kind,
# name), with the tests that exercise it. For the primitives inside the native
# equation programs, the public calls named here were observed to invoke the
# rule. A rule with no test lists its reason in UNCOVERED.
_RESPONSE = ("PublicRuleTests.test_response_element",)
_RETAINED_Z = ("PublicRuleTests.test_retained_z_transfer_zero",)
RULES = {
    ("numerics/determinant_ad.py", "custom_jvp", "polynomial"): ("DirectRuleTests.test_determinant",),
    ("numerics/matrix_functions.py", "custom_jvp", "root"): ("DirectRuleTests.test_spd_root",),
    ("numerics/native_compensated.py", "jvp", "_pair_p"): ("DirectRuleTests.test_compensated_pair",) + _RETAINED_Z,
    ("numerics/native_compensated.py", "transpose", "_pair_p"): ("DirectRuleTests.test_compensated_pair",) + _RETAINED_Z,
    ("numerics/native_compensated.py", "jvp", "_batch_p"): ("DirectRuleTests.test_compensated_pair_batched",),
    ("numerics/native_compensated.py", "transpose", "_batch_p"): ("DirectRuleTests.test_compensated_pair_batched",),
    ("numerics/assembly_program.py", "custom_jvp", "inverse"): _RESPONSE,
    ("numerics/native_sparse.py", "transpose", "_solve_p"): _RESPONSE + _RETAINED_Z,
    ("numerics/native_equation_jet.py", "jvp", "ns._solve_p"): _RESPONSE + _RETAINED_Z,
    ("numerics/native_equation_jet.py", "transpose", "_linear_p"): _RESPONSE,
    ("numerics/native_pair.py", "jvp", "_first_p"): _RETAINED_Z,
    ("numerics/native_pair.py", "transpose", "_residual_pair_p"): _RETAINED_Z,
    ("numerics/native_root.py", "custom_jvp", "root"): ("PublicRuleTests.test_diagonal_root",),
    ("numerics/native_root.py", "custom_jvp", "from_packet"): ("PublicRuleTests.test_diagonal_root",),
    ("numerics/native_root.py", "custom_jvp", "vector"): ("PublicRuleTests.test_diagonal_root",),
    ("numerics/native_transfer.py", "custom_jvp", "root"): ("PublicRuleTests.test_transfer_zero",) + _RETAINED_Z,
    ("numerics/native_transfer.py", "custom_jvp", "from_packet"): ("PublicRuleTests.test_transfer_zero",) + _RETAINED_Z,
}
_SUPERSEDED = "registered, then replaced at import by native_equation_jet's rule"
_NOT_REACHED = "not reached by the public calls of this module"
UNCOVERED = {
    ("numerics/native_sparse.py", "jvp", "_solve_p"): _SUPERSEDED,
    ("numerics/native_sparse.py", "jvp", "_batch_solve_p"): _SUPERSEDED,
    ("numerics/native_coupling.py", "custom_jvp", "branch_sqrt"):
        "no independent second-order oracle for the residue-normalized coupling derivative",
    **{("numerics/native_sparse.py", kind, name): _NOT_REACHED for kind, name in (
        ("jvp", "_primal_archive_p"), ("transpose", "_primal_archive_p"),
        ("transpose", "_batch_solve_p"), ("jvp", "_residual_solve_p"),
        ("jvp", "_batch_residual_solve_p"), ("transpose", "_residual_solve_p"),
        ("transpose", "_batch_residual_solve_p"))},
    ("numerics/native_equation_jet.py", "jvp", "ns._batch_solve_p"): _NOT_REACHED,
    ("numerics/native_equation_jet.py", "jvp", "_first_p"): _NOT_REACHED,
    ("numerics/native_equation_jet.py", "jvp", "_batch_first_p"): _NOT_REACHED,
    ("numerics/native_equation_jet.py", "transpose", "_batch_linear_p"): _NOT_REACHED,
    **{("numerics/native_pair.py", kind, name): _NOT_REACHED for kind, name in (
        ("jvp", "_pair_p"), ("jvp", "_batch_pair_p"), ("jvp", "_batch_first_p"),
        ("transpose", "_pair_p"), ("transpose", "_batch_pair_p"),
        ("jvp", "_residual_pair_p"), ("jvp", "_batch_residual_pair_p"),
        ("transpose", "_batch_residual_pair_p"),
        ("transpose", "_linear_p"), ("transpose", "_batch_linear_p"))},
}

_DECORATED = re.compile(r"@jax\.custom_(jvp|vjp)\s*\n\s*def (\w+)")
_ASSIGNED = re.compile(r"(\w+)\s*=\s*jax\.custom_(jvp|vjp)\(")
_REGISTERED = re.compile(r"ad\.primitive_(jvps|transposes)\[([\w.]+)\]")
_LOOP = re.compile(r"for primitive in \(([^)]*)\):")


def registered_rules(package: Path):
    rules = set()
    for path in sorted(package.rglob("*.py")):
        module = path.relative_to(package).as_posix()
        text = path.read_text(encoding="utf-8")
        for kind, name in _DECORATED.findall(text):
            rules.add((module, "custom_" + kind, name))
        for name, kind in _ASSIGNED.findall(text):
            rules.add((module, "custom_" + kind, name))
        loop = ()
        for line in text.splitlines():
            found = _LOOP.search(line)
            if found:
                loop = tuple(item.strip() for item in found.group(1).split(",") if item.strip())
            for table, name in _REGISTERED.findall(line):
                kind = "jvp" if table == "jvps" else "transpose"
                for resolved in (loop if name == "primitive" else (name,)):
                    rules.add((module, kind, resolved))
    return rules


def _exact(value) -> Exact:
    return value if isinstance(value, Exact) else Exact.of(complex(value))


def _observe(label, **values):
    print("AD_RULE " + json.dumps({"rule": label, **values}, sort_keys=True), flush=True)


def assert_taylor(test, label, product, oracle, sensitivity, step, unit):
    """product/oracle: (phi(x + h v), phi(x), J v); sensitivity: their S."""
    step = Fraction(step)
    remainder = _exact(product[0]) - _exact(product[1]) - _exact(product[2]) * Exact(step)
    expected = oracle[0] - oracle[1] - oracle[2] * Exact(step)
    condition = (sensitivity[0] + sensitivity[1] + abs(float(step)) * sensitivity[2]) / expected.magnitude()
    assert_forward(test, f"{label}_taylor_{float(step):.0e}", complex(remainder), complex(expected),
                   condition, unit)


def assert_identity(test, label, jvp_side: Exact, vjp_side: Exact, oracle: Exact, sensitivity, unit):
    """Both sides are the oracle's bilinear form; S is its sensitivity sum."""
    condition = sensitivity / oracle.magnitude()
    assert_forward(test, f"{label}_jvp_inner", complex(jvp_side), complex(oracle), condition, unit)
    assert_forward(test, f"{label}_vjp_inner", complex(vjp_side), complex(oracle), condition, unit)
    difference = (jvp_side - vjp_side).magnitude()
    bound = 2 * 100 * EPS * sensitivity
    _observe(label + "_identity", difference=difference, bound=bound, condition_bound=condition)
    test.assertLessEqual(difference, bound)


class InventoryTests(unittest.TestCase):
    def test_every_registered_rule_is_in_the_inventory(self):
        package = Path(scnsim.__file__).parent
        discovered = registered_rules(package)
        inventory = set(RULES) | set(UNCOVERED)
        self.assertEqual(set(RULES) & set(UNCOVERED), set())
        self.assertEqual(sorted(discovered - inventory), [], "rules missing from the inventory")
        self.assertEqual(sorted(inventory - discovered), [], "inventory names rules that do not exist")
        for rule, tests in RULES.items():
            self.assertTrue(tests, rule)
            for name in tests:
                case, method = name.split(".")
                self.assertTrue(callable(getattr(globals()[case], method, None)), (rule, name))
        _observe("inventory", covered=len(RULES),
                 uncovered=[[*rule, reason] for rule, reason in sorted(UNCOVERED.items())])


# ---------------------------------------------------------------------------
# Rules that are directly callable


def _det(matrix):
    """Laplace expansion over any field with +, -, * (exact or hyper-dual)."""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    total = ZERO
    for column in range(n):
        minor = [row[:column] + row[column + 1:] for row in matrix[1:]]
        term = matrix[0][column] * _det(minor)
        total = total + term if column % 2 == 0 else total - term
    return total


def _seeded_matrix(matrix, first=None, second=None, relative_second=True):
    """Hyper-dual matrix: e1 carries direction ``first`` (a matrix), e2 one entry."""
    result = []
    for i, row in enumerate(matrix):
        values = []
        for j, value in enumerate(row):
            b = first[i][j] if first is not None else ZERO
            c = ZERO
            if second is not None and second == (i, j):
                c = value if relative_second else ONE
            values.append(HyperDual(value, b, c))
        result.append(values)
    return result


def _unit(n, entry):
    return [[ONE if (i, j) == entry else ZERO for j in range(n)] for i in range(n)]


class DirectRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Let SCNSim initialize JAX with the suite's declared runtime before
        # these tests call JAX directly (as a standalone module run would not).
        plan, _, node = _diagonal_root_plan(80e-15, 8e-9)
        run = CircuitRun(plan=plan, storage="memory", precision="float64")
        try:
            run.evaluate(run.original.reduce(ReductionPipeline().retain(node.node)),
                         DiagonalRootSpec(coordinate=node.node, root_hint=6.3 * u.GHz))
        finally:
            run.close()

    def test_determinant(self):
        import jax
        import jax.numpy as jnp
        from scnsim.numerics.determinant_ad import determinant

        matrices = (
            np.array([[2.0 - 0.5j, 0.25 + 1.0j], [-1.5 + 0.75j, 3.0 + 0.125j]]),
            np.array([[4.0 + 1.0j, -1.0 + 0.5j, 0.5], [0.25j, 3.0 - 2.0j, 1.0 + 1.0j],
                      [-0.75 + 0.25j, 1.5, 2.0 - 0.5j]]),
        )
        weight = 0.75 - 1.25j
        for matrix in matrices:
            n = matrix.shape[0]
            direction = np.round(np.cos(np.arange(n * n) + 1.0).reshape(n, n), 3) * (1 - 0.5j)
            exact = [[_exact(value) for value in row] for row in matrix]
            exact_direction = [[_exact(value) for value in row] for row in direction]
            with self.subTest(n=n), jax.enable_x64(True):
                primal, tangent = jax.jvp(determinant, (jnp.asarray(matrix),), (jnp.asarray(direction),))
                _, pullback = jax.vjp(determinant, jnp.asarray(matrix))
                cotangent = np.asarray(pullback(jnp.asarray(weight, dtype=jnp.complex128))[0])
                value = _det(exact)
                derivative = _det(_seeded_matrix(exact, exact_direction)).b
                entries = [(i, j) for i in range(n) for j in range(n)]
                # S(det): each entry perturbed relatively; S(D): entries of M and V.
                value_s = sum(_det(_seeded_matrix(exact, second=e)).c.magnitude() for e in entries)
                derivative_s = sum(_det(_seeded_matrix(exact, exact_direction, second=e)).d.magnitude()
                                   for e in entries)
                derivative_s += sum((exact_direction[i][j] * _det(_seeded_matrix(exact, second=(i, j),
                                     relative_second=False)).c).magnitude() for i, j in entries)
                assert_forward(self, f"det{n}_primal", complex(primal), complex(value),
                               value_s / value.magnitude(), "1")
                assert_forward(self, f"det{n}_jvp", complex(tangent), complex(derivative),
                               derivative_s / derivative.magnitude(), "1")

                bilinear = _exact(weight) * derivative
                bilinear_s = abs(weight) * derivative_s + bilinear.magnitude()
                vjp_side = sum((_exact(cotangent[i, j]) * exact_direction[i][j] for i, j in entries), ZERO)
                assert_identity(self, f"det{n}", _exact(weight) * _exact(tangent), vjp_side,
                                bilinear, bilinear_s, "1")

                for step in (2.0 ** -6, 2.0 ** -9, 2.0 ** -12):
                    moved = matrix + step * direction
                    exact_moved = [[_exact(value) for value in row] for row in moved]
                    moved_s = sum(_det(_seeded_matrix(exact_moved, second=e)).c.magnitude() for e in entries)
                    assert_taylor(self, f"det{n}", (determinant(jnp.asarray(moved)), primal, tangent),
                                  (_det(exact_moved), value, derivative),
                                  (moved_s, value_s, derivative_s), step, "1")

    def test_spd_root(self):
        import jax
        import jax.numpy as jnp
        from scnsim.numerics.matrix_functions import spd_root

        matrix = np.array([[50.0, 12.5], [12.5, 75.0]])
        direction = np.array([[1.0, -0.375], [-0.375, 0.5]])
        weight = np.array([[0.5, 1.25], [1.25, -2.0]])
        with jax.enable_x64(True):
            primal, tangent = jax.jvp(spd_root, (jnp.asarray(matrix),), (jnp.asarray(direction),))
            _, pullback = jax.vjp(spd_root, jnp.asarray(matrix))
            cotangent = np.asarray(pullback(jnp.asarray(weight))[0])
            oracle = _SpdRootOracle(matrix)
            # Functional phi(A) = <W, sqrt(A)>; J v = <W, X(V)>.
            value, value_s = oracle.functional(weight)
            derivative, derivative_s = oracle.directional(direction, weight)
            assert_forward(self, "spd_root_functional", float(np.sum(weight * primal)), float(value),
                           value_s / abs(float(value)), "1")
            jvp_side = sum(Fraction(float(w)) * Fraction(float(x)) for w, x in zip(weight.flat, np.asarray(tangent).flat))
            vjp_side = sum(Fraction(float(g)) * Fraction(float(v)) for g, v in zip(cotangent.flat, direction.flat))
            exact_derivative = Exact(Fraction(derivative))
            assert_identity(self, "spd_root", Exact(jvp_side), Exact(vjp_side), exact_derivative,
                            derivative_s, "1")
            for step in (2.0 ** -6, 2.0 ** -9, 2.0 ** -12):
                moved = matrix + step * direction
                moved_value, moved_s = _SpdRootOracle(moved).functional(weight)
                product_moved = float(np.sum(weight * np.asarray(spd_root(jnp.asarray(moved)))))
                product_value = sum(Fraction(float(w)) * Fraction(float(x)) for w, x in zip(weight.flat, np.asarray(primal).flat))
                assert_taylor(self, "spd_root",
                              (complex(product_moved), complex(float(product_value)), complex(float(jvp_side))),
                              (Exact(Fraction(moved_value)), Exact(Fraction(value)), exact_derivative),
                              (moved_s, value_s, derivative_s), step, "1")

    def _compensated_case(self, batched):
        import jax
        import jax.numpy as jnp
        from scnsim.numerics import native_compensated as nc

        x = (1.0 / 3.0, 2.0 ** -56 / 3.0)
        y = (math.pi, 1.2246467991473532e-16)
        direction = (0.75, 2.0 ** -55, -1.25, 2.0 ** -54)
        weight = 0.625

        def product(xh, xl, yh, yl):
            return nc.project(nc.multiply((xh, xl), (yh, yl)))

        function = jax.vmap(product) if batched else product
        def lift(values):
            return tuple(jnp.asarray([value, value]) if batched else jnp.asarray(value) for value in values)
        def scalar(value):
            return float(np.asarray(value).reshape(-1)[0])

        with jax.enable_x64(True):
            primal, tangent = jax.jvp(function, lift(x + y), lift(direction))
            _, pullback = jax.vjp(function, *lift(x + y))
            cotangents = pullback(jnp.asarray([weight, weight]) if batched else jnp.asarray(weight))
            fx, fy = Fraction(x[0]) + Fraction(x[1]), Fraction(y[0]) + Fraction(y[1])
            dx = Fraction(direction[0]) + Fraction(direction[1])
            dy = Fraction(direction[2]) + Fraction(direction[3])
            value = fx * fy
            derivative = dx * fy + fx * dy
            value_s = float(sum(abs(Fraction(v)) for v in x) * abs(fy) + abs(fx) * sum(abs(Fraction(v)) for v in y))
            derivative_s = float(sum(abs(Fraction(v)) for v in x) * abs(dy) + sum(abs(Fraction(v)) for v in y) * abs(dx)
                                 + abs(fy) * sum(abs(Fraction(v)) for v in direction[:2])
                                 + abs(fx) * sum(abs(Fraction(v)) for v in direction[2:]))
            label = "compensated_multiply" + ("_batched" if batched else "")
            assert_forward(self, label + "_primal", scalar(primal), float(value), value_s / float(abs(value)), "1")
            assert_forward(self, label + "_jvp", scalar(tangent), float(derivative),
                           derivative_s / float(abs(derivative)), "1")
            jvp_side = Fraction(weight) * Fraction(scalar(tangent))
            vjp_side = sum(Fraction(scalar(c)) * Fraction(d) for c, d in zip(cotangents, direction))
            bilinear = Fraction(weight) * derivative
            assert_identity(self, label, Exact(jvp_side), Exact(vjp_side), Exact(bilinear),
                            weight * derivative_s + float(abs(bilinear)), "1")
            for step in (2.0 ** -6, 2.0 ** -9, 2.0 ** -12):
                moved = tuple(v + step * d for v, d in zip(x + y, direction))
                moved_value = (Fraction(moved[0]) + Fraction(moved[1])) * (Fraction(moved[2]) + Fraction(moved[3]))
                moved_s = float(sum(abs(Fraction(v)) for v in moved[:2]) * abs(Fraction(moved[2]) + Fraction(moved[3]))
                                + abs(Fraction(moved[0]) + Fraction(moved[1])) * sum(abs(Fraction(v)) for v in moved[2:]))
                assert_taylor(self, label, (scalar(function(*lift(moved))), scalar(primal), scalar(tangent)),
                              (Exact(moved_value), Exact(value), Exact(derivative)),
                              (moved_s, value_s, derivative_s), step, "1")

    def test_compensated_pair(self):
        self._compensated_case(batched=False)

    def test_compensated_pair_batched(self):
        self._compensated_case(batched=True)


class _SpdRootOracle:
    """sqrt of a symmetric positive definite 2x2 matrix in Decimal arithmetic.

    sqrt(A) = (A + s I) / t with s = sqrt(det A), t = sqrt(tr A + 2 s). The
    directional derivative X solves R X + X R = V. Primitives are the three
    symmetric entries a, b, c (and those of V and W for the bilinear form).
    """

    def __init__(self, matrix):
        with localcontext() as context:
            context.prec = PRECISION
            self.a = [[Decimal.from_float(float(value)) for value in row] for row in matrix]
            s = (self.a[0][0] * self.a[1][1] - self.a[0][1] * self.a[1][0]).sqrt()
            t = (self.a[0][0] + self.a[1][1] + 2 * s).sqrt()
            self.root = [[(self.a[i][j] + (s if i == j else 0)) / t for j in range(2)] for i in range(2)]

    def _sylvester(self, right):
        """Solve R X + X R = right (2x2) in Decimal by Cramer's rule on 4x4."""
        r = self.root
        rows = []
        for i in range(2):
            for j in range(2):
                row = [Decimal(0)] * 4
                for k in range(2):
                    row[2 * k + j] += r[i][k]
                    row[2 * i + k] += r[k][j]
                rows.append(row)
        vector = [right[i][j] for i in range(2) for j in range(2)]
        work = [rows[k] + [vector[k]] for k in range(4)]
        for column in range(4):
            pivot = max(range(column, 4), key=lambda k: abs(work[k][column]))
            work[column], work[pivot] = work[pivot], work[column]
            for k in range(4):
                if k != column:
                    factor = work[k][column] / work[column][column]
                    work[k] = [x - factor * y for x, y in zip(work[k], work[column])]
        solution = [work[k][4] / work[k][k] for k in range(4)]
        return [[solution[0], solution[1]], [solution[2], solution[3]]]

    @staticmethod
    def _symmetric_units(matrix):
        return ([[matrix[0][0], 0], [0, 0]], [[0, matrix[0][1]], [matrix[1][0], 0]],
                [[0, 0], [0, matrix[1][1]]])

    def functional(self, weight):
        with localcontext() as context:
            context.prec = PRECISION
            w = [[Decimal.from_float(float(value)) for value in row] for row in weight]
            value = sum(w[i][j] * self.root[i][j] for i in range(2) for j in range(2))
            sensitivity = 0
            for unit in self._symmetric_units(self.a):
                change = self._sylvester([[Decimal(x) for x in row] for row in unit])
                sensitivity += abs(sum(w[i][j] * change[i][j] for i in range(2) for j in range(2)))
            return value, float(sensitivity)

    def directional(self, direction, weight):
        with localcontext() as context:
            context.prec = PRECISION
            v = [[Decimal.from_float(float(value)) for value in row] for row in direction]
            w = [[Decimal.from_float(float(value)) for value in row] for row in weight]
            x = self._sylvester(v)
            def inner(matrix):
                return sum(w[i][j] * matrix[i][j] for i in range(2) for j in range(2))
            value = inner(x)
            sensitivity = 0
            for unit in self._symmetric_units(self.a):
                change_root = self._sylvester([[Decimal(item) for item in row] for row in unit])
                product = [[sum(change_root[i][k] * x[k][j] + x[i][k] * change_root[k][j] for k in range(2))
                            for j in range(2)] for i in range(2)]
                sensitivity += abs(inner(self._sylvester([[-item for item in row] for row in product])))
            for unit in self._symmetric_units(v):
                sensitivity += abs(inner(self._sylvester([[Decimal(item) for item in row] for row in unit])))
            sensitivity += abs(value)  # The weight W perturbed relatively as a whole.
            return value, float(sensitivity)


# ---------------------------------------------------------------------------
# Rules reached through public sensitivity calls


TWO_PORT = (("C", "a", None, 120e-15), ("L", "a", "m", 3.3e-9), ("C", "m", None, 250e-15),
            ("R", "m", None, 2.0e3), ("C", "m", "b", 15e-15), ("L", "b", None, 6.8e-9),
            ("C", "b", None, 90e-15), ("R", "a", "b", 7.5e3))
PARAMETERS = {1: ("inductance", u.H), 2: ("capacitance", u.F)}
FREQUENCY = 5.3e9
QUANTITIES = (("re_ab", 0, 1, "real"), ("im_aa", 0, 0, "imag"))
DIRECTION = {1: 1.0, 2: -0.5}           # Relative direction per parameter.
WEIGHTS = {"re_ab": 2.0, "im_aa": -0.25}  # 1/S per quantity.
STEPS = (1e-2, 1e-3, 1e-4)


def _two_port_plan():
    definitions = ParameterDefinitions(id="ad_parameters")
    parameters = {index: definitions.parameter(id=name, baseline=TWO_PORT[index][3] * unit,
                                               spec=ParameterSpec(unit=unit))
                  for index, (name, unit) in PARAMETERS.items()}
    plan = CircuitPlan(id="ad_two_port")
    buses = {name: plan.bus(id=name) for name in "amb"}
    for index, (kind, a, b, value) in enumerate(TWO_PORT):
        unit = {"C": u.F, "L": u.H, "R": u.ohm}[kind]
        quantity = parameters.get(index, value * unit)
        element = {"C": lambda q: components.capacitor(id=f"e{index}", capacitance=q),
                   "L": lambda q: components.inductor(id=f"e{index}", inductance=q),
                   "R": lambda q: components.resistor(id=f"e{index}", resistance=q)}[kind](quantity)
        plan.series(id=f"s{index}", start=buses[a], elements=(plan.add(element),),
                    end=buses[b] if b is not None else plan.ground)
    for name in "ab":
        plan.add_port(id=f"p{name}", at=buses[name], role="terminated", reference_impedance=50.0 * u.ohm)
    return plan, parameters


def _part(value: Exact, part):
    return Exact(value.re) if part == "real" else Exact(value.im)


def _two_port_functional(values):
    """phi = sum_k w_k f_k and its sensitivity S at the given branch values."""
    branches = tuple(Branch(kind, a, b, value) for (kind, a, b, _), value in zip(TWO_PORT, values))
    response = Network(branches, ("a", "b")).response(FREQUENCY)
    value, changes = ZERO, None
    for label, row, column, part in QUANTITIES:
        weight = Exact.of(WEIGHTS[label])
        value = value + weight * _part(response.matrix("Y")[row][column], part)
        local = [weight * _part(change, part) for change in response.first_changes("Y", row, column)]
        changes = local if changes is None else [x + y for x, y in zip(changes, local)]
    return response, value, sum(change.magnitude() for change in changes)


def _diagonal_root_plan(capacitance, inductance):
    definitions = ParameterDefinitions(id="root_parameters")
    parameter = definitions.parameter(id="capacitance", baseline=capacitance * u.F,
                                      spec=ParameterSpec(unit=u.F))
    plan = CircuitPlan(id="ad_diagonal_root")
    node = plan.bus(id="a")
    c = plan.add(components.capacitor(id="c", capacitance=parameter))
    l = plan.add(components.inductor(id="l", inductance=inductance * u.H))
    plan.parallel(id="lc", start=node, branches=((c,), (l,)), end=plan.ground)
    return plan, parameter, node


def _matched_zero_plan(capacitance, inductance):
    definitions = ParameterDefinitions(id="zero_parameters")
    parameter = definitions.parameter(id="capacitance", baseline=capacitance * u.F,
                                      spec=ParameterSpec(unit=u.F))
    plan = CircuitPlan(id="ad_matched_zero")
    node = plan.bus(id="p")
    r = plan.add(components.resistor(id="r", resistance=50.0 * u.ohm))
    c = plan.add(components.capacitor(id="c", capacitance=parameter))
    l = plan.add(components.inductor(id="l", inductance=inductance * u.H))
    plan.parallel(id="rlc", start=node, branches=((r,), (c,), (l,)), end=plan.ground)
    plan.add_port(id="p", at=node, role="terminated", reference_impedance=50.0 * u.ohm)
    return plan, parameter


def _coupled_pair_plan(capacitance, inductance):
    """Two LC modes coupled by C_M parallel L_M; C_M is the parameter."""
    definitions = ParameterDefinitions(id="coupled_parameters")
    parameter = definitions.parameter(id="coupling_capacitance", baseline=capacitance * u.F,
                                      spec=ParameterSpec(unit=u.F))
    plan = CircuitPlan(id="ad_retained_z_zero")
    a, b = plan.bus(id="a"), plan.bus(id="b")
    for name, node, c_value, l_value in (("a", a, 80e-15, 8e-9), ("b", b, 95e-15, 7e-9)):
        c = plan.add(components.capacitor(id=f"c_{name}", capacitance=c_value * u.F))
        l = plan.add(components.inductor(id=f"l_{name}", inductance=l_value * u.H))
        plan.parallel(id=f"mode_{name}", start=node, branches=((c,), (l,)), end=plan.ground)
    cm = plan.add(components.capacitor(id="c_m", capacitance=parameter))
    plan.series(id="coupling_c", start=a, elements=(cm,), end=b)
    # L_M is split exactly in two halves through an internal node, so the
    # retained View eliminates one node and Y_ab is unchanged.
    middle = plan.bus(id="m")
    for name, start, end in (("l_m1", a, middle), ("l_m2", middle, b)):
        half = plan.add(components.inductor(id=name, inductance=inductance / 2 * u.H))
        plan.series(id=f"coupling_{name}", start=start, elements=(half,), end=end)
    return plan, parameter, (a, b)


def _lc_frequency(capacitance, inductance):
    """Exact-to-60-digits f = 1 / (2 pi sqrt(L C)) at binary64 inputs."""
    with localcontext() as context:
        context.prec = PRECISION
        value = 1 / (2 * PI * (Decimal.from_float(inductance) * Decimal.from_float(capacitance)).sqrt())
        return Exact(Fraction(value))


class PublicRuleTests(unittest.TestCase):
    def test_response_element(self):
        plan, parameters = _two_port_plan()
        wrt = tuple(parameters[index] for index in sorted(parameters))
        run = CircuitRun(plan=plan, storage="memory", precision="float64")
        try:
            specs = {label: ResponseElementSpec(family="Y", input_coordinate=f"p{'ab'[row]}",
                                                output_coordinate=f"p{'ab'[column]}", frequency=FREQUENCY * u.Hz)
                     for label, row, column, _ in QUANTITIES}
            quantities = {label: getattr(spec, part).on(run.original)
                          for (label, _, _, part), spec in zip(QUANTITIES, specs.values())}
            direction = {parameters[i]: DIRECTION[i] * TWO_PORT[i][3] * PARAMETERS[i][1] for i in parameters}
            jvp = run.sensitivity(SensitivitySpec(quantities=quantities, wrt=wrt, derivative=JVP(direction=direction)))
            vjp = run.sensitivity(SensitivitySpec(quantities=quantities, wrt=wrt, derivative=VJP(
                weights={OutputCoordinate(quantity_label=label): WEIGHTS[label] / u.S for label in WEIGHTS})))
            blocks = {block.quantity_id: block for block in jvp.jvp.blocks}
            self.assertTrue(all(block.status == "complete" for block in blocks.values()))
            (block,) = vjp.vjp.blocks
            self.assertEqual(block.status, "complete")

            values = [value for _, _, _, value in TWO_PORT]
            response, phi, phi_s = _two_port_functional(values)
            # Exact J v per quantity, and the bilinear form with its sensitivity.
            bilinear, changes, jvp_side = ZERO, None, ZERO
            for label, row, column, part in QUANTITIES:
                weight = Exact.of(WEIGHTS[label])
                tangent, local_changes = ZERO, None
                for index, scale in DIRECTION.items():
                    tangent = tangent + Exact.of(scale) * _part(response.relative_derivative("Y", row, column, index), part)
                    second = [Exact.of(scale) * _part(term, part)
                              for term in response.derivative_changes("Y", row, column, index)]
                    local_changes = second if local_changes is None else [x + y for x, y in zip(local_changes, second)]
                actual = float(blocks[label].value.to(u.S).magnitude)
                tangent_s = sum(change.magnitude() for change in local_changes)
                tangent_s += sum(abs(scale) * _part(response.relative_derivative("Y", row, column, index), part).magnitude()
                                 for index, scale in DIRECTION.items())
                assert_forward(self, f"response_jvp_{label}", actual, complex(tangent),
                               tangent_s / tangent.magnitude(), "S", frequency_perturbed=True)
                bilinear = bilinear + weight * tangent
                jvp_side = jvp_side + weight * _exact(actual)
                weighted = [weight * change for change in local_changes]
                changes = weighted if changes is None else [x + y for x, y in zip(changes, weighted)]
            bilinear_s = sum(change.magnitude() for change in changes)
            bilinear_s += sum(abs(DIRECTION[index]) * sum(
                (Exact.of(WEIGHTS[label]) * _part(response.relative_derivative("Y", row, column, index), part)
                 for label, row, column, part in QUANTITIES), ZERO).magnitude() for index in DIRECTION)
            bilinear_s += sum((Exact.of(WEIGHTS[label]) * sum(
                (Exact.of(DIRECTION[index]) * _part(response.relative_derivative("Y", row, column, index), part)
                 for index in DIRECTION), ZERO)).magnitude() for label, row, column, part in QUANTITIES)
            vjp_side = sum((_exact(gradient) * _exact(DIRECTION[i] * TWO_PORT[i][3])
                            for gradient, i in zip(block.values, sorted(parameters))), ZERO)
            assert_identity(self, "response_element", jvp_side, vjp_side, bilinear, bilinear_s, "1")

            primal = sum((Exact.of(WEIGHTS[label]) * _part(_exact(
                run.evaluate(run.original, specs[label]).value.to(u.S).magnitude), part)
                for label, _, _, part in QUANTITIES), ZERO)
            for step in STEPS:
                moved = list(values)
                for index, scale in DIRECTION.items():
                    moved[index] = values[index] * (1 + step * scale)
                setting = ParameterSet({parameters[i]: moved[i] * PARAMETERS[i][1] for i in parameters})
                product_moved = sum((Exact.of(WEIGHTS[label]) * _part(_exact(
                    run.evaluate(run.original, specs[label], parameters=setting).value.to(u.S).magnitude), part)
                    for label, _, _, part in QUANTITIES), ZERO)
                _, moved_phi, moved_s = _two_port_functional(moved)
                # Both remainders use the moved binary64 values and the same h,
                # so the rounding of the moved parameters enters both alike.
                assert_taylor(self, "response_element", (product_moved, primal, jvp_side),
                              (moved_phi, phi, bilinear), (moved_s, phi_s, bilinear_s), step, "1")
        finally:
            run.close()

    def _scalar_root_case(self, label, plan, parameter, quantity, spec, capacitance, inductance,
                          primal_condition, derivative_condition, view):
        weight = -1.5  # 1/Hz
        run = CircuitRun(plan=plan, storage="memory", precision="float64")
        try:
            target = view(run)
            quantities = {label: quantity(target)}
            jvp = run.sensitivity(SensitivitySpec(quantities=quantities, wrt=(parameter,),
                                                  derivative=JVP(direction={parameter: capacitance * u.F})))
            vjp = run.sensitivity(SensitivitySpec(quantities=quantities, wrt=(parameter,), derivative=VJP(
                weights={OutputCoordinate(quantity_label=label): weight / u.Hz})))
            (jvp_block,), (vjp_block,) = jvp.jvp.blocks, vjp.vjp.blocks
            self.assertEqual((jvp_block.status, vjp_block.status), ("complete", "complete"))
            frequency = _lc_frequency(capacitance, inductance)
            derivative = frequency * Exact(Fraction(-1, 2))  # C df/dC = -f/2
            tangent = float(jvp_block.value.to(u.Hz).magnitude)
            assert_forward(self, label + "_jvp", tangent, complex(derivative), derivative_condition, "Hz")
            # The VJP is w df/dC: the weight and C each add 1 to kappa.
            assert_forward(self, label + "_vjp", float(vjp_block.values[0]),
                           weight * complex(derivative) / capacitance, derivative_condition + 2, "1/F")
            bilinear = Exact.of(weight) * derivative
            # S of w * (C df/dC) * 1: derivative condition plus w and the direction.
            sensitivity = bilinear.magnitude() * (derivative_condition + 2)
            assert_identity(self, label, Exact.of(weight) * _exact(tangent),
                            _exact(vjp_block.values[0]) * _exact(capacitance), bilinear, sensitivity, "1")
            primal = float(spec(run, target, None))
            for step in STEPS:
                moved = capacitance * (1 + step)
                product_moved = float(spec(run, target, ParameterSet({parameter: moved * u.F})))
                h = (Fraction(moved) - Fraction(capacitance)) / Fraction(capacitance)
                # Use the exact displacement as the step for both remainders.
                moved_frequency = _lc_frequency(moved, inductance)
                assert_taylor(self, label, (product_moved, primal, tangent),
                              (moved_frequency, frequency, derivative),
                              (moved_frequency.magnitude() * primal_condition,
                               frequency.magnitude() * primal_condition,
                               derivative.magnitude() * derivative_condition), h, "Hz")
        finally:
            run.close()

    def test_diagonal_root(self):
        capacitance, inductance = 80e-15, 8e-9
        plan, parameter, node = _diagonal_root_plan(capacitance, inductance)
        spec = DiagonalRootSpec(coordinate=node.node, root_hint=6.3 * u.GHz)
        def view(run):
            return run.original.reduce(ReductionPipeline().retain(node.node))
        def evaluate(run, target, parameters):
            result = run.evaluate(target, spec, **({} if parameters is None else {"parameters": parameters}))
            return result.frequency.to(u.Hz).magnitude
        # f = 1/(2 pi sqrt(LC)): C and L each contribute 1/2 to kappa(f);
        # C df/dC = -f/2 has the same relative sensitivities, so kappa = 1.
        self._scalar_root_case("diagonal_root", plan, parameter, lambda target: spec.frequency.on(target),
                               evaluate, capacitance, inductance, 1.0, 1.0, view)

    def test_transfer_zero(self):
        capacitance, inductance = 80e-15, 8e-9
        plan, parameter = _matched_zero_plan(capacitance, inductance)
        spec = TransferZeroSpec(anchor=6.3 * u.GHz, family="S", input_coordinate="p", output_coordinate="p")
        def evaluate(run, target, parameters):
            result = run.evaluate(target, spec, **({} if parameters is None else {"parameters": parameters}))
            return result.frequency.to(u.Hz).magnitude
        # With g = 1/R - 1/Z0 the zero solves C w^2 + i g w - 1/L = 0. At
        # g = 0: kappa(w) is matched_zero_oracle's bound, and for
        # J = C dw/dC = -w/2 the relative sensitivities are C: w/4, L: w/4,
        # R: 1/(2 C R), Z0: 1/(2 C Z0), so kappa(J) = 1 + 1/(w C R) + 1/(w C Z0).
        omega, primal_condition = matched_zero_oracle()
        derivative_condition = 1 + 2 / (omega * capacitance * 50.0)
        self._scalar_root_case("transfer_zero", plan, parameter, lambda target: spec.frequency.on(target),
                               evaluate, capacitance, inductance, primal_condition, derivative_condition,
                               lambda run: run.original)

    def test_retained_z_transfer_zero(self):
        # Z_ab = -Y_ab / det Y vanishes where Y_ab = i (omega C_M - 1/(omega L_M))
        # does, at f = 1/(2 pi sqrt(L_M C_M)), independently of the two modes.
        # kappa as for the diagonal root: C_M and L_M each contribute 1/2.
        capacitance, inductance = 2e-15, 500e-9
        plan, parameter, nodes = _coupled_pair_plan(capacitance, inductance)
        spec = TransferZeroSpec(anchor=5.0 * u.GHz, family="Z", input_coordinate=nodes[0].node,
                                output_coordinate=nodes[1].node)
        def view(run):
            return run.original.reduce(ReductionPipeline().retain(nodes[0].node, nodes[1].node))
        def evaluate(run, target, parameters):
            result = run.evaluate(target, spec, **({} if parameters is None else {"parameters": parameters}))
            return result.frequency.to(u.Hz).magnitude
        self._scalar_root_case("retained_z_zero", plan, parameter, lambda target: spec.frequency.on(target),
                               evaluate, capacitance, inductance, 1.0, 1.0, view)


if __name__ == "__main__":
    unittest.main()
