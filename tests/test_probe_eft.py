"""Error-free transformations of the installed platform kernel, normal and subnormal.

SCNSim's compensated certificates rest on two error-free transformations:
TwoSum (s + e == a + b exactly) and TwoProduct (p + e == a * b exactly). The
compiled ``_compensated_cpu`` kernel shipped in each platform wheel and the
JAX implementation in ``numerics.compensated`` are probed with exact Fraction
checks, so no tolerance is involved. The installed-wheel CI jobs run this
module on Linux, macOS and Windows.

Normal range: operands and results are normal numbers, so both
transformations must be exact, and the high part must equal the correctly
rounded IEEE result (NumPy's own addition or multiplication).

Subnormal range: subnormal operands or results. With gradual underflow TwoSum
stays exact, and its error term vanishes whenever the exact sum is
representable (always the case for a subnormal sum). For TwoProduct only
cases whose exact product is representable are used (checked on the oracle
side): the high part must equal that product and the error term must vanish.
The JAX implementation is probed in the normal range only; its subnormal
behaviour is reported on Issue 47.
"""
from __future__ import annotations

from fractions import Fraction
import unittest

import numpy as np

from scnsim import CircuitPlan, CircuitRun, DirectSolveSpec, components, units as u
from scnsim._native._compensated_cpu import csr_pair_product


DTYPES = (np.float32, np.float64)


def _csr(values, dtype):
    data = np.asarray(values, dtype=dtype)
    return data, np.arange(len(values), dtype=np.int64), np.array([0, len(values)], dtype=np.int64)


def _kernel(coefficients, rhs, dtype):
    """One CSR row: sum_k coefficients[k] * rhs[k] as a (high, low) pair."""
    high, indices, offsets = _csr(coefficients, dtype)
    low = np.zeros_like(high)
    x_high = np.asarray(rhs, dtype=dtype)
    out_high, out_low = np.zeros(1, dtype=dtype), np.zeros(1, dtype=dtype)
    csr_pair_product(high, indices, offsets, low, indices.copy(), offsets.copy(),
                     x_high, np.zeros_like(x_high), out_high, out_low)
    return out_high[0], out_low[0]


def native_two_sum(a, b, dtype):
    return _kernel([1, 1], [a, b], dtype)


def native_two_product(a, b, dtype):
    return _kernel([a], [b], dtype)


def _normal_cases(dtype, count=64):
    """Deterministic operands spread over the exponent range of dtype."""
    info = np.finfo(dtype)
    generator = np.random.default_rng(47)
    # Exponents stay within half the normal range, so products stay normal.
    exponent_limit = (info.maxexp - 2) // 2
    def draw():
        mantissa = generator.uniform(1.0, 2.0, count) * generator.choice((-1.0, 1.0), count)
        return np.ldexp(mantissa, generator.integers(-exponent_limit + 1, exponent_limit, count)).astype(dtype)
    return list(zip(draw(), draw()))


def _subnormal_sum_cases(dtype):
    info = np.finfo(dtype)
    tiny, smallest = info.tiny, info.smallest_subnormal
    values = [smallest * 7, smallest * 3, tiny / 3, tiny * 0.75, -tiny / 5, tiny * 3, smallest]
    return [(dtype(a), dtype(b)) for a in values for b in values]


def _subnormal_product_cases(dtype):
    info = np.finfo(dtype)
    smallest, mant = info.smallest_subnormal, info.nmant
    cases = [(smallest * 6, 0.5), (smallest * 1000, 3.0), (smallest * 12, -0.25),
             (np.ldexp(1.0, info.minexp - 2), 3.0),
             # Two normal operands whose exact product is subnormal.
             (np.ldexp(3.0, info.minexp // 2), np.ldexp(5.0, info.minexp // 2 - mant // 2))]
    return [(dtype(a), dtype(b)) for a, b in cases]


class NativeKernelProbeTests(unittest.TestCase):
    def _assert_exact_sum(self, a, b, dtype):
        high, low = native_two_sum(a, b, dtype)
        exact = Fraction(float(a)) + Fraction(float(b))
        self.assertEqual(Fraction(float(high)) + Fraction(float(low)), exact, (a, b))
        self.assertEqual(high, dtype(a) + dtype(b))
        if Fraction(float(dtype(exact))) == exact:  # A representable sum leaves no error.
            self.assertEqual(low, 0, (a, b))

    def test_normal_range(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype.__name__):
                for a, b in _normal_cases(dtype):
                    self._assert_exact_sum(a, b, dtype)
                    high, low = native_two_product(a, b, dtype)
                    self.assertEqual(Fraction(float(high)) + Fraction(float(low)),
                                     Fraction(float(a)) * Fraction(float(b)), (a, b))
                    self.assertEqual(high, dtype(a) * dtype(b))

    def test_subnormal_range(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype.__name__):
                for a, b in _subnormal_sum_cases(dtype):
                    self._assert_exact_sum(a, b, dtype)
                for a, b in _subnormal_product_cases(dtype):
                    exact = Fraction(float(a)) * Fraction(float(b))
                    # Oracle precondition: the exact product is a subnormal of dtype.
                    self.assertEqual(Fraction(float(dtype(exact))), exact)
                    self.assertLess(abs(exact), Fraction(float(np.finfo(dtype).tiny)))
                    high, low = native_two_product(a, b, dtype)
                    self.assertEqual(Fraction(float(high)), exact, (a, b))
                    self.assertEqual(low, 0)


class JaxCompensatedProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Let SCNSim initialize JAX with the suite's declared runtime first.
        plan = CircuitPlan(id="probe_eft_runtime")
        node = plan.bus(id="a")
        resistor = plan.add(components.resistor(id="r", resistance=50.0 * u.ohm))
        plan.series(id="load", start=node, elements=(resistor,), end=plan.ground)
        plan.add_port(id="a", at=node, role="terminated", reference_impedance=50.0 * u.ohm)
        run = CircuitRun(plan=plan, storage="memory", precision="float64")
        try:
            run.solve(run.original, DirectSolveSpec(frequencies=[1.0] * u.GHz))
        finally:
            run.close()

    def test_normal_range(self):
        import jax
        import jax.numpy as jnp
        from scnsim.numerics import compensated as cp

        kw = dict(xp=jnp, barrier=jax.lax.optimization_barrier)
        two_sum = jax.jit(lambda a, b: cp.two_sum(a, b, **kw))
        product = jax.jit(lambda a, b: cp.multiply(cp.from_value(a, xp=jnp), cp.from_value(b, xp=jnp), **kw))
        for dtype in DTYPES:
            with self.subTest(dtype=dtype.__name__), jax.enable_x64(dtype == np.float64):
                for a, b in _normal_cases(dtype):
                    high, low = (np.asarray(value) for value in two_sum(jnp.asarray(a), jnp.asarray(b)))
                    self.assertEqual(Fraction(float(high)) + Fraction(float(low)),
                                     Fraction(float(a)) + Fraction(float(b)), (a, b))
                    self.assertEqual(high, dtype(a) + dtype(b))
                    high, low = (np.asarray(value) for value in product(jnp.asarray(a), jnp.asarray(b)))
                    self.assertEqual(Fraction(float(high)) + Fraction(float(low)),
                                     Fraction(float(a)) * Fraction(float(b)), (a, b))
                    self.assertEqual(high, dtype(a) * dtype(b))


if __name__ == "__main__":
    unittest.main()
