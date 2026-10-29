"""H1/H5 A=I multiscale diagnostics; the aggregate gate remains authoritative."""
from __future__ import annotations
import unittest
import numpy as np


class SolveDiagnosticsTests(unittest.TestCase):
    def test_approximate_identity_solution_distinguishes_the_two_metrics(self):
        import jax
        import jax.numpy as jnp
        from scnsim.numerics import jax_core
        with jax.enable_x64(True):
            a = jnp.eye(2, dtype=jnp.float64)
            b = jnp.asarray([1.0, 1e-20], dtype=jnp.float64)
            x = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
            observed = jax_core.residual_with_diagnostics(a, x, b)
            self.assertEqual(float(observed.outcome), 5e-21)
            np.testing.assert_array_equal(observed.diagnostics.per_rhs, [1.0])
            self.assertTrue(observed.diagnostics.performed)
            self.assertIsNone(observed.diagnostics.unavailable_reason)
            # H5's diagnostic does not replace the original aggregate/tau gate.
            self.assertLessEqual(float(observed.outcome), float(jax_core.tau(2, jnp.float64)))
            self.assertGreater(float(observed.diagnostics.per_rhs[0]), float(jax_core.tau(2, jnp.float64)))

    def test_rhs_columns_keep_their_independent_componentwise_diagnostics(self):
        from scnsim.numerics.solve_diagnostics import componentwise_per_rhs
        a = np.eye(2)
        b = np.column_stack(([1.0, 1e-20], [1.0, 1e-20]))
        x = np.column_stack(([1.0, 0.0], [1.0, 1e-20]))
        error = a @ x - b
        scale = abs(a) @ abs(x) + abs(b)
        observed = componentwise_per_rhs(error, scale, xp=np)
        np.testing.assert_array_equal(observed.per_rhs, [1.0, 0.0])

    def test_dense_diagnostic_solve_returns_the_original_actual_outcome(self):
        import jax
        import jax.numpy as jnp
        from scnsim.numerics import jax_core
        with jax.enable_x64(True):
            a = jnp.eye(2, dtype=jnp.float64)
            b = jnp.asarray([1.0, 1e-20], dtype=jnp.float64)
            original = jax_core.solve(a, b, 2, 3, jnp.int32(0))
            observed = jax_core.solve_with_diagnostics(a, b, 2, 3, jnp.int32(0))
            for actual, expected in zip(observed.outcome, original, strict=True):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(int(observed.outcome[1]), 0)
            np.testing.assert_array_equal(observed.diagnostics.per_rhs, [0.0])

    def test_sparse_diagnostic_solve_keeps_four_scalar_aux_and_real_solution(self):
        from scnsim.compilation.program import SparsePattern
        from scnsim.numerics.sparse_solve import _host_solve, host_solve_with_diagnostics
        pattern, _ = SparsePattern.from_coo((2, 2), np.array([0, 1]), np.array([0, 1]))
        a = np.ones(2)
        b = np.array([1.0, 1e-20])
        original = _host_solve(pattern, a, b, np.int32(0), stage=3, mode='N', ordering=None)
        observed = host_solve_with_diagnostics(pattern, a, b, np.int32(0), stage=3)
        np.testing.assert_array_equal(observed.outcome.value, original.value)
        self.assertEqual(observed.outcome.aux, original.aux)
        self.assertEqual(len(observed.outcome.aux), 4)
        self.assertEqual(int(observed.outcome.aux.status), 0)
        np.testing.assert_array_equal(observed.diagnostics.per_rhs, [0.0])

    def test_compensated_identity_certificate_uses_the_same_residual_frame(self):
        from scipy.sparse import eye, csr_matrix
        from scnsim.numerics import compensated as cp
        from scnsim.numerics.sparse_solve import _host_pair_product
        a = eye(2, format='csr', dtype=np.float64)
        low = csr_matrix((2, 2), dtype=np.float64)
        x = cp.from_value(np.array([1.0, 0.0]), xp=np)
        b = cp.from_value(np.array([1.0, 1e-20]), xp=np)
        observed = cp.solve_certificate_with_diagnostics(
            a, low, x, b, real_dtype=np.float64, execute=_host_pair_product)
        original = cp.solve_certificate(a, low, x, b, real_dtype=np.float64, execute=_host_pair_product)
        self.assertEqual(observed.outcome, original)
        self.assertEqual(float(observed.outcome[0]), 5e-21)
        np.testing.assert_array_equal(observed.diagnostics.per_rhs, [1.0])

    def test_compensated_diagnostic_solve_preserves_actual_fixed_correction(self):
        from scnsim.compilation.program import SparsePattern
        from scnsim.numerics import compensated as cp
        from scnsim.numerics.sparse_solve import _host_pair_equation, host_pair_solve_with_diagnostics
        pattern, _ = SparsePattern.from_coo((2, 2), np.array([0, 1]), np.array([0, 1]))
        a = np.ones(2), np.zeros(2)
        b = cp.from_value(np.array([1.0, 1e-20]), xp=np)
        original = _host_pair_equation(pattern, a, b, np.int32(0), stage=3, ordering=None)
        observed = host_pair_solve_with_diagnostics(pattern, a, b, np.int32(0), stage=3)
        for actual, expected in zip(observed.outcome.value, original.value, strict=True):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(observed.outcome.aux, original.aux)
        self.assertEqual(len(observed.outcome.aux), 4)
        self.assertEqual(int(observed.outcome.aux.status), 0)
        np.testing.assert_array_equal(observed.diagnostics.per_rhs, [0.0])


if __name__ == '__main__':
    unittest.main()
