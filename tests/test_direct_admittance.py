"""Compatible coordinate Y/Z and matched TransferZero regressions."""
import math
import unittest
import numpy as np
from scnsim import CircuitRun, ReductionPipeline, ResponseElementSpec, TransferZeroSpec, units as u
from tests.analytic_conditions import assert_forward, matched_zero_oracle, response_oracle
from tests.test_direct_analysis import _advanced_direct_plan, _matched_zero_plan


class DirectAdmittanceTests(unittest.TestCase):
    def test_retained_nonport_scalar_y_and_z(self):
        run = CircuitRun(plan=_advanced_direct_plan(), storage="memory", backend="jax", precision="float64")
        try:
            view = run.original.reduce(ReductionPipeline().retain("a", "b"))
            omega = 2*math.pi*5.5e9
            expected_y, condition_y = response_oracle(omega)
            c = np.array([[82e-15, -2e-15], [-2e-15, 97e-15]])
            k = np.array([[1/8e-9+1/500e-9, -1/500e-9], [-1/500e-9, 1/7e-9+1/500e-9]])
            y = 1j*(k/omega-omega*c)
            det = y[0, 0]*y[1, 1]-y[0, 1]**2
            local_bounds = 2*(np.abs(k)/omega+omega*np.abs(c))/np.abs(y)
            det_condition = (abs(y[0, 0]*y[1, 1])*(local_bounds[0, 0]+local_bounds[1, 1])
                             +2*abs(y[0, 1])**2*local_bounds[0, 1])/abs(det)
            for family, expected, condition, unit in (
                    ("Y", expected_y, condition_y, u.S),
                    ("Z", -y[0, 1]/det, condition_y+det_condition, u.ohm)):
                result = run.evaluate(view, ResponseElementSpec(family=family,
                    input_coordinate="a", output_coordinate="b", frequency=5.5*u.GHz))
                assert_forward(self, "retained_"+family, complex(result.value.to(unit).magnitude),
                               expected, condition, str(unit), frequency_perturbed=True)
        finally:
            run.close()


    def test_matched_s_zero_consumes_analytic_frequency_jet(self):
        run = CircuitRun(plan=_matched_zero_plan(), storage="memory", backend="jax", precision="float64")
        try:
            result = run.evaluate(run.original, TransferZeroSpec(anchor=6.3*u.GHz,
                family="S", input_coordinate="p", output_coordinate="p"))
            expected, condition = matched_zero_oracle()
            actual = complex(result.zero.to("radian/second").magnitude)
            assert_forward(self, "DUT_matched_S_zero", actual, expected, condition, "rad/s")
            # One-Port S uses the existing bordered numerator of order q+1=2.
            numerator_order = 2
            transfer_tau = 256 * (numerator_order + 1) * np.finfo(np.float64).eps
            self.assertLessEqual(abs(actual.imag), transfer_tau * abs(actual))
        finally:
            run.close()


