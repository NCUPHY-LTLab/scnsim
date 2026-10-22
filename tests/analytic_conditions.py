"""Independent analytic condition bounds for the Human-assigned H2 fixtures.

Primitive component values vary independently and relatively. Explicit response
frequencies participate in their formation bounds. This is a triangle bound,
not a sampled estimate or an accuracy assertion about SCNSim's certificates.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import json
import math

import numpy as np


CA, CB, CM = 80.0e-15, 95.0e-15, 2.0e-15
LA, LB, LM = 8.0e-9, 7.0e-9, 500.0e-9


def assert_forward(test, label, actual, expected, condition, unit, *, frequency_perturbed=False):
    """Record each oracle bound/error ratio before enforcing H2's fixed c=100."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = float(np.max(np.abs(expected)))
    relative_error = float(np.max(np.abs(actual - expected))) / scale
    ratio = relative_error / (float(condition) * np.finfo(np.float64).eps)
    def values(array):
        return {"shape": list(array.shape),
                "real_f64": [float(x).hex() for x in array.real.flat],
                "imag_f64": [float(x).hex() for x in array.imag.flat]}
    observation = dict(observable=label, unit=unit, actual=values(actual), oracle=values(expected),
                       condition_bound=float(condition), relative_error=relative_error,
                       error_over_condition_u=ratio,
                       perturbations="independent relative authored physical primitives",
                       evaluation_frequency_perturbed=frequency_perturbed)
    print("H2 " + json.dumps(observation, sort_keys=True), flush=True)
    test.assertLessEqual(ratio, 100.0, json.dumps(observation, sort_keys=True))


def two_mode_matrices(omega):
    c = np.array([[CA + CM, -CM], [-CM, CB + CM]])
    k = np.array([[1 / LA + 1 / LM, -1 / LM], [-1 / LM, 1 / LB + 1 / LM]])
    return c, k, k - omega**2 * c


def operator_condition(omega):
    c, k, q = two_mode_matrices(omega)
    # The frequency derivative contributes another 2*omega**2*abs(C).
    return np.max(np.abs(k) + 3 * omega**2 * np.abs(c)) / np.max(np.abs(q))


def response_oracle(omega):
    a, b = omega * CM, 1 / (omega * LM)
    return 1j * (a - b), 2 * (a + b) / abs(a - b)


def two_mode_roots():
    # Closed quadratic det(K-lambda*C), using the same original binary64 SI
    # primitives with independent high-precision arithmetic (no producer solve).
    with localcontext() as context:
        context.prec = 50  # Oracle arithmetic precision, not a test tolerance.
        ca, cb, cm, la, lb, lm = map(Decimal.from_float, (CA, CB, CM, LA, LB, LM))
        aa, ab = ca + cm, cb + cm
        da, db, dm = 1 / la + 1 / lm, 1 / lb + 1 / lm, 1 / lm
        a = aa * ab - cm * cm
        b = da * ab + db * aa - 2 * dm * cm
        c = da * db - dm * dm
        discriminant = (b * b - 4 * a * c).sqrt()
        return tuple(float(((b + sign * discriminant) / (2 * a)).sqrt()) for sign in (-1, 1))


def pole_condition(omega):
    c, _, q = two_mode_matrices(omega)
    x = np.array([-q[0, 1], q[0, 0]])
    incidence = (np.array([1., 0.]), np.array([0., 1.]), np.array([1., -1.]))
    numerator = sum((1 / l + omega**2 * cap) * abs(vector @ x)**2
                    for l, cap, vector in zip((LA, LB, LM), (CA, CB, CM), incidence))
    return numerator / abs(2 * omega**2 * (x @ c @ x))


def coupling_oracle(omega):
    aa, ab, da, db, dm = CA + CM, CB + CM, 1 / LA + 1 / LM, 1 / LB + 1 / LM, 1 / LM
    n = omega**2 * CM - dm
    value = n / (2 * (aa * da * ab * db)**0.25)
    sensitivities = (-CA / (4 * aa), -CB / (4 * ab),
                    omega**2 * CM / n - CM / (4 * aa) - CM / (4 * ab),
                    (1 / LA) / (4 * da), (1 / LB) / (4 * db),
                    dm / n + dm / (4 * da) + dm / (4 * db),
                    2 * omega**2 * CM / n)
    return value, sum(abs(item) for item in sensitivities)


def residue_oracle(branch):
    capacitance, inductance = ((CA, LA), (CB, LB))[branch]
    a, d = capacitance + CM, 1 / inductance + 1 / LM
    value = -1 / (2 * math.sqrt(a * d))
    condition = capacitance / (2 * a) + CM / (2 * a) + (1 / inductance) / (2 * d) + (1 / LM) / (2 * d)
    return value, condition


def matched_zero_oracle():
    with localcontext() as context:
        context.prec = 50
        l, c = map(Decimal.from_float, (8.0e-9, 80.0e-15))
        omega = float(1 / (l * c).sqrt())
    # R and Z0 are independently authored even though both equal 50 ohms.
    condition = 1 + 1 / (2 * omega * 80.0e-15 * 50.0) + 1 / (2 * omega * 80.0e-15 * 50.0)
    return omega, condition


def mutual_stiffness_oracle(junction_incidence, loop_incidence):
    """Independent original two-SQUID stamp and its primitive condition bound."""
    la, lb, coefficient, lj = 4.0e-9, 9.0e-9, 0.25, 8.0e-9
    mutual = coefficient * math.sqrt(la * lb)
    determinant = la * lb - mutual**2
    inverse = np.array([[lb, -mutual], [-mutual, la]]) / determinant
    junction = sum(np.outer(vector, vector) / lj for vector in junction_incidence)
    expected = junction + loop_incidence @ inverse @ loop_incidence.T
    variations = (np.array([[la, mutual / 2], [mutual / 2, 0.]]),
                  np.array([[0., mutual / 2], [mutual / 2, lb]]),
                  np.array([[0., mutual], [mutual, 0.]]))
    bound = np.abs(junction)
    for variation in variations:
        bound = bound + np.abs(loop_incidence @ (-inverse @ variation @ inverse) @ loop_incidence.T)
    return expected, np.max(bound) / np.max(np.abs(expected))
