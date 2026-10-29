"""Shared declared-dtype residual scale and coupling reciprocity certificates.

Native equation callbacks own factors and solves. This module retains only
mathematical helpers consumed by those current native authorities; reciprocity
keeps the ordinary physical transpose and the existing rounding budget.
"""
from __future__ import annotations
import numpy as np


def maximum_abs(a, real_dtype):
    if hasattr(a, 'data') and hasattr(a, 'tocsc'):
        return real_dtype(np.max(np.abs(a.data))) if a.nnz else real_dtype(0)
    return real_dtype(np.max(np.abs(a)))

def tau(n, real_dtype):
    return real_dtype(real_dtype(256 * (n + 1)) * np.finfo(real_dtype).eps)

def ratio(num, den, real_dtype):
    if not (np.isfinite(num) and np.isfinite(den)):
        return real_dtype(np.inf)
    if den == 0:
        return real_dtype(0 if num == 0 else np.inf)
    return real_dtype(num / den)

def schur_reciprocity(system, state):
    """Explain solve-induced skew without changing the physical selected F.

    R=B-A X gives F-F.T = input_skew + X.T R-R.T X (ordinary
    transpose). Only this signed residual term is removed from the certificate.
    Bounds use real/imaginary component arithmetic in the declared dtype. A
    length-d complex dot and final subtraction have at most 2d+2 rounding
    stages on any term and 8d+2 scalar operations overall; the latter also
    bounds absolute subnormal-rounding errors. No input skew enters the budget.
    """
    F, _, Q, _, _, X, _, _ = state
    rd, cd = system.real_dtype, system.complex_dtype
    selected, eliminated = system.selected, system.eliminated
    u = rd(np.finfo(rd).eps / 2)
    subnormal = np.nextafter(rd(0), rd(1))

    def up(value):
        return np.nextafter(np.asarray(value, dtype=rd), rd(np.inf))

    def add(a, b):
        return up(np.asarray(a, dtype=rd) + np.asarray(b, dtype=rd))

    def mul(a, b):
        return up(np.asarray(a, dtype=rd) * np.asarray(b, dtype=rd))

    def magnitude(a):
        # L1 complex magnitude bounds both components without a square/sqrt.
        if hasattr(a, 'tocsc'):
            result = a.copy()
            result.data = add(np.abs(a.data.real), np.abs(a.data.imag))
            return result
        return add(np.abs(a.real), np.abs(a.imag))

    def degrees(a):
        if hasattr(a, 'tocsr'):
            return np.diff(a.tocsr().indptr)
        return np.full(a.shape[0], a.shape[1], dtype=np.int64)

    def constants(a):
        d = np.asarray(degrees(a), dtype=rd)[:, None]
        k = add(mul(rd(2), d), rd(2))
        ku = mul(k, u)
        lower = np.nextafter(rd(1)-ku, rd(-np.inf))
        gamma = np.where(lower > 0, up(ku/lower), rd(np.inf))
        operations = add(mul(rd(8), d), rd(2))
        floor = np.where(lower > 0, up(mul(operations, subnormal)/lower), rd(np.inf))
        return gamma, floor

    def dot_upper(a, b):
        # Enclose the exact positive product, including rounding in the bound
        # computation itself, rather than treating a rounded sum as an upper bound.
        gamma, floor = constants(a)
        product = np.asarray(magnitude(a) @ magnitude(b), dtype=rd)
        lower = np.nextafter(rd(1)-gamma, rd(-np.inf))
        return np.where(lower > 0, up(add(product, floor)/lower), rd(np.inf))

    def formation_error(a, x, b):
        gamma, floor = constants(a)
        return add(mul(gamma, add(magnitude(b), dot_upper(a, x))), floor)

    def subtraction_error(a, b):
        return add(mul(u, add(magnitude(a), magnitude(b))), mul(rd(2), subnormal))

    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        raw_skew = np.asarray(F-F.T, dtype=cd)
        D = Q[selected][:, selected].toarray()
        if len(eliminated):
            A = Q[eliminated][:, eliminated].tocsc()
            B = Q[eliminated][:, selected].toarray()
            C = Q[selected][:, eliminated]
            R = np.asarray(B-A @ X, dtype=cd)
            P = np.asarray(X.T @ R, dtype=cd)
            T = np.asarray(P-P.T, dtype=cd)
            er = formation_error(A, X, B)
            ef = formation_error(C, X, D)
            ep = formation_error(X.T, R, np.zeros_like(P))
            et = add(add(ep, ep.T), subtraction_error(P, P.T))
            residual_uncertainty = dot_upper(X.T, er)
            budget = add(add(ef, ef.T), add(residual_uncertainty, residual_uncertainty.T))
            budget = add(budget, et)
            input_skew = dict(eliminated=maximum_abs(A-A.T, rd),
                              cross=maximum_abs(C-B.T, rd))
        else:
            T = np.zeros_like(F)
            budget = np.zeros(F.shape, dtype=rd)
            input_skew = dict(eliminated=rd(0), cross=rd(0))
        H = np.asarray(raw_skew-T, dtype=cd)
        budget = add(budget, subtraction_error(F, F.T))
        budget = add(budget, subtraction_error(raw_skew, T))
        num = maximum_abs(raw_skew, rd)
        scale = maximum_abs(np.abs(F)+np.abs(F.T), rd)
        allowance = rd(add(mul(tau(len(selected), rd), scale), maximum_abs(budget, rd)))
        defect = maximum_abs(H, rd)
        # A nonfinite/unavailable arithmetic certificate is never an infinite
        # allowance for success; the caller retains the existing failure owner.
        closed = bool(np.isfinite(allowance) and np.isfinite(defect) and defect <= allowance)
        input_skew['retained'] = maximum_abs(D-D.T, rd)
    return closed, dict(numerator=num, denominator=scale, ratio=ratio(num, scale, rd),
                        raw_selected_matrix=F, raw_skew=raw_skew,
                        residual_contribution=T, corrected_defect=H,
                        corrected_defect_norm=defect, rounding_budget=budget,
                        allowance=allowance, unit_roundoff=u,
                        arithmetic_precision=np.dtype(rd).name, input_skew=input_skew,
                        rounding_model="gamma_(2d+2); (8d+2) subnormal errors; sparse row d",
                        input_skew_interpretation='input-origin skew remains signal; never credited to rounding budget')
