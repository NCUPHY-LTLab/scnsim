"""Finite intrinsic transmission-zero candidates from the existing RLC stamps.

This is a candidate provider, never a second physical evaluator or certificate.
Only the declared pure-inductive fixed nodal domain is admitted. The original
compensated transfer corrector certifies candidates; poles, infinite and static
pencil entries are retained as raw homogeneous data, not promoted to roots.
Factors/eigensolver buffers are call-local; this module owns no Run/cache/sink.
"""
from __future__ import annotations
from typing import NamedTuple
import numpy as np


class PolynomialSpectrum(NamedTuple):
    alpha: np.ndarray
    beta: np.ndarray
    omega_rad_s: np.ndarray
    defined: np.ndarray
    reference_omega_rad_s: float
    node_count: int
    pencil_size: int
    coefficient_dtype: str


def _readonly(value):
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(order='C'),dtype=value.dtype).reshape(value.shape)


def coefficient_function(assembly, view_index, dtype, candidate):
    """Stable candidate-only coefficient kernel; captures no runtime point."""
    return assembly.polynomial_coefficients(candidate.operands[view_index],
        view_status=candidate.views[view_index].status,dtype=dtype)


def transmission_zero_spectrum(pattern, coefficients, selected, *, input_index,
                               output_index, reference_omega_rad_s):
    """SISO I_input -> V_output Rosenbrock pencil, s=-i omega.

    Q=K+sG+s²C acts on nodal flux. The descriptor state is (flux, flux_rate),
    so voltage output is c.T*flux_rate, not a pole eigenproblem. Time and row
    scaling are invertible unit transformations; no numerical rank/epsilon,
    deduplication, search range or eligibility threshold is introduced here.
    Standard selected-dtype coefficient coalescing proposes candidates only;
    represented compensated equations remain the verification authority.
    """
    from scipy.linalg import eig
    from scipy.sparse import coo_matrix
    K_values,C_values,G_values = coefficients
    dtype = np.asarray(K_values).dtype
    real_dtype = np.empty((),dtype=dtype).real.dtype
    n = pattern.shape[0]
    K,C,G = tuple(coo_matrix((np.asarray(value),
        (pattern.rows,pattern.cols)),shape=pattern.shape).toarray()
        for value in (K_values,C_values,G_values))
    reference = real_dtype.type(reference_omega_rad_s)
    # The source root Hint already owns the nonzero reference. Native division
    # errors are not hidden behind a new numerical cutoff or alternative scale.
    Ks,Gs,Cs = K, reference*G, (reference*reference)*C
    scale = np.max(np.abs(Ks)+np.abs(Gs)+np.abs(Cs),axis=1)
    scale = np.where(scale==0,real_dtype.type(1),scale)
    Ks,Gs,Cs = tuple(value/scale[:,None] for value in (Ks,Gs,Cs))
    b,c = np.zeros(n,dtype=dtype),np.zeros(n,dtype=dtype)
    b[selected[input_index]]=1
    c[selected[output_index]]=1
    zeros,eye = np.zeros((n,n),dtype=dtype),np.eye(n,dtype=dtype)
    A = np.block([[zeros,eye],[-Ks,-Gs]])
    E = np.block([[eye,zeros],[zeros,Cs]])
    B = np.concatenate((np.zeros(n,dtype=dtype),b/scale))
    output = np.concatenate((np.zeros(n,dtype=dtype),c))
    size = 2*n+1
    Ap,Ep = np.zeros((size,size),dtype=dtype),np.zeros((size,size),dtype=dtype)
    Ap[:2*n,:2*n],Ap[:2*n,-1],Ap[-1,:2*n] = A,B,-output
    Ep[:2*n,:2*n] = E
    alpha,beta = eig(Ap,Ep,left=False,right=False,homogeneous_eigvals=True)
    defined = beta!=0
    # beta==0 is an exact homogeneous infinity/undefined representation, not a
    # near-rank classification. Keep alpha/beta untouched for every source row.
    omega = np.full(alpha.shape,np.nan,dtype=dtype)
    omega[defined] = dtype.type(1j)*reference*(alpha[defined]/beta[defined])
    return PolynomialSpectrum(_readonly(alpha),_readonly(beta),_readonly(omega),
        _readonly(defined),float(reference),n,size,str(dtype))
