"""Shared imaginary-root admission, with tolerance owned by the actual frame.

The comparison accepts both host scalars and traced scalar operands. It does
not select precision, infer matrix size, alter omega, or own a root solver.
"""
from __future__ import annotations


def passive_imaginary_root(omega, tolerance):
    """Accept a passive root within the supplied existing formation tolerance."""
    return omega.imag <= tolerance * abs(omega)
