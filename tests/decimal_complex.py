"""Decimal complex arithmetic for oracles that need irrational values.

Callers set the Decimal context precision (oracle arithmetic, never a
comparison tolerance). Only what the oracles use is provided: field
operations, square root and the exponential.
"""
from __future__ import annotations

from decimal import Decimal, getcontext


class DComplex:
    __slots__ = ("re", "im")

    def __init__(self, re, im=Decimal(0)):
        self.re, self.im = Decimal(re), Decimal(im)

    @staticmethod
    def of(value) -> "DComplex":
        if isinstance(value, DComplex):
            return value
        value = complex(value)
        return DComplex(Decimal.from_float(value.real), Decimal.from_float(value.imag))

    def __add__(self, o):
        o = DComplex.of(o)
        return DComplex(self.re + o.re, self.im + o.im)

    __radd__ = __add__

    def __neg__(self):
        return DComplex(-self.re, -self.im)

    def __sub__(self, o):
        o = DComplex.of(o)
        return DComplex(self.re - o.re, self.im - o.im)

    def __rsub__(self, o):
        return DComplex.of(o) - self

    def __mul__(self, o):
        o = DComplex.of(o)
        return DComplex(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = DComplex.of(o)
        scale = o.re * o.re + o.im * o.im
        return DComplex((self.re * o.re + self.im * o.im) / scale, (self.im * o.re - self.re * o.im) / scale)

    def __rtruediv__(self, o):
        return DComplex.of(o) / self

    def __complex__(self):
        return complex(float(self.re), float(self.im))

    def magnitude(self) -> Decimal:
        return (self.re * self.re + self.im * self.im).sqrt()

    def sqrt(self) -> "DComplex":
        """Principal square root."""
        modulus = self.magnitude()
        real = ((modulus + self.re) / 2).sqrt()
        imag = ((modulus - self.re) / 2).sqrt()
        return DComplex(real, imag if self.im >= 0 else -imag)

    def exp(self) -> "DComplex":
        scale = self.re.exp()
        return DComplex(scale * _cos(self.im), scale * _sin(self.im))


def _series(x: Decimal, start: int) -> Decimal:
    """sum_k (-1)^k x^(2k+start) / (2k+start)! until terms vanish."""
    getcontext().prec += 5
    term = Decimal(1)
    for k in range(1, start + 1):
        term = term * x / k
    total, k = term, start
    while True:
        term = -term * x * x / ((k + 1) * (k + 2))
        k += 2
        if total + term == total:
            break
        total += term
    getcontext().prec -= 5
    return +total


def _cos(x: Decimal) -> Decimal:
    return _series(x, 0)


def _sin(x: Decimal) -> Decimal:
    return _series(x, 1)
