"""Exact rational oracle for small lumped networks and their condition bounds.

Every binary64 primitive, and the binary64 angular frequency, is converted
exactly to a Fraction. Nodal admittance, Schur reduction, Z and S are then
formed in exact complex-rational arithmetic, so the oracle has no rounding.
First and second derivatives use exact hyper-dual numbers over the same
field, so they are exact as well.

The condition bound of an entry is the triangle bound used by
``analytic_conditions``: the sum over independently and relatively perturbed
primitives of the absolute first-order change, divided by the entry. The
angular frequency and, for S, the reference impedance count as primitives.

Branch admittances follow SCNSim's convention: 1/R, i/(omega L), -i omega C.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True)
class Exact:
    """One exact complex rational number."""

    re: Fraction
    im: Fraction = Fraction(0)

    @staticmethod
    def of(value) -> "Exact":
        if isinstance(value, Exact):
            return value
        value = complex(value)
        return Exact(Fraction(value.real), Fraction(value.imag))

    def __add__(self, other):
        if isinstance(other, HyperDual):
            return NotImplemented
        other = Exact.of(other)
        return Exact(self.re + other.re, self.im + other.im)

    __radd__ = __add__

    def __neg__(self):
        return Exact(-self.re, -self.im)

    def __sub__(self, other):
        if isinstance(other, HyperDual):
            return NotImplemented
        return self + (-Exact.of(other))

    def __rsub__(self, other):
        return Exact.of(other) - self

    def __mul__(self, other):
        if isinstance(other, HyperDual):
            return NotImplemented
        other = Exact.of(other)
        return Exact(self.re * other.re - self.im * other.im,
                     self.re * other.im + self.im * other.re)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, HyperDual):
            return NotImplemented
        other = Exact.of(other)
        scale = other.re * other.re + other.im * other.im
        return Exact((self.re * other.re + self.im * other.im) / scale,
                     (self.im * other.re - self.re * other.im) / scale)

    def __rtruediv__(self, other):
        return Exact.of(other) / self

    def conjugate(self):
        return Exact(self.re, -self.im)

    def is_zero(self) -> bool:
        return self.re == 0 and self.im == 0

    def __complex__(self) -> complex:
        return complex(float(self.re), float(self.im))

    def magnitude(self) -> float:
        return abs(complex(self))


ZERO, ONE, I = Exact(Fraction(0)), Exact(Fraction(1)), Exact(Fraction(0), Fraction(1))


@dataclass(frozen=True)
class HyperDual:
    """a + b e1 + c e2 + d e1 e2 with e1**2 = e2**2 = 0, over Exact."""

    a: Exact
    b: Exact = ZERO
    c: Exact = ZERO
    d: Exact = ZERO

    @staticmethod
    def of(value) -> "HyperDual":
        return value if isinstance(value, HyperDual) else HyperDual(Exact.of(value))

    def __add__(self, other):
        o = HyperDual.of(other)
        return HyperDual(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    __radd__ = __add__

    def __neg__(self):
        return HyperDual(-self.a, -self.b, -self.c, -self.d)

    def __sub__(self, other):
        return self + (-HyperDual.of(other))

    def __rsub__(self, other):
        return HyperDual.of(other) - self

    def __mul__(self, other):
        o = HyperDual.of(other)
        return HyperDual(self.a * o.a, self.a * o.b + self.b * o.a, self.a * o.c + self.c * o.a,
                         self.a * o.d + self.b * o.c + self.c * o.b + self.d * o.a)

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = ONE / self.a
        square = inverse * inverse
        return HyperDual(inverse, -self.b * square, -self.c * square,
                         -self.d * square + 2 * self.b * self.c * square * inverse)

    def __truediv__(self, other):
        return self * HyperDual.of(other).reciprocal()

    def __rtruediv__(self, other):
        return HyperDual.of(other) * self.reciprocal()

    def conjugate(self):
        return HyperDual(self.a.conjugate(), self.b.conjugate(), self.c.conjugate(), self.d.conjugate())

    def is_zero(self) -> bool:
        return self.a.is_zero() and self.b.is_zero() and self.c.is_zero() and self.d.is_zero()


def zeros(rows, columns):
    return [[ZERO for _ in range(columns)] for _ in range(rows)]


def identity(size):
    return [[ONE if row == column else ZERO for column in range(size)] for row in range(size)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), ZERO)
             for j in range(len(b[0]))] for i in range(len(a))]


def add(a, b, scale=ONE):
    return [[x + scale * y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def _pivot_nonzero(value) -> bool:
    return not (value.a.is_zero() if isinstance(value, HyperDual) else value.is_zero())


def solve(a, b):
    """Exact Gauss-Jordan elimination of a @ x = b (b is a matrix)."""
    n = len(a)
    work = [list(a[row]) + list(b[row]) for row in range(n)]
    for column in range(n):
        pivot = next(row for row in range(column, n) if _pivot_nonzero(work[row][column]))
        work[column], work[pivot] = work[pivot], work[column]
        inverse = ONE / work[column][column]
        work[column] = [value * inverse for value in work[column]]
        for row in range(n):
            if row != column and not work[row][column].is_zero():
                factor = work[row][column]
                work[row] = [x - factor * y for x, y in zip(work[row], work[column])]
    return [row[n:] for row in work]


def inverse(a):
    return solve(a, identity(len(a)))


@dataclass(frozen=True)
class Branch:
    """One two-terminal R, L or C element; ``None`` is the ground node."""

    kind: str
    a: str | None
    b: str | None
    value: float


def _admittance(kind, value, omega):
    if kind == "R":
        return ONE / value
    if kind == "L":
        return I / (omega * value)
    if kind == "C":
        return -I * omega * value
    raise ValueError(kind)


def _stamp(matrix, index, branch, value):
    ends = [(index[node], sign) for node, sign in ((branch.a, 1), (branch.b, -1)) if node is not None]
    for row, row_sign in ends:
        for column, column_sign in ends:
            term = value if row_sign == column_sign else -value
            matrix[row][column] = matrix[row][column] + term


@dataclass(frozen=True)
class Network:
    """Branches plus the ordered port nodes; other nodes are eliminated."""

    branches: tuple[Branch, ...]
    ports: tuple[str, ...]

    @property
    def nodes(self):
        internal = []
        for branch in self.branches:
            for node in (branch.a, branch.b):
                if node is not None and node not in self.ports and node not in internal:
                    internal.append(node)
        return tuple(self.ports) + tuple(internal)

    @property
    def primitive_count(self):
        # Branch values, then angular frequency, then reference impedance.
        return len(self.branches) + 2

    def families(self, values, omega, reference):
        """Y, Z and S of the ports for (possibly hyper-dual) primitives."""
        nodes = self.nodes
        index = {node: position for position, node in enumerate(nodes)}
        nodal = zeros(len(nodes), len(nodes))
        for branch, value in zip(self.branches, values, strict=True):
            _stamp(nodal, index, branch, _admittance(branch.kind, value, omega))
        p = len(self.ports)
        y = [row[:p] for row in nodal[:p]]
        if len(nodes) > p:
            a_te = [row[p:] for row in nodal[:p]]
            tail = solve([row[p:] for row in nodal[p:]], [row[:p] for row in nodal[p:]])
            y = add(y, matmul(a_te, tail), -ONE)
        m = inverse(add(identity(p), y, reference))
        s = matmul(add(identity(p), y, -reference), m)
        return {"Y": y, "Z": inverse(y), "S": s}

    def response(self, frequency_hz: float, reference: float = 50.0):
        """Exact responses at the binary64 omega SCNSim forms from frequency_hz."""
        return Response(self, 2 * math.pi * frequency_hz, reference)


def _part(value: complex, part):
    return value if part is None else getattr(value, part)


class Response:
    """Exact Y, Z, S, their derivatives and condition bounds at one point."""

    def __init__(self, network: Network, omega: float, reference: float):
        self.network = network
        self.primitives = tuple(branch.value for branch in network.branches) + (omega, reference)
        self._exact = network.families(*self._split([Exact.of(v) for v in self.primitives]))
        self._first = {}
        self._second = {}

    @property
    def y(self):
        return self._exact["Y"]

    @property
    def s(self):
        return self._exact["S"]

    def matrix(self, family):
        return self._exact[family]

    def entry(self, family, row, column) -> complex:
        return complex(self._exact[family][row][column])

    def _split(self, values):
        return values[:-2], values[-2], values[-1]

    def _seeded(self, first, second=None):
        values = []
        for position, primitive in enumerate(self.primitives):
            exact = Exact.of(primitive)
            # Relative seeds: the e-coefficients are p df/dp and p q d2f/dpdq.
            values.append(HyperDual(exact, exact if position == first else ZERO,
                                    exact if position == second else ZERO))
        return self.network.families(*self._split(values))

    def _first_changes(self, family):
        if family not in self._first:
            self._first[family] = [self._seeded(k)[family] for k in range(self.network.primitive_count)]
        return self._first[family]

    def relative_derivative(self, family, row, column, primitive) -> Exact:
        """Exact p * d(entry)/dp for one primitive index."""
        return self._first_changes(family)[primitive][row][column].b

    def condition(self, family, row, column, part=None):
        """Triangle condition bound of one entry or its real/imag part."""
        value = abs(_part(self.entry(family, row, column), part))
        changes = self._first_changes(family)
        total = sum(abs(_part(complex(change[row][column].b), part)) for change in changes)
        return total / value

    def first_changes(self, family, row, column):
        """Exact change of the entry per relative primitive, in primitive order."""
        return [change[row][column].b for change in self._first_changes(family)]

    def derivative_changes(self, family, row, column, primitive):
        """Exact change of p d(entry)/dp per relative primitive q."""
        key = (family, primitive)
        if key not in self._second:
            self._second[key] = [self._seeded(primitive, q)[family]
                                 for q in range(self.network.primitive_count)]
        derivative = self.relative_derivative(family, row, column, primitive)
        # q d(p df/dp)/dq = p q d2f/dpdq + [p == q] p df/dp
        return [change[row][column].d + (derivative if q == primitive else ZERO)
                for q, change in enumerate(self._second[key])]

    def derivative_condition(self, family, row, column, primitive, part=None):
        """Condition bound of p d(entry)/dp, from exact second derivatives."""
        derivative = complex(self.relative_derivative(family, row, column, primitive))
        total = sum(abs(_part(complex(term), part))
                    for term in self.derivative_changes(family, row, column, primitive))
        return total / abs(_part(derivative, part))
