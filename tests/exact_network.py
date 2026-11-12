"""Exact rational oracle for small lumped networks and their condition bounds.

Every binary64 primitive, and the binary64 angular frequency, is converted
exactly to a Fraction. Nodal admittance, Schur reduction, Z and S are then
formed in exact complex-rational arithmetic, so the oracle has no rounding.
The condition bound of an entry is the triangle bound used by
``analytic_conditions``: the sum over independently and relatively perturbed
primitives of the absolute first-order change, divided by the entry. The
frequency and the reference impedance count as primitives where they enter.

Branch admittances follow SCNSim's convention: 1/R, i/(omega L), -i omega C.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


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
        other = Exact.of(other)
        return Exact(self.re + other.re, self.im + other.im)

    __radd__ = __add__

    def __neg__(self):
        return Exact(-self.re, -self.im)

    def __sub__(self, other):
        return self + (-Exact.of(other))

    def __rsub__(self, other):
        return Exact.of(other) - self

    def __mul__(self, other):
        other = Exact.of(other)
        return Exact(self.re * other.re - self.im * other.im,
                     self.re * other.im + self.im * other.re)

    __rmul__ = __mul__

    def __truediv__(self, other):
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


def solve(a, b):
    """Exact Gauss-Jordan elimination of a @ x = b (b is a matrix)."""
    n = len(a)
    work = [list(a[row]) + list(b[row]) for row in range(n)]
    for column in range(n):
        pivot = next(row for row in range(column, n) if not work[row][column].is_zero())
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

    def admittance(self, omega: Exact) -> Exact:
        value = Exact.of(self.value)
        if self.kind == "R":
            return ONE / value
        if self.kind == "L":
            return I / (omega * value)
        if self.kind == "C":
            return -I * omega * value
        raise ValueError(self.kind)

    def omega_log_derivative(self, omega: Exact) -> Exact:
        """omega * d(admittance)/d(omega)."""
        return {"R": ZERO, "L": -self.admittance(omega), "C": self.admittance(omega)}[self.kind]


def stamp(nodes, branch, value):
    """Return the nodal matrix of one branch carrying admittance ``value``."""
    index = {node: position for position, node in enumerate(nodes)}
    matrix = zeros(len(nodes), len(nodes))
    ends = [(index[node], sign) for node, sign in ((branch.a, 1), (branch.b, -1)) if node is not None]
    for row, row_sign in ends:
        for column, column_sign in ends:
            matrix[row][column] = matrix[row][column] + value * (row_sign * column_sign)
    return matrix


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

    def nodal(self, omega, values=None):
        nodes = self.nodes
        total = zeros(len(nodes), len(nodes))
        for position, branch in enumerate(self.branches):
            value = branch.admittance(omega) if values is None else values[position]
            total = add(total, stamp(nodes, branch, value))
        return total

    def extension(self, nodal):
        """W = [I; -A_ee^-1 A_et]: all node voltages for unit port voltages."""
        p = len(self.ports)
        a_ee = [row[p:] for row in nodal[p:]]
        a_et = [row[:p] for row in nodal[p:]]
        if not a_ee:
            return identity(p)
        tail = solve(a_ee, a_et)
        return identity(p) + [[-value for value in row] for row in tail]

    def response(self, frequency_hz: float, reference: float = 50.0):
        """Exact Y, Z, S at the binary64 omega SCNSim forms from frequency_hz."""
        import math
        omega = Exact.of(2 * math.pi * frequency_hz)
        return Response(self, omega, Exact.of(reference))


class Response:
    def __init__(self, network: Network, omega: Exact, reference: Exact):
        self.network, self.omega, self.reference = network, omega, reference
        nodal = network.nodal(omega)
        self.w = network.extension(nodal)
        self.y = matmul(transpose(self.w), matmul(nodal, self.w))
        p = len(network.ports)
        self._m = inverse(add(identity(p), self.y, reference))
        self.s = matmul(add(identity(p), self.y, -reference), self._m)

    @property
    def z(self):
        return inverse(self.y)

    def _family_derivative(self, family, d_y):
        if family == "Y":
            return d_y
        if family == "Z":
            z = self.z
            return [[-value for value in row] for row in matmul(z, matmul(d_y, z))]
        p = len(self.network.ports)
        left = add(identity(p), self.s)
        product = matmul(left, matmul(d_y, self._m))
        return [[-self.reference * value for value in row] for row in product]

    def perturbations(self, family):
        """Exact first-order changes of the family per relative primitive."""
        network, omega = self.network, self.omega
        changes = []
        for position, branch in enumerate(network.branches):
            # p dY/dp is -Y for R and L and +Y for C (Y = 1/R, i/(wL), -iwC).
            sign = 1 if branch.kind == "C" else -1
            local = stamp(network.nodes, branch, branch.admittance(omega) * sign)
            changes.append(self._family_derivative(
                family, matmul(transpose(self.w), matmul(local, self.w))))
        total = zeros(len(network.nodes), len(network.nodes))
        for branch in network.branches:
            total = add(total, stamp(network.nodes, branch, branch.omega_log_derivative(omega)))
        changes.append(self._family_derivative(
            family, matmul(transpose(self.w), matmul(total, self.w))))
        if family == "S":
            p = len(network.ports)
            left = add(identity(p), self.s)
            product = matmul(left, matmul(self.y, self._m))
            changes.append([[-self.reference * value for value in row] for row in product])
        return changes

    def matrix(self, family):
        return {"Y": self.y, "Z": self.z, "S": self.s}[family]

    def condition(self, family, row, column):
        """Triangle condition bound of one entry (frequency perturbed)."""
        value = self.matrix(family)[row][column].magnitude()
        total = sum(change[row][column].magnitude() for change in self.perturbations(family))
        return total / value

    def entry(self, family, row, column) -> complex:
        return complex(self.matrix(family)[row][column])
