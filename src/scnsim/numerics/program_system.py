"""Call-local sparse boundary for physical-program quantity kernels.

The immutable AssemblyProgram supplies every physical coefficient and index.
This boundary adapts its actual synchronized operands to the existing checked
Newton/cofactor/SVD kernels; it never recompiles a physical model or retains a
factor, candidate point, Workspace or numerical history after the call.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np


@dataclass(frozen=True, slots=True)
class NumericalViewCapability:
    port_realizable: bool


class NoMeasurements:
    """Disabled numerical phases do not collect clocks or rows."""
    def phase(self, *args, **kwargs):
        return nullcontext()


class ObservedMeasurements:
    """Call-local forwarding of actual numerical phases to the operation owner."""
    def __init__(self, observer, *, equation):
        self.observer, self.equation = observer, equation

    def phase(self, name, **details):
        return self.observer(name, equation=self.equation)


class ProgramSystem:
    def __init__(self, assembly, view_payload, *, port_realizable, real_dtype, complex_dtype):
        from ..compilation.models import SparseMatrix
        from .sparse_solve import _host_pair_product
        self.real_dtype, self.complex_dtype = real_dtype, complex_dtype
        self.view = NumericalViewCapability(port_realizable)
        self.n = assembly.pattern.shape[0]
        self.rows, self.cols = assembly.pattern.rows, assembly.pattern.cols
        self.selected = np.asarray(assembly.view_program.selected, dtype=np.intp)
        self.eliminated = np.asarray(assembly.view_program.eliminated, dtype=np.intp)
        self.factor_facts = []
        self.compensation_evidence = None
        self.pair_product = _host_pair_product
        pattern = assembly.view_program.patterns[3]
        values = np.asarray(view_payload.model.B, dtype=real_dtype)
        self.B = SparseMatrix(pattern.shape, pattern.rows, pattern.cols, values)
        # Incidence and capacitance-cut coordinate weights are physical real
        # operands. Convert only the equation RHS to complex numerical dtype.
        B = np.zeros(pattern.shape, dtype=real_dtype)
        B[pattern.rows, pattern.cols] = values
        selected_map = np.asarray(view_payload.selected_map, dtype=real_dtype)
        Bk = B @ selected_map.T if port_realizable else None
        self.Bk_rhs = np.asarray(Bk, dtype=complex_dtype) if port_realizable else None
        self.Rk = np.asarray(view_payload.Rk, dtype=complex_dtype) if port_realizable else None
        self.Dk = np.asarray(view_payload.Dk, dtype=complex_dtype) if port_realizable else None
        self.Go = np.asarray(view_payload.Go, dtype=complex_dtype) if port_realizable else None
        slots = {(int(row), int(col)): index for index, (row, col) in enumerate(zip(self.rows, self.cols, strict=True))}
        def pairs(rows):
            return np.asarray([slots[int(row), int(col)] for row in rows for col in rows], dtype=np.intp)
        self.B_pair_map = pairs(self.B.rows)
        self.Bk, self.Bk_pair_map = None, None
        if port_realizable:
            support = np.unique(pattern.rows)
            rr = np.repeat(support, self.Bk_rhs.shape[1])
            cc = np.tile(np.arange(self.Bk_rhs.shape[1]), len(support))
            self.Bk = SparseMatrix(Bk.shape, rr, cc, Bk[rr, cc])
            self.Bk_pair_map = pairs(self.Bk.rows)

    def csc(self, values, measurements):
        from scipy.sparse import coo_matrix
        with measurements.phase('sparse_csc_build', nodes=self.n, nnz=len(values)):
            return coo_matrix((np.asarray(values), (self.rows, self.cols)), shape=(self.n, self.n)).tocsc()

    def stamp(self, matrix, coefficient, pair_map):
        weights = np.asarray(matrix.values, dtype=self.complex_dtype)
        columns = np.asarray(matrix.cols)
        values = weights[:, None] * coefficient[columns[:, None], columns[None, :]] * weights[None, :]
        result = np.zeros(len(self.rows), dtype=self.complex_dtype)
        np.add.at(result, pair_map, values.ravel())
        return result


def host_assembly(payload, *, compensated=False):
    """Existing four-value assembly, or six-value represented pair authority.

    The declared arithmetic route selects the tuple at this unique adapter;
    low components are retained exactly when the caller requested compensation.
    """
    high = (np.asarray(payload.values), np.asarray(payload.frequency_values),
            np.asarray(payload.bound), np.asarray(payload.status, dtype=np.int32))
    return high + (np.asarray(payload.low_values), np.asarray(payload.frequency_low_values)) if compensated else high
