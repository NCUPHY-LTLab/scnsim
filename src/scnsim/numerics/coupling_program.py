"""Residue coupling on finished branches of the SAME physical program.

The existing residue/rank/reciprocity/normalization certificates own primal
success. This layer adapts immutable assembly declarations and call-local root
results, never refinds poles or changes the physical transpose convention.
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
from .models import EvaluationResult, RootBranch, NumericalFailure
from .program_system import ProgramSystem, NoMeasurements, host_assembly
from .evidence import evidence_bytes, numerical_error


def coupling_value(root_program, inputs, branches, *, capability, frequency_index, midpoint):
    from .sparse_quantities import residue_coupling
    from .determinants import QuantityError
    rd, cd = (np.float32, np.complex64) if root_program.precision == 'float32' else (np.float64, np.complex128)
    branch_records = tuple(RootBranch(program.declaration.kind, program.declaration.coordinate_index,
                                     float(np.asarray(result.root_omega_rad_s).real / (2 * np.pi)), result)
                           for program, result in branches)
    roots = tuple(cd(record.result.root_omega_rad_s) for record in branch_records)
    evaluation = cd(cd(roots[0] + roots[1]) / cd(2)) if midpoint else cd(inputs.omega_rad_s[frequency_index])
    def assemble(omega, *, loaded, derivative=True, compensated=False):
        payload, _ = root_program.assemble(inputs.theta, cd(omega), loaded=loaded,
                                           compensated=compensated, capability=capability)
        return host_assembly(payload, compensated=compensated)
    view = root_program.assembly.view_program(inputs.theta, port_realizable=capability)
    system = ProgramSystem(root_program.assembly, view, port_realizable=capability, real_dtype=rd, complex_dtype=cd)
    job = SimpleNamespace(branches=branch_records, frequency_mode='complex_root_midpoint' if midpoint else 'fixed',
                          evaluation_omega_rad_s=evaluation)
    try:
        body, facts = residue_coupling(system, job, assemble, NoMeasurements())
    except QuantityError as error:
        raise numerical_error(NumericalFailure(error.kind, error.stage, error.detail, evidence_bytes(error.facts))) from error
    facts['derivative_equation'] = 'selected_residue_slope_with_checked_frequency_jet_and_local_null_gauge'
    return EvaluationResult('residue_coupling', **body, evidence_bytes=evidence_bytes(facts))


class CouplingFrame:
    """Adapter from parent point+finished-root operands to the residue graph."""
    def __init__(self, graph, output, original_inputs, *, frequency_index, midpoint):
        self.graph, self.output, self.original_inputs = graph, output, original_inputs
        self.frequency_index, self.midpoint = frequency_index, midpoint

    def _direction(self, directions):
        import jax.numpy as jnp
        from .physical_program import ProgramInputs
        point, first, second = directions
        evaluation = (first + second) / 2 if self.midpoint else point.omega_rad_s[self.frequency_index]
        return ProgramInputs(point.theta, jnp.asarray([first, second, evaluation], dtype=self.graph.inputs.omega_rad_s.dtype),
                             jnp.zeros_like(self.graph.inputs.branch_starts_rad_s))

    def jvp(self, directions):
        jets, certificates = self.graph.jvp(self._direction(directions))
        return jets[self.output], certificates

    def mixed(self, first, second, mixed):
        import jax
        a, b = self._direction(first), self._direction(second)
        _, _, values, certificates = self.graph.mixed(a, b)
        jets, more = self.graph.jvp(self._direction(mixed))
        return values[self.output] + jets[self.output], certificates + more

    def vjp(self, weight):
        import jax.numpy as jnp
        from .physical_program import ProgramInputs
        value, certificates = self.graph.vjp(((self.output, weight),))
        first, second = value.omega_rad_s[:2]
        omega = jnp.zeros_like(self.original_inputs.omega_rad_s)
        if self.midpoint:
            first, second = first + value.omega_rad_s[2] / 2, second + value.omega_rad_s[2] / 2
        else:
            omega = omega.at[self.frequency_index].set(value.omega_rad_s[2])
        point = ProgramInputs(value.theta, omega, jnp.zeros_like(self.original_inputs.branch_starts_rad_s))
        return (point, first, second), certificates
