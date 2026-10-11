"""Physical-program root values over the existing checked numerical kernels.

Topology, fields and View arithmetic come only from AssemblyProgram. Newton,
cofactor/SVD, physical transpose and closure conditions remain their existing
numerical authority. Stable compiled assembly functions capture declarations,
not points. A value call owns its temporary sparse boundary and factor lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
import numpy as np

from .models import EvaluationResult
from .program_system import ProgramSystem, NoMeasurements, host_assembly


@dataclass(frozen=True, slots=True)
class RootDeclaration:
    kind: str
    start_index: int
    quantity_key: str
    coordinate_index: int | None = None
    row_index: int | None = None
    column_index: int | None = None
    family: str | None = None
    output_index: int | None = None
    input_index: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class RootProgram:
    assembly: object
    declaration: RootDeclaration
    precision: str
    _assembly_functions: dict = field(default_factory=dict, repr=False)
    _assembly_lock: object = field(default_factory=RLock, repr=False)
    _prepared_executables: dict = field(default_factory=dict, repr=False)

    def assembly_function(self, *, loaded, compensated, capability):
        key = (loaded, compensated, capability)
        with self._assembly_lock:
            function = self._assembly_functions.get(key)
            if function is None:
                import jax
                assembly = self.assembly
                function = jax.jit(lambda theta, omega: assembly(theta, omega, loaded=loaded,
                                                                compensated=compensated, port_realizable=capability))
                self._assembly_functions[key] = function
            return function

    def install_prepared(self, key, executable):
        with self._assembly_lock:
            self._prepared_executables[key] = executable

    def assemble(self, theta, omega, *, loaded, compensated, capability):
        key = (loaded, compensated, capability)
        function = self._prepared_executables.get(key)
        if function is None:
            function = self.assembly_function(loaded=loaded, compensated=compensated, capability=capability)
        return function(theta, omega)

    def evaluate(self, inputs, *, capability, measurements=None, sparse_runner=None, prepared_bootstrap=None):
        from .determinants import QuantityError
        from .models import NumericalFailure
        from .evidence import numerical_error, evidence_bytes
        try:
            if sparse_runner is None:
                return self._value(inputs, capability=capability, measurements=measurements, prepared_bootstrap=prepared_bootstrap)
            return sparse_runner(self._value, (inputs,), dict(capability=capability, measurements=measurements, prepared_bootstrap=prepared_bootstrap))
        except QuantityError as error:
            return EvaluationResult(self.declaration.quantity_key, failure=NumericalFailure(
                error.kind, error.stage, error.detail, evidence_bytes(error.facts)))

    def value(self, inputs, *, capability, measurements=None, sparse_runner=None, prepared_bootstrap=None):
        from .evidence import numerical_error
        result = self.evaluate(inputs, capability=capability, measurements=measurements, sparse_runner=sparse_runner, prepared_bootstrap=prepared_bootstrap)
        if result.failure is not None:
            raise numerical_error(result.failure)
        return result

    def _value(self, inputs, *, capability, measurements=None, observer=None, prepared_bootstrap=None):
        from .sparse_root import diagonal_root
        from .sparse_quantities import evaluate_quantity, numerical_code
        from .evidence import evidence_bytes
        if observer is not None:
            from .program_system import ObservedMeasurements
            measurements = ObservedMeasurements(observer, equation=self.declaration.kind)
        else:
            measurements = NoMeasurements() if measurements is None else measurements
        rd, cd = (np.float32, np.complex64) if self.precision == 'float32' else (np.float64, np.complex128)
        declaration = self.declaration
        start = cd(np.asarray(inputs.branch_starts_rad_s)[declaration.start_index])
        theta = inputs.theta
        loaded = declaration.kind != 'transfer_zero' or not capability
        compensated = declaration.kind == 'transfer_zero' and declaration.family == 'Z' and not capability
        boot_function = self.assembly_function(loaded=loaded, compensated=compensated, capability=capability)
        boot, view = (self.assemble(theta, start, loaded=loaded, compensated=compensated, capability=capability)
                      if prepared_bootstrap is None else prepared_bootstrap)
        system = ProgramSystem(self.assembly, view, port_realizable=capability, real_dtype=rd, complex_dtype=cd)
        # Exactly one call-local prepared state may be reused by the first
        # Newton evaluation. No frequency cache or point enters the program.
        boot_key = (np.asarray(start, dtype=cd).tobytes(), loaded, compensated)
        first_state = [host_assembly(boot, compensated=compensated)]
        def assemble(omega, *, loaded, derivative=True, compensated=False):
            key = (np.asarray(omega, dtype=cd).tobytes(), loaded, compensated)
            if first_state and key == boot_key:
                return first_state.pop()
            function = self.assembly_function(loaded=loaded, compensated=compensated, capability=capability)
            payload, _ = self.assemble(theta, cd(omega), loaded=loaded, compensated=compensated, capability=capability)
            return host_assembly(payload, compensated=compensated)
        if declaration.kind == 'diagonal_root':
            omega, slope, certificate, code, steps = diagonal_root(system, start, declaration.coordinate_index,
                                                                   assemble, measurements)
            if code:
                numerical_code(code)
            body = dict(root_omega_rad_s=omega, root_slope=slope)
            facts = dict(certificate=certificate, newton_steps=steps)
        else:
            # The existing quantity kernels consume only these numerical job
            # fields. This is a real declared equation, not a fabricated View.
            job = NumericalRootJob(declaration, start)
            body, facts = evaluate_quantity(system, job, assemble, measurements)
        facts['derivative_equation'] = ('reduced_Y_entry_local_implicit' if declaration.kind == 'transfer_zero' and declaration.family == 'Y'
                                        else 'selected_signed_cofactor_local_implicit' if declaration.kind == 'transfer_zero' and declaration.family == 'Z'
                                        else 'selected_bordered_scattering_local_implicit' if declaration.kind == 'transfer_zero'
                                        else 'selected_determinant_local_implicit' if declaration.kind == 'hybridized_pole'
                                        else 'selected_operator_entry_local_implicit')
        facts['derivative_limit'] = 'physical local implicit derivative at a machine-closed root; not finite-Newton differentiation or an exact-zero/forward-accuracy guarantee'
        return EvaluationResult(declaration.quantity_key, **body, evidence_bytes=evidence_bytes(facts))


@dataclass(frozen=True, slots=True)
class NumericalRootJob:
    declaration: RootDeclaration
    omega_start_rad_s: object
    root_hint_hz: None = None

    @property
    def kind(self): return self.declaration.kind
    @property
    def row_index(self): return self.declaration.row_index
    @property
    def column_index(self): return self.declaration.column_index
    @property
    def family(self): return self.declaration.family
    @property
    def output_index(self): return self.declaration.output_index
    @property
    def input_index(self): return self.declaration.input_index
