"""Declared objective loss on the single checked physical Quantity graph.

The coordinator supplies finalized targets/scales and owns search/continuation.
One call builds one physical point and one reverse frame. Numerical derivatives
remain physical SI derivatives; optimizer-coordinate transformations are outside
this module. No native factors, points or derivative tapes survive the call.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple, Any
import hashlib
import json
import numpy as np

from ..canonical import canonical_json_bytes, float64_from_hex
from .quantity_program import compile_quantity_program
from ..numerics.objective_loss import compile_objective_loss
from ..numerics.physical_program import ProgramOutcome, _selected_precision
from ..numerics.evidence import require_primal


class ObjectiveOutcome(NamedTuple):
    primal: ProgramOutcome
    values: tuple
    normalized_residuals: tuple
    contributions: tuple
    total_cost: Any
    gradient: Any
    aux: Any


@dataclass(frozen=True, slots=True, eq=False)
class ObjectiveProgram:
    quantity_program: object
    loss: object
    identity_bytes: bytes
    _arithmetic_functions: dict = field(default_factory=dict, repr=False)

    @property
    def program(self): return self.quantity_program.program
    @property
    def precision(self): return self.program.precision
    @property
    def output_layout(self): return self.quantity_program.output_layout
    @property
    def program_sha256(self): return hashlib.sha256(self.identity_bytes).hexdigest()

    def inputs(self, parameter_set, *, branch_starts_rad_s=None):
        return self.quantity_program.inputs(parameter_set, branch_starts_rad_s=branch_starts_rad_s)

    def prepare_batch(self, inputs_sequence, *, sparse_runner=None):
        return self.program.prepare_batch(inputs_sequence, sparse_runner=sparse_runner)

    def _frame(self, inputs, sparse_runner, prepared=None):
        graph, output, aux = self.program._graph(self.program._cast(inputs), self.program._prepared_capabilities(inputs, prepared),
                                                sparse_runner=sparse_runner, prepared=prepared)
        # Finalized target/scale/weight arithmetic has its own static owner;
        # it must not alias another loss appended to the same Quantity graph.
        graph.arithmetic_functions = self._arithmetic_functions
        loss = self.loss
        payload = graph.pure(lambda *operands: loss(operands), *output)
        cost = graph.pure(lambda value: value.total_cost, payload)
        result = graph.values[payload]
        primal = ProgramOutcome(result.values, aux)
        require_primal(primal)
        return graph, output, cost, result, primal

    @_selected_precision
    def value(self, inputs, sparse_runner=None, *, prepared=None):
        _, _, _, result, primal = self._frame(inputs, sparse_runner, prepared)
        return ObjectiveOutcome(primal, result.values, result.normalized_residuals,
                                result.contributions, result.total_cost, None, primal.aux)

    @_selected_precision
    def value_and_gradient(self, inputs, *, active_keys, sparse_runner=None):
        import jax
        import jax.numpy as jnp
        graph, output, cost, result, primal = self._frame(inputs, sparse_runner)
        # Differentiate the complete cost. Exact abs-at-zero is not given a
        # standalone subgradient: its zero-target square has already been
        # lowered to raw components. Hinge boundaries have a zero cost covector.
        operands = tuple(graph.values[index] for index in output)
        _, pullback = jax.vjp(lambda *values: self.loss(values).total_cost, *operands)
        covectors = pullback(jnp.ones_like(result.total_cost))
        unavailable = False
        for available, reasons, covector, declaration in zip(primal.aux.observable_available,
                                                             primal.aux.observable_reasons,
                                                             covectors, self.loss.objectives, strict=True):
            if bool(available) or declaration.weight == 0:
                continue
            abs_only = bool(reasons) and all('absolute-value zero' in reason or ': magnitude ' in reason
                                             for reason in reasons)
            if not (abs_only and not np.any(np.asarray(covector) != 0)):
                unavailable = True
        if unavailable:
            return ObjectiveOutcome(primal, result.values, result.normalized_residuals,
                                    result.contributions, result.total_cost, None, primal.aux)
        derivative, certificates = graph.vjp(((cost, jnp.ones_like(result.total_cost)),))
        aux = self.program._derivative_aux(primal.aux, certificates)
        actual_available = int(np.asarray(aux.status)) == 0
        aux = aux._replace(derivative_available=actual_available)
        indices = self.program.model_program.fields.layout.input_indices(active_keys)
        gradient = derivative.theta[jnp.asarray(indices, dtype=jnp.int32)] if actual_available else None
        return ObjectiveOutcome(primal, result.values, result.normalized_residuals,
                                result.contributions, result.total_cost, gradient, aux)


def _smooth_operand(quantity):
    if quantity['type'] == 'quantity_absolute':
        return quantity['operand']
    if quantity.get('projection') == 'magnitude':
        return dict(quantity, projection=None)
    return None


def compile_objective_quantities(plan_document, optimization_spec, parameter_set, *, default_view=None, mesh=None,
                              bounds=None, precision='float64', authorized=frozenset(), program_owner=None, admit_program=False):
    objectives = tuple(optimization_spec['objectives'])
    declarations, modes = {}, []
    for objective in objectives:
        quantity = objective['quantity']
        raw = _smooth_operand(quantity)
        if (raw is not None and objective['comparison'] == 'target'
                and float64_from_hex(objective['target']['si_value_f64']) == 0):
            declarations[objective['id']] = raw
            modes.append('squared_magnitude_zero_target')
        else:
            declarations[objective['id']] = quantity
            modes.append('value')
    quantity_program = compile_quantity_program(plan_document, declarations, parameter_set,
                                                default_view=default_view, mesh=mesh,
                                                bounds=bounds, precision=precision, authorized=authorized, program_owner=program_owner, admit_program=admit_program)
    return quantity_program, tuple(modes)


def compile_objective_program(plan_document, optimization_spec, parameter_set, *, default_view=None, mesh=None,
                              bounds=None, precision='float64', authorized=frozenset(),
                              program_owner=None, admit_program=False):
    quantity_program, modes = compile_objective_quantities(
        plan_document, optimization_spec, parameter_set, default_view=default_view, mesh=mesh,
        bounds=bounds, precision=precision, authorized=authorized,
        program_owner=program_owner, admit_program=admit_program)
    objectives = tuple(optimization_spec['objectives'])
    loss = compile_objective_loss(objectives, operand_modes=modes)
    identity = {'schema': 'scnsim.objective_program', 'schema_version': 1,
                'quantity_program': json.loads(quantity_program.identity_bytes),
                'objectives': objectives, 'operand_modes': modes}
    return ObjectiveProgram(quantity_program, loss, canonical_json_bytes(identity))
