"""Measure the public evidence and failure surface of every operation family.

Run from the repository root:

    uv run --locked python measure.py <output.json> [case-name-prefix ...]

For each case the script runs the same public call in Memory and in a Workspace,
reads the Workspace Result again after `run.close()`, again through `run.resolve`
on a reopened Workspace, and again through `run.resolve` in a fresh process
(`measure.py --child ...`). Only public SCNSim calls are used. The summary of
each reading is written to <output.json>; `compare.py` builds the tables.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import (array_bytes, coupled_plan, failure_row, json_bytes, keys_of, open_run, primitive_plan,
                 public_names, tree, u)

from scnsim import (CMAESSpec, CircuitPlan, CostObjective, DiagonalRootSpec, DirectSolveSpec,
                    HybridizedPoleSpec, JVP, VJP, Jacobian, LBFGSBSpec, OperatorElementRootSpec, OperatorSpec,
                    OptimizationSpec, OptimizationVariable, OutputCoordinate, ParameterDefinitions,
                    ParameterSet, ParameterSpace, ParameterSpec, ReductionPipeline,
                    ResidueNormalizedCouplingSpec, ResponseElementSpec, SensitivitySpec,
                    StaticCapacitanceElementSpec, StaticCapacitanceSpec, TransferZeroSpec, components)

HUGE = 1e308  # fF; the value the evidence tests use to make a point fail


def sha(value):
    return hashlib.sha256(json.dumps(tree(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ fixtures
def repeated_mode_plan():
    """Two uncoupled identical tanks (tests/test_root_certificates.py): the pole is not simple."""
    plan = CircuitPlan(id="repeated_physical_pole")
    nodes = tuple(plan.bus(id=name) for name in ("a", "b"))
    for name, node in zip(("a", "b"), nodes):
        c = plan.add(components.capacitor(id=f"c_{name}", capacitance=80 * u.fF))
        l = plan.add(components.inductor(id=f"l_{name}", inductance=8 * u.nH))
        plan.parallel(id=f"lc_{name}", start=node, branches=((c,), (l,)), end=plan.ground)
    return plan, tuple(node.node for node in nodes), None


def constant_cross_plan():
    """A purely inductive cross entry (tests/test_root_certificates.py): zero frequency slope."""
    plan = CircuitPlan(id="constant_cross_entry")
    a, b = plan.bus(id="a"), plan.bus(id="b")
    for name, node in (("a", a), ("b", b)):
        c = plan.add(components.capacitor(id=f"c_{name}", capacitance=80 * u.fF))
        l = plan.add(components.inductor(id=f"l_{name}", inductance=8 * u.nH))
        plan.parallel(id=f"lc_{name}", start=node, branches=((c,), (l,)), end=plan.ground)
    cross = plan.add(components.inductor(id="cross", inductance=500 * u.nH))
    plan.series(id="cross_branch", start=a, elements=(cross,), end=b)
    return plan, (a.node, b.node), None


def underflow_plan():
    """Residue slopes whose product underflows (tests/test_root_certificates.py)."""
    plan = CircuitPlan(id="residue_denominator_underflow")
    a, b = plan.bus(id="a"), plan.bus(id="b")
    for name, node, inductance in (("a", a, 1e300), ("b", b, 2e300)):
        c = plan.add(components.capacitor(id=f"c_{name}", capacitance=1e-300 * u.farad))
        l = plan.add(components.inductor(id=f"l_{name}", inductance=inductance * u.henry))
        plan.parallel(id=f"lc_{name}", start=node, branches=((c,), (l,)), end=plan.ground)
    return plan, (a.node, b.node), None


class Request:
    def __init__(self, op, *args, **kwargs):
        self.op, self.args, self.kwargs = op, args, kwargs


def execute(run, request):
    if request.op == "solve":
        return run.solve(*request.args, **request.kwargs)
    if request.op == "evaluate":
        return run.evaluate(*request.args, **request.kwargs)
    if request.op == "sensitivity":
        return run.sensitivity(*request.args, **request.kwargs)
    if request.op == "optimize":
        return run.optimize(*request.args, progress=False, **request.kwargs)
    raise ValueError(request.op)


def resolve(run, request):
    return run.resolve(*request.args, **request.kwargs)


# ------------------------------------------------------------------ cases
def _grid(parameter, values):
    return ParameterSpace.grid(axes={parameter: values * u.fF})


def _point(parameter, value):
    return ParameterSet({parameter: value * u.fF})


def _view(run, *coordinates):
    return run.original.reduce(ReductionPipeline().retain(*coordinates))


def _root_spec(coordinate):
    return DiagonalRootSpec(coordinate=coordinate, root_hint=6.0 * u.GHz)


def _coupling_spec(coordinates, hint=6.2):
    return ResidueNormalizedCouplingSpec(
        branch_a=DiagonalRootSpec(coordinate=coordinates[0], root_hint=hint * u.GHz),
        branch_b=DiagonalRootSpec(coordinate=coordinates[1], root_hint=hint * u.GHz),
        frequency=5.5 * u.GHz)


def _zero_spec(family="S"):
    return TransferZeroSpec(anchor=6.0 * u.GHz, family=family, input_coordinate="signal_in",
                            output_coordinate="signal_in")


def _optimization(parameter, quantity, target, optimizer, bounds=(80.0, 140.0)):
    return OptimizationSpec(
        variables=(OptimizationVariable(parameter=parameter, bounds=(bounds[0] * u.fF, bounds[1] * u.fF)),),
        objectives=(CostObjective(id="objective", quantity=quantity, target=target,
                                  weight=1.0 * u.dimensionless),),
        optimizer=optimizer)


_CMA = lambda: CMAESSpec(seed=7, population_size=4, max_evaluations=8)
_LBFGSB = lambda: LBFGSBSpec(max_evaluations=8, max_iterations=4, ftol=1e-12, gtol=1e-9, max_line_search=5,
                             commit_every_iterations=1)

CASES = {}


def case(name, fixture):
    def register(builder):
        CASES[name] = (fixture, builder)
        return builder
    return register


# ---- Direct solve
@case("direct/single", primitive_plan)
def _(run, c, p):
    return Request("solve", run.original, DirectSolveSpec(frequencies=[5.5, 6.0, 6.0, 6.5] * u.GHz))


@case("direct/sweep", primitive_plan)
def _(run, c, p):
    return Request("solve", run.original, DirectSolveSpec(frequencies=[5.5, 6.0, 6.0, 6.5] * u.GHz),
                   parameters=_grid(p, [100.0, 110.0, 100.0]))


@case("direct/single-fail", primitive_plan)
def _(run, c, p):
    return Request("solve", run.original, DirectSolveSpec(frequencies=[5.5, 6.0] * u.GHz),
                   parameters=_point(p, HUGE))


@case("direct/single-fail-frequency", primitive_plan)
def _(run, c, p):
    return Request("solve", run.original, DirectSolveSpec(frequencies=[5.5e9, 6.0e9, 6.5e9, 1.0e308] * u.Hz))


@case("direct/sweep-fail", primitive_plan)
def _(run, c, p):
    return Request("solve", run.original, DirectSolveSpec(frequencies=[5.5, 6.0] * u.GHz),
                   parameters=_grid(p, [110.0, HUGE, 110.0]))


# ---- Operator
@case("operator/single", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), OperatorSpec(frequencies=[5.5, 6.0] * u.GHz))


@case("operator/sweep", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), OperatorSpec(frequencies=[5.5, 6.0, 6.5] * u.GHz),
                   parameters=_grid(p, [100.0, 110.0, 100.0]))


@case("operator/single-fail", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), OperatorSpec(frequencies=[5.5, 6.0] * u.GHz),
                   parameters=_point(p, HUGE))


@case("operator/sweep-fail", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), OperatorSpec(frequencies=[5.5, 6.0] * u.GHz),
                   parameters=_grid(p, [110.0, HUGE, 110.0]))


# ---- Response element
def _response(c, family="Z"):
    return ResponseElementSpec(family=family, input_coordinate=c, output_coordinate=c, frequency=6.0 * u.GHz)


@case("response/single", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _response(c))


@case("response/sweep", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _response(c), parameters=_grid(p, [100.0, 110.0, 100.0]))


@case("response/single-fail", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _response(c), parameters=_point(p, HUGE))


@case("response/sweep-fail", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _response(c), parameters=_grid(p, [110.0, HUGE, 110.0]))


# ---- Static capacitance
@case("static/matrix", primitive_plan)
def _(run, c, p):
    return Request("evaluate", run.original, StaticCapacitanceSpec())


@case("static/matrix-retained", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), StaticCapacitanceSpec())


@case("static/element", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), StaticCapacitanceElementSpec(row=c, column=c))


@case("static/matrix-fail", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), StaticCapacitanceSpec(), parameters=_point(p, HUGE))


@case("static/element-fail", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), StaticCapacitanceElementSpec(row=c, column=c),
                   parameters=_point(p, HUGE))


@case("static/sweep-refused", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), StaticCapacitanceSpec(), parameters=_grid(p, [100.0, 110.0]))


# ---- Diagonal root
@case("diagonal_root/single", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _root_spec(c))


@case("diagonal_root/single-at-point", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _root_spec(c), parameters=_point(p, 100.0))


@case("diagonal_root/sweep", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _root_spec(c), parameters=_grid(p, [100.0, 110.0, 100.0]))


@case("diagonal_root/single-fail-point", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _root_spec(c), parameters=_point(p, HUGE))


@case("diagonal_root/single-fail-baseline", lambda: primitive_plan(capacitance=HUGE))
def _(run, c, p):
    return Request("evaluate", _view(run, c), _root_spec(c))


@case("diagonal_root/sweep-fail-point", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _root_spec(c), parameters=_grid(p, [100.0, 110.0, HUGE, 100.0]))


@case("diagonal_root/sweep-fail-baseline", lambda: primitive_plan(capacitance=HUGE))
def _(run, c, p):
    return Request("evaluate", _view(run, c), _root_spec(c), parameters=_grid(p, [100.0, 110.0]))


# ---- Operator element root
def _element_spec(c):
    return OperatorElementRootSpec(row=c[0], column=c[0], root_hint=6.2 * u.GHz)


@case("operator_element_root/single", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _element_spec(c))


@case("operator_element_root/sweep", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _element_spec(c), parameters=_grid(p, [78.0, 80.0, 78.0]))


@case("operator_element_root/single-fail-point", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _element_spec(c), parameters=_point(p, HUGE))


@case("operator_element_root/single-fail-baseline", constant_cross_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), OperatorElementRootSpec(row=c[0], column=c[1],
                                                                       root_hint=6.3 * u.GHz))


@case("operator_element_root/sweep-fail-point", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _element_spec(c), parameters=_grid(p, [78.0, 80.0, HUGE, 78.0]))


# ---- Hybridized pole
def _pole_spec(c, anchor=6.1):
    return HybridizedPoleSpec(coordinates=c, anchor=anchor * u.GHz)


@case("pole/single", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _pole_spec(c))


@case("pole/single-at-point", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _pole_spec(c), parameters=_point(p, 78.0))


@case("pole/sweep", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _pole_spec(c), parameters=_grid(p, [78.0, 80.0, 78.0]))


@case("pole/single-fail-point", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _pole_spec(c), parameters=_point(p, HUGE))


@case("pole/single-fail-baseline", repeated_mode_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _pole_spec(c, anchor=6.3))


@case("pole/sweep-fail-point", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _pole_spec(c), parameters=_grid(p, [78.0, 80.0, HUGE, 78.0]))


# ---- Transfer zero
@case("transfer_zero/single", primitive_plan)
def _(run, c, p):
    return Request("evaluate", run.original, _zero_spec())


@case("transfer_zero/single-at-point", primitive_plan)
def _(run, c, p):
    return Request("evaluate", run.original, _zero_spec(), parameters=_point(p, 100.0))


@case("transfer_zero/sweep", primitive_plan)
def _(run, c, p):
    return Request("evaluate", run.original, _zero_spec(), parameters=_grid(p, [100.0, 110.0, 100.0]))


@case("transfer_zero/single-fail-point", primitive_plan)
def _(run, c, p):
    return Request("evaluate", run.original, _zero_spec(), parameters=_point(p, HUGE))


@case("transfer_zero/single-fail-baseline", primitive_plan)
def _(run, c, p):
    return Request("evaluate", run.original, _zero_spec("Y"))


@case("transfer_zero/sweep-fail-point", primitive_plan)
def _(run, c, p):
    return Request("evaluate", run.original, _zero_spec(), parameters=_grid(p, [100.0, 110.0, HUGE, 100.0]))


@case("transfer_zero/sweep-fail-baseline", primitive_plan)
def _(run, c, p):
    return Request("evaluate", run.original, _zero_spec("Y"), parameters=_grid(p, [100.0, 120.0]))


# ---- Residue-normalised coupling
@case("coupling/single", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _coupling_spec(c))


@case("coupling/single-at-point", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _coupling_spec(c), parameters=_point(p, 78.0))


@case("coupling/single-pole-branches", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), ResidueNormalizedCouplingSpec(
        branch_a=HybridizedPoleSpec(coordinates=c, anchor=6.1 * u.GHz),
        branch_b=HybridizedPoleSpec(coordinates=c, anchor=6.4 * u.GHz), frequency=5.5 * u.GHz))


@case("coupling/sweep", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _coupling_spec(c), parameters=_grid(p, [78.0, 80.0, 78.0]))


@case("coupling/single-fail-point", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _coupling_spec(c), parameters=_point(p, HUGE))


@case("coupling/single-fail-baseline", underflow_plan)
def _(run, c, p):
    import math
    return Request("evaluate", _view(run, *c), ResidueNormalizedCouplingSpec(
        branch_a=DiagonalRootSpec(coordinate=c[0], root_hint=(1 / (2 * math.pi)) * u.Hz),
        branch_b=DiagonalRootSpec(coordinate=c[1], root_hint=(1 / (math.sqrt(2) * 2 * math.pi)) * u.Hz),
        frequency=1 * u.Hz))


@case("coupling/sweep-fail-point", coupled_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, *c), _coupling_spec(c), parameters=_grid(p, [78.0, 80.0, HUGE, 78.0]))


# ---- Sensitivity
def _sensitivity(quantities, p, form, weight=None):
    label = next(iter(quantities))
    if form == "jacobian":
        derivative = Jacobian()
    elif form == "jvp":
        derivative = JVP(direction={p: p.baseline})
    else:
        derivative = VJP(weights={OutputCoordinate(quantity_label=label): weight})
    return SensitivitySpec(quantities=quantities, wrt=(p,), derivative=derivative)


for _form in ("jacobian", "jvp", "vjp"):
    def _make(form):
        @case("sensitivity/response-" + form, primitive_plan)
        def _(run, c, p):
            response = ResponseElementSpec(family="Y", input_coordinate="signal_in",
                                           output_coordinate="signal_in", frequency=5.5 * u.GHz)
            return Request("sensitivity", _sensitivity({"response": response.real.on(run.original)}, p,
                                                       form, 1.0 / u.S))

        @case("sensitivity/root-" + form, primitive_plan)
        def _(run, c, p):
            return Request("sensitivity", _sensitivity({"f": _root_spec(c).frequency.on(_view(run, c))}, p,
                                                       form, 1.0 / u.Hz))

        @case("sensitivity/pole-" + form, coupled_plan)
        def _(run, c, p):
            return Request("sensitivity", _sensitivity({"f": _pole_spec(c).frequency.on(_view(run, *c))}, p,
                                                       form, 1.0 / u.Hz))
    _make(_form)


@case("sensitivity/coupling-jacobian", coupled_plan)
def _(run, c, p):
    return Request("sensitivity", _sensitivity({"g": _coupling_spec(c).real.on(_view(run, *c))}, p, "jacobian"))


@case("sensitivity/zero-jacobian", primitive_plan)
def _(run, c, p):
    return Request("sensitivity", _sensitivity({"z": _zero_spec().frequency.on(run.original)}, p, "jacobian"))


@case("sensitivity/response-fail", primitive_plan)
def _(run, c, p):
    response = ResponseElementSpec(family="Z", input_coordinate=c, output_coordinate=c, frequency=6.0 * u.GHz)
    return Request("sensitivity", _sensitivity({"response": response.real.on(_view(run, c))}, p, "jacobian"),
                   parameters=_point(p, HUGE))


@case("sensitivity/root-fail-point", primitive_plan)
def _(run, c, p):
    return Request("sensitivity", _sensitivity({"f": _root_spec(c).frequency.on(_view(run, c))}, p, "jacobian"),
                   parameters=_point(p, HUGE))


# ---- Optimization
@case("optimization/root-cma", primitive_plan)
def _(run, c, p):
    return Request("optimize", _view(run, c), _optimization(p, _root_spec(c).frequency, 6.2 * u.GHz, _CMA()))


@case("optimization/root-lbfgsb", primitive_plan)
def _(run, c, p):
    return Request("optimize", _view(run, c), _optimization(p, _root_spec(c).frequency, 6.2 * u.GHz, _LBFGSB()))


@case("optimization/pole-cma", coupled_plan)
def _(run, c, p):
    return Request("optimize", _view(run, *c), _optimization(p, _pole_spec(c).frequency, 6.2 * u.GHz, _CMA(),
                                                           bounds=(70.0, 90.0)))


@case("optimization/coupling-cma", coupled_plan)
def _(run, c, p):
    return Request("optimize", _view(run, *c), _optimization(p, _coupling_spec(c).magnitude,
                                                           2.0e8 * u.rad / u.s, _CMA(), bounds=(70.0, 90.0)))


@case("optimization/zero-cma", primitive_plan)
def _(run, c, p):
    return Request("optimize", run.original, _optimization(p, _zero_spec().frequency, 6.2 * u.GHz, _CMA()))


@case("optimization/response-cma", primitive_plan)
def _(run, c, p):
    response = ResponseElementSpec(family="Y", input_coordinate="signal_in", output_coordinate="signal_in",
                                   frequency=5.5 * u.GHz)
    return Request("optimize", run.original, _optimization(p, response.imag, 1.0e-3 * u.S, _CMA()))


@case("optimization/root-cma-failing-trials", primitive_plan)
def _(run, c, p):
    return Request("optimize", _view(run, c), _optimization(p, _root_spec(c).frequency, 6.2 * u.GHz, _CMA(),
                                                          bounds=(80.0, HUGE)))


@case("optimization/root-cma-fail-baseline", lambda: primitive_plan(capacitance=HUGE))
def _(run, c, p):
    return Request("optimize", _view(run, c), _optimization(p, _root_spec(c).frequency, 6.2 * u.GHz, _CMA(),
                                                          bounds=(80.0, HUGE)))


# ---- Second batch: further failing inputs and explicit root search
def floating_plan():
    """The two-node fixture plus an isolated capacitor pair (benchmarks/circuits.py, F1-singular-static)."""
    plan, coordinate, parameter = primitive_plan(plan_id="primitive_with_floating_pair")
    a, b = plan.bus(id="floating_a"), plan.bus(id="floating_b")
    floating = plan.add(components.capacitor(id="isolated_capacitor", capacitance=6.0 * u.fF))
    plan.series(id="isolated_pair", start=a, elements=(floating,), end=b)
    return plan, coordinate, parameter


@case("static/matrix-fail-singular", floating_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), StaticCapacitanceSpec())


@case("static/element-fail-singular", floating_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), StaticCapacitanceElementSpec(row=c, column=c))


@case("operator/single-fail-singular", floating_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), OperatorSpec(frequencies=[5.5, 6.0] * u.GHz))


@case("operator/sweep-fail-singular", floating_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), OperatorSpec(frequencies=[5.5, 6.0] * u.GHz),
                   parameters=_grid(p, [100.0, 110.0]))


@case("response/single-fail-singular", floating_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _response(c))


@case("direct/single-fail-singular", floating_plan)
def _(run, c, p):
    return Request("solve", _view(run, c), DirectSolveSpec(frequencies=[5.5, 6.0] * u.GHz))


@case("diagonal_root/single-fail-hint", primitive_plan)
def _(run, c, p):
    # benchmarks/circuits.py, F3-root-nonconvergence: a finite far local seed.
    return Request("evaluate", _view(run, c), DiagonalRootSpec(coordinate=c, root_hint=1.0e30 * u.Hz))


def _searched_root(c):
    from scnsim import RootSearchSpec
    return DiagonalRootSpec(coordinate=c, root_hint=6.0 * u.GHz, search=RootSearchSpec(
        method="finite_seeds", tie="source_order", hint_frequency_factor=10, seeds=[6.0 * u.GHz]))


def _searched_zero():
    from scnsim import RootSearchSpec
    return TransferZeroSpec(anchor=6.0 * u.GHz, family="Z", input_coordinate="signal_in",
                            output_coordinate="signal_in", search=RootSearchSpec(
                                method="polynomial_transfer_zero", hint_frequency_factor=10, tie="unavailable"))


@case("diagonal_root/single-search", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _searched_root(c))


@case("diagonal_root/single-search-at-point", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, c), _searched_root(c), parameters=_point(p, 100.0))


@case("diagonal_root/single-search-fail-point", primitive_plan)
def _(run, c, p):
    # An explicit search initialises the branch at the requested point.
    return Request("evaluate", _view(run, c), _searched_root(c), parameters=_point(p, HUGE))


@case("transfer_zero/single-search-at-point", primitive_plan)
def _(run, c, p):
    # The call of docs/guides/roots-and-branch-selection.qmd (`evaluate_notch`).
    return Request("evaluate", _view(run, "signal_in"), _searched_zero(), parameters=_point(p, 100.0))


@case("transfer_zero/single-search-fail-point", primitive_plan)
def _(run, c, p):
    return Request("evaluate", _view(run, "signal_in"), _searched_zero(), parameters=_point(p, HUGE))


@case("sensitivity/root-fail-baseline", lambda: primitive_plan(capacitance=HUGE))
def _(run, c, p):
    return Request("sensitivity", _sensitivity({"f": _root_spec(c).frequency.on(_view(run, c))}, p, "jacobian"))


# ------------------------------------------------------------------ summaries
_VALUE_FIELDS = ("root", "frequency", "linewidth", "slope", "value", "magnitude", "real", "imag", "zero",
                 "numerator_slope", "denominator", "coupling", "branch_a_residue", "branch_b_residue",
                 "evaluation_omega", "family", "matrix", "row_coordinates", "column_coordinates",
                 "zero_charge", "row", "column")


def _nested_keys(value, depth=2):
    """Keys of an evidence mapping, with the keys of nested mappings to a small depth."""
    from lib import is_array_record
    if not isinstance(value, Mapping) or is_array_record(value) or depth == 0:
        return None
    row = {}
    for key in sorted(value, key=str):
        child = value[key]
        if isinstance(child, Mapping) and not is_array_record(child):
            row[str(key)] = _nested_keys(child, depth - 1)
        elif isinstance(child, (tuple, list)) and child and isinstance(child[0], Mapping) \
                and not is_array_record(child[0]):
            row[str(key)] = {"[%d]" % len(child): _nested_keys(child[0], depth - 1)}
        else:
            row[str(key)] = None
    return row


def evidence_row(value):
    if not isinstance(value, Mapping):
        return {"present": False, "type": type(value).__name__}
    return {"present": True, "key_count": len(value), "keys": keys_of(value),
            "nested_keys": _nested_keys(value), "array_bytes": array_bytes(value),
            "json_bytes": json_bytes(value), "sha": sha(value), "tree": tree(value),
            "per_key_array_bytes": {str(k): array_bytes(v) for k, v in value.items()}}


def evaluation_row(evaluation):
    """Public fields of the in-memory numerical body (`result.evaluation`, `trial.bodies[i]`)."""
    row = {"class": type(evaluation).__name__, "id": evaluation.id,
           "set_fields": [f.name for f in fields(evaluation)
                          if f.name not in ("id", "failure", "evidence_bytes")
                          and getattr(evaluation, f.name) is not None],
           "evidence_bytes_len": len(evaluation.evidence_bytes)}
    document = json.loads(evaluation.evidence_bytes)
    row["evidence_bytes"] = evidence_row(document)
    if evaluation.null_vector is not None:
        row["null_vector"] = tree(evaluation.null_vector)
    if evaluation.failure is not None:
        failure = evaluation.failure
        row["failure"] = {"kind": failure.kind, "stage": failure.stage, "detail": failure.detail,
                          "evidence_bytes_len": len(failure.evidence_bytes),
                          "evidence_bytes": evidence_row(json.loads(failure.evidence_bytes))}
    return row


def values_row(result):
    row = {}
    for name in _VALUE_FIELDS:
        if hasattr(result, name):
            value = getattr(result, name)
            if value is not None:
                row[name] = tree(value)
    for name in ("s", "y", "z"):
        family = getattr(result, name, None)
        if family is not None:
            row[name] = {"matrix": tree(family.view.matrix), "frequencies": tree(family.view.frequencies),
                         "coordinates": list(family.view.coordinates)}
    points = getattr(result, "operator_points", None) or (
        getattr(result, "points", None) if type(result).__name__ == "OperatorResult" else None)
    if points:
        row["operator_points"] = [{"frequency": tree(point.frequency), "matrix": tree(point.matrix),
                                   "coordinates": list(point.coordinates)} for point in points]
    return row


def analysis_row(result):
    """One point Result of solve/evaluate, in either storage."""
    row = {"class": type(result).__name__, "public": public_names(result)}
    values = values_row(result)
    row["values_sha"] = sha(values)
    row["values"] = {key: value for key, value in values.items() if key not in ("s", "y", "z", "operator_points")}
    row["evidence"] = evidence_row(getattr(result, "evidence", None))
    if hasattr(result, "evaluation"):  # in-memory analysis Result
        row["status"] = result.status
        row["succeeded"] = result.succeeded
        row["returned_failure"] = failure_row(result.failure)
        row["evaluation"] = evaluation_row(result.evaluation)
        row["dependencies"] = [{"locator": tree(item.locator), "evaluation": evaluation_row(item.result)}
                               for item in result.dependencies]
        row["context_keys"] = sorted(result.context)
    if hasattr(result, "metadata") and isinstance(result.metadata, Mapping):  # in-memory Direct
        row["metadata_keys"] = sorted(result.metadata)
    try:
        benchmark = result.benchmark
    except Exception:
        benchmark = None
    if benchmark is not None and is_dataclass(benchmark):
        row["benchmark"] = {"status": benchmark.status, "failure": tree(benchmark.failure)}
    return row


def sweep_row(result):
    row = {"class": type(result).__name__, "public": public_names(result), "points": []}
    for point in result.points:
        item = {"class": type(point).__name__, "public": public_names(point), "succeeded": point.succeeded,
                "source_index": tree(point.source_index)}
        if point.succeeded:
            item["result"] = analysis_row(point.result)
            item["failure_attribute"] = None if point.failure is None else "set"
        else:
            item["failure"] = failure_row(point.failure)
            try:
                point.result
                item["result_access"] = "returned"
            except BaseException as error:
                item["result_access"] = "raises " + type(error).__name__
            if hasattr(point, "analysis"):
                item["analysis"] = analysis_row(point.analysis)
        row["points"].append(item)
    if hasattr(result, "baseline"):  # in-memory sweep
        baseline = result.baseline
        row["baseline"] = None if baseline is None else {
            "succeeded": baseline.succeeded, "analysis": analysis_row(baseline.analysis)}
        row["metadata_keys"] = sorted(result.metadata)
        row["summary"] = tree(result.summary)
    return row


def _block(block):
    row = {"class": type(block).__name__, "block_id": block.block_id, "status": block.status,
           "failure": tree(block.failure)}
    for name in ("value", "values"):
        if hasattr(block, name) and getattr(block, name) is not None:
            row[name] = tree(getattr(block, name))
    return row


def _certificates(certificates):
    return {"count": len(certificates), "array_bytes": sum(array_bytes(tree_fields(c)) for c in certificates),
            "roles": dict(Counter(getattr(c, "role", "?") for c in certificates)),
            "classes": dict(Counter(type(c).__name__ for c in certificates)), "sha": sha([tree(c) for c in certificates])}


def tree_fields(value):
    if is_dataclass(value):
        return {f.name: getattr(value, f.name) for f in fields(value)}
    if hasattr(value, "_fields"):
        return {name: getattr(value, name) for name in value._fields}
    return {"value": value}


def sensitivity_row(result):
    row = {"class": type(result).__name__, "public": public_names(result), "status": result.status,
           "primal": [_block(block) for block in result.primal], "operations": {}}
    for kind, projection in result.operations.items():
        row["operations"][kind] = {"status": projection.status, "diagnostics": tree(projection.diagnostics),
                                   "blocks": [_block(block) for block in projection.blocks]}
    row["values_sha"] = sha({"primal": row["primal"], "operations": row["operations"]})
    if hasattr(result, "certificates"):
        row["certificates"] = _certificates(result.certificates)
        row["summary"] = tree(result.summary)
    return row


def _failure_mapping(value):
    if value is None:
        return None
    if isinstance(value, Mapping):
        row = {key: value[key] for key in ("kind", "stage", "category", "detail", "message", "reason")
               if key in value}
        row["keys"] = keys_of(value)
        evidence = value.get("evidence")
        if isinstance(evidence, Mapping):
            row["evidence"] = evidence_row(evidence)
        return row
    return failure_row(value)


def optimization_row(result):
    row = {"class": type(result).__name__, "public": public_names(result)}
    if hasattr(result, "trials"):  # in-memory
        row["outcome"] = result.outcome
        trials = []
        for trial in result.history:
            trials.append({
                "kind": trial.kind, "status": trial.status, "evaluation_ordinal": trial.evaluation_ordinal,
                "cost": tree(trial.cost), "failure": _failure_mapping(trial.failure),
                "bodies": {str(index): evaluation_row(body) for index, body in trial.bodies.items()},
                "certificates": _certificates(trial.certificates),
                "term_keys": sorted({key for component in trial.objective_components
                                     for term in component["terms"] for key in term})})
        row["history"] = trials
        row["baseline"] = None if result.baseline is None else result.baseline.status
        row["best_ordinal"] = None if result.best is None else result.best.evaluation_ordinal
        row["best_cost"] = None if result.best is None else tree(result.best.cost)
        row["summary"] = tree(result.summary)
        return row
    row["ledger_length"] = len(result.ledger)
    row["best_ordinal"] = None if result.best is None else result.best.evaluation_ordinal
    row["best_cost"] = None if result.best is None else tree(result.best.cost)
    history = []
    for stage in result.stages:
        for item in stage.history:
            entry = {"keys": sorted(item), "kind": item.get("kind", "evaluation"), "status": item["status"],
                     "evaluation_ordinal": item["evaluation_ordinal"], "cost_f64": item.get("cost_f64"),
                     "failure": _failure_mapping(item.get("failure"))}
            dependencies = item.get("dependencies")
            if isinstance(dependencies, Mapping):
                entry["dependencies"] = {}
                for identity, body in dependencies.items():
                    entry["dependencies"][identity[:12]] = {
                        "keys": sorted(body), "evidence": evidence_row(body.get("evidence")),
                        "array_bytes": array_bytes(body), "json_bytes": json_bytes(body)}
            terms = []
            for component in item.get("objectives", item.get("components", ())):
                for term in component.get("terms", ()):
                    terms.append({"keys": sorted(term),
                                  "evidence": evidence_row(term.get("evidence")) if "evidence" in term else None,
                                  "residue_coupling_evidence": sorted(term["residue_coupling_evidence"])
                                  if "residue_coupling_evidence" in term else None,
                                  "failure": _failure_mapping(term.get("failure"))})
            entry["terms"] = terms
            history.append(entry)
    row["history"] = history
    return row


def summarise(result):
    name = type(result).__name__
    if "Optimization" in name:
        return optimization_row(result)
    if "Sensitivity" in name:
        return sensitivity_row(result)
    if "Sweep" in name:
        return sweep_row(result)
    return analysis_row(result)


def reading(call):
    """Run one public call; describe what it returned or raised."""
    started = time.perf_counter()
    try:
        result = call()
    except BaseException as error:  # typed SCNSim failures and anything else
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return None, {"outcome": "raised", "failure": failure_row(error),
                      "seconds": round(time.perf_counter() - started, 3)}
    row = {"outcome": "returned", "result": summarise(result), "seconds": round(time.perf_counter() - started, 3)}
    return result, row


def child(name, directory):
    fixture, builder = CASES[name]
    plan, coordinates, parameter = fixture()
    run = open_run(plan, "workspace", directory)
    try:
        request = builder(run, coordinates, parameter)
        _, row = reading(lambda: resolve(run, request))
    finally:
        run.close()
    return row


def measure(name, root):
    fixture, builder = CASES[name]
    record = {}
    # Memory
    plan, coordinates, parameter = fixture()
    run = open_run(plan, "memory")
    try:
        request = builder(run, coordinates, parameter)
        result, record["memory"] = reading(lambda: execute(run, request))
    finally:
        run.close()
    if result is not None:
        record["memory_after_close"] = {"outcome": "returned", "result": summarise(result)}
    # Workspace
    directory = Path(root) / name.replace("/", "__")
    directory.mkdir(parents=True)
    plan, coordinates, parameter = fixture()
    run = open_run(plan, "workspace", directory)
    try:
        request = builder(run, coordinates, parameter)
        result, record["workspace"] = reading(lambda: execute(run, request))
    finally:
        run.close()
    if result is not None:
        record["workspace_after_close"] = {"outcome": "returned", "result": summarise(result)}
    # Reopened Workspace, same process
    record["workspace_reopened"] = child(name, directory)
    return record


def strip_seconds(value):
    if isinstance(value, Mapping):
        return {key: strip_seconds(item) for key, item in value.items() if key != "seconds"}
    if isinstance(value, list):
        return [strip_seconds(item) for item in value]
    return value


def main():
    if sys.argv[1] == "--child":
        names = json.loads(sys.argv[3])
        rows = {}
        for name in names:
            try:
                rows[name] = child(name, Path(sys.argv[2]) / name.replace("/", "__"))
            except BaseException as error:
                rows[name] = {"outcome": "script_error", "error": repr(error), "trace": traceback.format_exc()}
        sys.stdout.write("\n@@CHILD@@" + json.dumps(rows))
        return
    output = Path(sys.argv[1])
    prefixes = sys.argv[2:]
    # A selector ending in "$" names one case exactly; any other selector is a prefix.
    names = [name for name in CASES if not prefixes or any(
        name == prefix[:-1] if prefix.endswith("$") else name.startswith(prefix) for prefix in prefixes)]
    report = {}
    with TemporaryDirectory() as root:
        for name in names:
            started = time.perf_counter()
            try:
                report[name] = measure(name, root)
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                report[name] = {"script_error": repr(error), "trace": traceback.format_exc()}
            if "script_error" not in report[name]:
                # Fresh process: a new interpreter reopens the Workspace and resolves the same request.
                completed = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--child", root, json.dumps([name])],
                    capture_output=True, text=True, cwd=os.getcwd())
                if completed.returncode != 0 or "@@CHILD@@" not in completed.stdout:
                    print("fresh process failed", completed.returncode, completed.stderr[-3000:], flush=True)
                else:
                    report[name]["workspace_fresh_process"] = json.loads(
                        completed.stdout.split("@@CHILD@@", 1)[1])[name]
            print("%-48s %6.1fs  %s" % (name, time.perf_counter() - started, " | ".join(
                "%s=%s" % (key, value.get("outcome")) for key, value in report[name].items()
                if isinstance(value, Mapping))), flush=True)
            output.write_text(json.dumps(report, sort_keys=True, indent=1))
    print("wrote", output.name)


if __name__ == "__main__":
    main()
