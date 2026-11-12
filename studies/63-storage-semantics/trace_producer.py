"""Which producer path of PhysicalAnalysis runs for each storage and request form?

Observation only: a profile hook counts calls of a few named functions while a
public call runs. Nothing is replaced. Run from the repository root.
"""
import os
import sys
from collections import Counter
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import coupled_plan, open_run, primitive_plan, u
from scnsim import (DiagonalRootSpec, DirectSolveSpec, OperatorSpec, ParameterSet, ParameterSpace, ReductionPipeline,
                    ResidueNormalizedCouplingSpec, ResponseElementSpec, StaticCapacitanceSpec)

WATCH = {("physical_analysis.py", "_run_native_sweep"), ("physical_analysis.py", "_run_native_direct"),
         ("physical_analysis.py", "evaluate"), ("physical_analysis.py", "_evaluate_native_sweep_point"),
         ("physical_analysis.py", "run"), ("evidence.py", "require_primal"),
         ("evidence.py", "project_native_root"), ("physical_analysis.py", "project_primal_bodies")}


def observed(call):
    counts = Counter()

    def hook(frame, event, arg):
        if event == "call":
            code = frame.f_code
            key = (os.path.basename(code.co_filename), code.co_name)
            if key in WATCH and "scnsim" in code.co_filename:
                counts[key[1]] += 1

    sys.setprofile(hook)
    try:
        call()
    finally:
        sys.setprofile(None)
    return dict(sorted(counts.items()))


def requests(run, fixture):
    plan, coordinates, parameter = fixture
    if isinstance(coordinates, tuple):
        view = run.original.reduce(ReductionPipeline().retain(*coordinates))
        spec = ResidueNormalizedCouplingSpec(
            branch_a=DiagonalRootSpec(coordinate=coordinates[0], root_hint=6.2 * u.GHz),
            branch_b=DiagonalRootSpec(coordinate=coordinates[1], root_hint=6.2 * u.GHz), frequency=5.5 * u.GHz)
        grid = ParameterSpace.grid(axes={parameter: [78.0, 80.0] * u.fF})
        yield "coupling single", lambda: run.evaluate(view, spec)
        yield "coupling single at a point", lambda: run.evaluate(view, spec, parameters=ParameterSet({parameter: 78.0 * u.fF}))
        yield "coupling sweep", lambda: run.evaluate(view, spec, parameters=grid)
        return
    view = run.original.reduce(ReductionPipeline().retain(coordinates))
    root = DiagonalRootSpec(coordinate=coordinates, root_hint=6.0 * u.GHz)
    grid = ParameterSpace.grid(axes={parameter: [100.0, 110.0] * u.fF})
    point = ParameterSet({parameter: 100.0 * u.fF})
    yield "diagonal root single", lambda: run.evaluate(view, root)
    yield "diagonal root single at a point", lambda: run.evaluate(view, root, parameters=point)
    yield "diagonal root sweep", lambda: run.evaluate(view, root, parameters=grid)
    response = ResponseElementSpec(family="Z", input_coordinate=coordinates, output_coordinate=coordinates,
                                   frequency=6.0 * u.GHz)
    yield "response single", lambda: run.evaluate(view, response)
    yield "response sweep", lambda: run.evaluate(view, response, parameters=grid)
    operator = OperatorSpec(frequencies=[5.5, 6.0] * u.GHz)
    yield "operator single", lambda: run.evaluate(view, operator)
    yield "static capacitance single", lambda: run.evaluate(view, StaticCapacitanceSpec())
    direct = DirectSolveSpec(frequencies=[5.5, 6.0] * u.GHz)
    yield "direct single", lambda: run.solve(run.original, direct)
    yield "direct sweep", lambda: run.solve(run.original, direct, parameters=grid)


for fixture_builder in (primitive_plan, coupled_plan):
    for storage in ("memory", "workspace"):
        with TemporaryDirectory() as directory:
            fixture = fixture_builder()
            run = open_run(fixture[0], storage, directory)
            try:
                for label, call in requests(run, fixture):
                    print("%-9s %-34s %s" % (storage, label, observed(call)), flush=True)
            finally:
                run.close()
