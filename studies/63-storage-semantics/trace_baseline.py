"""Where is a failing single-point baseline root raised in each storage? Prints the tracebacks."""
import os
import sys
import traceback
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import open_run, primitive_plan, u
from scnsim import DiagonalRootSpec, ReductionPipeline, SCNSimError

for storage in ("memory", "workspace"):
    with TemporaryDirectory() as directory:
        plan, coordinate, capacitance = primitive_plan(capacitance=1e308)
        run = open_run(plan, storage, directory)
        try:
            view = run.original.reduce(ReductionPipeline().retain(coordinate))
            try:
                run.evaluate(view, DiagonalRootSpec(coordinate=coordinate, root_hint=6.0 * u.GHz))
            except SCNSimError as error:
                print("=====", storage, type(error).__name__, str(error), sorted(error.evidence))
                frames = traceback.extract_tb(error.__traceback__)
                for frame in frames:
                    if "scnsim" in frame.filename:
                        print("   ", frame.filename.split("src/scnsim/")[-1], frame.lineno, frame.name)
        finally:
            run.close()
