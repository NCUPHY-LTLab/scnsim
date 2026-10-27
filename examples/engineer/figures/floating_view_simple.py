"""Source-checked continuation for floating_model_simple; not executed."""
from scnsim import CircuitRun, DiagonalRootSpec, ReductionPipeline, units as u
from floating_model_simple import build_model

plan, plus, minus = build_model()
run = CircuitRun(plan=plan, storage="memory")
try:
    view = run.original.reduce(
        ReductionPipeline()
        .transform_pair(plus.node, minus.node, id="floating")
        .retain("floating.differential")
    )
    root = run.evaluate(view, DiagonalRootSpec(
        coordinate="floating.differential", root_hint=10.0*u.GHz,
    ))
    frequency = root.frequency
finally:
    run.close()

# differential = Phi_a - Phi_b
# common = (45*Phi_a + 42*Phi_b)/87
