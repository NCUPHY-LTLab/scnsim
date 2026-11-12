"""Let SCNSim initialize JAX before a test module calls JAX directly.

``tests/__init__.py`` declares the suite's runtime with
``configure_runtime(cpu_threads=10)``. SCNSim applies a declaration when its
own first numerical call initializes JAX. When other code has initialized JAX
first, SCNSim cannot tell which thread pool is in effect, and every later
SCNSim call in the process raises ``RuntimePreparationError`` ("JAX resources
are already initialized").

A test module that calls JAX directly therefore starts with one small SCNSim
solve. It can then run alone, first, or in any order with the other modules.
"""
from __future__ import annotations

from scnsim import CircuitPlan, CircuitRun, DirectSolveSpec, components, units as u


def initialize_jax_through_scnsim() -> None:
    """Solve one resistor to ground, as the first numerical call of a session would."""
    plan = CircuitPlan(id="jax_runtime")
    node = plan.bus(id="a")
    resistor = plan.add(components.resistor(id="r", resistance=50.0 * u.ohm))
    plan.series(id="load", start=node, elements=(resistor,), end=plan.ground)
    plan.add_port(id="a", at=node, role="terminated", reference_impedance=50.0 * u.ohm)
    run = CircuitRun(plan=plan, storage="memory", precision="float64")
    try:
        run.solve(run.original, DirectSolveSpec(frequencies=[1.0] * u.GHz))
    finally:
        run.close()
