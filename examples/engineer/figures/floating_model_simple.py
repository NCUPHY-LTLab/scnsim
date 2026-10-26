"""Four-component floating LC teaching Plan; no numerical execution."""
from scnsim import CircuitPlan, components, units as u


def build_model():
    plan = CircuitPlan(id="floating_lc_simple")
    plus = plan.bus(id="Phi_a")
    minus = plan.bus(id="Phi_b")
    Ca = plan.add(components.capacitor(id="Ca", capacitance=45.0*u.fF))
    Cb = plan.add(components.capacitor(id="Cb", capacitance=42.0*u.fF))
    Cm = plan.add(components.capacitor(id="Cm", capacitance=16.0*u.fF))
    L = plan.add(components.inductor(id="L", inductance=7.0*u.nH))
    plan.parallel(
        id="mutual_lc", start=plus, branches=((Cm,), (L,)), end=minus,
    )
    plan.branch(id="stray_a", at=plus, elements=(Ca,), end=plan.ground)
    plan.branch(id="stray_b", at=minus, elements=(Cb,), end=plan.ground)
    return plan, plus, minus
