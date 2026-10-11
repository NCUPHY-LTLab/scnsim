"""Call-local implicit differentiation of an already checked complex root.

Newton and its branch/closure certificates remain RootProgram's authority.
The derivative frame differentiates the physical zero equation at that finished
root; it never differentiates Newton history or reruns Newton for a direction.
Physical derivatives use ordinary complex products. Reverse covectors use JAX's
real-loss encoding at the checked graph boundary, not a physical conjugation.
"""
from __future__ import annotations
from dataclasses import dataclass



@dataclass(slots=True)
class ImplicitRootFrame:
    graph: object
    equation_node: int
    omega_index: int
    input_template: object
    _slope: object = None
    _slope_certificates: tuple = ()

    def _frequency_direction(self):
        import jax.numpy as jnp
        from .physical_program import ProgramInputs
        point = self.graph.inputs
        return ProgramInputs(jnp.zeros_like(point.theta),
                             jnp.zeros_like(point.omega_rad_s).at[self.omega_index].set(1),
                             jnp.zeros_like(point.branch_starts_rad_s))

    def _direction(self, direction, omega_dot=0):
        import jax.numpy as jnp
        from .physical_program import ProgramInputs
        return ProgramInputs(direction.theta,
                             jnp.zeros_like(self.graph.inputs.omega_rad_s).at[self.omega_index].set(omega_dot),
                             jnp.zeros_like(self.graph.inputs.branch_starts_rad_s))

    def slope(self):
        if self._slope is None:
            jets, certificates = self.graph.jvp(self._frequency_direction())
            self._slope = jets[self.equation_node]
            self._slope_certificates = certificates
        return self._slope, self._slope_certificates

    def _solve(self, rhs, certificates):
        import jax.numpy as jnp
        from .sparse_solve import SolveAux
        slope, slope_certificates = self.slope()
        prior = jnp.int32(0)
        for certificate in tuple(certificates) + slope_certificates:
            prior = jnp.where(prior != 0, prior, certificate.status)
        # This scalar implicit equation has the same selected-dtype residual
        # certificate as other derivative equations, without a native factor.
        value = -rhs / slope
        numerator = jnp.abs(slope * value + rhs)
        denominator = jnp.abs(slope) * jnp.abs(value) + jnp.abs(rhs)
        eta = jnp.where(jnp.isfinite(numerator) & jnp.isfinite(denominator),
                        jnp.where(denominator == 0, jnp.where(numerator == 0, 0, jnp.inf),
                                  numerator / denominator), jnp.inf)
        rd = value.real.dtype
        threshold = jnp.asarray(256 * 2, dtype=rd) * jnp.finfo(rd).eps
        bad = ~jnp.isfinite(value) | ~jnp.isfinite(eta) | (eta > threshold)
        status = jnp.where(prior != 0, prior, jnp.where(bad, jnp.int32(13), jnp.int32(0)))
        return value, tuple(certificates) + slope_certificates + (SolveAux(status, eta, numerator, denominator),)

    def jvp(self, direction):
        jets, certificates = self.graph.jvp(self._direction(direction))
        return self._solve(jets[self.equation_node], certificates)

    def vjp(self, weight):
        import jax.numpy as jnp
        from .physical_program import ProgramInputs
        slope, slope_certificates = self.slope()
        derivative, certificates = self.graph.vjp(((self.equation_node, -weight / slope),))
        # Initial hints select the local branch, not a continuous physical input
        # to the implicitly selected root on that branch.
        result = ProgramInputs(derivative.theta, jnp.zeros_like(self.input_template.omega_rad_s),
                               jnp.zeros_like(self.input_template.branch_starts_rad_s))
        _, checked = self._solve(jnp.asarray(0, dtype=slope.dtype), certificates)
        return result, slope_certificates + checked

    def mixed(self, first, second):
        a, ca = self.jvp(first)
        b, cb = self.jvp(second)
        _, _, mixed, certificates = self.graph.mixed(self._direction(first, a), self._direction(second, b))
        value, checked = self._solve(mixed[self.equation_node], ca + cb + certificates)
        return value, checked
