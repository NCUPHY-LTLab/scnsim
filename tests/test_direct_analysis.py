"""Retained explicit-Julia Direct, selected-network and analytic regressions.

The inherited Float64 comparisons are scoped to the original physical
fixtures. They are not JAX accuracy, new root-search or performance Gates.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from scnsim import (
    CircuitPlan, CircuitRun, DirectQuantityResult, DirectSolveSpec,
    DiagonalRootSpec, HybridizedPoleSpec, OperatorResult, OperatorSpec,
    ReductionPipeline, ResidueNormalizedCouplingSpec, ResponseElementSpec,
    RLGC, TransferZeroSpec, components, units as u,
)

def _coupled_rlgc() -> RLGC:
    return RLGC(
        conductors=("readout", "filter"),
        reference_conductor="ground",
        resistance_per_length=[[0.18, 0.0], [0.0, 0.22]] * u.ohm / u.meter,
        inductance_per_length=[[420.0, 75.0], [75.0, 395.0]] * u.nH / u.meter,
        conductance_per_length=[[0.0, 0.0], [0.0, 0.0]] * u.siemens / u.meter,
        capacitance_per_length=[[175.0, -22.0], [-22.0, 168.0]] * u.pF / u.meter,
        extraction_frequency=5.0 * u.GHz,
    )


def _advanced_direct_plan():
    plan = CircuitPlan(id="advanced_direct_stabilization")
    capacitor_a = plan.add(components.capacitor(id="capacitor_a", capacitance=80.0*u.fF))
    inductor_a = plan.add(components.inductor(id="inductor_a", inductance=8.0*u.nH))
    capacitor_b = plan.add(components.capacitor(id="capacitor_b", capacitance=95.0*u.fF))
    inductor_b = plan.add(components.inductor(id="inductor_b", inductance=7.0*u.nH))
    coupling_capacitor = plan.add(components.capacitor(id="coupling_capacitor", capacitance=2.0*u.fF))
    coupling_inductor = plan.add(components.inductor(id="coupling_inductor", inductance=500.0*u.nH))
    a, b = plan.bus(id="a"), plan.bus(id="b")
    plan.parallel(id="mode_a", start=a, branches=((capacitor_a,), (inductor_a,)), end=plan.ground)
    plan.parallel(id="mode_b", start=b, branches=((capacitor_b,), (inductor_b,)), end=plan.ground)
    plan.parallel(id="coupling", start=a, branches=((coupling_capacitor,), (coupling_inductor,)), end=b)
    return plan


def _matched_zero_plan():
    plan = CircuitPlan(id="transfer_zero_stabilization")
    resistor = plan.add(components.resistor(id="resistor", resistance=50.0*u.ohm))
    capacitor = plan.add(components.capacitor(id="capacitor", capacitance=80.0*u.fF))
    inductor = plan.add(components.inductor(id="inductor", inductance=8.0*u.nH))
    signal = plan.bus(id="p")
    plan.parallel(id="matched_lc", start=signal,
                  branches=((resistor,), (capacitor,), (inductor,)), end=plan.ground)
    plan.add_port(id="p", at=signal, role="terminated", reference_impedance=50.0*u.ohm)
    return plan


def _floating_probe_plan():
    plan = CircuitPlan(id="floating_probe_stabilization")
    feed = plan.add(components.capacitor(id="feed", capacitance=35.0*u.fF))
    coupler = plan.add(components.capacitor(id="coupler", capacitance=4.0*u.fF))
    qubit = plan.add(components.floating_parallel_linear_lc_resonator(
        id="qubit", terminal_1_to_reference_capacitance=45.0*u.fF,
        terminal_2_to_reference_capacitance=42.0*u.fF,
        terminal_mutual_capacitance=16.0*u.fF, inductance=7.0*u.nH))
    input_node, output_node = plan.bus(id="input"), plan.bus(id="output")
    plus, minus = plan.bus(id="qubit_plus"), plan.bus(id="qubit_minus")
    plan.series(id="feed_branch", start=input_node, elements=(feed,), end=output_node)
    plan.series(id="coupler_branch", start=output_node, elements=(coupler,), end=plus)
    plan.series(id="qubit_branch", start=plus,
                elements=(qubit.between(qubit.pin("terminal_1"), qubit.pin("terminal_2")),), end=minus)
    feed_in = plan.add_port(id="feedline_in", at=input_node, role="terminated", reference_impedance=50.0*u.ohm)
    plan.add_port(id="feedline_out", at=output_node, role="terminated", reference_impedance=50.0*u.ohm)
    probe_plus = plan.add_port(id="probe_plus", at=plus, role="nonloading_probe", reference_impedance=50.0*u.ohm)
    probe_minus = plan.add_port(id="probe_minus", at=minus, role="nonloading_probe", reference_impedance=50.0*u.ohm)
    return plan, feed_in, probe_plus, probe_minus, ("qubit_plus", "qubit_minus")


def _tau(order: int) -> float:
    return 256.0 * (order + 1) * np.finfo(np.float64).eps


@contextmanager
def _closed_run(**kwargs):
    run = CircuitRun(**kwargs)
    try:
        yield run
    finally:
        run.close()


class DirectDeclarationTests(unittest.TestCase):
    def test_transmission_line_and_reduction_grammar_preserve_declaration_order(self):
        rlgc = _coupled_rlgc()
        line = components.transmission_line(id="line", length=1.6*u.mm,
                                           rlgc=rlgc, n_sections=8)
        self.assertEqual(tuple(line.pins),
                         ("head.readout", "head.filter", "tail.readout", "tail.filter"))
        self.assertEqual(line.metadata["n_sections"], 8)
        self.assertEqual(rlgc.conductors, ("readout", "filter"))
        with self.assertRaises(ValueError):
            components.transmission_line(id="bad", length=1.0*u.mm,
                                        rlgc=_coupled_rlgc(), n_sections=0)
        plan, _, probe_plus, probe_minus, nodes = _floating_probe_plan()
        pipeline = (ReductionPipeline().ptc(probe_plus, probe_minus)
                    .transform_pair(*nodes, id="qubit")
                    .retain("feedline_in", "feedline_out", "qubit.differential"))
        self.assertIsNotNone(plan)
        with self.assertRaises(ValueError):
            pipeline.ptc(probe_plus)
        with self.assertRaises(ValueError):
            pipeline.transform_pair(*nodes, id="late")
        with self.assertRaises(ValueError):
            pipeline.retain("feedline_in")


@unittest.skipUnless(os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
                     "set SCNSIM_RUN_JULIA_TESTS=1 for the packaged Julia regressions")
class SelectedNetworkTests(unittest.TestCase):
    def test_n_trace_pi_ladder_and_n_port_direct_are_not_scalarized(self) -> None:
        plan = CircuitPlan(id="n_trace_direct")
        line = plan.add(components.transmission_line(id="line", length=1.6 * u.mm, rlgc=_coupled_rlgc(), n_sections=2))
        for end in ("head", "tail"):
            for conductor in ("readout", "filter"):
                node = plan.bus(id=f"{end}_{conductor}")
                plan.link(id=f"connect_{end}_{conductor}", endpoints=(node, line.pin(end, conductor=conductor)))
                plan.add_port(id=f"{end}_{conductor}", at=node, role="terminated", reference_impedance=50.0 * u.ohm)
        spec = DirectSolveSpec(frequencies=[1.0, 2.0] * u.GHz)
        with TemporaryDirectory() as workspace:
            with _closed_run(plan=plan, workspace=Path(workspace), backend="julia", precision="float64") as run:
                explanation = run.explain(run.original, spec)
                self.assertEqual(run.inventory().requests, ())
                compiled = explanation.evidence["compiled"]
                rows = compiled["expanded_branch_rows"]
                audits = [row for row in rows if row["kind"] == "transmission_line_audit"]
                self.assertEqual(len(audits), 1)
                self.assertEqual(audits[0]["conductors"], ("readout", "filter"))
                self.assertEqual(audits[0]["n_sections"], 2)
                self.assertEqual(len(audits[0]["stations"]), 6)
                self.assertTrue(any(row.get("row_conductor") != row.get("column_conductor") and not row.get("omitted_as_zero") for row in rows if row.get("kind") in {"series_inductance", "shunt_capacitance"}))
                result = run.solve(run.original, spec)
                self.assertEqual(result.s.view.matrix.shape, (2, 4, 4))
                self.assertEqual(result.y.view.matrix.shape, (2, 4, 4))
                self.assertEqual(result.z.view.matrix.shape, (2, 4, 4))
                self.assertEqual(tuple(result.s.view.coordinates), ("head_readout", "head_filter", "tail_readout", "tail_filter"))

    def test_ptc_transform_retain_uses_one_realized_lineage(self) -> None:
        plan, _, probe_plus, probe_minus, nodes = _floating_probe_plan()
        with TemporaryDirectory() as workspace:
            with _closed_run(plan=plan, workspace=Path(workspace), backend="julia", precision="float64") as run:
                view = run.original.reduce(
                    ReductionPipeline()
                    .ptc(probe_plus, probe_minus)
                    .transform_pair(*nodes, id="qubit")
                    .retain("feedline_in", "feedline_out", "qubit.differential")
                )
                spec = DirectSolveSpec(frequencies=[5.5] * u.GHz)
                explanation = run.explain(view, spec)
                lineage = explanation.evidence["compiled"]["ref_lineage"]
                port_nets = {port["id"]: port["net"]
                             for port in explanation.evidence["connectivity"]["ports"]}
                self.assertEqual(lineage["terminal_coordinates"],
                                 (port_nets["feedline_in"], port_nets["feedline_out"], "qubit.differential"))
                self.assertEqual(lineage["ptc"]["selected_ports"], ("probe_plus", "probe_minus"))
                self.assertEqual([row["port_id"] for row in lineage["ptc"]["loads"]], ["probe_plus", "probe_minus"])
                self.assertTrue(lineage["port_realizable"])
                result = run.solve(view, spec)
                self.assertEqual(result.s.view.matrix.shape, (1, 3, 3))
                self.assertEqual(result.s.view.probe_loads["probe_plus"], "compensated")
                self.assertEqual(result.s.view.probe_loads["probe_minus"], "compensated")


@unittest.skipUnless(os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
                     "set SCNSIM_RUN_JULIA_TESTS=1 for the packaged Julia regressions")
class DirectQuantityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = TemporaryDirectory()
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.workspace = Path(cls._temporary.name)
        cls.plan = _advanced_direct_plan()
        cls.circuit_run = CircuitRun(plan=cls.plan, workspace=cls.workspace, backend="julia", precision="float64")
        cls.addClassCleanup(cls.circuit_run.close)
        cls.view = cls.circuit_run.original.reduce(ReductionPipeline().retain("a", "b"))
        cls.operator_spec = OperatorSpec(frequencies=[5.0, 5.5] * u.GHz)
        cls.response_spec = ResponseElementSpec(
            family="Y", input_coordinate="a", output_coordinate="b", frequency=5.5 * u.GHz
        )
        cls.pole_spec = HybridizedPoleSpec(coordinates=("a", "b"), anchor=6.2 * u.GHz)
        cls.coupling_spec = ResidueNormalizedCouplingSpec(
            branch_a=DiagonalRootSpec(coordinate="a", root_hint=6.2 * u.GHz),
            branch_b=DiagonalRootSpec(coordinate="b", root_hint=6.2 * u.GHz),
            frequency=5.5 * u.GHz,
        )
        cls.operator = cls.circuit_run.evaluate(cls.view, cls.operator_spec)
        cls.response = cls.circuit_run.evaluate(cls.view, cls.response_spec)
        cls.pole = cls.circuit_run.evaluate(cls.view, cls.pole_spec)
        cls.coupling = cls.circuit_run.evaluate(cls.view, cls.coupling_spec)

        cls._zero_temporary = TemporaryDirectory()
        cls.addClassCleanup(cls._zero_temporary.cleanup)
        cls.zero_plan = _matched_zero_plan()
        cls.zero_run = CircuitRun(plan=cls.zero_plan, workspace=Path(cls._zero_temporary.name), backend="julia", precision="float64")
        cls.addClassCleanup(cls.zero_run.close)
        cls.zero_spec = TransferZeroSpec(
            anchor=6.3 * u.GHz,
            family="S",
            input_coordinate="p",
            output_coordinate="p",
        )
        cls.zero = cls.zero_run.evaluate(cls.zero_run.original, cls.zero_spec)

    @staticmethod
    def _matrices(omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        capacitance = np.asarray([[82.0, -2.0], [-2.0, 97.0]]) * 1.0e-15
        stiffness = np.asarray(
            [
                [1.0 / (8.0e-9) + 1.0 / (500.0e-9), -1.0 / (500.0e-9)],
                [-1.0 / (500.0e-9), 1.0 / (7.0e-9) + 1.0 / (500.0e-9)],
            ]
        )
        return capacitance, stiffness, stiffness - omega**2 * capacitance

    def test_operator_and_response_match_the_declared_dynamic_equations(self) -> None:
        self.assertIsInstance(self.operator, OperatorResult)
        self.assertEqual(tuple(point.frequency.to("gigahertz").magnitude for point in self.operator.points), (5.0, 5.5))
        for point in self.operator.points:
            omega = 2.0 * math.pi * float(point.frequency.to("hertz").magnitude)
            _, _, expected = self._matrices(omega)
            actual = np.asarray(point.matrix.to("siemens / second").magnitude)
            scale = np.max(np.abs(expected))
            self.assertLessEqual(np.max(np.abs(actual - expected)) / scale, _tau(2))

        self.assertIsInstance(self.response, DirectQuantityResult)
        omega = 2.0 * math.pi * 5.5e9
        expected = 1j * (omega * 2.0e-15 - 1.0 / (omega * 500.0e-9))
        actual = complex(self.response.value.to("siemens").magnitude)
        self.assertLessEqual(abs(actual - expected) / abs(expected), _tau(2))
        self.assertEqual(self.response.family, "Y")
        self.assertEqual(self.response.magnitude.to("siemens").magnitude, abs(actual))
        self.assertEqual(self.response.real.to("siemens").magnitude, actual.real)
        self.assertEqual(self.response.imag.to("siemens").magnitude, actual.imag)

    def test_hybridized_pole_and_residue_coupling_match_the_common_operator(self) -> None:
        self.assertIsInstance(self.pole, DirectQuantityResult)
        capacitance, stiffness, _ = self._matrices(0.0)
        roots = np.sqrt(np.linalg.eigvals(np.linalg.solve(capacitance, stiffness)))
        pole = complex(self.pole.root.to("radian / second").magnitude)
        self.assertLessEqual(min(abs(pole - root) for root in roots) / abs(pole), _tau(2))
        self.assertEqual(self.pole.frequency.to("hertz").magnitude, pole.real / (2.0 * math.pi))
        self.assertEqual(self.pole.linewidth.to("hertz").magnitude, -pole.imag / math.pi)

        self.assertIsInstance(self.coupling, DirectQuantityResult)
        omega_a = math.sqrt(stiffness[0, 0] / capacitance[0, 0])
        omega_b = math.sqrt(stiffness[1, 1] / capacitance[1, 1])
        slope_a = 2.0 * omega_a * capacitance[0, 0]
        slope_b = 2.0 * omega_b * capacitance[1, 1]
        omega = 2.0 * math.pi * 5.5e9
        expected = (stiffness[0, 1] - omega**2 * capacitance[0, 1]) / math.sqrt(slope_a * slope_b)
        actual = complex(self.coupling.coupling.to("radian / second").magnitude)
        self.assertLessEqual(abs(actual - expected) / abs(expected), _tau(2))
        self.assertEqual(self.coupling.magnitude.to("radian / second").magnitude, abs(actual))
        self.assertLessEqual(
            abs(complex(self.coupling.branch_a_residue.to("ohm").magnitude) + 1.0 / slope_a) / abs(1.0 / slope_a),
            _tau(2),
        )
        self.assertLessEqual(
            abs(complex(self.coupling.branch_b_residue.to("ohm").magnitude) + 1.0 / slope_b) / abs(1.0 / slope_b),
            _tau(2),
        )

    def test_transfer_zero_is_the_analytic_matched_parallel_lc_zero(self) -> None:
        self.assertIsInstance(self.zero, DirectQuantityResult)
        expected_frequency = 1.0 / (2.0 * math.pi * math.sqrt(8.0e-9 * 80.0e-15))
        actual_omega = complex(self.zero.zero.to("radian / second").magnitude)
        expected_omega = 2.0 * math.pi * expected_frequency
        self.assertLessEqual(abs(actual_omega - expected_omega) / expected_omega, _tau(1))
        self.assertEqual(self.zero.frequency.to("hertz").magnitude, actual_omega.real / (2.0 * math.pi))
        self.assertNotEqual(complex(self.zero.denominator.to("dimensionless").magnitude), 0.0j)
        self.assertNotEqual(complex(self.zero.numerator_slope.to("dimensionless").magnitude), 0.0j)
