"""Retained Julia/HB physics, typed failures and exact sealed readback.

Current structured authoring supplies equivalent physical fixtures. These tests
freeze no new Memory, native-AD, root-search or report-composition semantics.
"""

from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from contextlib import ExitStack
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import MappingProxyType
import unittest

import numpy as np
from tests.test_direct_analysis import _two_coordinate_plan

from scnsim import (
    BiasState,
    CircuitPlan,
    CircuitRun,
    CurrentDrive,
    DirectSolveSpec,
    EvidenceIntegrityError,
    HBCaseFailure,
    HBCaseSpec,
    HBSolveSpec,
    HBTruncation,
    HybridizedPoleSpec,
    PortRealizabilityError,
    PumpAxis,
    PumpState,
    ReductionPipeline,
    ReportSpec,
    RLGC,
    SCNSimCapabilityError,
    SCNSimValidationError,
    SParameterTrace,
    TransferZeroSpec,
    components,
    units as u,
)




def _one_port_jj():
    plan = CircuitPlan(id="hb_jj_stabilization")
    resonator = plan.add(components.grounded_parallel_single_junction_resonator(
        id="resonator", capacitance=80.0*u.fF,
        josephson_inductance=8.0*u.nH, junction_capacitance=2.0*u.fF))
    signal = plan.bus(id="signal")
    plan.link(id="resonator", endpoints=(signal, resonator.pin("terminal_1")))
    port = plan.add_port(id="p", at=signal, role="terminated", reference_impedance=50.0*u.ohm)
    return plan, port




def _hb_dc_spec(port: object, cases: tuple[tuple[str, object | None], ...]) -> HBSolveSpec:
    pump = PumpAxis(id="pump", frequency=6.0 * u.GHz)
    drive = CurrentDrive(id="dc", at=port, mode=(0,))
    return HBSolveSpec(
        pump_axes=(pump,),
        drives=(drive,),
        frequencies=[5.5] * u.GHz,
        cases=tuple(
            HBCaseSpec(id=identifier, currents={} if current is None else {drive: current})
            for identifier, current in cases
        ),
        truncation=HBTruncation(
            pump_harmonics=(0,),
            modulation_harmonics=(0,),
            three_wave_mixing=False,
            four_wave_mixing=False,
        ),
        traces=(
            SParameterTrace(
                id="reflection",
                input_port="p",
                input_mode=(0,),
                output_port="p",
                output_mode=(0,),
            ),
        ),
    )

def _floating_probe_plan():
    plan = CircuitPlan(id="hb_floating_probe_stabilization")
    feed = plan.add(components.capacitor(id="feed", capacitance=35.0*u.fF))
    coupler = plan.add(components.capacitor(id="coupler", capacitance=4.0*u.fF))
    qubit = plan.add(components.floating_parallel_single_junction_resonator(
        id="qubit", terminal_1_to_reference_capacitance=45.0*u.fF,
        terminal_2_to_reference_capacitance=42.0*u.fF,
        terminal_mutual_capacitance=16.0*u.fF,
        josephson_inductance=7.0*u.nH, junction_capacitance=2.0*u.fF))
    input_node = plan.bus(id="input")
    output_node = plan.bus(id="output")
    plus = plan.bus(id="qubit_plus")
    minus = plan.bus(id="qubit_minus")
    plan.series(id="feed", start=input_node, elements=(feed,), end=output_node)
    plan.series(id="coupler", start=output_node, elements=(coupler,), end=qubit.pin("terminal_1"))
    plan.link(id="qubit_plus", endpoints=(plus, qubit.pin("terminal_1")))
    plan.link(id="qubit_minus", endpoints=(minus, qubit.pin("terminal_2")))
    feed_in = plan.add_port(id="feedline_in", at=input_node, role="terminated", reference_impedance=50.0*u.ohm)
    plan.add_port(id="feedline_out", at=output_node, role="terminated", reference_impedance=50.0*u.ohm)
    probe_plus = plan.add_port(id="probe_plus", at=plus, role="nonloading_probe", reference_impedance=50.0*u.ohm)
    probe_minus = plan.add_port(id="probe_minus", at=minus, role="nonloading_probe", reference_impedance=50.0*u.ohm)
    return plan, feed_in, probe_plus, probe_minus, plus.node, minus.node


class HBDeclarationTests(unittest.TestCase):

    def test_hb_specs_are_immutable_ordered_and_defensively_copied(self) -> None:
        plan, port = _one_port_jj()
        pump = PumpAxis(id="pump", frequency=6.0 * u.GHz)
        first = CurrentDrive(id="first", at=port, mode=(0,))
        second = CurrentDrive(id="second", at=port, mode=(1,))
        frequencies = np.asarray([5.0, 5.5], dtype=np.float64) * u.GHz
        case = HBCaseSpec(id="biased", currents={second: (2.0 + 1.0j) * u.nA, first: 1.0 * u.nA})
        spec = HBSolveSpec(
            pump_axes=(pump,),
            drives=(first, second),
            frequencies=frequencies,
            cases=(case, HBCaseSpec(id="off", currents={})),
            truncation=HBTruncation(
                pump_harmonics=(1,),
                modulation_harmonics=(1,),
                three_wave_mixing=True,
                four_wave_mixing=True,
            ),
        )
        self.assertEqual([drive.id for drive in spec.drives], ["first", "second"])
        self.assertEqual([case.id for case in spec.cases], ["biased", "off"])
        self.assertEqual(spec.cases[0].currents[first], 1.0*u.nA)
        self.assertEqual(spec.cases[0].currents[second], (2.0+1.0j)*u.nA)
        frequencies.magnitude[0] = 9.0
        self.assertEqual(tuple(spec.frequencies.to("gigahertz").magnitude), (5.0, 5.5))
        self.assertFalse(np.asarray(spec.frequencies.magnitude).flags.writeable)
        self.assertIsInstance(case.currents, MappingProxyType)
        with self.assertRaises(FrozenInstanceError):
            pump.id = "changed"

    def test_hb_spec_validation_is_closed(self) -> None:
        _, port = _one_port_jj()
        pump = PumpAxis(id="pump", frequency=6.0 * u.GHz)
        drive = CurrentDrive(id="drive", at=port, mode=(1,))
        truncation = HBTruncation(
            pump_harmonics=(1,),
            modulation_harmonics=(1,),
            three_wave_mixing=True,
            four_wave_mixing=True,
        )
        case = HBCaseSpec(id="case", currents={drive: 1.0 * u.nA})
        with self.assertRaises(ValueError):
            HBSolveSpec(pump_axes=(pump, pump), drives=(drive,), frequencies=[5.0] * u.GHz, cases=(case,), truncation=HBTruncation(pump_harmonics=(1, 1), modulation_harmonics=(1, 1), three_wave_mixing=True, four_wave_mixing=True))
        with self.assertRaises(ValueError):
            HBSolveSpec(pump_axes=(pump,), drives=(CurrentDrive(id="rank", at=port, mode=(1, 0)),), frequencies=[5.0] * u.GHz, cases=(HBCaseSpec(id="case", currents={}),), truncation=truncation)
        with self.assertRaises(ValueError):
            HBSolveSpec(pump_axes=(pump,), drives=(drive, CurrentDrive(id="conjugate", at=port, mode=(-1,))), frequencies=[5.0] * u.GHz, cases=(HBCaseSpec(id="case", currents={}),), truncation=truncation)
        dc = CurrentDrive(id="dc", at=port, mode=(0,))
        with self.assertRaises(ValueError):
            HBSolveSpec(pump_axes=(pump,), drives=(dc,), frequencies=[5.0] * u.GHz, cases=(HBCaseSpec(id="case", currents={dc: (1.0 + 1.0j) * u.nA}),), truncation=truncation)
        twin = CurrentDrive(id="drive", at=port, mode=(1,))
        with self.assertRaises(ValueError):
            HBSolveSpec(pump_axes=(pump,), drives=(drive,), frequencies=[5.0] * u.GHz, cases=(HBCaseSpec(id="case", currents={twin: 1.0 * u.nA}),), truncation=truncation)
        with self.assertRaises(ValueError):
            HBSolveSpec(pump_axes=(pump,), drives=(drive,), frequencies=[5.0, 5.0] * u.GHz, cases=(case,), truncation=truncation)

    def test_hb_binding_failures_create_no_request_or_attempt(self) -> None:
        foreign_plan, foreign_port = _one_port_jj()
        foreign_spec = _hb_dc_spec(foreign_port, (("off", None),))
        local_plan, local_port = _one_port_jj()
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            run = CircuitRun(backend="julia", plan=local_plan, workspace=Path(workspace))
            owners.callback(run.close)
            with self.assertRaises(SCNSimValidationError):
                run.solve(run.original, foreign_spec)
            self.assertEqual(run.inventory().requests, ())

        bad_trace = HBSolveSpec(
            pump_axes=(PumpAxis(id="pump", frequency=6.0 * u.GHz),),
            drives=(CurrentDrive(id="dc", at=local_port, mode=(0,)),),
            frequencies=[5.5] * u.GHz,
            cases=(HBCaseSpec(id="off", currents={}),),
            truncation=HBTruncation(pump_harmonics=(0,), modulation_harmonics=(0,), three_wave_mixing=False, four_wave_mixing=False),
            traces=(SParameterTrace(id="bad", input_port="missing", input_mode=(0,), output_port="p", output_mode=(0,)),),
        )
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            run = CircuitRun(backend="julia", plan=local_plan, workspace=Path(workspace))
            owners.callback(run.close)
            with self.assertRaises(PortRealizabilityError):
                run.solve(run.original, bad_trace)
            self.assertEqual(run.inventory().requests, ())

@unittest.skipUnless(
    os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
    "set SCNSIM_RUN_JULIA_TESTS=1 to run packaged Julia HB preflight regressions",
)
class HBPreflightTests(unittest.TestCase):
    def test_hb_lattice_rejects_zero_frequency_and_exact_mode_collision(self) -> None:
        plan, port = _one_port_jj()
        pump = PumpAxis(id="pump", frequency=6.0 * u.GHz)
        drive = CurrentDrive(id="drive", at=port, mode=(1,))
        zero = HBSolveSpec(
            pump_axes=(pump,),
            drives=(drive,),
            frequencies=[6.0] * u.GHz,
            cases=(HBCaseSpec(id="off", currents={}),),
            truncation=HBTruncation(pump_harmonics=(1,), modulation_harmonics=(1,), three_wave_mixing=True, four_wave_mixing=True),
        )
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            run = CircuitRun(backend="julia", plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            with self.assertRaises(PortRealizabilityError) as raised:
                run.explain(run.original, zero)
            self.assertIn("zero", str(raised.exception))
            self.assertEqual(run.inventory().requests, ())

        plan, port = _one_port_jj()
        axes = (PumpAxis(id="first", frequency=6.0 * u.GHz), PumpAxis(id="second", frequency=6.0 * u.GHz))
        drive = CurrentDrive(id="drive", at=port, mode=(1, 0))
        collision = HBSolveSpec(
            pump_axes=axes,
            drives=(drive,),
            frequencies=[5.5] * u.GHz,
            cases=(HBCaseSpec(id="off", currents={}),),
            truncation=HBTruncation(pump_harmonics=(1, 1), modulation_harmonics=(1, 1), three_wave_mixing=True, four_wave_mixing=True),
        )
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            run = CircuitRun(backend="julia", plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            with self.assertRaises(PortRealizabilityError) as raised:
                run.explain(run.original, collision)
            self.assertIn("collision", str(raised.exception))
            self.assertEqual(run.inventory().requests, ())

    def test_hb_lowers_both_squid_junctions_and_rejects_mutual_series_resistance(self) -> None:
        plan = CircuitPlan(id="hb_squid_preflight")
        resonator = plan.add(
            components.grounded_parallel_symmetric_squid_resonator(
                id="resonator",
                capacitance=80.0 * u.fF,
                josephson_inductance=8.0 * u.nH,
                junction_capacitance=2.0 * u.fF,
                loop_inductance=1.0 * u.nH,
            )
        )
        node = plan.bus(id="signal")
        plan.link(id="resonator", endpoints=(node, resonator.pin("terminal_1")))
        port = plan.add_port(id="p", at=node, role="terminated", reference_impedance=50.0 * u.ohm)
        spec = _hb_dc_spec(port, (("off", None),))
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            run = CircuitRun(backend="julia", plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            compiled = run.explain(run.original, spec).evidence["compiled"]
            nonlinear = [row for row in compiled["expanded_branch_rows"] if row["kind"] == "josephson_inductance"]
            self.assertEqual(
                [row["component_path"] for row in nonlinear],
                [("resonator", "squid", "junction_1"), ("resonator", "squid", "junction_2")],
            )
            self.assertEqual(run.inventory().requests, ())

        rlgc = RLGC(
            conductors=("a", "b"),
            reference_conductor="ground",
            resistance_per_length=[[1.0, 0.1], [0.1, 1.0]] * u.ohm / u.meter,
            inductance_per_length=[[2.0, 0.2], [0.2, 2.0]] * u.nH / u.meter,
            conductance_per_length=[[0.0, 0.0], [0.0, 0.0]] * u.siemens / u.meter,
            capacitance_per_length=[[2.0, -0.2], [-0.2, 2.0]] * u.pF / u.meter,
        )
        plan = CircuitPlan(id="hb_mutual_resistance")
        line = plan.add(components.transmission_line(id="line", length=1.0 * u.mm, rlgc=rlgc, n_sections=1))
        ports = []
        for end in ("head", "tail"):
            for conductor in rlgc.conductors:
                node = plan.bus(id=f"{end}_{conductor}")
                plan.link(id=f"{end}_{conductor}", endpoints=(node, line.pin(end, conductor=conductor)))
                ports.append(plan.add_port(id=f"{end}_{conductor}", at=node, role="terminated", reference_impedance=50.0 * u.ohm))
        pump = PumpAxis(id="pump", frequency=6.0 * u.GHz)
        drive = CurrentDrive(id="dc", at=ports[0], mode=(0,))
        spec = HBSolveSpec(
            pump_axes=(pump,),
            drives=(drive,),
            frequencies=[5.5] * u.GHz,
            cases=(HBCaseSpec(id="off", currents={}),),
            truncation=HBTruncation(pump_harmonics=(0,), modulation_harmonics=(0,), three_wave_mixing=False, four_wave_mixing=False),
        )
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            run = CircuitRun(backend="julia", plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            with self.assertRaises(SCNSimCapabilityError):
                run.explain(run.original, spec)
            self.assertEqual(run.inventory().requests, ())

    def test_hb_explain_is_no_attempt_and_driven_ptc_requires_authorization(self) -> None:
        plan, feed_in, probe_plus, probe_minus, plus, minus = _floating_probe_plan()
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            run = CircuitRun(backend="julia", plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            view = run.original.reduce(
                ReductionPipeline()
                .ptc(probe_plus, probe_minus)
                .transform_pair(plus, minus, id="qubit")
                .retain("feedline_in", "feedline_out", "qubit.differential")
            )
            pump = PumpAxis(id="pump", frequency=9.0 * u.GHz)
            drive = CurrentDrive(id="pump_drive", at=feed_in, mode=(1,))

            def spec(allowed: bool) -> HBSolveSpec:
                return HBSolveSpec(
                    pump_axes=(pump,),
                    drives=(drive,),
                    frequencies=[5.5] * u.GHz,
                    cases=(HBCaseSpec(id="driven", currents={drive: 1.0 * u.nA}),),
                    truncation=HBTruncation(pump_harmonics=(1,), modulation_harmonics=(1,), three_wave_mixing=True, four_wave_mixing=True),
                    allow_driven_ptc=allowed,
                )

            with self.assertRaises(PortRealizabilityError):
                run.explain(view, spec(False))
            self.assertEqual(run.inventory().requests, ())
            explanation = run.explain(view, spec(True))
            self.assertEqual(run.inventory().requests, ())
            evidence = explanation.evidence["compiled"]["hb_preflight"]["topology_evidence"]
            self.assertEqual(evidence["nonlinear_balance"]["load_state"], "loaded")
            self.assertEqual(evidence["response_linearization"]["load_state"], "compensated")

    def test_hb_numerical_classifier_keeps_linearization_and_response_stages_narrow(self) -> None:
        from scnsim.execution.preparation import _child_environment, packaged_julia_resources, prepare_runtime

        prepared = prepare_runtime()
        program = r'''
using LinearAlgebra
using SCNSimBackend

linearization = SCNSimBackend.hb_numeric_exception(SingularException(1), "linearization")
@assert linearization isa SCNSimBackend.HBCaseNumericalFailure
println(linearization.stage)

response = try
    SCNSimBackend.hb_checked_solve(
        zeros(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1),
        "forced singular response",
        1,
    )
    nothing
catch error
    error
end
@assert response isa SCNSimBackend.HBCaseNumericalFailure
println(response.stage)

propagated = try
    error("not an accepted numerical failure")
catch original
    try
        SCNSimBackend.hb_numeric_exception(original, "linearization")
        nothing
    catch error
        error
    end
end
@assert propagated isa ErrorException
println(nameof(typeof(propagated)))
'''
        with packaged_julia_resources() as (project, _, _):
            completed = subprocess.run(
                [
                    str(prepared.executable),
                    "--startup-file=no",
                    "--history-file=no",
                    "--threads=1",
                    f"--project={project}",
                    "-e",
                    program,
                ],
                cwd=project,
                env=_child_environment(),
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            completed.stdout.splitlines(),
            ["linearization", "response_formation", "ErrorException"],
        )

@unittest.skipUnless(
    os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
    "set SCNSIM_RUN_JULIA_TESTS=1 to run packaged Julia complex-anchor regressions",
)
class JuliaComplexAnchorDecodingTests(unittest.TestCase):
    def test_julia_complex_anchor_decode_preserves_bits(self) -> None:
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            plan, coordinates = _two_coordinate_plan()
            run = CircuitRun(backend="julia", plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            view = run.original.reduce(ReductionPipeline().retain(*coordinates))
            spec = HybridizedPoleSpec(coordinates=coordinates, anchor=(6.0 - 1.0e-12j) * u.GHz)
            request = run.explain(view, spec).evidence
            encoded = request["spec"]["anchor"]
            from scnsim.execution.preparation import _child_environment, packaged_julia_resources, prepare_runtime

            prepared = prepare_runtime()
            program = f'''
using SCNSimBackend
record = Dict{{String,Any}}(
    "type" => "complex_quantity_f64",
    "real_si_f64" => "{encoded['real_si_f64']}",
    "imag_si_f64" => "{encoded['imag_si_f64']}",
    "si_unit" => "hertz",
    "dimensionality" => "frequency",
)
decoded = SCNSimBackend.complex_frequency_value(record)
@assert SCNSimBackend.f64_hex(real(decoded)) == "{encoded['real_si_f64']}"
@assert SCNSimBackend.f64_hex(imag(decoded)) == "{encoded['imag_si_f64']}"
println(SCNSimBackend.f64_hex(real(decoded)))
println(SCNSimBackend.f64_hex(imag(decoded)))
'''
            with packaged_julia_resources() as (project, _, _):
                decoded = subprocess.run(
                    [
                        str(prepared.executable),
                        "--startup-file=no",
                        "--history-file=no",
                        "--threads=1",
                        f"--project={project}",
                        "-e",
                        program,
                    ],
                    cwd=project,
                    env=_child_environment(),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(
                decoded.stdout.splitlines(),
                [encoded["real_si_f64"], encoded["imag_si_f64"]],
            )

@unittest.skipUnless(
    os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
    "set SCNSIM_RUN_JULIA_TESTS=1 to run packaged Julia HB regressions",
)
class HBRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = TemporaryDirectory()
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.workspace = Path(cls._temporary.name)
        cls.plan, cls.port = _one_port_jj()
        cls.circuit_run = CircuitRun(backend="julia", plan=cls.plan, workspace=cls.workspace)
        cls.addClassCleanup(cls.circuit_run.close)
        cls.partial_spec = _hb_dc_spec(cls.port, (("off", None), ("small", 1.0 * u.nA), ("overcritical", 10.0 * u.uA)))
        cls.all_failure_spec = _hb_dc_spec(cls.port, (("overcritical", 10.0 * u.uA), ("far_overcritical", 20.0 * u.uA)))
        cls.partial = cls.circuit_run.solve(cls.circuit_run.original, cls.partial_spec)
        cls.all_failure = cls.circuit_run.solve(cls.circuit_run.original, cls.all_failure_spec)

    def test_hb_partial_and_all_failure_batches_are_completed_results(self) -> None:
        self.assertEqual(tuple(self.partial.cases), ("off", "small", "overcritical"))
        self.assertIsInstance(self.partial.cases, MappingProxyType)
        self.assertTrue(self.partial.cases["off"].succeeded)
        self.assertTrue(self.partial.cases["small"].succeeded)
        failure = self.partial.cases["overcritical"]
        self.assertFalse(failure.succeeded)
        self.assertEqual(failure.failure.stage, "operating_point")
        with self.assertRaises(HBCaseFailure) as raised:
            _ = failure.s
        self.assertIs(raised.exception, failure.failure)
        success = self.partial.cases["small"]
        self.assertEqual(success.bias_state, BiasState.ON)
        self.assertEqual(success.pump_state, PumpState.OFF)
        self.assertEqual(success.s.view.matrix.shape, (1, 1, 1))
        self.assertEqual(success.y.view.matrix.shape, (1, 1, 1))
        self.assertEqual(success.z.view.matrix.shape, (1, 1, 1))
        self.assertEqual(success.states.shape[1], len(success.state_node_map))
        trace = np.asarray(success.traces["reflection"].value.magnitude)
        selected = np.asarray(success.s.view.matrix.magnitude)[:, 0, 0]
        self.assertTrue(np.array_equal(trace.view(np.uint64), selected.view(np.uint64)))
        self.assertFalse(np.asarray(success.states.magnitude).flags.writeable)
        self.assertEqual(len(success.effective_sources), 1)
        effective = success.effective_sources[0]
        self.assertEqual(effective["drive_id"], "dc")
        self.assertEqual(effective["mode"], (0,))
        self.assertEqual(effective["coefficient"], (1.0e-9 + 0.0j) * u.ampere)
        self.assertEqual(effective["generated_conjugate"]["mode"], (0,))
        self.assertEqual(
            effective["backend_binding"]["coefficient_convention"],
            "exp_plus_i_m_dot_omega_t_josephsoncircuits_source",
        )
        self.assertEqual(len(effective["injection_map_sha256"]), 64)

        topology = self.partial.topology_evidence
        self.assertEqual(
            set(topology),
            {
                "allow_driven_ptc",
                "intrinsic_compiled_graph_sha256",
                "nonlinear_balance",
                "response_linearization",
            },
        )
        self.assertFalse(topology["allow_driven_ptc"])
        self.assertEqual(topology["nonlinear_balance"]["load_state"], "loaded")
        self.assertEqual(topology["response_linearization"]["load_state"], "raw")
        self.assertEqual(
            topology["nonlinear_balance"]["lineage_sha256"],
            topology["response_linearization"]["lineage_sha256"],
        )
        self.assertEqual(len(topology["intrinsic_compiled_graph_sha256"]), 64)

        native = success.s.backend_native
        reconciliation = success.s.reconciliation
        self.assertIsNotNone(native)
        self.assertIsNotNone(reconciliation)
        self.assertEqual(native.coordinates, success.s.view.coordinates)
        self.assertEqual(native.input_channels, success.s.view.input_channels)
        self.assertEqual(native.output_channels, success.s.view.output_channels)
        self.assertTrue(reconciliation.comparable)
        self.assertIsNone(reconciliation.reason)
        self.assertTrue(np.isfinite(reconciliation.residual))
        self.assertEqual(
            reconciliation.last_comparable_ancestor,
            topology["response_linearization"]["lineage_sha256"],
        )
        self.assertEqual(len(reconciliation.evidence_sha256), 64)
        self.assertEqual(len(success.state_node_map), 1)
        self.assertEqual(success.state_node_map[0]["state_index"], 0)
        self.assertTrue(success.state_node_map[0]["compiler_node_id"])
        self.assertEqual(success.state_node_map[0]["source"], {
            "kind": "plan_node",
            "plan_node_id": success.state_node_map[0]["compiler_node_id"],
            "visibility": "public",
        })
        with self.assertRaises(FrozenInstanceError):
            success.id = "changed"
        with self.assertRaises(FrozenInstanceError):
            self.partial.cases = {}

        result_path = next(
            self.workspace.glob(
                f"leaves/*/requests/{self.partial.identity.request_sha256}/attempts/000001/result.json"
            )
        )
        document = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [(case["case_ordinal"], case["case_id"], case["status"]) for case in document["cases"]],
            [(1, "off", "success"), (2, "small", "success"), (3, "overcritical", "failure")],
        )
        artifact_roles = {
            "s", "y", "z", "backend_native_s", "backend_native_z", "states", "effective_source_vectors"
        }
        for ordinal, case in enumerate(document["cases"][:2], start=1):
            self.assertEqual(set(case["artifacts"]), artifact_roles)
            for role, artifact in case["artifacts"].items():
                self.assertEqual(artifact["id"], role)
                self.assertEqual(artifact["path"], f"artifacts/cases/{ordinal:06d}/{role}.zarr")
                self.assertEqual(
                    artifact["file_manifest"],
                    f"artifacts/cases/{ordinal:06d}/{role}.manifest.json",
                )
                self.assertEqual(len(artifact["sha256"]), 64)
        self.assertNotIn("artifacts", document["cases"][2])
        self.assertEqual(document["cases"][2]["failure"]["kind"], "hb_case_failure")
        self.assertEqual(document["cases"][2]["failure"]["stage"], "operating_point")

        self.assertTrue(all(not outcome.succeeded for outcome in self.all_failure.cases.values()))
        self.assertEqual({outcome.failure.stage for outcome in self.all_failure.cases.values()}, {"operating_point"})
        failure_table = self.all_failure.plot().data[0]
        self.assertEqual(tuple(failure_table.cells.values[0]), ("overcritical", "far_overcritical"))
        self.assertEqual(tuple(failure_table.cells.values[1]), ("failure", "failure"))
        rows = {row["request_sha256"]: row for row in self.circuit_run.inventory().requests}
        self.assertEqual(rows[self.all_failure.identity.request_sha256]["status"], "succeeded")

    def test_hb_cache_resolve_inventory_report_and_integrity_boundaries(self) -> None:
        repeated = self.circuit_run.solve(self.circuit_run.original, self.partial_spec)
        self.assertEqual(repeated.identity, self.partial.identity)
        resolved = self.circuit_run.resolve(self.circuit_run.original, self.partial_spec)
        self.assertEqual(resolved.identity, self.partial.identity)
        inventory = self.circuit_run.inventory().requests
        self.assertEqual([row["request_sha256"] for row in inventory], sorted(row["request_sha256"] for row in inventory))
        self.assertTrue(all(row["attempts"] == ("000001",) for row in inventory))

        report = self.circuit_run.build_report(ReportSpec(inputs=(self.partial, self.all_failure)))
        self.assertIn("HB batch", report.html)
        self.assertIn("overcritical", report.html)
        self.assertIn("operating_point", report.html)
        self.assertNotIn(str(self.workspace), report.html)

        script = """
import json
from pathlib import Path
from contextlib import ExitStack
from tests.test_harmonic_balance import _hb_dc_spec, _one_port_jj
from scnsim import CircuitRun, units as u
plan, port = _one_port_jj()
run = CircuitRun(backend="julia", plan=plan, workspace=Path(__import__('sys').argv[1]))
spec = _hb_dc_spec(port, ((\"off\", None), (\"small\", 1.0 * u.nA), (\"overcritical\", 10.0 * u.uA)))
result = run.resolve(run.original, spec)
print(json.dumps({
    'plan_sha256': result.identity.plan_sha256,
    'request_sha256': result.identity.request_sha256,
    'attempt_sha256': result.identity.attempt_sha256,
    'result_sha256': result.identity.result_sha256,
}, sort_keys=True))
"""
        output = subprocess.check_output([sys.executable, "-c", script, str(self.workspace)], cwd=Path(__file__).parents[1], text=True)
        self.assertEqual(json.loads(output)["result_sha256"], self.partial.identity.result_sha256)

        with TemporaryDirectory() as corrupted_root, ExitStack() as owners:
            corrupted = Path(corrupted_root, "workspace")
            shutil.copytree(self.workspace, corrupted)
            result_path = next(corrupted.glob(f"leaves/*/requests/{self.partial.identity.request_sha256}/attempts/000001/result.json"))
            result_path.write_text("{}", encoding="utf-8")
            plan, port = _one_port_jj()
            run = CircuitRun(backend="julia", plan=plan, workspace=corrupted)
            owners.callback(run.close)
            spec = _hb_dc_spec(port, (("off", None), ("small", 1.0 * u.nA), ("overcritical", 10.0 * u.uA)))
            with self.assertRaises(EvidenceIntegrityError):
                run.resolve(run.original, spec)
            with self.assertRaises(EvidenceIntegrityError):
                run.inventory()

    def test_nonzero_three_wave_pump_executes_with_generated_conjugate(self) -> None:
        plan, port = _one_port_jj()
        pump = PumpAxis(id="pump", frequency=6.0 * u.GHz)
        drive = CurrentDrive(id="pump_drive", at=port, mode=(2,))
        spec = HBSolveSpec(
            pump_axes=(pump,),
            drives=(drive,),
            frequencies=[5.5] * u.GHz,
            cases=(HBCaseSpec(id="pumped", currents={drive: 1.0 * u.nA}),),
            truncation=HBTruncation(
                pump_harmonics=(2,),
                modulation_harmonics=(2,),
                three_wave_mixing=True,
                four_wave_mixing=False,
            ),
        )
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            run = CircuitRun(backend="julia", plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            lattice = run.explain(run.original, spec).evidence["compiled"]["hb_preflight"]["lattice"]
            self.assertEqual([row["mode"] for row in lattice["operating_point_modes"]], [(2,)])
            self.assertEqual([row["mode"] for row in lattice["input_modes"]], [(0,), (1,), (-1,)])
            outcome = run.solve(run.original, spec).cases["pumped"]
            self.assertTrue(outcome.succeeded)
            self.assertEqual(outcome.bias_state, BiasState.OFF)
            self.assertEqual(outcome.pump_state, PumpState.ON)
            self.assertEqual(outcome.effective_sources[0]["mode"], (2,))
            self.assertEqual(outcome.effective_sources[0]["generated_conjugate"]["mode"], (-2,))
            self.assertEqual(outcome.effective_sources[0]["coefficient"], (1.0e-9 + 0.0j) * u.ampere)
            self.assertTrue(outcome.s.reconciliation.comparable)

    def test_driven_ptc_four_wave_and_direct_hb_independent_grids(self) -> None:
        plan, feed_in, probe_plus, probe_minus, plus, minus = _floating_probe_plan()
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            run = CircuitRun(backend="julia", plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            view = run.original.reduce(
                ReductionPipeline()
                .ptc(probe_plus, probe_minus)
                .transform_pair(plus, minus, id="qubit")
                .retain("feedline_in", "feedline_out", "qubit.differential")
            )
            direct = run.solve(
                view,
                DirectSolveSpec(frequencies=[5.4, 6.0, 6.6] * u.GHz),
            )
            pump = PumpAxis(id="pump", frequency=9.0 * u.GHz)
            drive = CurrentDrive(id="pump_drive", at=feed_in, mode=(1,))
            hb_spec = HBSolveSpec(
                pump_axes=(pump,),
                drives=(drive,),
                frequencies=[6.0] * u.GHz,
                cases=(HBCaseSpec(id="driven", currents={drive: 1.0 * u.pA}),),
                truncation=HBTruncation(
                    pump_harmonics=(3,),
                    modulation_harmonics=(1,),
                    three_wave_mixing=False,
                    four_wave_mixing=True,
                ),
                allow_driven_ptc=True,
            )
            explanation = run.explain(view, hb_spec)
            nonlinear_rows = [
                row
                for row in explanation.evidence["compiled"]["expanded_branch_rows"]
                if row["kind"] == "josephson_inductance"
            ]
            self.assertEqual(len(nonlinear_rows), 1)
            hb = run.solve(view, hb_spec)
            outcome = hb.cases["driven"]
            self.assertTrue(outcome.succeeded)
            self.assertEqual(outcome.pump_state, PumpState.ON)
            self.assertEqual(hb.topology_evidence["nonlinear_balance"]["load_state"], "loaded")
            self.assertEqual(hb.topology_evidence["response_linearization"]["load_state"], "compensated")
            self.assertEqual(outcome.s.view.probe_loads[probe_plus.id], "compensated")
            self.assertEqual(outcome.s.view.probe_loads[probe_minus.id], "compensated")
            self.assertEqual(outcome.s.view.probe_loads["feedline_in"], "raw")
            self.assertEqual(outcome.s.view.probe_loads["feedline_out"], "raw")
            self.assertTrue(all(state == "raw" for state in outcome.s.backend_native.probe_loads.values()))
            self.assertFalse(outcome.s.reconciliation.comparable)
            self.assertEqual(outcome.s.reconciliation.reason, "load_or_ptc")
            self.assertIsNone(outcome.s.reconciliation.residual)
            self.assertTrue(
                np.array_equal(
                    direct.frequencies.to("hertz").magnitude,
                    ([5.4, 6.0, 6.6] * u.GHz).to("hertz").magnitude,
                )
            )
            self.assertTrue(
                np.array_equal(
                    outcome.s.view.frequencies.to("hertz").magnitude,
                    ([6.0] * u.GHz).to("hertz").magnitude,
                )
            )
            self.assertEqual(direct.s.view.matrix.shape, (3, 3, 3))
            self.assertEqual(outcome.s.view.matrix.shape, (1, 3, 3))
            self.assertNotEqual(direct.identity.request_sha256, hb.identity.request_sha256)

            report = run.build_report(ReportSpec(inputs=(direct, hb)))
            self.assertIs(report.inputs[0], direct)
            self.assertIs(report.inputs[1], hb)
            self.assertIn(direct.identity.result_sha256, report.html)
            self.assertIn(hb.identity.result_sha256, report.html)
            self.assertNotIn(str(workspace), report.html)


if __name__ == "__main__":
    unittest.main()
