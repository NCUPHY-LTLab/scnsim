"""JAX Workspace versioning and separately retained Julia attempt readback."""

from __future__ import annotations

import json
import os
from pathlib import Path
from contextlib import ExitStack
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import scnsim.execution.coordinator as coordinator

from scnsim import (
    BackendProtocolError,
    CMAESSpec,
    CircuitPlan,
    CircuitRun,
    CostObjective,
    DiagonalRootSpec,
    DirectSolveSpec,
    EvidenceIntegrityError,
    OptimizationSpec,
    ParameterDefinitions,
    ParameterSpec,
    OptimizationVariable,
    ReductionPipeline,
    ResultUnavailableError,
    WorkspaceVersioningDowngradeForbidden,
    components,
    units as u,
)


def _primitive_plan(*, plan_id="primitive_full_v1", capacitance=110.0):
    plan = CircuitPlan(id=plan_id)
    inputs = ParameterDefinitions(id="primitive_inputs")
    capacitance_ref = inputs.parameter(id="capacitance", baseline=capacitance*u.fF,
                                      spec=ParameterSpec(unit=u.fF))
    capacitor = plan.add(components.capacitor(id="resonator_cap", capacitance=capacitance_ref))
    inductor = plan.add(components.inductor(id="resonator_ind", inductance=5.8*u.nH))
    node = plan.bus(id="resonator_node")
    plan.parallel(id="parallel_lc", start=node, branches=((capacitor,), (inductor,)), end=plan.ground)
    coordinate = node.node
    boundary = plan.bus(id="boundary")
    coupling = plan.add(components.capacitor(id="coupling_cap", capacitance=6.0*u.fF))
    plan.series(id="coupler", start=boundary, elements=(coupling,), end=node)
    plan.add_port(id="signal_in", at=boundary, role="terminated", reference_impedance=50.0*u.ohm)
    return plan, coordinate, capacitance_ref


def _primitive_requests(run, resonator, capacitance):
    direct = DirectSolveSpec(frequencies=[5.5, 6.0, 6.5]*u.GHz)
    root = DiagonalRootSpec(coordinate=resonator, root_hint=6.0*u.GHz)
    optimization = OptimizationSpec(
        variables=(OptimizationVariable(parameter=capacitance, bounds=(80.0*u.fF, 140.0*u.fF)),),
        objectives=(CostObjective(id="resonance_frequency", quantity=root.frequency,
                                  target=6.2*u.GHz, weight=1.0*u.dimensionless),),
        optimizer=CMAESSpec(seed=17, max_evaluations=5))
    view = run.original.reduce(ReductionPipeline().retain(resonator))
    return direct, root, optimization, view


class WorkspaceVersioningTests(unittest.TestCase):
    def test_versioning_upgrade_preserves_leaf_and_forbids_downgrade(self):
        # Documented Workspace identities, not Run's private binding object.
        with TemporaryDirectory() as directory, ExitStack() as owners:
            workspace = Path(directory)
            plan, _, _ = _primitive_plan(plan_id="first")
            first = CircuitRun(plan=plan, workspace=workspace, backend="jax")
            owners.callback(first.close)
            original = json.loads((workspace / "workspace.json").read_bytes())["active_leaf"]
            plan, _, _ = _primitive_plan(plan_id="first")
            upgraded = CircuitRun(plan=plan, workspace=workspace, versioned=True, backend="jax")
            owners.callback(upgraded.close)
            index = json.loads((workspace / "workspace.json").read_bytes())
            self.assertEqual(index["iterations"][0]["workspace_instance_id"], original["workspace_instance_id"])
            self.assertEqual(index["iterations"][0]["plan_sha256"], original["plan_sha256"])
            plan, _, _ = _primitive_plan(plan_id="second", capacitance=120.0)
            second = CircuitRun(plan=plan, workspace=workspace, versioned=True, backend="jax")
            owners.callback(second.close)
            index = json.loads((workspace / "workspace.json").read_bytes())
            self.assertNotEqual(index["iterations"][1]["workspace_instance_id"], original["workspace_instance_id"])
            with self.assertRaises(WorkspaceVersioningDowngradeForbidden):
                plan, _, _ = _primitive_plan(plan_id="second", capacitance=120.0)
                CircuitRun(plan=plan, workspace=workspace, versioned=False, backend="jax")


@unittest.skipUnless(os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
                     "deferred Julia Workspace inventory/version evidence")
class JuliaWorkspaceVersioningInventoryTests(unittest.TestCase):
    def test_versioning_upgrade_preserves_leaf_and_forbids_downgrade(self):
        # Documented Workspace identities, not Run's private binding object.
        with TemporaryDirectory() as directory, ExitStack() as owners:
            workspace = Path(directory)
            plan, _, _ = _primitive_plan(plan_id="first")
            first = CircuitRun(plan=plan, workspace=workspace, backend="julia")
            owners.callback(first.close)
            original = json.loads((workspace / "workspace.json").read_bytes())["active_leaf"]
            plan, _, _ = _primitive_plan(plan_id="first")
            upgraded = CircuitRun(plan=plan, workspace=workspace, versioned=True, backend="julia")
            owners.callback(upgraded.close)
            index = json.loads((workspace / "workspace.json").read_bytes())
            self.assertEqual(index["iterations"][0]["workspace_instance_id"], original["workspace_instance_id"])
            self.assertEqual(index["iterations"][0]["plan_sha256"], original["plan_sha256"])
            self.assertEqual(first.inventory().requests, ())
            self.assertEqual(upgraded.inventory().requests, ())
            plan, _, _ = _primitive_plan(plan_id="second", capacitance=120.0)
            second = CircuitRun(plan=plan, workspace=workspace, versioned=True, backend="julia")
            owners.callback(second.close)
            index = json.loads((workspace / "workspace.json").read_bytes())
            self.assertNotEqual(index["iterations"][1]["workspace_instance_id"], original["workspace_instance_id"])
            self.assertEqual(first.inventory().requests, ())
            self.assertEqual(second.inventory().requests, ())
            with self.assertRaises(WorkspaceVersioningDowngradeForbidden):
                plan, _, _ = _primitive_plan(plan_id="second", capacitance=120.0)
                CircuitRun(plan=plan, workspace=workspace, versioned=False, backend="julia")


@unittest.skipUnless(
    os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
    "set SCNSIM_RUN_JULIA_TESTS=1 for retained packaged Julia execution",
)
class WorkspaceReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = TemporaryDirectory()
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.workspace = Path(cls._temporary.name)
        cls.plan, cls.resonator, cls.capacitor = _primitive_plan()
        cls.circuit_run = CircuitRun(backend="julia", plan=cls.plan, workspace=cls.workspace)
        cls.addClassCleanup(cls.circuit_run.close)
        cls.direct_spec, cls.root_spec, cls.optimization_spec, cls.quantity_view = _primitive_requests(
            cls.circuit_run, cls.resonator, cls.capacitor
        )
        cls.direct = cls.circuit_run.solve(cls.circuit_run.original, cls.direct_spec)
        cls.root = cls.circuit_run.evaluate(cls.quantity_view, cls.root_spec)
        cls.optimization = cls.circuit_run.optimize(cls.quantity_view, cls.optimization_spec)

    def test_direct_root_and_cma_share_one_plan_without_identity_aliasing(self) -> None:
        identities = (self.direct.identity, self.root.identity, self.optimization.identity)
        self.assertEqual(len({identity.plan_sha256 for identity in identities}), 1)
        self.assertEqual(len({identity.request_sha256 for identity in identities}), 3)
        self.assertEqual(self.direct.s.view.matrix.shape, (3, 1, 1))
        self.assertTrue(np.isfinite(self.root.frequency.to("hertz").magnitude))
        self.assertTrue(np.isfinite(self.root.linewidth.to("hertz").magnitude))
        self.assertTrue(self.optimization.ledger)
        self.assertEqual(self.optimization.best.parameters.allow_extrapolation, ())
        best = self.optimization.best.parameters.values[self.capacitor].to("fF").magnitude
        self.assertGreaterEqual(best, 80.0)
        self.assertLessEqual(best, 140.0)

    def test_success_reuse_inventory_and_fresh_process_resolve_are_exact(self) -> None:
        self.assertEqual(
            self.circuit_run.solve(self.circuit_run.original, self.direct_spec).identity,
            self.direct.identity,
        )
        self.assertEqual(
            self.circuit_run.evaluate(self.quantity_view, self.root_spec).identity,
            self.root.identity,
        )
        self.assertEqual(
            self.circuit_run.optimize(self.quantity_view, self.optimization_spec).identity,
            self.optimization.identity,
        )
        inventory = self.circuit_run.inventory().requests
        self.assertEqual(len(inventory), 3)
        self.assertTrue(all(row["status"] == "succeeded" for row in inventory))
        self.assertTrue(all(row["attempts"] == ("000001",) for row in inventory))

        script = """
import json
from pathlib import Path
from tests.test_workspace_readback import _primitive_plan, _primitive_requests
from scnsim import CircuitRun
from unittest.mock import patch
import scnsim.execution.coordinator as coordinator
plan, node, capacitor = _primitive_plan()
run = CircuitRun(backend="julia", plan=plan, workspace=Path(__import__('sys').argv[1]))
direct, root, optimization, view = _primitive_requests(run, node, capacitor)
with patch.object(coordinator, "run_terminal", side_effect=AssertionError("resolve must not execute")):
    results = (
        run.resolve(run.original, direct),
        run.resolve(view, root),
        run.resolve(view, optimization),
    )
run.close()
print(json.dumps([result.identity.result_sha256 for result in results]))
"""
        output = subprocess.check_output(
            [sys.executable, "-c", script, str(self.workspace)],
            cwd=Path(__file__).parents[1],
            text=True,
        )
        self.assertEqual(
            json.loads(output),
            [self.direct.identity.result_sha256, self.root.identity.result_sha256, self.optimization.identity.result_sha256],
        )

    def test_failed_and_interrupted_attempts_are_visible_and_retry_appends_success(self) -> None:
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            plan, _, _ = _primitive_plan(plan_id="attempt_history")
            run = CircuitRun(backend="julia", plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            spec = DirectSolveSpec(frequencies=[5.5] * u.GHz)

            with patch.object(
                coordinator,
                "run_terminal",
                side_effect=BackendProtocolError("deterministic launch failure", stage="process_start"),
            ):
                with self.assertRaises(BackendProtocolError):
                    run.solve(run.original, spec)
            failed = run.inventory().requests
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["status"], "failed")
            self.assertEqual(failed[0]["attempts"], ("000001",))
            with self.assertRaises(ResultUnavailableError):
                run.resolve(run.original, spec)
            self.assertEqual(run.inventory().requests[0]["attempts"], ("000001",))

            with patch.object(coordinator, "run_terminal", side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    run.solve(run.original, spec)
            interrupted = run.inventory().requests[0]
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["attempts"], ("000001", "000002"))
            with self.assertRaises(ResultUnavailableError):
                run.resolve(run.original, spec)
            self.assertEqual(run.inventory().requests[0]["attempts"], interrupted["attempts"])

            result = run.solve(run.original, spec)
            completed = run.inventory().requests[0]
            self.assertEqual(completed["status"], "succeeded")
            self.assertEqual(completed["attempts"], ("000001", "000002", "000003"))
            attempt_root, = Path(workspace).glob(f"leaves/*/requests/{completed['request_sha256']}/attempts")
            self.assertEqual(
                [json.loads((attempt_root / ordinal / "receipt.json").read_bytes())["outcome"] for ordinal in completed["attempts"]],
                ["failure", "interrupted", "success"],
            )
            self.assertEqual(run.resolve(run.original, spec).identity, result.identity)
            self.assertEqual(run.solve(run.original, spec).identity, result.identity)
            self.assertEqual(run.inventory().requests[0]["attempts"], completed["attempts"])


if __name__ == "__main__":
    unittest.main()
