"""Retained ordinary Julia CMA retry/replay, distinct from JAX resume_from.

A real transport run writes real generation artifacts before the test injects
interruption. Retry uses actual Julia; no checkpoint data or success is forged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from contextlib import ExitStack
import unittest
from unittest.mock import patch

from scnsim import CircuitRun, EvidenceIntegrityError
from scnsim.canonical import canonical_json_bytes, sha256_hex
import scnsim.execution.coordinator as coordinator
from tests.test_workspace_readback import _primitive_plan, _primitive_requests


@unittest.skipUnless(
    os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
    "set SCNSIM_RUN_JULIA_TESTS=1 for retained packaged Julia execution",
)
class JuliaReplayTests(unittest.TestCase):
    def test_cma_interrupted_generation_replays_exactly_and_corruption_fails_closed(self) -> None:
        with TemporaryDirectory() as workspace, TemporaryDirectory() as reference_directory, ExitStack() as owners:
            plan, resonator, capacitor = _primitive_plan()
            run = CircuitRun(backend="julia", plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            _, _, optimization, view = _primitive_requests(run, resonator, capacitor)
            reference_workspace = Path(reference_directory)
            reference_plan, reference_node, reference_capacitance = _primitive_plan()
            reference_run = CircuitRun(backend="julia", plan=reference_plan, workspace=reference_workspace)
            owners.callback(reference_run.close)
            _, _, reference_spec, reference_view = _primitive_requests(reference_run, reference_node, reference_capacitance)
            reference = reference_run.optimize(reference_view, reference_spec)
            real_run_terminal = coordinator.run_terminal

            def complete_generation_then_interrupt(*args: object, **kwargs: object):
                real_run_terminal(*args, **kwargs)
                interruption = KeyboardInterrupt()
                interruption.termination = "terminated"  # type: ignore[attr-defined]
                raise interruption

            with patch.object(coordinator, "run_terminal", complete_generation_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run.optimize(view, optimization)

            interrupted = run.inventory().requests[0]
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["attempts"], ("000001",))
            request_sha = interrupted["request_sha256"]
            first_attempt, = Path(workspace).glob(f"leaves/*/requests/{request_sha}/attempts/000001")
            receipt_path = first_attempt / "receipt.json"
            receipt = json.loads(receipt_path.read_bytes())
            self.assertEqual(receipt["outcome"], "interrupted")
            self.assertEqual([row["id"] for row in receipt["artifacts"]], ["generation_000001"])
            ledger_path = first_attempt / "artifacts" / "generations" / "000001.json"
            ledger_bytes = ledger_path.read_bytes()
            ledger_sha = sha256_hex(ledger_bytes)
            self.assertEqual(receipt["artifacts"][0]["sha256"], ledger_sha)

            with TemporaryDirectory() as corrupted_root, ExitStack() as corrupted_owners:
                corrupted = Path(corrupted_root, "workspace")
                shutil.copytree(workspace, corrupted)
                corrupt_ledger_path = next(
                    corrupted.glob(
                        f"leaves/*/requests/{request_sha}/attempts/000001/artifacts/generations/000001.json"
                    )
                )
                corrupt_receipt_path = corrupt_ledger_path.parents[2] / "receipt.json"
                corrupt_ledger = json.loads(corrupt_ledger_path.read_bytes())
                corrupt_ledger["candidates"][0]["evaluation_ordinal"] += 1
                corrupt_ledger_bytes = canonical_json_bytes(corrupt_ledger)
                corrupt_ledger_path.write_bytes(corrupt_ledger_bytes)
                corrupt_receipt = json.loads(corrupt_receipt_path.read_bytes())
                corrupt_receipt["artifacts"][0]["sha256"] = sha256_hex(corrupt_ledger_bytes)
                corrupt_receipt_path.write_bytes(canonical_json_bytes(corrupt_receipt))

                corrupt_plan, corrupt_resonator, corrupt_capacitor = _primitive_plan()
                corrupt_run = CircuitRun(backend="julia", plan=corrupt_plan, workspace=corrupted)
                corrupted_owners.callback(corrupt_run.close)
                _, _, corrupt_spec, corrupt_view = _primitive_requests(
                    corrupt_run, corrupt_resonator, corrupt_capacitor
                )
                with self.assertRaises(EvidenceIntegrityError):
                    corrupt_run.optimize(corrupt_view, corrupt_spec)
                with self.assertRaises(EvidenceIntegrityError):
                    corrupt_run.inventory()
                self.assertFalse((corrupt_ledger_path.parents[3] / "000002").exists())

            resumed = run.optimize(view, optimization)
            completed = run.inventory().requests[0]
            self.assertEqual(completed["status"], "succeeded")
            self.assertEqual(completed["attempts"], ("000001", "000002"))
            second_attempt = first_attempt.parent / "000002"
            second_attempt_document = json.loads((second_attempt / "attempt.json").read_bytes())
            self.assertEqual(second_attempt_document["resume_ledger_sha256"], ledger_sha)
            self.assertEqual((second_attempt / "artifacts" / "generations" / "000001.json").read_bytes(), ledger_bytes)
            self.assertEqual(canonical_json_bytes(resumed.ledger[0]), ledger_bytes)
            self.assertEqual(resumed.best.cost.hex(), reference.best.cost.hex())
            resumed_best = resumed.best.parameters.values[capacitor].to("fF").magnitude
            reference_best = reference.best.parameters.values[reference_capacitance].to("fF").magnitude
            self.assertEqual(float(resumed_best).hex(), float(reference_best).hex())


if __name__ == "__main__":
    unittest.main()
