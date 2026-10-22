"""Retained Affine-support behavior and Julia CMA evidence regressions.

These cases preserve the accepted physical validation and CMA purposes from
the original Dev5 suite.  They do not freeze the separate, still-converging
authoring-override contract.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scnsim import (
    AffineMap,
    CMAESSpec,
    CircuitPlan,
    CircuitRun,
    CompositePlan,
    CostObjective,
    InvalidCandidatePhysicalParameter,
    SCNSimValidationError,
    Library,
    OperatorSpec,
    OptimizationSpec,
    OptimizationVariable,
    ParameterDefinitions,
    ParameterSet,
    ParameterSpec,
    ResponseElementSpec,
    ReductionPipeline,
    components,
    units as u,
)


class _AffineCatalog(Library):
    """Two accepted affine consumers with one exposed length parameter."""

    def mapped_resonator(self, *, id: str):
        composite = CompositePlan(id=id, library=self)
        definitions = ParameterDefinitions(id=f"{id}_definitions")
        length = definitions.parameter(
            id="length", baseline=1.0 * u.meter, spec=ParameterSpec(unit=u.meter)
        )
        first = composite.add(
            components.capacitor(
                id="first",
                capacitance=AffineMap(
                    input=length,
                    slope=20.0 * u.fF / u.meter,
                    intercept=10.0 * u.fF,
                    support=(0.99 * u.meter, 1.01 * u.meter),
                ),
            )
        )
        second = composite.add(
            components.capacitor(
                id="second",
                capacitance=AffineMap(
                    input=length,
                    slope=30.0 * u.fF / u.meter,
                    intercept=5.0 * u.fF,
                    support=(0.99 * u.meter, 1.01 * u.meter),
                ),
            )
        )
        inductor = composite.add(
            components.inductor(id="inductor", inductance=8.0 * u.nH)
        )
        terminal = composite.bus(id="terminal")
        composite.parallel(
            id="parallel_resonator",
            start=terminal,
            branches=((first,), (second,), (inductor,)),
            end=composite.ground,
        )
        composite.expose_parameter(id="length", parameter=length)
        composite.expose_pin(id="terminal", at=terminal)
        return composite.build()


def _affine_plan() -> tuple[CircuitPlan, object]:
    plan = CircuitPlan(id="affine_extrapolation_stabilization")
    resonator = plan.add(_AffineCatalog().mapped_resonator(id="mapped"))
    signal = plan.bus(id="signal")
    plan.link(
        id="resonator_boundary",
        endpoints=(signal, resonator.pin("terminal")),
    )
    plan.add_port(
        id="p",
        at=signal,
        role="terminated",
        reference_impedance=50.0 * u.ohm,
    )
    return plan, resonator.parameter("length")


def _receipt_for_result(workspace: Path, result: object) -> dict[str, object]:
    """Read the one sealed Julia receipt bound to this public Result identity."""
    identity = result.identity
    request_dir = workspace / "leaves"
    paths = tuple(
        request_dir.glob(
            f"*/requests/{identity.request_sha256}/attempts/*/receipt.json"
        )
    )
    matches = []
    for path in paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("attempt_sha256") == identity.attempt_sha256:
            matches.append(receipt)
    if len(matches) != 1:
        raise AssertionError("Result identity did not select exactly one sealed receipt")
    return matches[0]


class AffinePhysicalValidationTests(unittest.TestCase):
    def test_ordinary_unauthorized_extrapolation_is_an_authoring_error(self) -> None:
        plan, parameter = _affine_plan()
        with TemporaryDirectory() as workspace:
            run = CircuitRun(
                plan=plan,
                workspace=Path(workspace),
                backend="jax",
                precision="float64",
            )
            try:
                view = run.original.reduce(ReductionPipeline().retain("signal"))
                with self.assertRaises(SCNSimValidationError) as raised:
                    run.evaluate(
                        view,
                        OperatorSpec(frequencies=[5.0] * u.GHz),
                        parameters=ParameterSet({parameter: 1.05 * u.meter}),
                    )
                self.assertEqual(raised.exception.stage, "authoring")
            finally:
                run.close()


@unittest.skipUnless(
    os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
    "set SCNSIM_RUN_JULIA_TESTS=1 to run packaged Julia Affine regressions",
)
class JuliaAffineEvidenceTests(unittest.TestCase):
    def test_unauthorized_extrapolation_is_typed_and_creates_no_request(self) -> None:
        plan, parameter = _affine_plan()
        with TemporaryDirectory() as workspace:
            run = CircuitRun(
                plan=plan,
                workspace=Path(workspace),
                backend="julia",
                precision="float64",
            )
            try:
                view = run.original.reduce(ReductionPipeline().retain("signal"))
                with self.assertRaises(SCNSimValidationError) as raised:
                    run.evaluate(
                        view,
                        OperatorSpec(frequencies=[5.0] * u.GHz),
                        parameters=ParameterSet({parameter: 1.05 * u.meter}),
                    )
                self.assertEqual(raised.exception.stage, "authoring")
                self.assertEqual(run.inventory().requests, ())
            finally:
                run.close()

    def test_authorized_extrapolation_receipt_records_each_fanout_edge(self) -> None:
        plan, parameter = _affine_plan()
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            run = CircuitRun(
                plan=plan,
                workspace=root,
                backend="julia",
                precision="float64",
            )
            try:
                view = run.original.reduce(ReductionPipeline().retain("signal"))
                spec = OperatorSpec(frequencies=[5.0] * u.GHz)
                parameters = ParameterSet(
                    {parameter: 1.05 * u.meter},
                    allow_extrapolation=(parameter,),
                )
                result = run.evaluate(view, spec, parameters=parameters)
                receipt = _receipt_for_result(root, result)
                evidence = receipt["evidence"]["extrapolation_evidence"]
                self.assertEqual(len(evidence), 2)
                self.assertEqual(
                    [tuple(row["consumer_target"]["path"]) for row in evidence],
                    [("mapped", "first"), ("mapped", "second")],
                )
                self.assertTrue(all(row["side"] == "upper" for row in evidence))
                self.assertTrue(
                    all(
                        row["authorization_source"] == "parameter_set"
                        for row in evidence
                    )
                )
                resolved = run.resolve(view, spec, parameters=parameters)
                self.assertEqual(resolved.identity, result.identity)
            finally:
                run.close()

    def test_cma_retains_affine_rejections_and_authorized_candidates(self) -> None:
        plan, parameter = _affine_plan()
        with TemporaryDirectory() as workspace:
            run = CircuitRun(
                plan=plan,
                workspace=Path(workspace),
                backend="julia",
                precision="float64",
            )
            try:
                view = run.original.reduce(ReductionPipeline().retain("signal"))
                response = ResponseElementSpec(
                    family="Y",
                    input_coordinate="signal",
                    output_coordinate="signal",
                    frequency=5.0 * u.GHz,
                )
                spec = OptimizationSpec(
                    variables=(
                        OptimizationVariable(
                            parameter=parameter,
                            bounds=(0.9 * u.meter, 1.1 * u.meter),
                        ),
                    ),
                    objectives=(
                        CostObjective(
                            id="response",
                            quantity=response.magnitude,
                            target=1.0 * u.mS,
                            weight=1.0 * u.dimensionless,
                        ),
                    ),
                    optimizer=CMAESSpec(seed=23, max_evaluations=5),
                )

                result = run.optimize(view, spec)
                candidates = [
                    candidate
                    for generation in result.ledger
                    for candidate in generation["candidates"]
                ]
                rejected = [
                    candidate
                    for candidate in candidates
                    if candidate["outcome"]["status"] == "failure"
                ]
                self.assertTrue(rejected)
                for candidate in rejected:
                    self.assertEqual(candidate["outcome"]["penalty"], "positive_infinity")
                    self.assertEqual(
                        candidate["outcome"]["failure"]["kind"],
                        "invalid_candidate_physical_parameter",
                    )
                    self.assertEqual(
                        candidate["outcome"]["failure"]["stage"], "affine_support"
                    )
                    evidence = candidate["extrapolation_evidence"]
                    self.assertEqual(len(evidence), 2)
                    self.assertEqual(
                        [row["consumer_target"]["path"] for row in evidence],
                        [("mapped", "first"), ("mapped", "second")],
                    )
                    self.assertTrue(
                        all(row["authorization_source"] == "none" for row in evidence)
                    )
                self.assertEqual(run.resolve(view, spec).identity, result.identity)
                self.assertEqual(result.best.parameters.allow_extrapolation, ())

                authorized = run.optimize(
                    view,
                    spec.with_variable_overrides(
                        bounds={}, allow_extrapolation=(parameter,)
                    ),
                )
                authorized_candidates = [
                    candidate
                    for generation in authorized.ledger
                    for candidate in generation["candidates"]
                    if candidate["extrapolation_evidence"]
                ]
                self.assertTrue(authorized_candidates)
                for candidate in authorized_candidates:
                    self.assertEqual(candidate["outcome"]["status"], "success")
                    evidence = candidate["extrapolation_evidence"]
                    self.assertEqual(len(evidence), 2)
                    self.assertEqual(
                        [row["consumer_target"]["path"] for row in evidence],
                        [("mapped", "first"), ("mapped", "second")],
                    )
                    self.assertTrue(
                        all(
                            row["authorization_source"] == "optimization_spec"
                            for row in evidence
                        )
                    )
                    self.assertEqual(candidate["parameters"]["allow_extrapolation"], ())
                self.assertEqual(authorized.best.parameters.allow_extrapolation, ())
            finally:
                run.close()


if __name__ == "__main__":
    unittest.main()
