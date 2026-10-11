"""Same-process ordinary execution through the shared physical program.

Request/cache lookup, leases, durable sinks and result publication belong to the
coordinator. This module only expands declared parameter sources and dispatches
ordinary analysis; optimization has its separate stage/controller owner.
"""
from __future__ import annotations

import itertools

from ..compilation.compiler import parameter_key


def resolved_points(source: dict) -> list[dict]:
    kind = source["kind"]
    if kind == "point":
        return [source["parameters"]]
    if kind == "points":
        # Listed records are validated sparse overrides of the complete baseline.
        # Expand operands here without rewriting their canonical request source.
        baseline = source["baseline_parameters"]
        order = [parameter_key(row["parameter"]) for row in baseline["bindings"]]
        points = []
        for supplied in source["points"]:
            bindings = {parameter_key(row["parameter"]): row for row in baseline["bindings"]}
            bindings.update((parameter_key(row["parameter"]), row) for row in supplied["bindings"])
            authorizations = {parameter_key(ref): ref for ref in baseline["allow_extrapolation"]}
            authorizations.update((parameter_key(ref), ref) for ref in supplied["allow_extrapolation"])
            points.append(dict(baseline, bindings=[bindings[key] for key in order],
                               allow_extrapolation=[authorizations[key] for key in sorted(authorizations)]))
        return points
    if kind == "grid":
        points = []
        for combination in itertools.product(*(axis["values"] for axis in source["axes"])):
            base = {parameter_key(binding["parameter"]): binding for binding in source["base_parameters"]["bindings"]}
            for axis, value in zip(source["axes"], combination):
                base[parameter_key(axis["parameter"])] = {"parameter": axis["parameter"], "value": value}
            points.append(dict(source["base_parameters"], bindings=list(base.values())))
        return points
    raise ValueError(f"unsupported parameter source {kind!r}")





def execute_analysis(plan_document, prepared_analysis, *, backend, emit, trace,
                     checkpoint=None, checkpoint_policy="generation",
                     operation_resources=None, worker_backend_factory=None,
                     program_owner=None) -> dict:
    """Execute one ordinary request inside its caller's existing operation scope."""
    from .physical_analysis import PhysicalAnalysis
    analysis = prepared_analysis.request()
    if analysis['operation'] == 'optimize_direct':
        raise ValueError('optimize_direct belongs to the physical optimization coordinator')
    points = resolved_points(analysis['parameter_source'])
    with trace.span('numerical_host', details={'operation': analysis['operation']}):
        return PhysicalAnalysis(plan_document, analysis, backend, trace, emit, points,
                                program_owner=program_owner).run(points)
