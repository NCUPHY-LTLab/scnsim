"""Same-process ordinary execution through the shared physical program.

Request/cache lookup, leases, durable sinks and result publication belong to the
coordinator. This module only expands declared parameter sources and dispatches
ordinary analysis; optimization has its separate stage/controller owner.
"""
from __future__ import annotations

import itertools

from ..compilation.compiler import parameter_key


def iter_resolved_points(source: dict):
    """Expand one declared point at a time without duplicating a whole sweep."""
    kind = source["kind"]
    if kind == "point":
        yield source["parameters"]
        return
    if kind == "points":
        # Listed records are validated sparse overrides of the complete baseline.
        # Expand operands here without rewriting their canonical request source.
        baseline = source["baseline_parameters"]
        order = [parameter_key(row["parameter"]) for row in baseline["bindings"]]
        for supplied in source["points"]:
            bindings = {parameter_key(row["parameter"]): row for row in baseline["bindings"]}
            bindings.update((parameter_key(row["parameter"]), row) for row in supplied["bindings"])
            authorizations = {parameter_key(ref): ref for ref in baseline["allow_extrapolation"]}
            authorizations.update((parameter_key(ref), ref) for ref in supplied["allow_extrapolation"])
            yield dict(baseline, bindings=[bindings[key] for key in order],
                       allow_extrapolation=[authorizations[key] for key in sorted(authorizations)])
        return
    if kind == "grid":
        for combination in itertools.product(*(axis["values"] for axis in source["axes"])):
            base = {parameter_key(binding["parameter"]): binding for binding in source["base_parameters"]["bindings"]}
            for axis, value in zip(source["axes"], combination):
                base[parameter_key(axis["parameter"])] = {"parameter": axis["parameter"], "value": value}
            yield dict(source["base_parameters"], bindings=list(base.values()))
        return
    raise ValueError(f"unsupported parameter source {kind!r}")


def resolved_points(source: dict) -> list[dict]:
    """Return the existing eager preparation view used by warmup only."""
    return list(iter_resolved_points(source))





def execute_analysis(plan_document, prepared_analysis, *, backend, emit, trace,
                     checkpoint=None, checkpoint_policy="generation",
                     operation_resources=None, worker_backend_factory=None,
                     program_owner=None, execution_policy=None, result_sink=None) -> dict:
    """Execute one ordinary request inside its caller's existing operation scope."""
    from .physical_analysis import PhysicalAnalysis
    analysis = prepared_analysis.request()
    if analysis['operation'] == 'optimize_direct':
        raise ValueError('optimize_direct belongs to the physical optimization coordinator')
    points = iter_resolved_points(analysis['parameter_source'])
    try:
        first_point = next(points)
    except StopIteration:
        first_point = None
    points = iter(()) if first_point is None else itertools.chain((first_point,), points)
    with trace.span('numerical_host', details={'operation': analysis['operation']}):
        return PhysicalAnalysis(plan_document, analysis, backend, trace, emit, points,
                                program_owner=program_owner,
                                execution_policy=execution_policy,
                                result_sink=result_sink,
                                operation_resources=operation_resources,
                                first_point=first_point).run(points)
