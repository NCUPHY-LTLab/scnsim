"""Pure selectors over already-materialized numerical Results.

This module owns exact channel, frequency, and optimization-history lookup.
It has no plotting dependency so specifications and presentation can share
the same selection behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pint import Quantity

Channel = str | tuple[str, tuple[int, ...]]


def _optimizer_unavailability_text(value: object) -> str:
    """Keep optimizer rejection visible without changing the primal status."""

    if value is None:
        return ""
    if not isinstance(value, Mapping):
        raise TypeError("Optimization optimizer_unavailability projection is malformed")
    role = value["role"]
    kind = value["kind"]
    detail = value["detail"]
    evidence = value["evidence"]
    if not isinstance(role, str) or not isinstance(kind, str) or not isinstance(detail, str):
        raise TypeError("Optimization optimizer_unavailability projection is malformed")
    if not isinstance(evidence, Mapping):
        raise TypeError("Optimization optimizer_unavailability evidence is malformed")
    return f"optimizer unavailable ({role}; {kind}): {detail}; evidence={evidence!r}"


def _channel_index(
    channels: Sequence[tuple[str, tuple[int, ...]]],
    selector: Channel | None,
    *,
    role: str,
    default: bool,
) -> int:
    if not channels:
        raise ValueError(f"matrix has no {role} channels")
    if selector is None:
        if default:
            return 0
        raise ValueError(f"{role}_channel is required")
    if isinstance(selector, str):
        matches = [index for index, (coordinate, _) in enumerate(channels) if coordinate == selector]
        if not matches:
            raise ValueError(f"unknown {role} channel coordinate: {selector}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous {role} channel coordinate; use (coordinate, mode)")
        return matches[0]
    if (
        not isinstance(selector, tuple)
        or len(selector) != 2
        or not isinstance(selector[0], str)
        or not isinstance(selector[1], tuple)
        or any(not isinstance(value, int) or isinstance(value, bool) for value in selector[1])
    ):
        raise TypeError(f"{role}_channel must be a coordinate string or (coordinate, mode) tuple")
    try:
        return channels.index(selector)
    except ValueError as error:
        raise ValueError(f"unknown {role} channel: {selector!r}") from error


def _frequency_index(view: Any, frequency: Quantity | None) -> int:
    if not isinstance(frequency, Quantity) or np.asarray(frequency.magnitude).ndim != 0:
        raise TypeError("frequency must be one scalar Quantity")
    try:
        wanted = float(frequency.to(view.frequencies.units).magnitude)
    except Exception as error:
        raise ValueError("frequency is incompatible with the stored grid") from error
    values = np.asarray(view.frequencies.magnitude)
    matches = np.flatnonzero(values == wanted)
    if matches.size != 1:
        raise KeyError("frequency was not materialized exactly once")
    return int(matches[0])


def _optimization_series(
    result: Any,
    *,
    kind: Literal["history", "objective", "residual", "parameter"],
    objective: str | None,
    parameter: Any | None,
    stage_id: str | None = None,
) -> tuple[list[int], list[float | None], list[str], str, str, list[str]]:
    from ..canonical import float64_from_hex
    from ..authoring import ParameterRef
    from .memory_optimization import InMemoryOptimizationResult
    from .optimization_projection import _candidate_cost

    if isinstance(result, InMemoryOptimizationResult):
        return _in_memory_optimization_series(
            result,
            kind=kind,
            objective=objective,
            parameter=parameter,
            stage_id=stage_id,
        )

    selected_stage = None
    if result.stages:
        selected_stage = result.stage(stage_id)
    elif stage_id is not None:
        raise ValueError("stage_id is unavailable for this Optimization Result")

    if kind in {"objective", "residual"}:
        if not isinstance(objective, str) or not objective:
            raise ValueError(f"kind={kind!r} requires objective=<objective ID>")
        if parameter is not None:
            raise ValueError("parameter is valid only for kind='parameter'")
    elif kind == "parameter":
        if not isinstance(parameter, ParameterRef):
            raise TypeError("kind='parameter' requires parameter=ParameterRef")
        if objective is not None:
            raise ValueError("objective is valid only for objective or residual history")
        if selected_stage is None:
            if result.best is None:
                raise ValueError("parameter history requires a declared Optimization stage")
            retained = next(
                (
                    item
                    for item in result.best.parameters.values
                    if item.definitions_id == parameter.definitions_id
                    and item.id == parameter.id
                ),
                None,
            )
        else:
            retained = next(
                (
                    item
                    for item in selected_stage.variable_refs
                    if item.definitions_id == parameter.definitions_id
                    and item.id == parameter.id
                ),
                None,
            )
        if retained is None:
            raise ValueError(
                "selected parameter history is absent from the selected Optimization model: "
                f"{parameter.definitions_id}.{parameter.id}"
            )
        if retained._definition_record() != parameter._definition_record():
            raise ValueError(
                "selected parameter definition disagrees with the Optimization Result"
            )
        parameter = retained
    elif objective is not None or parameter is not None:
        raise ValueError("objective and parameter are invalid for total-cost history")

    selector = (
        objective if kind in {"objective", "residual"}
        else None if kind != "parameter"
        else {"definitions_id": parameter.definitions_id, "parameter_id": parameter.id}
    )
    if selected_stage is None:
        rows = result._fixed_reader.project(kind, selector)
    else:
        if kind in {"objective", "residual"} and objective not in selected_stage.objective_ids:
            raise ValueError(
                f"selected objective is absent from Optimization stage {selected_stage.stage_id!r}: {objective}"
            )
        rows = selected_stage.history
    ordinals: list[int] = []
    values: list[float | None] = []
    statuses: list[str] = []
    details: list[str] = []
    found = False
    unit = "dimensionless"
    for row in rows:
        ordinal = row["evaluation_ordinal"]
        value: float | None = None
        status = str(row["status"])
        component_details = "; ".join(
            f"{item.get('objective_id')}: {item.get('status')} "
            f"({len(item.get('terms', ())) } terms)"
            for item in row.get("components", ())
        )
        failure = row.get("failure")
        failure_detail = ""
        if isinstance(failure, Mapping):
            kind_name = failure.get("kind")
            stage_name = failure.get("stage")
            message = failure.get("detail")
            if message is None:
                message = failure.get("message")
            failure_detail = ": ".join(
                item for item in (
                    " @ ".join(str(part) for part in (kind_name, stage_name) if part),
                    None if message is None else str(message),
                ) if item
            )
        optimizer_detail = _optimizer_unavailability_text(
            row.get("optimizer_unavailability")
        )
        details.append("; ".join(
            item for item in (component_details, failure_detail, optimizer_detail) if item
        ))
        if kind == "history":
            encoded = row.get("cost_f64")
            value = _candidate_cost(row, status) if isinstance(encoded, str) else None
        elif kind in {"objective", "residual"}:
            components = row.get("components", ())
            component = next(
                (item for item in components if item["objective_id"] == objective),
                None,
            )
            if component is not None:
                found = True
                status = str(component["status"])
                field = "weighted_cost_f64" if kind == "objective" else "normalized_residual_f64"
                encoded = component.get(field)
                value = float64_from_hex(encoded) if isinstance(encoded, str) else None
        else:
            bindings = row["parameters"]["bindings"]
            key = {"definitions_id": parameter.definitions_id, "parameter_id": parameter.id}
            binding = next(
                (item for item in bindings if item["parameter"] == key),
                None,
            )
            if binding is not None:
                found = True
                envelope = binding["value"]
                quantity = Quantity(
                    float64_from_hex(envelope["si_value_f64"]), envelope["si_unit"]
                ).to(parameter.spec.si_unit)
                value = float(quantity.magnitude)
                unit = str(quantity.units)
        ordinals.append(ordinal)
        values.append(value)
        statuses.append(status)
    if kind in {"objective", "residual", "parameter"} and not found:
        selected = objective if parameter is None else f"{parameter.definitions_id}.{parameter.id}"
        raise ValueError(f"selected {kind} history is absent from the completed ledger: {selected}")
    if kind == "history":
        label = "total cost"
    elif kind == "objective":
        label = f"{objective} weighted cost"
    elif kind == "residual":
        label = f"{objective} normalized residual"
    else:
        label = f"{parameter.definitions_id}.{parameter.id}"
    return ordinals, values, statuses, label, unit, details


def _in_memory_optimization_series(
    result: Any,
    *,
    kind: Literal["history", "objective", "residual", "parameter"],
    objective: str | None,
    parameter: Any | None,
    stage_id: str | None,
) -> tuple[list[int], list[float | None], list[str], str, str, list[str]]:
    """Project already-materialized trial values without a fixed reader."""

    from ..authoring import ParameterRef

    selected_stage = result.stage(stage_id) if result.stages else None
    if not result.stages and stage_id is not None:
        raise ValueError("stage_id is unavailable for this in-memory Optimization Result")
    if kind in {"objective", "residual"}:
        if not isinstance(objective, str) or not objective:
            raise ValueError(f"kind={kind!r} requires objective=<objective ID>")
        if parameter is not None:
            raise ValueError("parameter is valid only for kind='parameter'")
    elif kind == "parameter":
        if not isinstance(parameter, ParameterRef):
            raise TypeError("kind='parameter' requires parameter=ParameterRef")
        if objective is not None:
            raise ValueError("objective is valid only for objective or residual history")
        if selected_stage is not None:
            retained = next(
                (ref for ref in selected_stage.variable_refs
                 if ref.definitions_id == parameter.definitions_id and ref.id == parameter.id),
                None,
            )
        elif result.best is not None:
            retained = next(
                (ref for ref, _ in result.best.parameters.values.items()
                 if ref.definitions_id == parameter.definitions_id and ref.id == parameter.id),
                None,
            )
        else:
            retained = None
        if retained is None:
            raise ValueError(
                "selected parameter history is absent from the in-memory Optimization model: "
                f"{parameter.definitions_id}.{parameter.id}"
            )
        if retained._definition_record() != parameter._definition_record():
            raise ValueError("selected parameter definition disagrees with the Optimization Result")
        parameter = retained
    elif objective is not None or parameter is not None:
        raise ValueError("objective and parameter are invalid for total-cost history")

    if selected_stage is not None:
        if kind in {"objective", "residual"} and objective not in selected_stage.objective_ids:
            raise ValueError(
                f"selected objective is absent from Optimization stage {selected_stage.stage_id!r}: {objective}"
            )
        rows = selected_stage.trials
    else:
        rows = result.trials

    ordinals: list[int] = []
    values: list[float | None] = []
    statuses: list[str] = []
    details: list[str] = []
    found = False
    unit = "dimensionless"
    for row in rows:
        status = row.status
        value: float | None = None
        component_details = "; ".join(
            f"{item['id']}: {item['status']} ({len(item['terms'])} terms)"
            for item in row.objective_components
        )
        failure = row.failure
        failure_detail = ""
        if failure is not None:
            failure_detail = ": ".join(
                str(part) for part in (
                    " @ ".join(str(item) for item in (failure.get("kind"), failure.get("stage")) if item),
                    failure.get("detail"),
                ) if part
            )
        optimizer_detail = _optimizer_unavailability_text(
            row.optimizer_unavailability
        )
        details.append("; ".join(
            item for item in (component_details, failure_detail, optimizer_detail) if item
        ))
        if kind == "history":
            value = row.cost
        elif kind in {"objective", "residual"}:
            component = next(
                (item for item in row.objective_components if item["id"] == objective),
                None,
            )
            if component is not None:
                found = True
                status = str(component["status"])
                if status == "success":
                    field = "weighted_cost" if kind == "objective" else "normalized_residual"
                    value = float(component[field])
        else:
            selected = next(
                (raw for ref, raw in row.parameters.values.items()
                 if ref.definitions_id == parameter.definitions_id and ref.id == parameter.id),
                None,
            )
            if selected is not None:
                found = True
                selected = selected.to(parameter.spec.si_unit)
                value = float(selected.magnitude)
                unit = str(selected.units)
        ordinals.append(row.evaluation_ordinal)
        values.append(value)
        statuses.append(status)
    if kind in {"objective", "residual", "parameter"} and not found:
        selected = objective if parameter is None else f"{parameter.definitions_id}.{parameter.id}"
        raise ValueError(f"selected {kind} history is absent from the in-memory ledger: {selected}")
    if kind == "history":
        label = "total cost"
    elif kind == "objective":
        label = f"{objective} weighted cost"
    elif kind == "residual":
        label = f"{objective} normalized residual"
    else:
        label = f"{parameter.definitions_id}.{parameter.id}"
    return ordinals, values, statuses, label, unit, details


__all__ = ["Channel", "_channel_index", "_frequency_index", "_optimization_series"]
