from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ..authoring import ParameterRef, ParameterSet
from ..construction import unavailable
from ..errors import SCNSimStateError
from ..visualization.presentation import Theme
from .base import AnalysisResult, LineDiscretization

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

@dataclass(frozen=True, slots=True)
class OptimizationBest:
    """A saved point selected under one explicit Optimization model."""

    parameters: ParameterSet
    cost: float
    discretization: tuple[LineDiscretization, ...] | None = None
    evaluation_ordinal: int | None = None
    status: str = "complete"
    model_identity: Mapping[str, object] | None = None
    locator: Mapping[str, object] | None = None

    def __init__(self) -> None:
        unavailable("OptimizationBest construction")


@dataclass(frozen=True, slots=True)
class OptimizationStage:
    """Compact, model-bound state and lazy history for one declared stage."""

    stage_id: str
    strategy: str
    model_identity: Mapping[str, object]
    objective_ids: tuple[str, ...]
    variable_refs: tuple[ParameterRef, ...]
    observed_best: OptimizationBest | None
    accepted_iterate: OptimizationBest | None
    termination_reason: str | None
    status: str
    evaluation_count: int
    iteration_count: int
    history: Sequence[Mapping[str, object]] = field(repr=False, compare=False)

    def __init__(self) -> None:
        unavailable("OptimizationStage construction")

    def __repr__(self) -> str:
        return (
            "OptimizationStage("
            f"id={self.stage_id!r}, strategy={self.strategy!r}, "
            f"status={self.status!r}, evaluations={self.evaluation_count}, "
            f"iterations={self.iteration_count})"
        )

@dataclass(frozen=True, slots=True)
class OptimizationResult(AnalysisResult):
    """Verified staged Optimization with model-specific lazy histories."""

    _fixed_reader: object = field(repr=False, compare=False)
    best: OptimizationBest | None = None
    stages: tuple[OptimizationStage, ...] = ()
    comparison_model_identity: Mapping[str, object] | None = None
    comparison_stage_id: str | None = None
    ledger: Sequence[Mapping[str, object]] = ()
    candidate_discretization: Sequence[tuple[LineDiscretization, ...] | None] = ()
    _presentation: Mapping[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __init__(self) -> None:
        unavailable("OptimizationResult construction")

    def __repr__(self) -> str:
        """Keep interactive inspection bounded without materializing history."""
        identity = self.identity
        best = "unavailable" if self.best is None else f"{self.best.cost:.12g}"
        candidate_count = (
            sum(stage.evaluation_count for stage in self.stages)
            if self.stages else len(self.candidate_discretization)
        )
        return (
            "OptimizationResult("
            f"request={identity.request_sha256[:12]}, "
            f"stages={len(self.stages)}, "
            f"candidates={candidate_count}, "
            f"best_cost={best})"
        )

    def stage(self, stage_id: str | None = None) -> OptimizationStage:
        """Select one exact stage; never combine histories across models."""

        if stage_id is None:
            if len(self.stages) == 1:
                return self.stages[0]
            if not self.stages:
                raise SCNSimStateError(
                    "Optimization Result has no stage projection",
                    stage="result_access",
                    evidence={"stage_count": 0},
                )
            raise SCNSimStateError(
                "stage_id is required when an Optimization Result has multiple stages",
                stage="result_access",
                evidence={"stage_ids": tuple(stage.stage_id for stage in self.stages)},
            )
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)

    def plot(
        self,
        *,
        kind: Literal["history", "objective", "residual", "parameter", "table", "comparison"] = "history",
        objective: str | None = None,
        parameter: ParameterRef | None = None,
        stage_id: str | None = None,
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ..visualization.plots.optimization import optimization_plot

        return optimization_plot(
            self, kind=kind, objective=objective, parameter=parameter,
            stage_id=stage_id, theme=theme
        )

    def show(self, **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        kind: Literal["history", "objective", "residual", "parameter", "table", "comparison"],
        objective: str | None = None,
        parameter: ParameterRef | None = None,
        stage_id: str | None = None,
    ) -> Figure:
        from ..visualization.plots.optimization import optimization_add_to

        return optimization_add_to(
            self,
            fig,
            row=row,
            col=col,
            kind=kind,
            objective=objective,
            parameter=parameter,
            stage_id=stage_id,
        )
