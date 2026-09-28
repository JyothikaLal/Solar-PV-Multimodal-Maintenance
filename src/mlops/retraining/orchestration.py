from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from src.mlops.retraining.persistence import (
    PersistenceThresholds,
    PersistentTriggerEvaluation,
    SignalPersistence,
    apply_persistence_policy,
    build_persistence_state,
)
from src.mlops.retraining.trigger_contract import (
    DriftSeverity,
    QualityThresholds,
    TriggerDecision,
)
from src.mlops.retraining.trigger_evaluator import (
    DriftSignal,
    QualityStatus,
    evaluate_retraining_trigger,
)


class RetrainingAction(str, Enum):
    NO_ACTION = "no_action"
    MONITOR = "monitor"
    RETRAIN = "retrain"
    QUALITY_BLOCK = "quality_block"


@dataclass(frozen=True)
class RetrainingOrchestrationInput:
    quality: QualityStatus
    drift_signals: Sequence[DriftSignal]
    previous_persistence: Sequence[SignalPersistence] = ()


@dataclass(frozen=True)
class RetrainingOrchestrationResult:
    action: RetrainingAction
    trigger_evaluation: object
    persistence_evaluation: PersistentTriggerEvaluation
    persistence_state: tuple[SignalPersistence, ...]


def _map_action(decision: TriggerDecision) -> RetrainingAction:
    mapping = {
        TriggerDecision.NO_ACTION: RetrainingAction.NO_ACTION,
        TriggerDecision.MONITOR: RetrainingAction.MONITOR,
        TriggerDecision.RETRAIN: RetrainingAction.RETRAIN,
        TriggerDecision.QUALITY_BLOCK: RetrainingAction.QUALITY_BLOCK,
    }
    return mapping[decision]


def evaluate_retraining_orchestration(
    request: RetrainingOrchestrationInput,
    *,
    quality_thresholds: QualityThresholds | None = None,
    persistence_thresholds: PersistenceThresholds | None = None,
) -> RetrainingOrchestrationResult:
    trigger_kwargs = {
        "quality": request.quality,
        "drift_signals": request.drift_signals,
    }

    if quality_thresholds is not None:
        trigger_kwargs["quality_thresholds"] = quality_thresholds

    trigger_evaluation = evaluate_retraining_trigger(**trigger_kwargs)

    persistence_state = build_persistence_state(
        previous_state=request.previous_persistence,
        current_signals=request.drift_signals,
    )

    persistence_kwargs = {
        "evaluation": trigger_evaluation,
        "persistence": persistence_state,
    }

    if persistence_thresholds is not None:
        persistence_kwargs["thresholds"] = persistence_thresholds

    persistence_evaluation = apply_persistence_policy(**persistence_kwargs)

    return RetrainingOrchestrationResult(
        action=_map_action(persistence_evaluation.decision),
        trigger_evaluation=trigger_evaluation,
        persistence_evaluation=persistence_evaluation,
        persistence_state=tuple(persistence_state),
    )
