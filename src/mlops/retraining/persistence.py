from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.mlops.retraining.trigger_contract import (
    DriftSeverity,
    TriggerDecision,
)
from src.mlops.retraining.trigger_evaluator import (
    DriftSignal,
    TriggerEvaluation,
)


@dataclass(frozen=True)
class PersistenceThresholds:
    """Configurable temporal persistence policy."""

    significant_consecutive_windows: int = 2
    strong_consecutive_windows: int = 2

    def __post_init__(self) -> None:
        if self.significant_consecutive_windows < 1:
            raise ValueError(
                "significant_consecutive_windows must be at least 1."
            )

        if self.strong_consecutive_windows < 1:
            raise ValueError(
                "strong_consecutive_windows must be at least 1."
            )


DEFAULT_PERSISTENCE_THRESHOLDS = PersistenceThresholds()


@dataclass(frozen=True)
class SignalPersistence:
    """Persistence state for one named drift signal."""

    signal_name: str
    consecutive_significant_windows: int = 0
    consecutive_strong_windows: int = 0

    def __post_init__(self) -> None:
        if not self.signal_name.strip():
            raise ValueError("signal_name must not be empty.")

        if self.consecutive_significant_windows < 0:
            raise ValueError(
                "consecutive_significant_windows cannot be negative."
            )

        if self.consecutive_strong_windows < 0:
            raise ValueError(
                "consecutive_strong_windows cannot be negative."
            )


@dataclass(frozen=True)
class PersistentTriggerEvaluation:
    """Retraining decision after applying temporal persistence."""

    decision: TriggerDecision
    current_evaluation: TriggerEvaluation
    persistence: tuple[SignalPersistence, ...]
    persistence_reasons: tuple[str, ...]


def update_signal_persistence(
    previous: SignalPersistence,
    current_severity: DriftSeverity,
) -> SignalPersistence:
    """Update one signal's consecutive drift counters."""

    if current_severity == DriftSeverity.STRONG:
        return SignalPersistence(
            signal_name=previous.signal_name,
            consecutive_significant_windows=(
                previous.consecutive_significant_windows + 1
            ),
            consecutive_strong_windows=(
                previous.consecutive_strong_windows + 1
            ),
        )

    if current_severity == DriftSeverity.SIGNIFICANT:
        return SignalPersistence(
            signal_name=previous.signal_name,
            consecutive_significant_windows=(
                previous.consecutive_significant_windows + 1
            ),
            consecutive_strong_windows=0,
        )

    return SignalPersistence(
        signal_name=previous.signal_name,
        consecutive_significant_windows=0,
        consecutive_strong_windows=0,
    )


def build_persistence_state(
    previous_state: Iterable[SignalPersistence],
    current_signals: Iterable[DriftSignal],
) -> tuple[SignalPersistence, ...]:
    """Update persistence counters for the current monitoring window."""

    previous_by_name = {
        state.signal_name: state
        for state in previous_state
    }

    current_states: list[SignalPersistence] = []

    for signal in current_signals:
        previous = previous_by_name.get(
            signal.name,
            SignalPersistence(signal_name=signal.name),
        )

        current_states.append(
            update_signal_persistence(
                previous=previous,
                current_severity=signal.severity,
            )
        )

    return tuple(current_states)


def apply_persistence_policy(
    evaluation: TriggerEvaluation,
    persistence: Sequence[SignalPersistence],
    thresholds: PersistenceThresholds = (
        DEFAULT_PERSISTENCE_THRESHOLDS
    ),
) -> PersistentTriggerEvaluation:
    """Apply temporal persistence to a single-window evaluation."""

    persistence_by_name = {
        state.signal_name: state
        for state in persistence
    }

    reasons: list[str] = []

    # Data quality is never delayed by persistence.
    if evaluation.decision == TriggerDecision.QUALITY_BLOCK:
        return PersistentTriggerEvaluation(
            decision=TriggerDecision.QUALITY_BLOCK,
            current_evaluation=evaluation,
            persistence=tuple(persistence),
            persistence_reasons=(),
        )

    # A strong signal already satisfies the existing single-window
    # retraining policy. Persistence is therefore a guard for repeated
    # significant/strong drift, not a delay on an isolated strong event.
    for signal in evaluation.drift_signals:
        state = persistence_by_name[signal.name]

        if (
            signal.severity == DriftSeverity.STRONG
            and state.consecutive_strong_windows
            >= thresholds.strong_consecutive_windows
        ):
            reasons.append(
                f"strong_drift_persisted:{signal.name}"
            )

        if (
            signal.severity == DriftSeverity.SIGNIFICANT
            and state.consecutive_significant_windows
            >= thresholds.significant_consecutive_windows
        ):
            reasons.append(
                f"significant_drift_persisted:{signal.name}"
            )

    if reasons:
        return PersistentTriggerEvaluation(
            decision=TriggerDecision.RETRAIN,
            current_evaluation=evaluation,
            persistence=tuple(persistence),
            persistence_reasons=tuple(reasons),
        )

    return PersistentTriggerEvaluation(
        decision=evaluation.decision,
        current_evaluation=evaluation,
        persistence=tuple(persistence),
        persistence_reasons=(),
    )
