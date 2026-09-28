from __future__ import annotations

from src.mlops.retraining.orchestration import (
    RetrainingAction,
    RetrainingOrchestrationInput,
    evaluate_retraining_orchestration,
)
from src.mlops.retraining.persistence import SignalPersistence
from src.mlops.retraining.trigger_contract import (
    DriftSeverity,
    TriggerDecision,
)
from src.mlops.retraining.trigger_evaluator import (
    DriftSignal,
    QualityStatus,
)


def good_quality() -> QualityStatus:
    return QualityStatus()


def signal(name: str, severity: DriftSeverity) -> DriftSignal:
    return DriftSignal(
        name=name,
        severity=severity,
    )


def test_no_drift_results_in_no_action():
    result = evaluate_retraining_orchestration(
        RetrainingOrchestrationInput(
            quality=good_quality(),
            drift_signals=[
                signal("feature_a", DriftSeverity.NONE),
            ],
        )
    )

    assert result.action == RetrainingAction.NO_ACTION
    assert result.persistence_state


def test_monitor_drift_results_in_monitor():
    result = evaluate_retraining_orchestration(
        RetrainingOrchestrationInput(
            quality=good_quality(),
            drift_signals=[
                signal("feature_a", DriftSeverity.MONITOR),
            ],
        )
    )

    assert result.action == RetrainingAction.MONITOR


def test_two_significant_signals_trigger_retraining():
    result = evaluate_retraining_orchestration(
        RetrainingOrchestrationInput(
            quality=good_quality(),
            drift_signals=[
                signal("feature_a", DriftSeverity.SIGNIFICANT),
                signal("feature_b", DriftSeverity.SIGNIFICANT),
            ],
        )
    )

    assert result.action == RetrainingAction.RETRAIN


def test_single_significant_signal_first_window_is_monitor():
    result = evaluate_retraining_orchestration(
        RetrainingOrchestrationInput(
            quality=good_quality(),
            drift_signals=[
                signal("feature_a", DriftSeverity.SIGNIFICANT),
            ],
        )
    )

    assert result.action == RetrainingAction.MONITOR


def test_single_significant_signal_second_window_retrains():
    previous = (
        SignalPersistence(
            signal_name="feature_a",
            consecutive_significant_windows=1,
            consecutive_strong_windows=0,
        ),
    )

    result = evaluate_retraining_orchestration(
        RetrainingOrchestrationInput(
            quality=good_quality(),
            drift_signals=[
                signal("feature_a", DriftSeverity.SIGNIFICANT),
            ],
            previous_persistence=previous,
        )
    )

    assert result.action == RetrainingAction.RETRAIN


def test_strong_signal_triggers_retraining():
    result = evaluate_retraining_orchestration(
        RetrainingOrchestrationInput(
            quality=good_quality(),
            drift_signals=[
                signal("feature_a", DriftSeverity.STRONG),
            ],
        )
    )

    assert result.action == RetrainingAction.RETRAIN


def test_quality_failure_blocks_retraining():
    result = evaluate_retraining_orchestration(
        RetrainingOrchestrationInput(
            quality=QualityStatus(
                invalid_timestamp_rate=1.0,
            ),
            drift_signals=[
                signal("feature_a", DriftSeverity.STRONG),
            ],
        )
    )

    assert result.action == RetrainingAction.QUALITY_BLOCK


def test_retraining_decision_does_not_imply_promotion():
    result = evaluate_retraining_orchestration(
        RetrainingOrchestrationInput(
            quality=good_quality(),
            drift_signals=[
                signal("feature_a", DriftSeverity.STRONG),
            ],
        )
    )

    assert result.action == RetrainingAction.RETRAIN
    assert result.action != RetrainingAction.NO_ACTION
    assert not hasattr(result, "promotion_version")


def test_persistence_state_is_returned_for_next_window():
    result = evaluate_retraining_orchestration(
        RetrainingOrchestrationInput(
            quality=good_quality(),
            drift_signals=[
                signal("feature_a", DriftSeverity.SIGNIFICANT),
            ],
        )
    )

    assert len(result.persistence_state) == 1
    assert result.persistence_state[0].signal_name == "feature_a"
    assert (
        result.persistence_state[0].consecutive_significant_windows == 1
    )
