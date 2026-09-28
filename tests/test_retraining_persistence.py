from __future__ import annotations

from src.mlops.retraining.persistence import (
    PersistenceThresholds,
    SignalPersistence,
    apply_persistence_policy,
    build_persistence_state,
    update_signal_persistence,
)
from src.mlops.retraining.trigger_contract import (
    DriftSeverity,
    TriggerDecision,
)
from src.mlops.retraining.trigger_evaluator import (
    DriftSignal,
    evaluate_drift,
)


def _signal(
    name: str,
    severity: DriftSeverity,
) -> DriftSignal:
    return DriftSignal(
        name=name,
        severity=severity,
    )


def test_first_significant_window_does_not_persist():
    signal = _signal("telemetry.gpoa", DriftSeverity.SIGNIFICANT)

    evaluation = evaluate_drift((signal,))

    state = build_persistence_state(
        previous_state=(),
        current_signals=(signal,),
    )

    result = apply_persistence_policy(
        evaluation=evaluation,
        persistence=state,
    )

    assert result.decision == TriggerDecision.MONITOR
    assert state[0].consecutive_significant_windows == 1


def test_second_consecutive_significant_window_retrains():
    signal = _signal("telemetry.gpoa", DriftSeverity.SIGNIFICANT)

    first = build_persistence_state(
        previous_state=(),
        current_signals=(signal,),
    )

    second = build_persistence_state(
        previous_state=first,
        current_signals=(signal,),
    )

    evaluation = evaluate_drift((signal,))

    result = apply_persistence_policy(
        evaluation=evaluation,
        persistence=second,
    )

    assert result.decision == TriggerDecision.RETRAIN
    assert result.persistence_reasons == (
        "significant_drift_persisted:telemetry.gpoa",
    )


def test_non_drift_resets_persistence():
    signal = _signal("telemetry.gpoa", DriftSeverity.SIGNIFICANT)

    previous = SignalPersistence(
        signal_name="telemetry.gpoa",
        consecutive_significant_windows=1,
        consecutive_strong_windows=1,
    )

    current = build_persistence_state(
        previous_state=(previous,),
        current_signals=(
            _signal("telemetry.gpoa", DriftSeverity.NONE),
        ),
    )

    assert current[0].consecutive_significant_windows == 0
    assert current[0].consecutive_strong_windows == 0


def test_monitor_level_drift_resets_significant_counter():
    previous = SignalPersistence(
        signal_name="telemetry.gpoa",
        consecutive_significant_windows=1,
        consecutive_strong_windows=0,
    )

    current = build_persistence_state(
        previous_state=(previous,),
        current_signals=(
            _signal("telemetry.gpoa", DriftSeverity.MONITOR),
        ),
    )

    assert current[0].consecutive_significant_windows == 0


def test_strong_drift_retrains_immediately():
    signal = _signal("thermal.embedding", DriftSeverity.STRONG)

    evaluation = evaluate_drift((signal,))

    state = build_persistence_state(
        previous_state=(),
        current_signals=(signal,),
    )

    result = apply_persistence_policy(
        evaluation=evaluation,
        persistence=state,
    )

    assert result.decision == TriggerDecision.RETRAIN


def test_quality_block_is_immediate():
    signal = _signal("telemetry.gpoa", DriftSeverity.SIGNIFICANT)

    from src.mlops.retraining.trigger_evaluator import QualityStatus
    from src.mlops.retraining.trigger_contract import QualityThresholds
    from src.mlops.retraining.trigger_evaluator import evaluate_retraining_trigger

    evaluation = evaluate_retraining_trigger(
        quality=QualityStatus(required_columns_valid=False),
        drift_signals=(signal,),
        quality_thresholds=QualityThresholds(),
    )

    state = build_persistence_state(
        previous_state=(),
        current_signals=(signal,),
    )

    result = apply_persistence_policy(
        evaluation=evaluation,
        persistence=state,
    )

    assert result.decision == TriggerDecision.QUALITY_BLOCK


def test_custom_persistence_threshold():
    signal = _signal("prediction.psi", DriftSeverity.SIGNIFICANT)

    first = build_persistence_state(
        previous_state=(),
        current_signals=(signal,),
    )

    evaluation = evaluate_drift((signal,))

    result = apply_persistence_policy(
        evaluation=evaluation,
        persistence=first,
        thresholds=PersistenceThresholds(
            significant_consecutive_windows=1,
            strong_consecutive_windows=1,
        ),
    )

    assert result.decision == TriggerDecision.RETRAIN


def test_multiple_significant_signals_retrain_without_persistence():
    signals = (
        _signal("feature.a", DriftSeverity.SIGNIFICANT),
        _signal("feature.b", DriftSeverity.SIGNIFICANT),
    )

    evaluation = evaluate_drift(signals)

    state = build_persistence_state(
        previous_state=(),
        current_signals=signals,
    )

    result = apply_persistence_policy(
        evaluation=evaluation,
        persistence=state,
    )

    assert result.decision == TriggerDecision.RETRAIN


def test_strong_counter_increments_only_for_consecutive_strong():
    previous = SignalPersistence(
        signal_name="thermal.psi",
        consecutive_significant_windows=1,
        consecutive_strong_windows=1,
    )

    current = update_signal_persistence(
        previous=previous,
        current_severity=DriftSeverity.SIGNIFICANT,
    )

    assert current.consecutive_significant_windows == 2
    assert current.consecutive_strong_windows == 0
