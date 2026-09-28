from __future__ import annotations

import pytest

from src.mlops.retraining.trigger_contract import (
    DriftSeverity,
    DriftThresholds,
    QualityThresholds,
    TriggerDecision,
    classify_js,
    classify_psi,
)
from src.mlops.retraining.trigger_evaluator import (
    DriftSignal,
    QualityStatus,
    evaluate_retraining_trigger,
    js_signal,
    psi_signal,
)


def test_psi_classification() -> None:
    thresholds = DriftThresholds()

    assert classify_psi(0.05, thresholds) == DriftSeverity.NONE
    assert classify_psi(0.10, thresholds) == DriftSeverity.MONITOR
    assert classify_psi(0.20, thresholds) == DriftSeverity.SIGNIFICANT
    assert classify_psi(0.25, thresholds) == DriftSeverity.STRONG


def test_js_classification() -> None:
    thresholds = DriftThresholds()

    assert classify_js(0.04, thresholds) == DriftSeverity.NONE
    assert classify_js(0.05, thresholds) == DriftSeverity.MONITOR
    assert classify_js(0.10, thresholds) == DriftSeverity.SIGNIFICANT
    assert classify_js(0.20, thresholds) == DriftSeverity.STRONG


def test_single_moderate_signal_only_monitors() -> None:
    result = evaluate_retraining_trigger(
        quality=QualityStatus(),
        drift_signals=(
            psi_signal("telemetry_feature", 0.15),
        ),
    )

    assert result.decision == TriggerDecision.MONITOR
    assert result.significant_signal_count == 0


def test_single_significant_signal_does_not_retrain() -> None:
    result = evaluate_retraining_trigger(
        quality=QualityStatus(),
        drift_signals=(
            psi_signal("telemetry_feature", 0.20),
        ),
    )

    assert result.decision == TriggerDecision.MONITOR
    assert result.significant_signal_count == 1


def test_multiple_significant_signals_trigger_retraining() -> None:
    result = evaluate_retraining_trigger(
        quality=QualityStatus(),
        drift_signals=(
            psi_signal("feature_1", 0.20),
            psi_signal("prediction", 0.22),
        ),
    )

    assert result.decision == TriggerDecision.RETRAIN
    assert result.significant_signal_count == 2


def test_single_strong_signal_triggers_retraining() -> None:
    result = evaluate_retraining_trigger(
        quality=QualityStatus(),
        drift_signals=(
            psi_signal("prediction", 0.30),
        ),
    )

    assert result.decision == TriggerDecision.RETRAIN
    assert result.strong_signal_count == 1


def test_quality_failure_blocks_retraining() -> None:
    result = evaluate_retraining_trigger(
        quality=QualityStatus(
            required_columns_valid=False,
        ),
        drift_signals=(
            psi_signal("prediction", 0.30),
        ),
    )

    assert result.decision == TriggerDecision.QUALITY_BLOCK
    assert "required_columns_invalid" in result.quality_block_reasons


def test_invalid_timestamp_blocks_retraining() -> None:
    result = evaluate_retraining_trigger(
        quality=QualityStatus(
            invalid_timestamp_rate=0.01,
        ),
        drift_signals=(),
    )

    assert result.decision == TriggerDecision.QUALITY_BLOCK
    assert "invalid_timestamp_rate" in result.quality_block_reasons


def test_custom_thresholds_are_supported() -> None:
    thresholds = DriftThresholds(
        psi_monitor=0.05,
        psi_significant=0.10,
        psi_strong=0.15,
    )

    result = evaluate_retraining_trigger(
        quality=QualityStatus(),
        drift_signals=(
            psi_signal("prediction", 0.16, thresholds),
        ),
        drift_thresholds=thresholds,
    )

    assert result.decision == TriggerDecision.RETRAIN


def test_quality_thresholds_are_configurable() -> None:
    thresholds = QualityThresholds(
        max_invalid_timestamp_rate=0.05,
    )

    result = evaluate_retraining_trigger(
        quality=QualityStatus(
            invalid_timestamp_rate=0.02,
        ),
        drift_signals=(),
        quality_thresholds=thresholds,
    )

    assert result.decision == TriggerDecision.NO_ACTION


def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValueError):
        DriftThresholds(
            psi_monitor=0.20,
            psi_significant=0.10,
            psi_strong=0.30,
        )


def test_negative_psi_is_rejected() -> None:
    with pytest.raises(ValueError):
        classify_psi(-0.01)


def test_negative_js_is_rejected() -> None:
    with pytest.raises(ValueError):
        classify_js(-0.01)
