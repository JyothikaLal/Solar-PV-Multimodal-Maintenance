from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.mlops.retraining.trigger_contract import (
    DEFAULT_DRIFT_THRESHOLDS,
    DEFAULT_QUALITY_THRESHOLDS,
    DriftSeverity,
    DriftThresholds,
    QualityThresholds,
    TriggerDecision,
    classify_js,
    classify_psi,
)


@dataclass(frozen=True)
class DriftSignal:
    """One independently measured drift signal."""

    name: str
    severity: DriftSeverity


@dataclass(frozen=True)
class QualityStatus:
    """Quality status used by the retraining trigger gate."""

    required_columns_valid: bool = True
    invalid_timestamp_rate: float = 0.0
    duplicate_key_rate: float = 0.0
    non_finite_rate: float = 0.0
    missing_required_columns: int = 0


@dataclass(frozen=True)
class TriggerEvaluation:
    """Final result of one retraining-trigger evaluation."""

    decision: TriggerDecision
    drift_signals: tuple[DriftSignal, ...]
    quality_block_reasons: tuple[str, ...]
    significant_signal_count: int
    strong_signal_count: int


def evaluate_quality(
    quality: QualityStatus,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
) -> tuple[str, ...]:
    """Return hard data-quality block reasons."""

    reasons: list[str] = []

    if not quality.required_columns_valid:
        reasons.append("required_columns_invalid")

    if (
        quality.missing_required_columns
        > thresholds.max_missing_required_columns
    ):
        reasons.append("missing_required_columns")

    if (
        quality.invalid_timestamp_rate
        > thresholds.max_invalid_timestamp_rate
    ):
        reasons.append("invalid_timestamp_rate")

    if (
        quality.duplicate_key_rate
        > thresholds.max_duplicate_key_rate
    ):
        reasons.append("duplicate_key_rate")

    if quality.non_finite_rate > thresholds.max_non_finite_rate:
        reasons.append("non_finite_rate")

    return tuple(reasons)


def evaluate_drift(
    signals: Iterable[DriftSignal],
    *,
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
) -> TriggerEvaluation:
    """
    Evaluate independently measured drift signals.

    A single monitor-level signal results in MONITOR.
    Multiple significant signals or any strong signal result in RETRAIN.
    """

    normalized_signals = tuple(signals)

    significant_count = sum(
        signal.severity == DriftSeverity.SIGNIFICANT
        for signal in normalized_signals
    )

    strong_count = sum(
        signal.severity == DriftSeverity.STRONG
        for signal in normalized_signals
    )

    if strong_count > 0:
        decision = TriggerDecision.RETRAIN
    elif significant_count >= thresholds.min_retraining_signals:
        decision = TriggerDecision.RETRAIN
    elif significant_count > 0:
        decision = TriggerDecision.MONITOR
    elif any(
        signal.severity == DriftSeverity.MONITOR
        for signal in normalized_signals
    ):
        decision = TriggerDecision.MONITOR
    else:
        decision = TriggerDecision.NO_ACTION

    return TriggerEvaluation(
        decision=decision,
        drift_signals=normalized_signals,
        quality_block_reasons=(),
        significant_signal_count=significant_count,
        strong_signal_count=strong_count,
    )


def evaluate_retraining_trigger(
    *,
    quality: QualityStatus,
    drift_signals: Iterable[DriftSignal],
    drift_thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
    quality_thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
) -> TriggerEvaluation:
    """
    Evaluate data quality first, then statistical drift.

    Quality failures always block automatic retraining.
    """

    quality_block_reasons = evaluate_quality(
        quality,
        quality_thresholds,
    )

    drift_evaluation = evaluate_drift(
        drift_signals,
        thresholds=drift_thresholds,
    )

    if quality_block_reasons:
        return TriggerEvaluation(
            decision=TriggerDecision.QUALITY_BLOCK,
            drift_signals=drift_evaluation.drift_signals,
            quality_block_reasons=quality_block_reasons,
            significant_signal_count=(
                drift_evaluation.significant_signal_count
            ),
            strong_signal_count=drift_evaluation.strong_signal_count,
        )

    return drift_evaluation


def psi_signal(
    name: str,
    psi: float,
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
) -> DriftSignal:
    """Build a PSI-based drift signal."""

    return DriftSignal(
        name=name,
        severity=classify_psi(psi, thresholds),
    )


def js_signal(
    name: str,
    divergence: float,
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
) -> DriftSignal:
    """Build a JS-divergence-based drift signal."""

    return DriftSignal(
        name=name,
        severity=classify_js(divergence, thresholds),
    )
