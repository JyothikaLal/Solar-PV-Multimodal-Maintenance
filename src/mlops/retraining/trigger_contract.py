from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TriggerDecision(str, Enum):
    """High-level decision produced by the retraining trigger policy."""

    NO_ACTION = "NO_ACTION"
    MONITOR = "MONITOR"
    RETRAIN = "RETRAIN"
    QUALITY_BLOCK = "QUALITY_BLOCK"


class DriftSeverity(str, Enum):
    """Severity assigned to an individual drift measurement."""

    NONE = "NONE"
    MONITOR = "MONITOR"
    SIGNIFICANT = "SIGNIFICANT"
    STRONG = "STRONG"


@dataclass(frozen=True)
class DriftThresholds:
    """
    Engineering policy thresholds for drift interpretation.

    These values are configurable policy defaults. They are not claimed
    to have been statistically derived from production history.
    """

    psi_monitor: float = 0.10
    psi_significant: float = 0.20
    psi_strong: float = 0.25

    js_monitor: float = 0.05
    js_significant: float = 0.10
    js_strong: float = 0.20

    min_retraining_signals: int = 2

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.psi_monitor
            < self.psi_significant
            < self.psi_strong
        ):
            raise ValueError(
                "PSI thresholds must satisfy "
                "0 <= monitor < significant < strong."
            )

        if not (
            0.0 <= self.js_monitor
            < self.js_significant
            < self.js_strong
        ):
            raise ValueError(
                "JS thresholds must satisfy "
                "0 <= monitor < significant < strong."
            )

        if self.min_retraining_signals < 1:
            raise ValueError(
                "min_retraining_signals must be at least 1."
            )


@dataclass(frozen=True)
class QualityThresholds:
    """
    Hard data-quality conditions.

    These are intentionally separate from statistical drift thresholds.
    """

    max_invalid_timestamp_rate: float = 0.0
    max_duplicate_key_rate: float = 0.0
    max_non_finite_rate: float = 0.0
    max_missing_required_columns: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            (
                "max_invalid_timestamp_rate",
                self.max_invalid_timestamp_rate,
            ),
            (
                "max_duplicate_key_rate",
                self.max_duplicate_key_rate,
            ),
            (
                "max_non_finite_rate",
                self.max_non_finite_rate,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if self.max_missing_required_columns < 0:
            raise ValueError(
                "max_missing_required_columns cannot be negative."
            )


DEFAULT_DRIFT_THRESHOLDS = DriftThresholds()
DEFAULT_QUALITY_THRESHOLDS = QualityThresholds()


def classify_psi(
    psi: float,
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
) -> DriftSeverity:
    """Classify one PSI measurement."""

    if psi < 0.0:
        raise ValueError("PSI cannot be negative.")

    if psi >= thresholds.psi_strong:
        return DriftSeverity.STRONG

    if psi >= thresholds.psi_significant:
        return DriftSeverity.SIGNIFICANT

    if psi >= thresholds.psi_monitor:
        return DriftSeverity.MONITOR

    return DriftSeverity.NONE


def classify_js(
    divergence: float,
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
) -> DriftSeverity:
    """Classify one Jensen-Shannon divergence measurement."""

    if divergence < 0.0:
        raise ValueError(
            "Jensen-Shannon divergence cannot be negative."
        )

    if divergence >= thresholds.js_strong:
        return DriftSeverity.STRONG

    if divergence >= thresholds.js_significant:
        return DriftSeverity.SIGNIFICANT

    if divergence >= thresholds.js_monitor:
        return DriftSeverity.MONITOR

    return DriftSeverity.NONE
