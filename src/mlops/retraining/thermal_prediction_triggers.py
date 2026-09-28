from __future__ import annotations

from typing import Iterable

from src.mlops.monitoring.thermal_prediction_drift import (
    monitor_thermal_prediction_drift,
)
from src.mlops.retraining.trigger_contract import (
    DEFAULT_DRIFT_THRESHOLDS,
    DriftThresholds,
)
from src.mlops.retraining.trigger_evaluator import (
    DriftSignal,
    evaluate_retraining_trigger,
    js_signal,
)


def build_raptormaps_prediction_drift_signal(
    reference_probabilities: Iterable[Iterable[float]],
    current_probabilities: Iterable[Iterable[float]],
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
) -> DriftSignal:
    """Convert RaptorMaps probability JS divergence into a signal."""
    result = monitor_thermal_prediction_drift(
        reference_probabilities=reference_probabilities,
        current_probabilities=current_probabilities,
    )

    return js_signal(
        name="raptormaps.thermal_prediction.js",
        divergence=float(result["js_divergence"]),
        thresholds=thresholds,
    )


def evaluate_raptormaps_prediction_drift(
    reference_probabilities: Iterable[Iterable[float]],
    current_probabilities: Iterable[Iterable[float]],
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
):
    """Evaluate the RaptorMaps thermal prediction-drift signal."""
    signal = build_raptormaps_prediction_drift_signal(
        reference_probabilities=reference_probabilities,
        current_probabilities=current_probabilities,
        thresholds=thresholds,
    )

    return evaluate_retraining_trigger(
        signals=(signal,),
    )
