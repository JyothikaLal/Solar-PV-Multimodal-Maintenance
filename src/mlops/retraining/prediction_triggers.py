from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.mlops.monitoring.prediction_drift import (
    monitor_prediction_drift,
)
from src.mlops.retraining.trigger_contract import (
    DEFAULT_DRIFT_THRESHOLDS,
    DriftThresholds,
)
from src.mlops.retraining.trigger_evaluator import (
    DriftSignal,
    evaluate_retraining_trigger,
    psi_signal,
)


def build_tecnalia_prediction_drift_signal(
    current_predictions: pd.Series | np.ndarray | Sequence[float],
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
) -> DriftSignal:
    """Convert TECNALIA prediction PSI into a retraining signal.

    The existing prediction monitor remains responsible for calculating
    PSI and Wasserstein distance. This adapter applies only the
    retraining policy to PSI.
    """
    result = monitor_prediction_drift(
        current_predictions=current_predictions,
    )

    return psi_signal(
        name="tecnalia.prediction.psi",
        psi=float(result["psi"]),
        thresholds=thresholds,
    )


def evaluate_tecnalia_prediction_drift(
    current_predictions: pd.Series | np.ndarray | Sequence[float],
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
):
    """Evaluate the TECNALIA prediction-drift trigger."""
    signal = build_tecnalia_prediction_drift_signal(
        current_predictions=current_predictions,
        thresholds=thresholds,
    )

    return evaluate_retraining_trigger(
        signals=(signal,),
    )
