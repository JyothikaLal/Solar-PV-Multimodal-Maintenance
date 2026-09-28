from __future__ import annotations

import numpy as np
import pandas as pd

from src.mlops.retraining.prediction_triggers import (
    build_tecnalia_prediction_drift_signal,
)


def test_reference_predictions_have_no_drift():
    reference = pd.read_csv(
        "reports/results/tecnalia/monitoring/"
        "tecnalia_prediction_reference_samples.csv"
    )["reference_value"]

    signal = build_tecnalia_prediction_drift_signal(reference)

    assert signal.name == "tecnalia.prediction.psi"
    assert signal.severity.value == "NONE"


def test_prediction_shift_produces_drift():
    reference = pd.read_csv(
        "reports/results/tecnalia/monitoring/"
        "tecnalia_prediction_reference_samples.csv"
    )["reference_value"]

    shifted = reference + 0.15

    signal = build_tecnalia_prediction_drift_signal(shifted)

    assert signal.severity.value in {
        "MONITOR",
        "SIGNIFICANT",
        "STRONG",
    }


def test_prediction_signal_accepts_numpy_array():
    reference = pd.read_csv(
        "reports/results/tecnalia/monitoring/"
        "tecnalia_prediction_reference_samples.csv"
    )["reference_value"].to_numpy()

    signal = build_tecnalia_prediction_drift_signal(reference)

    assert signal.severity.value == "NONE"


def test_invalid_prediction_population_is_rejected():
    invalid = pd.Series([np.nan, np.nan, np.nan])

    try:
        build_tecnalia_prediction_drift_signal(invalid)
    except ValueError as exc:
        assert "no valid values" in str(exc)
    else:
        raise AssertionError(
            "Expected invalid prediction population to be rejected."
        )
