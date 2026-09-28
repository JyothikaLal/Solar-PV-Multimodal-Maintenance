from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mlops.monitoring.prediction_drift import (
    REFERENCE_SAMPLE_PATH,
    REFERENCE_SUMMARY_PATH,
    load_prediction_reference_samples,
    load_prediction_reference_summary,
    monitor_prediction_drift,
)


def load_reference_values() -> np.ndarray:
    reference = load_prediction_reference_samples(
        REFERENCE_SAMPLE_PATH
    )

    return reference["reference_value"].to_numpy(
        dtype=float
    )


def test_prediction_reference_summary_is_valid():
    reference = load_prediction_reference_summary(
        REFERENCE_SUMMARY_PATH
    )

    assert len(reference) == 1
    assert (
        reference["reference_split"].iloc[0]
        == "validation"
    )

    assert (
        reference["model"].iloc[0]
        == "gradient_boosting_tuned"
    )

    assert int(reference["count"].iloc[0]) == 14530


def test_prediction_reference_samples_are_valid():
    reference = load_prediction_reference_samples(
        REFERENCE_SAMPLE_PATH
    )

    assert len(reference) == 10000
    assert (
        reference["reference_split"].iloc[0]
        == "validation"
    )

    values = reference["reference_value"].to_numpy(
        dtype=float
    )

    assert np.isfinite(values).all()


def test_prediction_self_reference_has_zero_drift():
    reference_values = load_reference_values()

    result = monitor_prediction_drift(
        reference_values,
    )

    assert result["reference_split"] == "validation"
    assert result["reference_model"] == (
        "gradient_boosting_tuned"
    )
    assert result["current_count"] == 10000
    assert result["current_missing_count"] == 0

    assert result["psi"] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert result["wasserstein_distance"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_prediction_shift_produces_measurable_drift():
    reference_values = load_reference_values()

    shifted = reference_values + 0.15

    result = monitor_prediction_drift(
        shifted,
    )

    assert result["current_count"] == 10000

    assert result["current_mean"] == pytest.approx(
        reference_values.mean() + 0.15,
        abs=1e-10,
    )

    assert result["wasserstein_distance"] == pytest.approx(
        0.15,
        abs=1e-10,
    )

    assert result["psi"] > 0.0


def test_prediction_missing_values_are_reported():
    reference_values = load_reference_values()

    current = pd.Series(reference_values)
    current.iloc[0] = np.nan
    current.iloc[1] = np.nan

    result = monitor_prediction_drift(
        current,
    )

    assert result["current_missing_count"] == 2
    assert result["current_count"] == 9998
    assert result["current_missing_rate"] == pytest.approx(
        2 / 10000
    )


def test_prediction_all_missing_values_are_rejected():
    current = np.full(
        100,
        np.nan,
    )

    with pytest.raises(
        ValueError,
        match="no valid values",
    ):
        monitor_prediction_drift(current)


def test_prediction_non_finite_values_are_rejected():
    reference_values = load_reference_values()

    current = reference_values.copy()
    current[0] = np.inf

    with pytest.raises(
        ValueError,
        match="non-finite values",
    ):
        monitor_prediction_drift(current)


def test_prediction_reference_summary_requires_validation_split(
    tmp_path,
):
    summary = pd.read_csv(
        REFERENCE_SUMMARY_PATH
    )

    summary.loc[0, "reference_split"] = "test"

    path = tmp_path / "prediction_reference.csv"
    summary.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="validation split",
    ):
        load_prediction_reference_summary(path)


def test_prediction_reference_samples_require_validation_split(
    tmp_path,
):
    samples = pd.read_csv(
        REFERENCE_SAMPLE_PATH
    )

    samples.loc[0, "reference_split"] = "test"

    path = tmp_path / "prediction_samples.csv"
    samples.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="one split",
    ):
        load_prediction_reference_samples(path)
