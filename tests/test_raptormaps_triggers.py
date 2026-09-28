from __future__ import annotations

import numpy as np
import pytest

from src.mlops.monitoring.thermal_prediction_drift import (
    EXPECTED_CLASS_COUNT,
    monitor_thermal_prediction_drift,
)
from src.mlops.retraining.embedding_triggers import (
    build_raptormaps_embedding_drift_signal,
)
from src.mlops.retraining.thermal_prediction_triggers import (
    build_raptormaps_prediction_drift_signal,
)


def _probability_population(
    rows: int = 20,
    dominant_class: int = 0,
) -> np.ndarray:
    values = np.full(
        (rows, EXPECTED_CLASS_COUNT),
        0.01,
        dtype=float,
    )

    values[:, dominant_class] = (
        1.0 - 0.01 * (EXPECTED_CLASS_COUNT - 1)
    )

    return values


def test_thermal_self_reference_has_no_drift():
    reference = _probability_population()

    result = monitor_thermal_prediction_drift(
        reference_probabilities=reference,
        current_probabilities=reference.copy(),
    )

    assert result["class_count"] == 12
    assert result["js_divergence"] == pytest.approx(0.0)


def test_thermal_shift_produces_measurable_drift():
    reference = _probability_population(dominant_class=0)
    current = _probability_population(dominant_class=1)

    result = monitor_thermal_prediction_drift(
        reference_probabilities=reference,
        current_probabilities=current,
    )

    assert result["js_divergence"] > 0.0


def test_thermal_trigger_classifies_strong_shift():
    reference = _probability_population(dominant_class=0)
    current = _probability_population(dominant_class=1)

    signal = build_raptormaps_prediction_drift_signal(
        reference_probabilities=reference,
        current_probabilities=current,
    )

    assert signal.name == "raptormaps.thermal_prediction.js"
    assert signal.severity.value == "STRONG"


def test_thermal_probability_dimension_is_validated():
    reference = _probability_population()
    current = np.ones((20, 11), dtype=float) / 11

    with pytest.raises(ValueError, match="exactly 12 classes"):
        monitor_thermal_prediction_drift(
            reference_probabilities=reference,
            current_probabilities=current,
        )


def test_thermal_probability_sum_is_validated():
    reference = _probability_population()
    current = reference.copy()
    current[0, 0] += 0.1

    with pytest.raises(ValueError, match="sum to 1"):
        monitor_thermal_prediction_drift(
            reference_probabilities=reference,
            current_probabilities=current,
        )


def test_thermal_non_finite_values_are_rejected():
    reference = _probability_population()
    current = reference.copy()
    current[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        monitor_thermal_prediction_drift(
            reference_probabilities=reference,
            current_probabilities=current,
        )


def test_embedding_reference_self_population_has_no_drift():
    from pathlib import Path

    reference_path = Path(
        "reports/results/raptormaps/monitoring/"
        "resnet18_finetuned_reference.npz"
    )

    reference = np.load(reference_path)["embedding_sample"]

    signal = build_raptormaps_embedding_drift_signal(
        model_name="resnet18_finetuned",
        current_embeddings=reference,
    )

    assert signal.name == (
        "raptormaps.embedding.psi:resnet18_finetuned"
    )
    assert signal.severity.value == "NONE"
