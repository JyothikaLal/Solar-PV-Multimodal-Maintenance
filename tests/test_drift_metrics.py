from __future__ import annotations

import numpy as np
import pytest

from src.mlops.monitoring.drift_metrics import (
    jensen_shannon_divergence,
    population_stability_index,
    wasserstein_distance,
)


def test_identical_numeric_distributions_have_zero_drift():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    assert population_stability_index(values, values) == pytest.approx(
        0.0
    )
    assert wasserstein_distance(values, values) == pytest.approx(
        0.0
    )


def test_shifted_numeric_distribution_has_positive_drift():
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    current = np.array([10.0, 11.0, 12.0, 13.0])

    assert population_stability_index(
        reference,
        current,
    ) > 0.0

    assert wasserstein_distance(
        reference,
        current,
    ) > 0.0


def test_identical_probability_distributions_have_zero_js():
    probabilities = [0.7, 0.2, 0.1]

    assert jensen_shannon_divergence(
        probabilities,
        probabilities,
    ) == pytest.approx(0.0)


def test_different_probability_distributions_have_positive_js():
    reference = [0.9, 0.05, 0.05]
    current = [0.1, 0.8, 0.1]

    assert jensen_shannon_divergence(
        reference,
        current,
    ) > 0.0


def test_empty_numeric_reference_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        population_stability_index([], [1.0])


def test_non_finite_numeric_values_are_rejected():
    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        wasserstein_distance(
            [1.0, np.nan],
            [1.0, 2.0],
        )


def test_negative_probability_is_rejected():
    with pytest.raises(
        ValueError,
        match="negative",
    ):
        jensen_shannon_divergence(
            [0.8, 0.2],
            [1.1, -0.1],
        )


def test_probability_category_count_must_match():
    with pytest.raises(
        ValueError,
        match="same number of categories",
    ):
        jensen_shannon_divergence(
            [0.5, 0.5],
            [0.5, 0.3, 0.2],
        )
