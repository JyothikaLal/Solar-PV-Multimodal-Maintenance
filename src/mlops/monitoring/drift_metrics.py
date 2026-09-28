from __future__ import annotations

from math import log
from typing import Iterable

import numpy as np


EPSILON = 1e-12


def _validate_numeric_arrays(
    reference: Iterable[float],
    current: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    reference_array = np.asarray(list(reference), dtype=float)
    current_array = np.asarray(list(current), dtype=float)

    if reference_array.size == 0:
        raise ValueError("Reference distribution must not be empty.")

    if current_array.size == 0:
        raise ValueError("Current distribution must not be empty.")

    if not np.isfinite(reference_array).all():
        raise ValueError(
            "Reference distribution contains non-finite values."
        )

    if not np.isfinite(current_array).all():
        raise ValueError(
            "Current distribution contains non-finite values."
        )

    return reference_array, current_array


def _build_histogram_probabilities(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    combined = np.concatenate([reference, current])

    minimum = float(combined.min())
    maximum = float(combined.max())

    if minimum == maximum:
        return (
            np.array([1.0], dtype=float),
            np.array([1.0], dtype=float),
        )

    edges = np.linspace(minimum, maximum, bins + 1)

    reference_counts, _ = np.histogram(
        reference,
        bins=edges,
    )
    current_counts, _ = np.histogram(
        current,
        bins=edges,
    )

    reference_probabilities = (
        reference_counts.astype(float)
        / reference_counts.sum()
    )
    current_probabilities = (
        current_counts.astype(float)
        / current_counts.sum()
    )

    reference_probabilities = np.clip(
        reference_probabilities,
        EPSILON,
        None,
    )
    current_probabilities = np.clip(
        current_probabilities,
        EPSILON,
        None,
    )

    reference_probabilities /= reference_probabilities.sum()
    current_probabilities /= current_probabilities.sum()

    return reference_probabilities, current_probabilities


def population_stability_index(
    reference: Iterable[float],
    current: Iterable[float],
    bins: int = 20,
) -> float:
    """Calculate PSI between two numeric distributions."""
    reference_array, current_array = _validate_numeric_arrays(
        reference,
        current,
    )

    reference_probabilities, current_probabilities = (
        _build_histogram_probabilities(
            reference_array,
            current_array,
            bins=bins,
        )
    )

    return float(
        np.sum(
            (
                current_probabilities
                - reference_probabilities
            )
            * np.log(
                current_probabilities
                / reference_probabilities
            )
        )
    )


def wasserstein_distance(
    reference: Iterable[float],
    current: Iterable[float],
) -> float:
    """Calculate the one-dimensional Wasserstein distance."""
    reference_array, current_array = _validate_numeric_arrays(
        reference,
        current,
    )

    reference_sorted = np.sort(reference_array)
    current_sorted = np.sort(current_array)

    quantiles = np.linspace(0.0, 1.0, 1001)

    reference_quantiles = np.quantile(
        reference_sorted,
        quantiles,
    )
    current_quantiles = np.quantile(
        current_sorted,
        quantiles,
    )

    return float(
        np.trapezoid(
            np.abs(
                reference_quantiles
                - current_quantiles
            ),
            quantiles,
        )
    )


def _validate_probability_distribution(
    probabilities: Iterable[float],
) -> np.ndarray:
    values = np.asarray(list(probabilities), dtype=float)

    if values.size == 0:
        raise ValueError("Probability distribution must not be empty.")

    if not np.isfinite(values).all():
        raise ValueError(
            "Probability distribution contains non-finite values."
        )

    if (values < 0).any():
        raise ValueError(
            "Probability distribution cannot contain negative values."
        )

    total = float(values.sum())

    if total <= 0:
        raise ValueError(
            "Probability distribution must have positive total mass."
        )

    return values / total


def jensen_shannon_divergence(
    reference: Iterable[float],
    current: Iterable[float],
) -> float:
    """Calculate Jensen-Shannon divergence between distributions."""
    reference_probabilities = _validate_probability_distribution(
        reference
    )
    current_probabilities = _validate_probability_distribution(
        current
    )

    if reference_probabilities.shape != current_probabilities.shape:
        raise ValueError(
            "Reference and current distributions must have "
            "the same number of categories."
        )

    reference_probabilities = np.clip(
        reference_probabilities,
        EPSILON,
        None,
    )
    current_probabilities = np.clip(
        current_probabilities,
        EPSILON,
        None,
    )

    reference_probabilities /= reference_probabilities.sum()
    current_probabilities /= current_probabilities.sum()

    midpoint = (
        reference_probabilities
        + current_probabilities
    ) / 2.0

    divergence = 0.5 * (
        np.sum(
            reference_probabilities
            * np.log(
                reference_probabilities / midpoint
            )
        )
        + np.sum(
            current_probabilities
            * np.log(
                current_probabilities / midpoint
            )
        )
    )

    return float(divergence)
