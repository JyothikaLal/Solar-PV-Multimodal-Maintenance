from __future__ import annotations

from typing import Iterable

import numpy as np

from src.models.fusion.contracts import RAPTORMAPS_CLASS_NAMES
from src.mlops.monitoring.drift_metrics import (
    jensen_shannon_divergence,
)


EXPECTED_CLASS_NAMES = tuple(RAPTORMAPS_CLASS_NAMES)
EXPECTED_CLASS_COUNT = len(EXPECTED_CLASS_NAMES)


def _validate_probability_population(
    probabilities: Iterable[Iterable[float]],
    population_name: str,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)

    if values.ndim != 2:
        raise ValueError(
            f"{population_name} probability population must be 2-D."
        )

    if values.shape[0] == 0:
        raise ValueError(
            f"{population_name} probability population must not be empty."
        )

    if values.shape[1] != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"{population_name} probability population must contain "
            f"exactly {EXPECTED_CLASS_COUNT} classes; "
            f"received {values.shape[1]}."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{population_name} probability population contains "
            "non-finite values."
        )

    if (values < 0).any() or (values > 1).any():
        raise ValueError(
            f"{population_name} probabilities must be within [0, 1]."
        )

    row_sums = values.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            f"{population_name} probability rows must sum to 1."
        )

    return values


def monitor_thermal_prediction_drift(
    reference_probabilities: Iterable[Iterable[float]],
    current_probabilities: Iterable[Iterable[float]],
) -> dict[str, object]:
    """Measure population-level RaptorMaps probability drift.

    The class-level population distribution is the mean probability
    assigned to each of the twelve RaptorMaps anomaly classes.
    """
    reference = _validate_probability_population(
        reference_probabilities,
        "Reference",
    )

    current = _validate_probability_population(
        current_probabilities,
        "Current",
    )

    reference_distribution = reference.mean(
        axis=0,
        dtype=np.float64,
    )

    current_distribution = current.mean(
        axis=0,
        dtype=np.float64,
    )

    js_divergence = jensen_shannon_divergence(
        reference_distribution,
        current_distribution,
    )

    return {
        "reference_count": int(reference.shape[0]),
        "current_count": int(current.shape[0]),
        "class_count": EXPECTED_CLASS_COUNT,
        "class_names": EXPECTED_CLASS_NAMES,
        "reference_distribution": reference_distribution,
        "current_distribution": current_distribution,
        "js_divergence": float(js_divergence),
    }
