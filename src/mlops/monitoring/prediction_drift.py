from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.mlops.monitoring.drift_metrics import (
    population_stability_index,
    wasserstein_distance,
)


REFERENCE_SUMMARY_PATH = Path(
    "reports/results/tecnalia/monitoring/"
    "tecnalia_prediction_reference.csv"
)

REFERENCE_SAMPLE_PATH = Path(
    "reports/results/tecnalia/monitoring/"
    "tecnalia_prediction_reference_samples.csv"
)


def load_prediction_reference_summary(
    path: str | Path = REFERENCE_SUMMARY_PATH,
) -> pd.DataFrame:
    reference = pd.read_csv(path)

    required = {
        "reference_split",
        "model",
        "candidate_id",
        "count",
        "sample_count",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "p99",
        "max",
    }

    missing = required - set(reference.columns)

    if missing:
        raise ValueError(
            f"Prediction reference summary is missing columns: "
            f"{sorted(missing)}"
        )

    if len(reference) != 1:
        raise ValueError(
            "Prediction reference summary must contain exactly one row."
        )

    if reference["reference_split"].iloc[0] != "validation":
        raise ValueError(
            "Prediction reference must use the validation split."
        )

    return reference


def load_prediction_reference_samples(
    path: str | Path = REFERENCE_SAMPLE_PATH,
) -> pd.DataFrame:
    reference = pd.read_csv(path)

    required = {
        "reference_split",
        "reference_value",
    }

    missing = required - set(reference.columns)

    if missing:
        raise ValueError(
            f"Prediction reference samples are missing columns: "
            f"{sorted(missing)}"
        )

    if reference["reference_split"].nunique() != 1:
        raise ValueError(
            "Prediction reference samples must use one split."
        )

    if reference["reference_split"].iloc[0] != "validation":
        raise ValueError(
            "Prediction reference samples must use validation."
        )

    values = pd.to_numeric(
        reference["reference_value"],
        errors="coerce",
    )

    if values.isna().any():
        raise ValueError(
            "Prediction reference samples contain invalid values."
        )

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(
            "Prediction reference samples contain non-finite values."
        )

    return reference


def monitor_prediction_drift(
    current_predictions: pd.Series | np.ndarray | list[float],
    reference_summary_path: str | Path = REFERENCE_SUMMARY_PATH,
    reference_sample_path: str | Path = REFERENCE_SAMPLE_PATH,
) -> dict[str, float | int | str]:
    reference_summary = load_prediction_reference_summary(
        reference_summary_path
    )

    reference_samples = load_prediction_reference_samples(
        reference_sample_path
    )

    current = pd.to_numeric(
        pd.Series(current_predictions),
        errors="coerce",
    )

    current_missing_count = int(current.isna().sum())

    current_clean = current.dropna()

    if current_clean.empty:
        raise ValueError(
            "Current prediction population contains no valid values."
        )

    current_array = current_clean.to_numpy(dtype=float)

    if not np.isfinite(current_array).all():
        raise ValueError(
            "Current prediction population contains non-finite values."
        )

    reference_array = reference_samples[
        "reference_value"
    ].to_numpy(dtype=float)

    result = {
        "reference_split": str(
            reference_summary["reference_split"].iloc[0]
        ),
        "reference_model": str(
            reference_summary["model"].iloc[0]
        ),
        "reference_candidate_id": int(
            reference_summary["candidate_id"].iloc[0]
        ),
        "reference_count": int(
            reference_summary["count"].iloc[0]
        ),
        "current_count": int(current_clean.size),
        "current_missing_count": current_missing_count,
        "current_missing_rate": float(
            current.isna().mean()
        ),
        "current_mean": float(current_clean.mean()),
        "current_std": float(current_clean.std()),
        "current_min": float(current_clean.min()),
        "current_p01": float(current_clean.quantile(0.01)),
        "current_p05": float(current_clean.quantile(0.05)),
        "current_p25": float(current_clean.quantile(0.25)),
        "current_median": float(current_clean.median()),
        "current_p75": float(current_clean.quantile(0.75)),
        "current_p95": float(current_clean.quantile(0.95)),
        "current_p99": float(current_clean.quantile(0.99)),
        "current_max": float(current_clean.max()),
        "psi": population_stability_index(
            reference_array,
            current_array,
        ),
        "wasserstein_distance": wasserstein_distance(
            reference_array,
            current_array,
        ),
    }

    return result
