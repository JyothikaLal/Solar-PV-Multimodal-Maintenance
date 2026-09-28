from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.tecnalia_regression import (
    BASE_NUMERIC_FEATURES,
    MODULE_COLUMN,
)

from src.mlops.monitoring.drift_metrics import (
    jensen_shannon_divergence,
    population_stability_index,
    wasserstein_distance,
)


REFERENCE_SUMMARY_PATH = Path(
    "reports/results/tecnalia/monitoring/"
    "telemetry_feature_reference.csv"
)

REFERENCE_SAMPLE_PATH = Path(
    "reports/results/tecnalia/monitoring/"
    "telemetry_feature_reference_samples.csv"
)


def load_reference_summary(
    path: str | Path = REFERENCE_SUMMARY_PATH,
) -> pd.DataFrame:
    reference = pd.read_csv(path)

    required = {
        "feature_name",
        "reference_split",
        "count",
        "missing_count",
        "missing_rate",
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
            f"Telemetry reference summary is missing columns: "
            f"{sorted(missing)}"
        )

    if set(reference["feature_name"]) != set(BASE_NUMERIC_FEATURES):
        raise ValueError(
            "Telemetry reference feature set does not match the "
            "frozen TECNALIA feature contract."
        )

    if reference["reference_split"].nunique() != 1:
        raise ValueError(
            "Telemetry reference must come from exactly one split."
        )

    if reference["reference_split"].iloc[0] != "train":
        raise ValueError(
            "Telemetry feature reference must use the training split."
        )

    return reference


def load_reference_samples(
    path: str | Path = REFERENCE_SAMPLE_PATH,
) -> pd.DataFrame:
    reference = pd.read_csv(path)

    required = {
        "feature_name",
        "reference_split",
        "reference_value",
    }

    missing = required - set(reference.columns)

    if missing:
        raise ValueError(
            f"Telemetry reference samples are missing columns: "
            f"{sorted(missing)}"
        )

    if set(reference["feature_name"]) != set(BASE_NUMERIC_FEATURES):
        raise ValueError(
            "Telemetry reference sample feature set does not match "
            "the frozen TECNALIA feature contract."
        )

    if reference["reference_split"].nunique() != 1:
        raise ValueError(
            "Telemetry reference samples must come from exactly one split."
        )

    if reference["reference_split"].iloc[0] != "train":
        raise ValueError(
            "Telemetry reference samples must use the training split."
        )

    if reference["reference_value"].isna().any():
        raise ValueError(
            "Telemetry reference samples contain missing values."
        )

    return reference


def _numeric_drift(
    reference_values: pd.Series,
    current_values: pd.Series,
) -> dict[str, float | int]:
    reference_clean = pd.to_numeric(
        reference_values,
        errors="coerce",
    ).dropna()

    current_numeric = pd.to_numeric(
        current_values,
        errors="coerce",
    )

    current_clean = current_numeric.dropna()

    if reference_clean.empty:
        raise ValueError(
            "Reference feature contains no valid values."
        )

    if current_clean.empty:
        raise ValueError(
            "Current feature contains no valid values."
        )

    return {
        "reference_count": int(reference_clean.size),
        "current_count": int(current_clean.size),
        "current_missing_count": int(current_numeric.isna().sum()),
        "current_missing_rate": float(current_numeric.isna().mean()),
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
            reference_clean,
            current_clean,
        ),
        "wasserstein_distance": wasserstein_distance(
            reference_clean,
            current_clean,
        ),
    }


def monitor_tecnalia_numeric_features(
    current: pd.DataFrame,
    reference_summary_path: str | Path = REFERENCE_SUMMARY_PATH,
    reference_sample_path: str | Path = REFERENCE_SAMPLE_PATH,
) -> pd.DataFrame:
    reference_summary = load_reference_summary(
        reference_summary_path
    )

    reference_samples = load_reference_samples(
        reference_sample_path
    )

    missing_features = (
        set(BASE_NUMERIC_FEATURES) - set(current.columns)
    )

    if missing_features:
        raise ValueError(
            f"Current telemetry frame is missing features: "
            f"{sorted(missing_features)}"
        )

    summary_by_feature = reference_summary.set_index(
        "feature_name"
    )

    samples_by_feature = reference_samples.groupby(
        "feature_name"
    )["reference_value"]

    rows: list[dict[str, object]] = []

    for feature in BASE_NUMERIC_FEATURES:
        result = _numeric_drift(
            reference_values=samples_by_feature.get_group(feature),
            current_values=current[feature],
        )

        result.update(
            {
                "feature_name": feature,
                "reference_split": summary_by_feature.loc[
                    feature,
                    "reference_split",
                ],
                "reference_missing_rate": float(
                    summary_by_feature.loc[
                        feature,
                        "missing_rate",
                    ]
                ),
            }
        )

        rows.append(result)

    return pd.DataFrame(
        rows,
        columns=[
            "feature_name",
            "reference_split",
            "reference_count",
            "current_count",
            "reference_missing_rate",
            "current_missing_count",
            "current_missing_rate",
            "current_mean",
            "current_std",
            "current_min",
            "current_p01",
            "current_p05",
            "current_p25",
            "current_median",
            "current_p75",
            "current_p95",
            "current_p99",
            "current_max",
            "psi",
            "wasserstein_distance",
        ],
    )


def monitor_tecnalia_module_distribution(
    reference_modules: pd.Series,
    current_modules: pd.Series,
) -> dict[str, float | int]:
    reference_modules = reference_modules.dropna().astype(str)
    current_modules = current_modules.dropna().astype(str)

    categories = sorted(
        set(reference_modules.unique())
        | set(current_modules.unique())
    )

    reference_counts = (
        reference_modules
        .value_counts()
        .reindex(categories, fill_value=0)
    )

    current_counts = (
        current_modules
        .value_counts()
        .reindex(categories, fill_value=0)
    )

    reference_distribution = reference_counts.to_numpy(
        dtype=float
    )

    current_distribution = current_counts.to_numpy(
        dtype=float
    )

    return {
        "reference_count": int(reference_modules.size),
        "current_count": int(current_modules.size),
        "js_divergence": jensen_shannon_divergence(
            reference_distribution,
            current_distribution,
        ),
    }
