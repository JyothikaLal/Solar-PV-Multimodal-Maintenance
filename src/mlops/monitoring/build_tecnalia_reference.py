from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.tecnalia_regression import (
    BASE_NUMERIC_FEATURES,
    MODULE_PATHS,
    RAW_ROOT,
    SPLIT_MANIFEST,
    attach_frozen_tecnalia_split,
    build_tecnalia_regression_frame,
    engineer_tecnalia_regression_modules,
    load_tecnalia_regression_modules,
    split_tecnalia_regression_frame,
)


OUTPUT_PATH = Path(
    "reports/results/tecnalia/monitoring/"
    "telemetry_feature_reference_samples.csv"
)

SUMMARY_PATH = Path(
    "reports/results/tecnalia/monitoring/"
    "telemetry_feature_reference.csv"
)

REFERENCE_SAMPLE_SIZE = 10_000
REFERENCE_RANDOM_STATE = 42


def build_training_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    modules = load_tecnalia_regression_modules(
        raw_root=RAW_ROOT,
        module_paths=MODULE_PATHS,
    )

    engineered = engineer_tecnalia_regression_modules(modules)

    regression_frame = build_tecnalia_regression_frame(engineered)

    split_manifest = pd.read_csv(SPLIT_MANIFEST)

    regression_frame = attach_frozen_tecnalia_split(
        regression_frame,
        split_manifest,
    )

    splits = split_tecnalia_regression_frame(regression_frame)
    train = splits["train"].copy()

    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for feature in BASE_NUMERIC_FEATURES:
        numeric = pd.to_numeric(
            train[feature],
            errors="coerce",
        )

        values = numeric.dropna()

        if values.empty:
            raise ValueError(
                f"Training reference feature has no valid values: {feature}"
            )

        sample = values.sample(
            n=min(REFERENCE_SAMPLE_SIZE, len(values)),
            random_state=REFERENCE_RANDOM_STATE,
        )

        for value in sample:
            sample_rows.append(
                {
                    "feature_name": feature,
                    "reference_split": "train",
                    "reference_value": float(value),
                }
            )

        summary_rows.append(
            {
                "feature_name": feature,
                "reference_split": "train",
                "count": int(values.size),
                "missing_count": int(numeric.isna().sum()),
                "missing_rate": float(numeric.isna().mean()),
                "sample_count": int(sample.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "p05": float(values.quantile(0.05)),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    samples = pd.DataFrame(sample_rows)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    samples.to_csv(OUTPUT_PATH, index=False)

    return summary, samples


if __name__ == "__main__":
    summary, samples = build_training_reference()

    print(f"Summary artifact: {SUMMARY_PATH}")
    print(f"Sample artifact: {OUTPUT_PATH}")
    print(f"Features: {summary['feature_name'].nunique()}")
    print(f"Reference samples: {len(samples)}")
    print()
    print(summary.to_string(index=False))
