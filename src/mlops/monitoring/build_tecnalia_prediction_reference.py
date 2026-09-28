from __future__ import annotations

from pathlib import Path

import pandas as pd


PREDICTION_PATH = Path(
    "reports/results/tecnalia/final_models/validation/"
    "gradient_boosting_validation_predictions.csv"
)

SUMMARY_PATH = Path(
    "reports/results/tecnalia/monitoring/"
    "tecnalia_prediction_reference.csv"
)

SAMPLE_PATH = Path(
    "reports/results/tecnalia/monitoring/"
    "tecnalia_prediction_reference_samples.csv"
)

REFERENCE_SAMPLE_SIZE = 10_000
REFERENCE_RANDOM_STATE = 42


def build_prediction_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTION_PATH)

    required = {
        "module_name",
        "Fecha",
        "y_pred",
        "model",
        "candidate_id",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Prediction artifact is missing columns: {sorted(missing)}"
        )

    values = pd.to_numeric(
        predictions["y_pred"],
        errors="coerce",
    )

    if values.isna().any():
        raise ValueError(
            "Prediction reference contains missing or non-numeric y_pred."
        )

    if not values.map(pd.notna).all():
        raise ValueError(
            "Prediction reference contains non-finite y_pred."
        )

    sample = values.sample(
        n=min(REFERENCE_SAMPLE_SIZE, len(values)),
        random_state=REFERENCE_RANDOM_STATE,
    )

    summary = pd.DataFrame(
        [
            {
                "reference_split": "validation",
                "model": predictions["model"].iloc[0],
                "candidate_id": int(
                    predictions["candidate_id"].iloc[0]
                ),
                "count": int(values.size),
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
        ]
    )

    sample_frame = pd.DataFrame(
        {
            "reference_split": "validation",
            "reference_value": sample.to_numpy(dtype=float),
        }
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    sample_frame.to_csv(SAMPLE_PATH, index=False)

    return summary, sample_frame


if __name__ == "__main__":
    summary, samples = build_prediction_reference()

    print(f"Summary artifact: {SUMMARY_PATH}")
    print(f"Sample artifact: {SAMPLE_PATH}")
    print(f"Reference rows: {summary['count'].iloc[0]}")
    print(f"Reference samples: {len(samples)}")
    print()
    print(summary.to_string(index=False))
