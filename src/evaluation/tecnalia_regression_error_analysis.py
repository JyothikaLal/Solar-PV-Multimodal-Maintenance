from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PREDICTIONS_PATH = Path(
    "reports/results/tecnalia/regression_baselines/test_predictions.csv"
)

OUTPUT_ROOT = Path(
    "reports/results/tecnalia/regression_baselines/error_analysis"
)

GPOA_COLUMN = "Front GPOA (W/m²)"
TARGET_COLUMN = "normalized_pmpp"
MODULE_COLUMN = "module_name"
TIMESTAMP_COLUMN = "Fecha"


def calculate_group_metrics(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Calculate regression error metrics for each group."""
    rows = []

    for group_value, group in df.groupby(group_column, sort=True):
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()

        rows.append(
            {
                group_column: group_value,
                "n": len(group),
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": np.sqrt(
                    mean_squared_error(y_true, y_pred)
                ),
                "mean_residual": np.mean(group["residual"]),
                "median_abs_error": np.median(
                    np.abs(group["residual"])
                ),
            }
        )

    return pd.DataFrame(rows)


def create_gpoa_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Attach interpretable operating-condition GPOA bins."""
    result = df.copy()

    bins = [200, 400, 600, 800, 1000, 1200, np.inf]
    labels = [
        "200-400",
        "400-600",
        "600-800",
        "800-1000",
        "1000-1200",
        "1200+",
    ]

    result["gpoa_bin"] = pd.cut(
        result[GPOA_COLUMN],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    return result


def calculate_residual_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall residual statistics by model."""
    rows = []

    for model_name, group in df.groupby("model", sort=True):
        residuals = group["residual"].to_numpy()

        rows.append(
            {
                "model": model_name,
                "n": len(group),
                "mean_residual": np.mean(residuals),
                "median_residual": np.median(residuals),
                "std_residual": np.std(residuals),
                "mae": np.mean(np.abs(residuals)),
                "p95_abs_error": np.percentile(
                    np.abs(residuals),
                    95,
                ),
                "max_abs_error": np.max(
                    np.abs(residuals)
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_largest_errors(
    df: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """Return the observations with the largest absolute residuals."""
    result = df.copy()

    result["absolute_error"] = np.abs(result["residual"])

    return (
        result.sort_values(
            "absolute_error",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def run_error_analysis(
    predictions_path: Path = PREDICTIONS_PATH,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, pd.DataFrame]:
    """Run the complete baseline regression error analysis."""
    output_root.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(
        predictions_path,
        parse_dates=[TIMESTAMP_COLUMN],
    )

    required_columns = {
        MODULE_COLUMN,
        TIMESTAMP_COLUMN,
        "y_true",
        "y_pred",
        "residual",
        "model",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing required columns: "
            f"{sorted(missing)}"
        )

    # The prediction artifact intentionally does not contain GPOA.
    # Recover it from the audited raw TECNALIA data through the
    # existing preprocessing path.
    from src.data.tecnalia_regression import (
        MODULE_PATHS,
        RAW_ROOT,
        attach_frozen_tecnalia_split,
        build_tecnalia_regression_frame,
        engineer_tecnalia_regression_modules,
        load_tecnalia_regression_modules,
    )

    modules = load_tecnalia_regression_modules(
        RAW_ROOT,
        MODULE_PATHS,
    )

    modules = engineer_tecnalia_regression_modules(modules)

    regression_frame = build_tecnalia_regression_frame(modules)

    regression_frame = regression_frame[
        [
            MODULE_COLUMN,
            TIMESTAMP_COLUMN,
            GPOA_COLUMN,
            TARGET_COLUMN,
        ]
    ]

    regression_frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        regression_frame[TIMESTAMP_COLUMN]
    )

    merged = predictions.merge(
        regression_frame,
        on=[MODULE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        validate="many_to_one",
    )

    if merged[GPOA_COLUMN].isna().any():
        raise ValueError(
            "Some predictions could not be matched to TECNALIA "
            "operating-condition data."
        )

    merged = create_gpoa_bins(merged)

    residual_summary = calculate_residual_summary(merged)

    module_metrics = pd.concat(
        [
            calculate_group_metrics(
                group,
                MODULE_COLUMN,
            ).assign(model=model_name)
            for model_name, group in merged.groupby(
                "model",
                sort=True,
            )
        ],
        ignore_index=True,
    )

    gpoa_metrics = pd.concat(
        [
            calculate_group_metrics(
                group.dropna(subset=["gpoa_bin"]),
                "gpoa_bin",
            ).assign(model=model_name)
            for model_name, group in merged.groupby(
                "model",
                sort=True,
            )
        ],
        ignore_index=True,
    )

    largest_errors = calculate_largest_errors(merged)

    merged.to_csv(
        output_root / "merged_test_predictions_with_conditions.csv",
        index=False,
    )

    residual_summary.to_csv(
        output_root / "residual_summary.csv",
        index=False,
    )

    module_metrics.to_csv(
        output_root / "module_error_metrics.csv",
        index=False,
    )

    gpoa_metrics.to_csv(
        output_root / "gpoa_error_metrics.csv",
        index=False,
    )

    largest_errors.to_csv(
        output_root / "largest_prediction_errors.csv",
        index=False,
    )

    return {
        "merged": merged,
        "residual_summary": residual_summary,
        "module_metrics": module_metrics,
        "gpoa_metrics": gpoa_metrics,
        "largest_errors": largest_errors,
    }


if __name__ == "__main__":
    results = run_error_analysis()

    print("\nResidual summary:")
    print(results["residual_summary"].to_string(index=False))

    print("\nModule-level metrics:")
    print(results["module_metrics"].to_string(index=False))

    print("\nGPOA-bin metrics:")
    print(results["gpoa_metrics"].to_string(index=False))
