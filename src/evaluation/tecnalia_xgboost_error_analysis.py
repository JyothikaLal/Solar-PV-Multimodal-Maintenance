from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.tecnalia_regression_error_analysis import (
    calculate_group_metrics,
    calculate_largest_errors,
    calculate_residual_summary,
    create_gpoa_bins,
)
from src.data.tecnalia_regression import (
    MODULE_PATHS,
    RAW_ROOT,
    attach_frozen_tecnalia_split,
    build_tecnalia_regression_frame,
    engineer_tecnalia_regression_modules,
    load_tecnalia_regression_modules,
)


PREDICTIONS_PATH = Path(
    "reports/results/tecnalia/xgboost/test_predictions.csv"
)

OUTPUT_ROOT = Path(
    "reports/results/tecnalia/xgboost/error_analysis"
)

GPOA_COLUMN = "Front GPOA (W/m²)"
TARGET_COLUMN = "normalized_pmpp"
MODULE_COLUMN = "module_name"
TIMESTAMP_COLUMN = "Fecha"


def load_xgboost_predictions(
    predictions_path: Path = PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Load XGBoost predictions and map them to the evaluation contract."""

    predictions = pd.read_csv(
        predictions_path,
        parse_dates=[TIMESTAMP_COLUMN],
    )

    required_columns = {
        MODULE_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        "prediction",
        "residual",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(
            "XGBoost prediction file is missing required columns: "
            f"{sorted(missing)}"
        )

    result = predictions[
        [
            MODULE_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            "prediction",
            "residual",
        ]
    ].copy()

    result = result.rename(
        columns={
            TARGET_COLUMN: "y_true",
            "prediction": "y_pred",
        }
    )

    result["model"] = "xgboost"

    return result[
        [
            "model",
            MODULE_COLUMN,
            TIMESTAMP_COLUMN,
            "y_true",
            "y_pred",
            "residual",
        ]
    ]


def attach_tecnalia_conditions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Recover audited TECNALIA operating conditions."""

    modules = load_tecnalia_regression_modules(
        RAW_ROOT,
        MODULE_PATHS,
    )

    modules = engineer_tecnalia_regression_modules(modules)

    regression_frame = build_tecnalia_regression_frame(
        modules
    )

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
            "Some XGBoost predictions could not be matched to "
            "TECNALIA operating-condition data."
        )

    return merged


def run_xgboost_error_analysis(
    predictions_path: Path = PREDICTIONS_PATH,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, pd.DataFrame]:
    """Run XGBoost error analysis using the established Task 19 methodology."""

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = load_xgboost_predictions(
        predictions_path
    )

    merged = attach_tecnalia_conditions(
        predictions
    )

    merged = create_gpoa_bins(
        merged
    )

    residual_summary = calculate_residual_summary(
        merged
    )

    module_metrics = calculate_group_metrics(
        merged,
        MODULE_COLUMN,
    ).assign(model="xgboost")

    gpoa_metrics = calculate_group_metrics(
        merged.dropna(subset=["gpoa_bin"]),
        "gpoa_bin",
    ).assign(model="xgboost")

    largest_errors = calculate_largest_errors(
        merged
    )

    merged.to_csv(
        output_root
        / "merged_test_predictions_with_conditions.csv",
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
    results = run_xgboost_error_analysis()

    print("\nResidual summary:")
    print(
        results["residual_summary"].to_string(
            index=False
        )
    )

    print("\nModule-level metrics:")
    print(
        results["module_metrics"].to_string(
            index=False
        )
    )

    print("\nGPOA-bin metrics:")
    print(
        results["gpoa_metrics"].to_string(
            index=False
        )
    )
