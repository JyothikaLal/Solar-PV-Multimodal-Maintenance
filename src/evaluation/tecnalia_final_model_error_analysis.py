from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.tecnalia_regression import (
    MODULE_PATHS,
    RAW_ROOT,
)


PREDICTIONS_PATH = Path(
    "reports/results/tecnalia/final_models/"
    "gradient_boosting_test_predictions.csv"
)

OUTPUT_ROOT = Path(
    "reports/results/tecnalia/final_models/"
    "error_analysis"
)


GPOA_COLUMN = "Front GPOA (W/m²)"

GPOA_BINS = [
    200,
    400,
    600,
    800,
    1000,
    1200,
    np.inf,
]

GPOA_LABELS = [
    "200-400",
    "400-600",
    "600-800",
    "800-1000",
    "1000-1200",
    "1200+",
]


def load_test_predictions() -> pd.DataFrame:
    df = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["Fecha"],
    )

    required_columns = {
        "module_name",
        "Fecha",
        "y_true",
        "y_pred",
        "residual",
        "model",
    }

    missing = required_columns.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    if len(df) != 12925:
        raise ValueError(
            f"Expected 12925 test predictions, got {len(df)}"
        )

    if not np.allclose(
        df["residual"].to_numpy(),
        df["y_true"].to_numpy()
        - df["y_pred"].to_numpy(),
    ):
        raise ValueError(
            "Residual column does not equal y_true - y_pred"
        )

    return df


def load_gpoa_data() -> pd.DataFrame:
    frames = []

    for module_name, relative_path in MODULE_PATHS.items():

        path = RAW_ROOT / relative_path

        raw = pd.read_csv(
            path,
            sep=";",
        )

        raw["Fecha"] = pd.to_datetime(
            raw["Fecha"],
            errors="coerce",
        )

        raw["module_name"] = module_name

        frames.append(
            raw[
                [
                    "module_name",
                    "Fecha",
                    GPOA_COLUMN,
                ]
            ]
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def attach_gpoa(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    gpoa = load_gpoa_data()

    if gpoa.duplicated(
        ["module_name", "Fecha"]
    ).any():
        raise ValueError(
            "Duplicate module/timestamp keys found in raw GPOA data"
        )

    merged = predictions.merge(
        gpoa,
        on=["module_name", "Fecha"],
        how="left",
        validate="one_to_one",
    )

    missing_gpoa = merged[GPOA_COLUMN].isna().sum()

    if missing_gpoa:
        raise ValueError(
            f"Missing GPOA values after merge: {missing_gpoa}"
        )

    return merged


def calculate_metrics(
    df: pd.DataFrame,
) -> dict:

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    residual = df["residual"].to_numpy()

    return {
        "n": len(df),
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "mean_residual": residual.mean(),
        "median_residual": np.median(residual),
        "std_residual": residual.std(),
        "p95_abs_residual": np.percentile(
            np.abs(residual),
            95,
        ),
        "max_abs_residual": np.max(
            np.abs(residual)
        ),
    }


def calculate_group_metrics(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    rows = []

    for group_name, group in df.groupby(
        group_column,
        dropna=False,
    ):

        metrics = calculate_metrics(group)

        rows.append(
            {
                group_column: group_name,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = load_test_predictions()

    merged = attach_gpoa(
        predictions
    )

    merged["gpoa_bin"] = pd.cut(
        merged[GPOA_COLUMN],
        bins=GPOA_BINS,
        labels=GPOA_LABELS,
        right=False,
    )

    overall = pd.DataFrame(
        [calculate_metrics(merged)]
    )

    module_metrics = calculate_group_metrics(
        merged,
        "module_name",
    )

    gpoa_metrics = calculate_group_metrics(
        merged,
        "gpoa_bin",
    )

    largest_errors = (
        merged.assign(
            abs_residual=lambda x:
                x["residual"].abs()
        )
        .sort_values(
            "abs_residual",
            ascending=False,
        )
        .head(100)
        [
            [
                "module_name",
                "Fecha",
                GPOA_COLUMN,
                "y_true",
                "y_pred",
                "residual",
                "abs_residual",
            ]
        ]
    )

    residual_quantiles = pd.DataFrame(
        {
            "quantile": [
                0.00,
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
                1.00,
            ],
            "residual": merged[
                "residual"
            ].quantile(
                [
                    0.00,
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                    1.00,
                ]
            ).to_numpy(),
            "absolute_residual": merged[
                "residual"
            ].abs().quantile(
                [
                    0.00,
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                    1.00,
                ]
            ).to_numpy(),
        }
    )

    overall.to_csv(
        OUTPUT_ROOT / "overall_metrics.csv",
        index=False,
    )

    module_metrics.to_csv(
        OUTPUT_ROOT / "module_error_metrics.csv",
        index=False,
    )

    gpoa_metrics.to_csv(
        OUTPUT_ROOT / "gpoa_error_metrics.csv",
        index=False,
    )

    largest_errors.to_csv(
        OUTPUT_ROOT / "largest_prediction_errors.csv",
        index=False,
    )

    residual_quantiles.to_csv(
        OUTPUT_ROOT / "residual_quantiles.csv",
        index=False,
    )

    merged.to_csv(
        OUTPUT_ROOT / "merged_test_predictions_with_conditions.csv",
        index=False,
    )

    print("\nOverall metrics:")
    print(
        overall.to_string(
            index=False
        )
    )

    print("\nModule metrics:")
    print(
        module_metrics.to_string(
            index=False
        )
    )

    print("\nGPOA metrics:")
    print(
        gpoa_metrics.to_string(
            index=False
        )
    )

    print(
        "\nSaved error-analysis artifacts to:"
    )
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()