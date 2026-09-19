from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from src.data.tecnalia_regression import (
    attach_frozen_tecnalia_split,
    build_tecnalia_regression_frame,
    MODULE_PATHS,
    RAW_ROOT,
    SPLIT_MANIFEST,
    engineer_tecnalia_regression_modules,
    get_regression_feature_columns,
    load_tecnalia_regression_modules,
    split_tecnalia_regression_frame,
)
from src.models.regression.baselines import create_regression_baseline_models
from src.models.regression.preprocessing import create_regression_preprocessor


OUTPUT_ROOT = Path("reports/results/tecnalia/regression_baselines")


def calculate_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate MAPE while excluding zero-valued targets."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    non_zero = y_true != 0

    if not np.any(non_zero):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (y_true[non_zero] - y_pred[non_zero])
                / y_true[non_zero]
            )
        )
        * 100.0
    )


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate the common regression evaluation metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": calculate_mape(y_true, y_pred),
    }


def prepare_regression_data():
    """Build the frozen TECNALIA regression train/val/test datasets."""
    modules = load_tecnalia_regression_modules(
        RAW_ROOT,
        MODULE_PATHS,
    )

    modules = engineer_tecnalia_regression_modules(modules)

    regression_frame = build_tecnalia_regression_frame(modules)

    split_manifest = pd.read_csv(SPLIT_MANIFEST)

    regression_frame = attach_frozen_tecnalia_split(
        regression_frame,
        split_manifest,
    )

    return split_tecnalia_regression_frame(regression_frame)


def train_and_evaluate_baselines() -> pd.DataFrame:
    """Train and evaluate all TECNALIA regression baseline models."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    splits = prepare_regression_data()

    numeric_features, categorical_features = (
        get_regression_feature_columns()
    )

    models = create_regression_baseline_models()

    results = []
    prediction_frames = []

    X_train = splits["train"][
        numeric_features + categorical_features
    ]
    y_train = splits["train"]["normalized_pmpp"]

    X_validation = splits["validation"][
        numeric_features + categorical_features
    ]
    y_validation = splits["validation"]["normalized_pmpp"]

    X_test = splits["test"][
        numeric_features + categorical_features
    ]
    y_test = splits["test"]["normalized_pmpp"]

    for model_name, model in models.items():
        preprocessor = create_regression_preprocessor(
            numeric_features,
            categorical_features,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        validation_predictions = pipeline.predict(X_validation)
        test_predictions = pipeline.predict(X_test)

        validation_metrics = calculate_regression_metrics(
            y_validation.to_numpy(),
            validation_predictions,
        )

        test_metrics = calculate_regression_metrics(
            y_test.to_numpy(),
            test_predictions,
        )

        results.append(
            {
                "model": model_name,
                "split": "validation",
                **validation_metrics,
            }
        )

        results.append(
            {
                "model": model_name,
                "split": "test",
                **test_metrics,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "module_name": splits["test"]["module_name"].to_numpy(),
                    "Fecha": splits["test"]["Fecha"].to_numpy(),
                    "y_true": y_test.to_numpy(),
                    "y_pred": test_predictions,
                    "residual": (
                        y_test.to_numpy() - test_predictions
                    ),
                    "model": model_name,
                }
            )
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_ROOT / "regression_baseline_metrics.csv",
        index=False,
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions_df.to_csv(
        OUTPUT_ROOT / "test_predictions.csv",
        index=False,
    )

    return results_df


if __name__ == "__main__":
    metrics = train_and_evaluate_baselines()

    print("\nTECNALIA regression baseline results:")
    print(metrics.to_string(index=False))
