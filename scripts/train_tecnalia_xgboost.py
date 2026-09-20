from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.tecnalia_regression import (
    MODULE_PATHS,
    RAW_ROOT,
    SPLIT_MANIFEST,
    attach_frozen_tecnalia_split,
    engineer_tecnalia_regression_modules,
    load_tecnalia_regression_modules,
)
from src.models.regression.advanced_features import (
    CATEGORICAL_FEATURES,
    get_advanced_regression_feature_columns,
)
from src.models.regression.xgboost_model import create_xgboost_regressor
from src.models.regression.xgboost_preprocessing import (
    create_xgboost_preprocessor,
)


OUTPUT_DIR = Path("reports/results/tecnalia/xgboost")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    residuals = y_pred - y_true

    non_zero = y_true != 0

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": float(
            np.mean(
                np.abs(
                    (y_true[non_zero] - y_pred[non_zero])
                    / y_true[non_zero]
                )
            )
            * 100
        ),
        "mean_residual": float(np.mean(residuals)),
    }


def main() -> None:
    print("Loading TECNALIA modules...")
    modules = load_tecnalia_regression_modules(
        RAW_ROOT,
        MODULE_PATHS,
    )

    print("Engineering TECNALIA features...")
    modules = engineer_tecnalia_regression_modules(modules)

    print("Building regression frame...")
    frame = pd.concat(modules.values(), ignore_index=True)

    split_manifest = pd.read_csv(SPLIT_MANIFEST)

    frame = attach_frozen_tecnalia_split(
        frame,
        split_manifest,
    )

    frame = frame.loc[
        frame["Front GPOA (W/m²)"] >= 200
    ].copy()

    frame["normalized_pmpp"] = (
        frame["Pmpp (W)"] / frame["rated_power"]
    )

    feature_columns = get_advanced_regression_feature_columns(frame)

    categorical_features = [
        column
        for column in feature_columns
        if column in CATEGORICAL_FEATURES
    ]

    numeric_features = [
        column
        for column in feature_columns
        if column not in categorical_features
    ]

    print(f"Rows: {len(frame):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()

    X_train = train[feature_columns]
    y_train = train["normalized_pmpp"]

    X_validation = validation[feature_columns]
    y_validation = validation["normalized_pmpp"]

    X_test = test[feature_columns]
    y_test = test["normalized_pmpp"]

    preprocessor = create_xgboost_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    print("Fitting preprocessing on training data only...")
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming validation and test data...")
    X_validation_processed = preprocessor.transform(X_validation)
    X_test_processed = preprocessor.transform(X_test)

    model = create_xgboost_regressor()

    print("Training XGBoost...")
    model.fit(
        X_train_processed,
        y_train,
        eval_set=[
            (X_validation_processed, y_validation),
        ],
        verbose=False,
    )

    validation_predictions = model.predict(X_validation_processed)
    test_predictions = model.predict(X_test_processed)

    validation_metrics = calculate_metrics(
        y_validation.to_numpy(),
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        y_test.to_numpy(),
        test_predictions,
    )

    metrics_rows = [
        {
            "model": "xgboost",
            "split": "validation",
            **validation_metrics,
        },
        {
            "model": "xgboost",
            "split": "test",
            **test_metrics,
        },
    ]

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_path = OUTPUT_DIR / "xgboost_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_df = test[
        [
            "module_name",
            "Fecha",
            "Front GPOA (W/m²)",
            "Pmpp (W)",
            "rated_power",
            "normalized_pmpp",
        ]
    ].copy()

    predictions_df["prediction"] = test_predictions
    predictions_df["residual"] = (
        predictions_df["prediction"]
        - predictions_df["normalized_pmpp"]
    )

    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    config = {
        "model": "xgboost",
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
        "min_child_weight": model.min_child_weight,
        "subsample": model.subsample,
        "colsample_bytree": model.colsample_bytree,
        "random_state": model.random_state,
        "feature_count": len(feature_columns),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "operating_threshold_gpoa": 200,
    }

    with open(
        OUTPUT_DIR / "xgboost_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("\nValidation metrics:")
    print(metrics_df[metrics_df["split"] == "validation"].to_string(index=False))

    print("\nTest metrics:")
    print(metrics_df[metrics_df["split"] == "test"].to_string(index=False))

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
