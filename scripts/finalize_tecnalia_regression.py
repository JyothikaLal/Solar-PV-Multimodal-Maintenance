from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from src.data.tecnalia_regression import (
    MODULE_PATHS,
    RAW_ROOT,
    SPLIT_MANIFEST,
    attach_frozen_tecnalia_split,
    build_tecnalia_regression_frame,
    engineer_tecnalia_regression_modules,
    get_regression_feature_columns,
    load_tecnalia_regression_modules,
)

from src.models.regression.advanced_features import (
    get_advanced_regression_feature_columns,
)

from src.models.regression.preprocessing import (
    create_regression_preprocessor,
)

from src.models.regression.xgboost_preprocessing import (
    create_xgboost_preprocessor,
)

from src.models.regression.tuning import (
    calculate_regression_metrics,
)


OUTPUT_ROOT = Path(
    "reports/results/tecnalia/final_models"
)

METRICS_PATH = (
    OUTPUT_ROOT / "final_model_metrics.csv"
)

CONFIG_PATH = (
    OUTPUT_ROOT / "final_model_config.json"
)

PREDICTIONS_PATH = (
    OUTPUT_ROOT / "gradient_boosting_test_predictions.csv"
)


def prepare_data() -> pd.DataFrame:
    modules = load_tecnalia_regression_modules(
        RAW_ROOT,
        MODULE_PATHS,
    )

    modules = engineer_tecnalia_regression_modules(
        modules
    )

    frame = build_tecnalia_regression_frame(
        modules
    )

    split_manifest = pd.read_csv(
        SPLIT_MANIFEST
    )

    return attach_frozen_tecnalia_split(
        frame,
        split_manifest,
    )


def evaluate_gradient_boosting(
    frame: pd.DataFrame,
):
    (
        numeric_features,
        categorical_features,
    ) = get_regression_feature_columns()

    feature_columns = (
        numeric_features
        + categorical_features
    )

    development = frame[
        frame["split"].isin(
            ["train", "validation"]
        )
    ].copy()

    test = frame[
        frame["split"] == "test"
    ].copy()

    X_development = development[
        feature_columns
    ]

    y_development = development[
        "normalized_pmpp"
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        "normalized_pmpp"
    ]

    preprocessor = (
        create_regression_preprocessor(
            numeric_features,
            categorical_features,
        )
    )

    X_development_processed = (
        preprocessor.fit_transform(
            X_development
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )

    model.fit(
        X_development_processed,
        y_development,
    )

    predictions = model.predict(
        X_test_processed
    )

    metrics = calculate_regression_metrics(
        y_test.to_numpy(),
        predictions,
    )

    return (
        model,
        preprocessor,
        metrics,
        test,
        predictions,
        {
            "model_family": "gradient_boosting",
            "selection_candidate": 5,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_samples_leaf": 5,
            "features": feature_columns,
        },
    )


def evaluate_xgboost(
    frame: pd.DataFrame,
):
    feature_columns = (
        get_advanced_regression_feature_columns(
            frame
        )
    )

    numeric_features = [
        column
        for column in feature_columns
        if column != "module_name"
    ]

    categorical_features = [
        "module_name"
    ]

    development = frame[
        frame["split"].isin(
            ["train", "validation"]
        )
    ].copy()

    test = frame[
        frame["split"] == "test"
    ].copy()

    X_development = development[
        feature_columns
    ]

    y_development = development[
        "normalized_pmpp"
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        "normalized_pmpp"
    ]

    preprocessor = (
        create_xgboost_preprocessor(
            numeric_features,
            categorical_features,
        )
    )

    X_development_processed = (
        preprocessor.fit_transform(
            X_development
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_development_processed,
        y_development,
    )

    predictions = model.predict(
        X_test_processed
    )

    metrics = calculate_regression_metrics(
        y_test.to_numpy(),
        predictions,
    )

    return (
        model,
        preprocessor,
        metrics,
        test,
        predictions,
        {
            "model_family": "xgboost",
            "selection_candidate": 0,
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "features": feature_columns,
        },
    )


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = prepare_data()

    print(
        f"Development rows: "
        f"{frame['split'].isin(['train', 'validation']).sum()}"
    )

    print(
        f"Test rows: "
        f"{(frame['split'] == 'test').sum()}"
    )

    (
        gb_model,
        gb_preprocessor,
        gb_metrics,
        gb_test,
        gb_predictions,
        gb_config,
    ) = evaluate_gradient_boosting(
        frame
    )

    (
        xgb_model,
        xgb_preprocessor,
        xgb_metrics,
        xgb_test,
        xgb_predictions,
        xgb_config,
    ) = evaluate_xgboost(
        frame
    )

    results = pd.DataFrame(
        [
            {
                "model": "gradient_boosting_tuned",
                "split": "test",
                **gb_metrics,
            },
            {
                "model": "xgboost_tuned",
                "split": "test",
                **xgb_metrics,
            },
        ]
    )

    results.to_csv(
        METRICS_PATH,
        index=False,
    )

    config = {
        "gradient_boosting_tuned": gb_config,
        "xgboost_tuned": xgb_config,
        "selection_rule": (
            "Validation MAE; "
            "test used only for final evaluation"
        ),
    }

    gb_prediction_frame = pd.DataFrame(
        {
            "module_name": gb_test["module_name"].to_numpy(),
            "Fecha": gb_test["Fecha"].to_numpy(),
            "y_true": gb_test["normalized_pmpp"].to_numpy(),
            "y_pred": gb_predictions,
        }
    )

    gb_prediction_frame["residual"] = (
        gb_prediction_frame["y_true"]
        - gb_prediction_frame["y_pred"]
    )

    gb_prediction_frame["model"] = (
        "gradient_boosting_tuned"
    )

    gb_prediction_frame.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    print(
        "\nFinal test results:"
    )

    print(
        results.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {METRICS_PATH}"
    )

    print(
        f"Saved: {CONFIG_PATH}"
    )

    print(
        f"Saved: {PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()