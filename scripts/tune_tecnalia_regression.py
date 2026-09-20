from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
    create_gradient_boosting_candidates,
    create_xgboost_candidates,
)


OUTPUT_ROOT = Path(
    "reports/results/tecnalia/tuning"
)

RESULTS_PATH = OUTPUT_ROOT / "tuning_results.csv"

BEST_PATH = OUTPUT_ROOT / "best_validation_models.json"


def prepare_data() -> pd.DataFrame:
    """Load TECNALIA data and attach the frozen split."""

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

    frame = attach_frozen_tecnalia_split(
        frame,
        split_manifest,
    )

    return frame


def prepare_xy(
    frame: pd.DataFrame,
    feature_columns: list[str],
):
    """Create train/validation feature and target sets."""

    train = frame[
        frame["split"] == "train"
    ].copy()

    validation = frame[
        frame["split"] == "validation"
    ].copy()

    X_train = train[feature_columns]
    y_train = train["normalized_pmpp"]

    X_validation = validation[
        feature_columns
    ]
    y_validation = validation[
        "normalized_pmpp"
    ]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


def run_gradient_boosting(
    frame: pd.DataFrame,
):
    """
    Tune Gradient Boosting using the existing
    Task 19 feature/preprocessing contract.
    """

    (
        numeric_features,
        categorical_features,
    ) = get_regression_feature_columns()

    feature_columns = (
        numeric_features
        + categorical_features
    )

    print(
        f"Gradient Boosting numeric features: "
        f"{len(numeric_features)}"
    )

    print(
        f"Gradient Boosting categorical features: "
        f"{len(categorical_features)}"
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = prepare_xy(
        frame,
        feature_columns,
    )

    preprocessor = create_regression_preprocessor(
        numeric_features,
        categorical_features,
    )

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    X_validation_processed = (
        preprocessor.transform(X_validation)
    )

    results = []

    for candidate_id, model, params in (
        create_gradient_boosting_candidates()
    ):
        model.fit(
            X_train_processed,
            y_train,
        )

        predictions = model.predict(
            X_validation_processed
        )

        metrics = calculate_regression_metrics(
            y_validation.to_numpy(),
            predictions,
        )

        result = {
            "model_family": "gradient_boosting",
            "candidate_id": candidate_id,
            **params,
            **metrics,
        }

        results.append(result)

        print(
            f"Gradient Boosting "
            f"candidate {candidate_id}: "
            f"MAE={metrics['mae']:.6f}, "
            f"RMSE={metrics['rmse']:.6f}, "
            f"R2={metrics['r2']:.6f}, "
            f"MAPE={metrics['mape_percent']:.4f}%"
        )

    return results


def run_xgboost(
    frame: pd.DataFrame,
):
    """
    Tune XGBoost using the validated advanced
    53-feature contract.
    """

    feature_columns = (
        get_advanced_regression_feature_columns(
            frame
        )
    )

    print(
        f"XGBoost features: "
        f"{len(feature_columns)}"
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = prepare_xy(
        frame,
        feature_columns,
    )

    numeric_features = [
        column
        for column in feature_columns
        if column != "module_name"
    ]

    categorical_features = [
        "module_name"
    ]

    preprocessor = create_xgboost_preprocessor(
        numeric_features,
        categorical_features,
    )

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    X_validation_processed = (
        preprocessor.transform(X_validation)
    )

    results = []

    for candidate_id, model, params in (
        create_xgboost_candidates()
    ):
        model.fit(
            X_train_processed,
            y_train,
        )

        predictions = model.predict(
            X_validation_processed
        )

        metrics = calculate_regression_metrics(
            y_validation.to_numpy(),
            predictions,
        )

        result = {
            "model_family": "xgboost",
            "candidate_id": candidate_id,
            **params,
            **metrics,
        }

        results.append(result)

        print(
            f"XGBoost candidate {candidate_id}: "
            f"MAE={metrics['mae']:.6f}, "
            f"RMSE={metrics['rmse']:.6f}, "
            f"R2={metrics['r2']:.6f}, "
            f"MAPE={metrics['mape_percent']:.4f}%"
        )

    return results


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = prepare_data()

    print(
        f"\nTotal rows: {len(frame)}"
    )

    print(
        "Split counts:"
    )

    print(
        frame["split"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    results = []

    results.extend(
        run_gradient_boosting(
            frame
        )
    )

    results.extend(
        run_xgboost(
            frame
        )
    )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        ["mae", "rmse"],
        ascending=[True, True],
    ).reset_index(
        drop=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    best_models = {}

    for family in [
        "gradient_boosting",
        "xgboost",
    ]:

        family_results = (
            results_df[
                results_df[
                    "model_family"
                ] == family
            ]
            .sort_values(
                ["mae", "rmse"]
            )
        )

        best = family_results.iloc[0]

        best_models[family] = {
            "candidate_id": int(
                best["candidate_id"]
            ),
            "mae": float(
                best["mae"]
            ),
            "rmse": float(
                best["rmse"]
            ),
            "r2": float(
                best["r2"]
            ),
            "mape_percent": float(
                best["mape_percent"]
            ),
        }

    BEST_PATH.write_text(
        json.dumps(
            best_models,
            indent=2,
        )
    )

    print(
        "\nAll validation candidates:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nBest validation models:"
    )

    for family, result in (
        best_models.items()
    ):
        print(
            f"{family}: "
            f"candidate={result['candidate_id']}, "
            f"MAE={result['mae']:.6f}, "
            f"RMSE={result['rmse']:.6f}, "
            f"R2={result['r2']:.6f}, "
            f"MAPE={result['mape_percent']:.4f}%"
        )

    print(
        f"\nSaved: {RESULTS_PATH}"
    )

    print(
        f"Saved: {BEST_PATH}"
    )


if __name__ == "__main__":
    main()