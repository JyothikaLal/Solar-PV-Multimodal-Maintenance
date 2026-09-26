from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.tecnalia_regression import (
    get_regression_feature_columns,
)
from src.models.regression.preprocessing import (
    create_regression_preprocessor,
)
from src.models.regression.tuning import (
    create_gradient_boosting_candidates,
)
from src.training.train_tecnalia_regression import (
    prepare_regression_data,
)


OUTPUT_ROOT = Path(
    "reports/results/tecnalia/final_models/validation"
)


SELECTED_CANDIDATE_ID = 5


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = prepare_regression_data()

    numeric_features, categorical_features = (
        get_regression_feature_columns()
    )

    feature_columns = (
        numeric_features + categorical_features
    )

    train = splits["train"]
    validation = splits["validation"]

    X_train = train[feature_columns]
    y_train = train["normalized_pmpp"]

    X_validation = validation[feature_columns]
    y_validation = validation["normalized_pmpp"]

    preprocessor = create_regression_preprocessor(
        numeric_features,
        categorical_features,
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_validation_processed = preprocessor.transform(
        X_validation
    )

    candidates = create_gradient_boosting_candidates()

    selected = [
        item
        for item in candidates
        if item[0] == SELECTED_CANDIDATE_ID
    ]

    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one Gradient Boosting "
            f"candidate with id {SELECTED_CANDIDATE_ID}."
        )

    candidate_id, model, params = selected[0]

    model.fit(
        X_train_processed,
        y_train,
    )

    predictions = model.predict(
        X_validation_processed
    )

    output = validation[
        ["module_name", "Fecha", "normalized_pmpp"]
    ].copy()

    output = output.rename(
        columns={
            "normalized_pmpp": "y_true",
        }
    )

    output["y_pred"] = predictions
    output["residual"] = (
        output["y_true"] - output["y_pred"]
    )
    output["abs_residual"] = (
        output["residual"].abs()
    )

    output["model"] = "gradient_boosting_tuned"
    output["candidate_id"] = candidate_id

    prediction_path = (
        OUTPUT_ROOT
        / "gradient_boosting_validation_predictions.csv"
    )

    output.to_csv(
        prediction_path,
        index=False,
    )

    residual = output["residual"]

    summary = pd.DataFrame(
        [
            {
                "model": "gradient_boosting_tuned",
                "candidate_id": candidate_id,
                "n": len(output),
                "mae": output["abs_residual"].mean(),
                "rmse": (
                    (output["residual"] ** 2).mean()
                    ** 0.5
                ),
                "mean_residual": residual.mean(),
                "median_residual": residual.median(),
                "std_residual": residual.std(),
                "q01_residual": residual.quantile(0.01),
                "q05_residual": residual.quantile(0.05),
                "q10_residual": residual.quantile(0.10),
                "q25_residual": residual.quantile(0.25),
                "q50_residual": residual.quantile(0.50),
                "q75_residual": residual.quantile(0.75),
                "q90_residual": residual.quantile(0.90),
                "q95_residual": residual.quantile(0.95),
                "q99_residual": residual.quantile(0.99),
                "q01_abs_residual": output[
                    "abs_residual"
                ].quantile(0.01),
                "q05_abs_residual": output[
                    "abs_residual"
                ].quantile(0.05),
                "q50_abs_residual": output[
                    "abs_residual"
                ].quantile(0.50),
                "q95_abs_residual": output[
                    "abs_residual"
                ].quantile(0.95),
                "q99_abs_residual": output[
                    "abs_residual"
                ].quantile(0.99),
            }
        ]
    )

    summary_path = (
        OUTPUT_ROOT
        / "validation_residual_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    metadata = pd.DataFrame(
        [
            {
                "model": "gradient_boosting_tuned",
                "candidate_id": candidate_id,
                **params,
                "feature_count": len(feature_columns),
                "numeric_feature_count": len(
                    numeric_features
                ),
                "categorical_feature_count": len(
                    categorical_features
                ),
                "training_rows": len(train),
                "validation_rows": len(validation),
                "split_source": (
                    "frozen TECNALIA split manifest"
                ),
                "selection_rule": (
                    "Validation MAE; model already "
                    "locked before test evaluation"
                ),
            }
        ]
    )

    metadata.to_csv(
        OUTPUT_ROOT / "validation_artifact_metadata.csv",
        index=False,
    )

    print(
        "\nValidation artifact generation completed."
    )
    print(
        f"Candidate: {candidate_id}"
    )
    print(
        f"Training rows: {len(train)}"
    )
    print(
        f"Validation rows: {len(validation)}"
    )
    print(
        f"Validation MAE: "
        f"{output['abs_residual'].mean():.6f}"
    )
    print(
        f"Validation RMSE: "
        f"{((output['residual'] ** 2).mean() ** 0.5):.6f}"
    )
    print(
        f"Validation mean residual: "
        f"{residual.mean():.6f}"
    )
    print(
        f"Validation median residual: "
        f"{residual.median():.6f}"
    )
    print(
        f"Predictions: {prediction_path}"
    )
    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
