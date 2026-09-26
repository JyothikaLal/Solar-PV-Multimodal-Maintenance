"""Register the finalized TECNALIA Gradient Boosting experiment in MLflow.

This script is retrospective: it reads already-generated experiment outputs
and does not retrain the model or regenerate predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.mlops.mlflow_tracking import (
    EXPERIMENT_TECNALIA_REGRESSION,
    build_standard_tags,
    log_artifact,
    log_dataset_metadata,
    log_metrics,
    log_params,
    start_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_MODEL_DIR = (
    PROJECT_ROOT
    / "reports"
    / "results"
    / "tecnalia"
    / "final_models"
)

CONFIG_PATH = FINAL_MODEL_DIR / "final_model_config.json"
METRICS_PATH = FINAL_MODEL_DIR / "final_model_metrics.csv"
SELECTION_PATH = FINAL_MODEL_DIR / "model_selection_summary.json"
PREDICTIONS_PATH = (
    FINAL_MODEL_DIR / "gradient_boosting_test_predictions.csv"
)

SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "results"
    / "tecnalia"
    / "tecnalia_split_manifest.csv"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    required_files = [
        CONFIG_PATH,
        METRICS_PATH,
        SELECTION_PATH,
        PREDICTIONS_PATH,
        SPLIT_MANIFEST_PATH,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    config = load_json(CONFIG_PATH)
    selection = load_json(SELECTION_PATH)

    metrics_df = pd.read_csv(METRICS_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    manifest_df = pd.read_csv(SPLIT_MANIFEST_PATH)

    model_name = selection["selected_model"]
    model_config = config[model_name]

    selected_metrics = metrics_df.loc[
        metrics_df["model"] == model_name
    ]

    if selected_metrics.empty:
        raise ValueError(
            f"No metrics found for selected model: {model_name}"
        )

    test_metrics = selected_metrics.loc[
        selected_metrics["split"] == "test"
    ]

    if test_metrics.empty:
        raise ValueError(
            f"No test metrics found for selected model: {model_name}"
        )

    test_metrics = test_metrics.iloc[0]

    if predictions_df.empty:
        raise ValueError("Final prediction artifact is empty.")

    expected_model_name = model_name
    observed_model_names = predictions_df["model"].unique().tolist()

    if observed_model_names != [expected_model_name]:
        raise ValueError(
            "Prediction artifact model identifier does not match "
            f"selected model. Expected {expected_model_name!r}, "
            f"found {observed_model_names!r}."
        )

    split_counts = (
        manifest_df["split"]
        .value_counts()
        .to_dict()
    )

    tags = build_standard_tags(
        dataset="TECNALIA",
        modality="telemetry",
        task="regression",
        model_family=model_config["model_family"],
        stage="final",
        run_type="retrospective",
        split_strategy="frozen_tecnalia_split_manifest",
    )

    params = {
        "target": "normalized_pmpp",
        "selected_model": model_name,
        "selection_candidate_id": selection["selected_candidate_id"],
        "selection_metric": selection["primary_selection_metric"],
        "selection_direction": selection["selection_direction"],
        "n_estimators": model_config["n_estimators"],
        "learning_rate": model_config["learning_rate"],
        "max_depth": model_config["max_depth"],
        "min_samples_leaf": model_config["min_samples_leaf"],
        "random_state": selection["selected_hyperparameters"]["random_state"],
        "feature_count": len(model_config["features"]),
        "features": model_config["features"],
        "manifest_row_count": len(manifest_df),
        "evaluated_test_rows": len(predictions_df),
    }

    validation_metrics = selection["validation_results"]

    metrics = {
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_r2": validation_metrics["r2"],
        "validation_mape_percent": validation_metrics["mape_percent"],
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_r2": float(test_metrics["r2"]),
        "test_mape_percent": float(test_metrics["mape_percent"]),
    }

    with start_run(
        experiment_name=EXPERIMENT_TECNALIA_REGRESSION,
        run_name="tecnalia_gradient_boosting_final_retrospective",
        tags=tags,
    ) as run:
        log_params(params)

        log_dataset_metadata(
            dataset_name="TECNALIA PV Performance",
            dataset_version="project_locked",
            manifest_path=SPLIT_MANIFEST_PATH,
            split_strategy="frozen_tecnalia_split_manifest",
            sample_counts={
                "manifest_train": int(split_counts.get("train", 0)),
                "manifest_validation": int(
                    split_counts.get("validation", 0)
                ),
                "manifest_test": int(split_counts.get("test", 0)),
                "evaluated_test": len(predictions_df),
            },
            metadata={
                "target_name": "normalized_pmpp",
                "target_construction": "Pmpp / rated_power",
            },
        )

        log_metrics(metrics)

        for artifact in [
            CONFIG_PATH,
            METRICS_PATH,
            SELECTION_PATH,
            PREDICTIONS_PATH,
        ]:
            log_artifact(artifact, artifact_path="tecnalia_final_model")

        print("MLflow retrospective run created successfully.")
        print("Run ID:", run.info.run_id)
        print("Experiment:", EXPERIMENT_TECNALIA_REGRESSION)
        print("Run name:", run.info.run_name)
        print("Selected model:", model_name)
        print("Evaluated test rows:", len(predictions_df))
        print("Validation MAE:", validation_metrics["mae"])
        print("Test MAE:", float(test_metrics["mae"]))
        print("Test RMSE:", float(test_metrics["rmse"]))
        print("Test R2:", float(test_metrics["r2"]))


if __name__ == "__main__":
    main()
