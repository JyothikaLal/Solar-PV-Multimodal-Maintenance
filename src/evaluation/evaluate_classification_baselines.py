from __future__ import annotations

from pathlib import Path

import pandas as pd
import joblib

from src.data.raptormaps_classification import (
    flatten_raptormaps_images,
    load_raptormaps_split,
)
from src.evaluation.classification import evaluate_classification


RESULTS_DIR = Path("reports/results/raptormaps/classification_baselines")
MODELS_DIR = Path("models/raptormaps/classification_baselines")

MODEL_NAMES = (
    "dummy",
    "logistic_regression",
    "decision_tree",
    "random_forest",
)


def evaluate_saved_baseline_models() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test, test_metadata = load_raptormaps_split("test")
    X_test_flat = flatten_raptormaps_images(X_test)

    class_names = sorted(test_metadata["anomaly_class"].unique())

    summary_rows = []

    for model_name in MODEL_NAMES:
        print(f"\nEvaluating {model_name}...")

        model_path = MODELS_DIR / f"{model_name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found: {model_path}"
            )

        model = joblib.load(model_path)

        y_pred = model.predict(X_test_flat)

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_flat)

        evaluation = evaluate_classification(
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
            y_proba=y_proba,
        )

        metrics = evaluation["overall"]

        summary_row = {
            "model": model_name,
            **metrics,
        }

        summary_rows.append(summary_row)

        per_class_path = (
            RESULTS_DIR / f"{model_name}_test_per_class.csv"
        )

        confusion_path = (
            RESULTS_DIR / f"{model_name}_test_confusion_matrix.csv"
        )

        pd.DataFrame(evaluation["per_class"]).to_csv(
            per_class_path,
            index=False,
        )

        pd.DataFrame(
            evaluation["confusion_matrix"],
            index=class_names,
            columns=class_names,
        ).to_csv(confusion_path)

        print(f"{model_name} test metrics:")

        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        by="macro_f1",
        ascending=False,
    )

    summary_path = RESULTS_DIR / "baseline_test_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nBaseline test summary:")
    print(summary.to_string(index=False))

    return summary


if __name__ == "__main__":
    evaluate_saved_baseline_models()