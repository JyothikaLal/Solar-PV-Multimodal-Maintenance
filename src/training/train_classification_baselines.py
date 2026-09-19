from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data.raptormaps_classification import (
    flatten_raptormaps_images,
    load_raptormaps_split,
)
from src.evaluation.classification import (
    evaluate_classification,
)
from src.models.classification.baselines import (
    create_baseline_models,
)


RESULTS_DIR = Path(
    "reports/results/raptormaps/classification_baselines"
)

MODELS_DIR = Path(
    "models/raptormaps/classification_baselines"
)


def train_and_evaluate_baselines() -> pd.DataFrame:
    """
    Train and evaluate all classical RaptorMaps
    classification baseline models.

    Training is performed only on the frozen training split.
    Model selection is performed using the validation split.
    The test split is not used here.
    """
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train_images, y_train, train_metadata = (
        load_raptormaps_split("train")
    )

    X_val_images, y_val, val_metadata = (
        load_raptormaps_split("validation")
    )

    X_train = flatten_raptormaps_images(
        X_train_images
    )

    X_val = flatten_raptormaps_images(
        X_val_images
    )

    class_names = sorted(
        train_metadata["anomaly_class"].unique()
    )

    models = create_baseline_models()

    summary_rows = []

    for model_name, model in models.items():
        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train,
            y_train,
        )

        y_pred = model.predict(
            X_val
        )

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(
                X_val
            )
        else:
            y_proba = None

        evaluation = evaluate_classification(
            y_true=y_val,
            y_pred=y_pred,
            class_names=class_names,
            y_proba=y_proba,
        )

        overall = evaluation["overall"]

        summary_rows.append(
            {
                "model": model_name,
                **overall,
            }
        )

        per_class = evaluation[
            "per_class"
        ]

        per_class.to_csv(
            RESULTS_DIR
            / f"{model_name}_per_class.csv",
            index=False,
        )

        confusion_matrix = evaluation[
            "confusion_matrix"
        ]

        pd.DataFrame(
            confusion_matrix,
            index=class_names,
            columns=class_names,
        ).to_csv(
            RESULTS_DIR
            / f"{model_name}_confusion_matrix.csv"
        )

        joblib.dump(
            model,
            MODELS_DIR
            / f"{model_name}.joblib",
        )

        print(
            f"{model_name} validation metrics:"
        )

        for metric_name, value in overall.items():
            print(
                f"  {metric_name}: "
                f"{value:.4f}"
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    summary.to_csv(
        RESULTS_DIR
        / "baseline_summary.csv",
        index=False,
    )

    return summary


if __name__ == "__main__":
    summary = train_and_evaluate_baselines()

    print(
        "\nBaseline validation summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )