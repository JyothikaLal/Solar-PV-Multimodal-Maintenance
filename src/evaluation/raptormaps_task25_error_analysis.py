from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data.raptormaps_classification import (
    MANIFEST_PATH,
    load_raptormaps_manifest,
)


RESULTS_DIR = Path(
    "reports/results/raptormaps/custom_cnn_task25"
)

ERROR_ANALYSIS_DIR = (
    RESULTS_DIR / "error_analysis"
)

PREDICTIONS_PATH = (
    RESULTS_DIR / "test_predictions.csv"
)

PER_CLASS_PATH = (
    RESULTS_DIR / "test_per_class_metrics.csv"
)

OVERALL_PATH = (
    RESULTS_DIR / "test_overall_metrics.csv"
)

OVERALL_ERROR_PATH = (
    ERROR_ANALYSIS_DIR / "overall_error_summary.csv"
)

CLASS_ERROR_PATH = (
    ERROR_ANALYSIS_DIR / "class_error_analysis.csv"
)

TOP_CONFUSIONS_PATH = (
    ERROR_ANALYSIS_DIR / "top_confusions.csv"
)

NORMALIZED_CM_PATH = (
    ERROR_ANALYSIS_DIR
    / "normalized_confusion_matrix.csv"
)

HIGH_CONFIDENCE_ERRORS_PATH = (
    ERROR_ANALYSIS_DIR
    / "high_confidence_errors.csv"
)

CONFUSION_PLOT_PATH = (
    ERROR_ANALYSIS_DIR / "confusion_matrix.png"
)


def load_task25_predictions() -> pd.DataFrame:
    """Load and validate Task 25 test predictions."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{PREDICTIONS_PATH}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    required_columns = {
        "sample_index",
        "y_true_index",
        "y_pred_index",
        "y_true_class",
        "y_pred_class",
        "confidence",
    }

    missing = (
        required_columns
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            "Prediction file is missing columns: "
            f"{sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError(
            "Prediction file contains no rows."
        )

    if not np.isfinite(
        predictions["confidence"]
    ).all():
        raise ValueError(
            "Confidence values contain non-finite values."
        )

    if (
        (predictions["confidence"] < 0)
        | (predictions["confidence"] > 1)
    ).any():
        raise ValueError(
            "Confidence values must be in [0,1]."
        )

    return predictions


def get_class_names(
    predictions: pd.DataFrame,
) -> list[str]:
    """Return classes in stable class-index order."""

    indexed = (
        predictions[
            ["y_true_index", "y_true_class"]
        ]
        .drop_duplicates()
        .sort_values("y_true_index")
    )

    return indexed["y_true_class"].tolist()


def calculate_overall_error_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate aggregate test error statistics."""

    total = len(predictions)

    correct = (
        predictions["y_true_index"]
        == predictions["y_pred_index"]
    )

    errors = ~correct

    summary = {
        "total_samples": total,
        "correct_predictions": int(correct.sum()),
        "incorrect_predictions": int(errors.sum()),
        "error_rate": float(errors.mean()),
        "mean_confidence": float(
            predictions["confidence"].mean()
        ),
        "mean_correct_confidence": float(
            predictions.loc[
                correct,
                "confidence",
            ].mean()
        ),
        "mean_error_confidence": float(
            predictions.loc[
                errors,
                "confidence",
            ].mean()
        ),
        "high_confidence_error_count": int(
            (
                errors
                & (predictions["confidence"] >= 0.80)
            ).sum()
        ),
        "high_confidence_error_rate": float(
            (
                errors
                & (predictions["confidence"] >= 0.80)
            ).sum()
            / total
        ),
    }

    return pd.DataFrame([summary])


def calculate_class_error_analysis(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate class-level error statistics."""

    rows = []

    for class_index, group in predictions.groupby(
        "y_true_index",
        sort=True,
    ):
        class_name = group[
            "y_true_class"
        ].iloc[0]

        errors = (
            group["y_pred_index"]
            != group["y_true_index"]
        )

        rows.append(
            {
                "class_index": int(class_index),
                "class_name": class_name,
                "support": len(group),
                "errors": int(errors.sum()),
                "error_rate": float(errors.mean()),
                "mean_confidence": float(
                    group["confidence"].mean()
                ),
                "mean_error_confidence": (
                    float(
                        group.loc[
                            errors,
                            "confidence",
                        ].mean()
                    )
                    if errors.any()
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_top_confusions(
    predictions: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    """Calculate the most frequent incorrect class pairs."""

    errors = predictions[
        predictions["y_true_index"]
        != predictions["y_pred_index"]
    ].copy()

    grouped = (
        errors.groupby(
            [
                "y_true_index",
                "y_true_class",
                "y_pred_index",
                "y_pred_class",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    grouped.insert(
        0,
        "rank",
        np.arange(1, len(grouped) + 1),
    )

    total_errors = max(len(errors), 1)

    grouped["error_percentage"] = (
        grouped["count"] / total_errors * 100.0
    )

    return grouped.head(25)


def calculate_normalized_confusion_matrix(
    predictions: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    """Create row-normalized confusion matrix."""

    labels = np.arange(len(class_names))

    matrix = confusion_matrix(
        predictions["y_true_index"],
        predictions["y_pred_index"],
        labels=labels,
    )

    row_sums = matrix.sum(axis=1, keepdims=True)

    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_sums != 0,
    )

    return pd.DataFrame(
        normalized,
        index=class_names,
        columns=class_names,
    )


def attach_manifest_information(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach source image information to predictions."""

    manifest = load_raptormaps_manifest(
        manifest_path=MANIFEST_PATH,
    )

    test_manifest = manifest[
        manifest["split"] == "test"
    ].reset_index(drop=True)

    if len(test_manifest) != len(predictions):
        raise ValueError(
            "Test manifest and prediction counts differ: "
            f"{len(test_manifest)} vs "
            f"{len(predictions)}."
        )

    indexed_manifest = test_manifest[
        [
            "image_filepath",
            "anomaly_class",
        ]
    ].copy()

    indexed_manifest.insert(
        0,
        "sample_index",
        np.arange(len(indexed_manifest)),
    )

    merged = predictions.merge(
        indexed_manifest,
        on="sample_index",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_manifest",
        ),
    )

    if merged["image_filepath"].isna().any():
        raise ValueError(
            "Some predictions could not be mapped "
            "back to test manifest rows."
        )

    return merged


def run_error_analysis() -> None:
    """Run the complete Task 25 error analysis."""

    ERROR_ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = load_task25_predictions()

    class_names = get_class_names(
        predictions
    )

    enriched = attach_manifest_information(
        predictions
    )

    overall = calculate_overall_error_summary(
        predictions
    )

    class_errors = calculate_class_error_analysis(
        predictions
    )

    top_confusions = calculate_top_confusions(
        predictions,
        class_names,
    )

    normalized_matrix = (
        calculate_normalized_confusion_matrix(
            predictions,
            class_names,
        )
    )

    errors = enriched[
        enriched["y_true_index"]
        != enriched["y_pred_index"]
    ].copy()

    high_confidence_errors = (
        errors[
            errors["confidence"] >= 0.80
        ]
        .sort_values(
            "confidence",
            ascending=False,
        )
        .head(100)
    )

    overall.to_csv(
        OVERALL_ERROR_PATH,
        index=False,
    )

    class_errors.to_csv(
        CLASS_ERROR_PATH,
        index=False,
    )

    top_confusions.to_csv(
        TOP_CONFUSIONS_PATH,
        index=False,
    )

    normalized_matrix.to_csv(
        NORMALIZED_CM_PATH
    )

    high_confidence_errors.to_csv(
        HIGH_CONFIDENCE_ERRORS_PATH,
        index=False,
    )

    # Plot the raw confusion matrix.
    labels = np.arange(len(class_names))

    raw_matrix = confusion_matrix(
        predictions["y_true_index"],
        predictions["y_pred_index"],
        labels=labels,
    )

    figure, axis = plt.subplots(
        figsize=(12, 10)
    )

    image = axis.imshow(
        raw_matrix,
        interpolation="nearest",
        aspect="auto",
    )

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=labels,
        yticks=labels,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Task 25 Custom CNN — Test Confusion Matrix",
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_PLOT_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Task 25 error analysis complete."
    )

    print(
        f"Total errors: "
        f"{int(overall['incorrect_predictions'].iloc[0])}"
    )

    print(
        f"Error rate: "
        f"{overall['error_rate'].iloc[0]:.4f}"
    )

    print(
        f"High-confidence errors: "
        f"{int(overall['high_confidence_error_count'].iloc[0])}"
    )

    print()
    print("Top confusion pairs:")
    print(
        top_confusions[
            [
                "rank",
                "y_true_class",
                "y_pred_class",
                "count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run_error_analysis()
