from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    y_proba: np.ndarray | None = None,
) -> dict[str, object]:
    """
    Evaluate a multiclass classification model.

    Args:
        y_true:
            Ground-truth integer class labels.

        y_pred:
            Predicted integer class labels.

        class_names:
            Class names ordered according to the class indices.

        y_proba:
            Optional predicted class probabilities with shape
            (n_samples, n_classes).

    Returns:
        Dictionary containing overall metrics, per-class metrics,
        confusion matrix, and optional probability-based metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError(
            f"y_true must be 1-dimensional. "
            f"Received shape: {y_true.shape}."
        )

    if y_pred.ndim != 1:
        raise ValueError(
            f"y_pred must be 1-dimensional. "
            f"Received shape: {y_pred.shape}."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same number "
            "of samples."
        )

    if not class_names:
        raise ValueError(
            "class_names must not be empty."
        )

    labels = np.arange(len(class_names))

    overall_metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
    }

    precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    support = np.bincount(
        y_true,
        minlength=len(class_names),
    )

    per_class_metrics = pd.DataFrame(
        {
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    result: dict[str, object] = {
        "overall": overall_metrics,
        "per_class": per_class_metrics,
        "confusion_matrix": matrix,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

        expected_shape = (
            len(y_true),
            len(class_names),
        )

        if y_proba.shape != expected_shape:
            raise ValueError(
                "y_proba has an unexpected shape: "
                f"{y_proba.shape}. "
                f"Expected {expected_shape}."
            )

        y_true_one_hot = np.eye(
            len(class_names)
        )[y_true]

        overall_metrics["roc_auc_ovr_macro"] = (
            roc_auc_score(
                y_true_one_hot,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        )

        overall_metrics["pr_auc_macro"] = (
            average_precision_score(
                y_true_one_hot,
                y_proba,
                average="macro",
            )
        )

    return result