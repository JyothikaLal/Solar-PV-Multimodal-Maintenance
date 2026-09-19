import numpy as np
import pandas as pd
import pytest

from src.evaluation.classification import (
    evaluate_classification,
)


CLASS_NAMES = [
    "class_a",
    "class_b",
    "class_c",
]


def test_classification_evaluation_returns_expected_metrics():
    y_true = np.array([
        0, 0, 1, 1, 2, 2
    ])

    y_pred = np.array([
        0, 1, 1, 1, 2, 0
    ])

    result = evaluate_classification(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    assert set(result["overall"]) == {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    }


def test_per_class_metrics_have_expected_structure():
    y_true = np.array([
        0, 0, 1, 1, 2, 2
    ])

    y_pred = np.array([
        0, 1, 1, 1, 2, 0
    ])

    result = evaluate_classification(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    per_class = result["per_class"]

    assert isinstance(
        per_class,
        pd.DataFrame,
    )

    assert list(per_class.columns) == [
        "class_name",
        "precision",
        "recall",
        "f1",
        "support",
    ]

    assert len(per_class) == 3


def test_confusion_matrix_has_expected_shape():
    y_true = np.array([
        0, 1, 2, 0, 1, 2
    ])

    y_pred = np.array([
        0, 1, 1, 2, 1, 2
    ])

    result = evaluate_classification(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    matrix = result["confusion_matrix"]

    assert matrix.shape == (3, 3)
    assert matrix.sum() == len(y_true)


def test_probability_metrics_are_computed():
    y_true = np.array([
        0, 1, 2, 0, 1, 2
    ])

    y_pred = np.array([
        0, 1, 2, 0, 1, 2
    ])

    y_proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
    ])

    result = evaluate_classification(
        y_true,
        y_pred,
        CLASS_NAMES,
        y_proba,
    )

    overall = result["overall"]

    assert "roc_auc_ovr_macro" in overall
    assert "pr_auc_macro" in overall

    assert 0.0 <= overall["roc_auc_ovr_macro"] <= 1.0
    assert 0.0 <= overall["pr_auc_macro"] <= 1.0


def test_mismatched_prediction_lengths_are_rejected():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError):
        evaluate_classification(
            y_true,
            y_pred,
            CLASS_NAMES,
        )


def test_invalid_probability_shape_is_rejected():
    y_true = np.array([
        0, 1, 2
    ])

    y_pred = np.array([
        0, 1, 2
    ])

    y_proba = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])

    with pytest.raises(ValueError):
        evaluate_classification(
            y_true,
            y_pred,
            CLASS_NAMES,
            y_proba,
        )