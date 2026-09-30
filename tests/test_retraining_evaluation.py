from __future__ import annotations

from pathlib import Path

from src.mlops.retraining.acceptance_contract import AcceptanceDecision
from src.mlops.retraining.evaluation import (
    evaluate_raptormaps_candidate,
    evaluate_tecnalia_candidate,
)
from src.mlops.retraining.training import (
    ClassificationTrainingResult,
    RegressionTrainingResult,
)


def test_tecnalia_evaluation_maps_training_metrics_to_acceptance_contract():
    result = RegressionTrainingResult(
        model_family="gradient_boosting_tuned",
        validation_metrics={
            "mae": 0.025,
            "rmse": 0.058,
            "r2": 0.92,
        },
        test_metrics={
            "mae": 0.031,
            "rmse": 0.065,
            "r2": 0.91,
        },
        model=object(),
        preprocessor=object(),
    )

    evaluation = evaluate_tecnalia_candidate(
        result,
        {
            "validation_mae": 0.029539,
            "validation_rmse": 0.061526,
            "validation_r2": 0.910609,
        },
    )

    assert evaluation.acceptance.decision == AcceptanceDecision.ACCEPT
    assert evaluation.candidate_validation_metrics == {
        "validation_mae": 0.025,
        "validation_rmse": 0.058,
        "validation_r2": 0.92,
    }


def test_tecnalia_evaluation_rejects_validation_regression():
    result = RegressionTrainingResult(
        model_family="gradient_boosting_tuned",
        validation_metrics={
            "mae": 0.031,
            "rmse": 0.060,
            "r2": 0.92,
        },
        test_metrics={
            "mae": 0.020,
            "rmse": 0.050,
            "r2": 0.95,
        },
        model=object(),
        preprocessor=object(),
    )

    evaluation = evaluate_tecnalia_candidate(
        result,
        {
            "validation_mae": 0.029539,
            "validation_rmse": 0.061526,
            "validation_r2": 0.910609,
        },
    )

    assert evaluation.acceptance.decision == AcceptanceDecision.REJECT


def test_raptormaps_evaluation_uses_best_validation_macro_f1():
    result = ClassificationTrainingResult(
        model_family="resnet18_finetuned",
        validation_metrics={
            "macro_f1": 0.67,
            "accuracy": 0.80,
            "balanced_accuracy": 0.63,
        },
        test_metrics={
            "macro_f1": 0.20,
            "accuracy": 0.30,
            "balanced_accuracy": 0.25,
        },
        best_validation_macro_f1=0.67,
        checkpoint_path=Path("best_model.pt"),
        history=[],
        class_names=["Cell", "No-Anomaly"],
    )

    evaluation = evaluate_raptormaps_candidate(
        result,
        {
            "best_validation_macro_f1": 0.6507379837195436,
            "validation_balanced_accuracy": 0.62,
        },
    )

    assert evaluation.acceptance.decision == AcceptanceDecision.ACCEPT
    assert (
        evaluation.candidate_validation_metrics[
            "best_validation_macro_f1"
        ]
        == 0.67
    )


def test_raptormaps_evaluation_rejects_macro_f1_regression():
    result = ClassificationTrainingResult(
        model_family="resnet18_finetuned",
        validation_metrics={
            "macro_f1": 0.64,
            "accuracy": 0.80,
            "balanced_accuracy": 0.63,
        },
        test_metrics={
            "macro_f1": 0.99,
            "accuracy": 0.99,
            "balanced_accuracy": 0.99,
        },
        best_validation_macro_f1=0.64,
        checkpoint_path=Path("best_model.pt"),
        history=[],
        class_names=["Cell", "No-Anomaly"],
    )

    evaluation = evaluate_raptormaps_candidate(
        result,
        {
            "best_validation_macro_f1": 0.6507379837195436,
        },
    )

    assert evaluation.acceptance.decision == AcceptanceDecision.REJECT
