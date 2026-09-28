from __future__ import annotations

from src.mlops.retraining.acceptance_contract import (
    AcceptanceDecision,
    ClassificationAcceptance,
    RegressionAcceptance,
    evaluate_classification_candidate,
    evaluate_regression_candidate,
)


def regression_metrics(mae, rmse, r2):
    return {
        "validation_mae": mae,
        "validation_rmse": rmse,
        "validation_r2": r2,
    }


def classification_metrics(f1, **kwargs):
    return {
        "best_validation_macro_f1": f1,
        **kwargs,
    }


def test_regression_candidate_accepts_when_primary_and_guardrails_improve():
    result = evaluate_regression_candidate(
        RegressionAcceptance(
            candidate_metrics=regression_metrics(0.025, 0.058, 0.92),
            production_metrics=regression_metrics(0.029539, 0.061526, 0.910609),
        )
    )

    assert result.decision == AcceptanceDecision.ACCEPT
    assert result.passed


def test_regression_candidate_rejects_mae_regression():
    result = evaluate_regression_candidate(
        RegressionAcceptance(
            candidate_metrics=regression_metrics(0.031, 0.060, 0.92),
            production_metrics=regression_metrics(0.029539, 0.061526, 0.910609),
        )
    )

    assert result.decision == AcceptanceDecision.REJECT


def test_regression_candidate_rejects_rmse_regression():
    result = evaluate_regression_candidate(
        RegressionAcceptance(
            candidate_metrics=regression_metrics(0.025, 0.070, 0.92),
            production_metrics=regression_metrics(0.029539, 0.061526, 0.910609),
        )
    )

    assert result.decision == AcceptanceDecision.REJECT


def test_regression_candidate_rejects_r2_regression():
    result = evaluate_regression_candidate(
        RegressionAcceptance(
            candidate_metrics=regression_metrics(0.025, 0.058, 0.90),
            production_metrics=regression_metrics(0.029539, 0.061526, 0.910609),
        )
    )

    assert result.decision == AcceptanceDecision.REJECT


def test_regression_missing_metric_is_incomplete():
    result = evaluate_regression_candidate(
        RegressionAcceptance(
            candidate_metrics={"validation_mae": 0.02},
            production_metrics=regression_metrics(0.029539, 0.061526, 0.910609),
        )
    )

    assert result.decision == AcceptanceDecision.INCOMPLETE


def test_classification_candidate_accepts_macro_f1_improvement():
    result = evaluate_classification_candidate(
        ClassificationAcceptance(
            candidate_metrics=classification_metrics(0.68),
            production_metrics=classification_metrics(0.650738),
        )
    )

    assert result.decision == AcceptanceDecision.ACCEPT


def test_classification_candidate_rejects_macro_f1_regression():
    result = evaluate_classification_candidate(
        ClassificationAcceptance(
            candidate_metrics=classification_metrics(0.64),
            production_metrics=classification_metrics(0.650738),
        )
    )

    assert result.decision == AcceptanceDecision.REJECT


def test_classification_guardrail_regression_rejects():
    result = evaluate_classification_candidate(
        ClassificationAcceptance(
            candidate_metrics=classification_metrics(
                0.68,
                validation_balanced_accuracy=0.60,
            ),
            production_metrics=classification_metrics(
                0.650738,
                validation_balanced_accuracy=0.62,
            ),
        )
    )

    assert result.decision == AcceptanceDecision.REJECT


def test_classification_missing_primary_metric_is_incomplete():
    result = evaluate_classification_candidate(
        ClassificationAcceptance(
            candidate_metrics={},
            production_metrics=classification_metrics(0.650738),
        )
    )

    assert result.decision == AcceptanceDecision.INCOMPLETE


def test_minority_class_regression_rejects():
    result = evaluate_classification_candidate(
        ClassificationAcceptance(
            candidate_metrics=classification_metrics(0.68),
            production_metrics=classification_metrics(0.650738),
            candidate_minority_f1={"Hot-Spot": 0.20},
            production_minority_f1={"Hot-Spot": 0.25},
        )
    )

    assert result.decision == AcceptanceDecision.REJECT
