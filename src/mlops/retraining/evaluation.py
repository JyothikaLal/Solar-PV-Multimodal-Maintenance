from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.mlops.retraining.acceptance_contract import (
    AcceptanceResult,
    ClassificationAcceptance,
    RegressionAcceptance,
    evaluate_classification_candidate,
    evaluate_regression_candidate,
)
from src.mlops.retraining.training import (
    ClassificationTrainingResult,
    RegressionTrainingResult,
)


@dataclass(frozen=True)
class CandidateEvaluation:
    model_family: str
    acceptance: AcceptanceResult
    candidate_validation_metrics: Mapping[str, float]
    production_validation_metrics: Mapping[str, float]


def evaluate_tecnalia_candidate(
    result: RegressionTrainingResult,
    production_validation_metrics: Mapping[str, float],
) -> CandidateEvaluation:
    """Evaluate a TECNALIA candidate using the existing Task 39 contract."""

    candidate_metrics = {
        "validation_mae": float(result.validation_metrics["mae"]),
        "validation_rmse": float(result.validation_metrics["rmse"]),
        "validation_r2": float(result.validation_metrics["r2"]),
    }

    production_metrics = {
        "validation_mae": float(
            production_validation_metrics["validation_mae"]
        ),
        "validation_rmse": float(
            production_validation_metrics["validation_rmse"]
        ),
        "validation_r2": float(
            production_validation_metrics["validation_r2"]
        ),
    }

    acceptance = evaluate_regression_candidate(
        RegressionAcceptance(
            candidate_metrics=candidate_metrics,
            production_metrics=production_metrics,
        )
    )

    return CandidateEvaluation(
        model_family=result.model_family,
        acceptance=acceptance,
        candidate_validation_metrics=candidate_metrics,
        production_validation_metrics=production_metrics,
    )


def evaluate_raptormaps_candidate(
    result: ClassificationTrainingResult,
    production_validation_metrics: Mapping[str, float],
) -> CandidateEvaluation:
    """Evaluate a RaptorMaps candidate using the existing Task 39 contract."""

    candidate_metrics = {
        "best_validation_macro_f1": float(
            result.best_validation_macro_f1
        ),
        "validation_balanced_accuracy": float(
            result.validation_metrics["balanced_accuracy"]
        ),
    }

    production_metrics = {
        "best_validation_macro_f1": float(
            production_validation_metrics["best_validation_macro_f1"]
        ),
    }

    if "validation_balanced_accuracy" in production_validation_metrics:
        production_metrics["validation_balanced_accuracy"] = float(
            production_validation_metrics["validation_balanced_accuracy"]
        )

    acceptance = evaluate_classification_candidate(
        ClassificationAcceptance(
            candidate_metrics=candidate_metrics,
            production_metrics=production_metrics,
        )
    )

    return CandidateEvaluation(
        model_family=result.model_family,
        acceptance=acceptance,
        candidate_validation_metrics=candidate_metrics,
        production_validation_metrics=production_metrics,
    )
