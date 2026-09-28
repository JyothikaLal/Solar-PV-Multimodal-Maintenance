from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class AcceptanceDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class RegressionAcceptance:
    candidate_metrics: Mapping[str, float]
    production_metrics: Mapping[str, float]


@dataclass(frozen=True)
class ClassificationAcceptance:
    candidate_metrics: Mapping[str, float]
    production_metrics: Mapping[str, float]
    candidate_minority_f1: Mapping[str, float] | None = None
    production_minority_f1: Mapping[str, float] | None = None


@dataclass(frozen=True)
class AcceptanceResult:
    decision: AcceptanceDecision
    passed: bool
    reasons: tuple[str, ...]


REGRESSION_PRIMARY = "validation_mae"

REGRESSION_GUARDRAILS = (
    "validation_rmse",
    "validation_r2",
)

CLASSIFICATION_PRIMARY = "best_validation_macro_f1"

CLASSIFICATION_GUARDRAILS = (
    "validation_balanced_accuracy",
    "validation_macro_precision",
    "validation_macro_recall",
    "validation_weighted_f1",
    "validation_roc_auc_ovr_macro",
    "validation_pr_auc_macro",
)


def evaluate_regression_candidate(
    candidate: RegressionAcceptance,
) -> AcceptanceResult:
    required = (REGRESSION_PRIMARY, *REGRESSION_GUARDRAILS)

    missing = [
        metric
        for metric in required
        if metric not in candidate.candidate_metrics
        or metric not in candidate.production_metrics
    ]

    if missing:
        return AcceptanceResult(
            decision=AcceptanceDecision.INCOMPLETE,
            passed=False,
            reasons=(f"Missing validation metrics: {sorted(set(missing))}",),
        )

    reasons: list[str] = []

    if not (
        candidate.candidate_metrics[REGRESSION_PRIMARY]
        < candidate.production_metrics[REGRESSION_PRIMARY]
    ):
        reasons.append("Candidate validation MAE did not improve.")

    if (
        candidate.candidate_metrics["validation_rmse"]
        > candidate.production_metrics["validation_rmse"]
    ):
        reasons.append("Candidate validation RMSE regressed.")

    if (
        candidate.candidate_metrics["validation_r2"]
        < candidate.production_metrics["validation_r2"]
    ):
        reasons.append("Candidate validation R² regressed.")

    if reasons:
        return AcceptanceResult(
            decision=AcceptanceDecision.REJECT,
            passed=False,
            reasons=tuple(reasons),
        )

    return AcceptanceResult(
        decision=AcceptanceDecision.ACCEPT,
        passed=True,
        reasons=("Candidate passed the TECNALIA validation gate.",),
    )


def evaluate_classification_candidate(
    candidate: ClassificationAcceptance,
) -> AcceptanceResult:
    if (
        CLASSIFICATION_PRIMARY not in candidate.candidate_metrics
        or CLASSIFICATION_PRIMARY not in candidate.production_metrics
    ):
        return AcceptanceResult(
            decision=AcceptanceDecision.INCOMPLETE,
            passed=False,
            reasons=("Missing validation Macro-F1.",),
        )

    reasons: list[str] = []

    if not (
        candidate.candidate_metrics[CLASSIFICATION_PRIMARY]
        > candidate.production_metrics[CLASSIFICATION_PRIMARY]
    ):
        reasons.append(
            "Candidate validation Macro-F1 did not improve."
        )

    for metric in CLASSIFICATION_GUARDRAILS:
        candidate_value = candidate.candidate_metrics.get(metric)
        production_value = candidate.production_metrics.get(metric)

        if candidate_value is None or production_value is None:
            continue

        if candidate_value < production_value:
            reasons.append(
                f"Candidate validation {metric} regressed."
            )

    if (
        candidate.candidate_minority_f1 is not None
        and candidate.production_minority_f1 is not None
    ):
        common_classes = set(candidate.candidate_minority_f1).intersection(
            candidate.production_minority_f1
        )

        for class_name in sorted(common_classes):
            if (
                candidate.candidate_minority_f1[class_name]
                < candidate.production_minority_f1[class_name]
            ):
                reasons.append(
                    f"Minority-class validation F1 regressed for {class_name}."
                )

    if reasons:
        return AcceptanceResult(
            decision=AcceptanceDecision.REJECT,
            passed=False,
            reasons=tuple(reasons),
        )

    return AcceptanceResult(
        decision=AcceptanceDecision.ACCEPT,
        passed=True,
        reasons=("Candidate passed the RaptorMaps validation gate.",),
    )
