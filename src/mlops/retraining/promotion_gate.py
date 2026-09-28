from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from src.mlops.model_registry import (
    LIFECYCLE_CANDIDATE,
    LIFECYCLE_PRODUCTION,
    validate_model_compatibility,
)
from src.mlops.retraining.acceptance_contract import (
    AcceptanceDecision,
)


class PromotionDecision(str, Enum):
    ELIGIBLE = "eligible"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class PromotionGateResult:
    decision: PromotionDecision
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PromotionRequest:
    registered_model_name: str
    candidate_version: str
    candidate_metadata: Mapping[str, str]
    acceptance_decision: AcceptanceDecision
    current_lifecycle_stage: str


def evaluate_promotion_gate(
    request: PromotionRequest,
) -> PromotionGateResult:
    reasons: list[str] = []

    if request.current_lifecycle_stage != LIFECYCLE_CANDIDATE:
        reasons.append(
            "Only a candidate model version may enter the promotion gate."
        )

    if request.acceptance_decision != AcceptanceDecision.ACCEPT:
        reasons.append(
            "Candidate-vs-production acceptance gate did not return ACCEPT."
        )

    try:
        validate_model_compatibility(dict(request.candidate_metadata))
    except (KeyError, ValueError) as exc:
        reasons.append(f"Model compatibility validation failed: {exc}")

    if reasons:
        return PromotionGateResult(
            decision=PromotionDecision.BLOCKED,
            reasons=tuple(reasons),
        )

    return PromotionGateResult(
        decision=PromotionDecision.ELIGIBLE,
        reasons=(
            "Candidate passed compatibility and acceptance gates.",
            "Explicit registry promotion is still required.",
        ),
    )
