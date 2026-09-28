from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class RollbackDecision(str, Enum):
    ELIGIBLE = "rollback_eligible"
    BLOCKED = "rollback_blocked"


class RollbackReason(str, Enum):
    PRODUCTION_REGRESSION = "production_regression"
    DATA_QUALITY_FAILURE = "data_quality_failure"
    MODEL_CONTRACT_VIOLATION = "model_contract_violation"


@dataclass(frozen=True)
class RollbackRequest:
    registered_model_name: str
    current_production_version: str
    rollback_target_version: str | None
    rollback_target_metadata: Mapping[str, str] | None
    reason: RollbackReason


@dataclass(frozen=True)
class RollbackResult:
    decision: RollbackDecision
    reasons: tuple[str, ...]


def evaluate_rollback_request(
    request: RollbackRequest,
) -> RollbackResult:
    reasons: list[str] = []

    if not request.current_production_version:
        reasons.append(
            "A current production version is required."
        )

    if not request.rollback_target_version:
        reasons.append(
            "A previous production rollback target is required."
        )

    if (
        request.rollback_target_version
        == request.current_production_version
    ):
        reasons.append(
            "Rollback target cannot equal the current production version."
        )

    if request.rollback_target_metadata is None:
        reasons.append(
            "Rollback target metadata is required."
        )

    if reasons:
        return RollbackResult(
            decision=RollbackDecision.BLOCKED,
            reasons=tuple(reasons),
        )

    return RollbackResult(
        decision=RollbackDecision.ELIGIBLE,
        reasons=(
            f"Rollback target {request.rollback_target_version} "
            "is eligible for explicit rollback.",
        ),
    )
