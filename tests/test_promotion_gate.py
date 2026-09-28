from __future__ import annotations

import pytest

from src.mlops.model_registry import (
    LIFECYCLE_CANDIDATE,
    LIFECYCLE_PRODUCTION,
    RAPTORMAPS_CLASS_NAMES,
    RAPTORMAPS_SPATIAL_SIZE,
)
from src.mlops.retraining.acceptance_contract import AcceptanceDecision
from src.mlops.retraining.promotion_gate import (
    PromotionDecision,
    PromotionRequest,
    evaluate_promotion_gate,
)


def valid_metadata() -> dict[str, str]:
    import json

    return {
        "dataset": "RaptorMaps",
        "modality": "thermal",
        "task": "classification",
        "class_count": "12",
        "class_names": json.dumps(RAPTORMAPS_CLASS_NAMES),
        "spatial_size": json.dumps(RAPTORMAPS_SPATIAL_SIZE),
    }


def request(
    *,
    stage=LIFECYCLE_CANDIDATE,
    decision=AcceptanceDecision.ACCEPT,
    metadata=None,
):
    return PromotionRequest(
        registered_model_name="SolarPV_RaptorMaps_ResNet18",
        candidate_version="2",
        candidate_metadata=metadata or valid_metadata(),
        acceptance_decision=decision,
        current_lifecycle_stage=stage,
    )


def test_accepted_compatible_candidate_is_eligible():
    result = evaluate_promotion_gate(request())

    assert result.decision == PromotionDecision.ELIGIBLE
    assert result.reasons


def test_rejected_candidate_is_blocked():
    result = evaluate_promotion_gate(
        request(decision=AcceptanceDecision.REJECT)
    )

    assert result.decision == PromotionDecision.BLOCKED


def test_incomplete_candidate_is_blocked():
    result = evaluate_promotion_gate(
        request(decision=AcceptanceDecision.INCOMPLETE)
    )

    assert result.decision == PromotionDecision.BLOCKED


def test_production_version_cannot_enter_promotion_gate():
    result = evaluate_promotion_gate(
        request(stage=LIFECYCLE_PRODUCTION)
    )

    assert result.decision == PromotionDecision.BLOCKED


def test_dataset_mismatch_is_blocked():
    metadata = valid_metadata()
    metadata["dataset"] = "WrongDataset"

    result = evaluate_promotion_gate(
        request(metadata=metadata)
    )

    assert result.decision == PromotionDecision.BLOCKED
    assert any(
        "compatibility" in reason.lower()
        for reason in result.reasons
    )


def test_modality_mismatch_is_blocked():
    metadata = valid_metadata()
    metadata["modality"] = "telemetry"

    result = evaluate_promotion_gate(
        request(metadata=metadata)
    )

    assert result.decision == PromotionDecision.BLOCKED


def test_class_count_mismatch_is_blocked():
    metadata = valid_metadata()
    metadata["class_count"] = "11"

    result = evaluate_promotion_gate(
        request(metadata=metadata)
    )

    assert result.decision == PromotionDecision.BLOCKED


def test_class_contract_mismatch_is_blocked():
    import json

    metadata = valid_metadata()
    metadata["class_names"] = json.dumps(
        RAPTORMAPS_CLASS_NAMES[:-1]
    )

    result = evaluate_promotion_gate(
        request(metadata=metadata)
    )

    assert result.decision == PromotionDecision.BLOCKED


def test_spatial_contract_mismatch_is_blocked():
    import json

    metadata = valid_metadata()
    metadata["spatial_size"] = json.dumps([24, 40])

    result = evaluate_promotion_gate(
        request(metadata=metadata)
    )

    assert result.decision == PromotionDecision.BLOCKED


def test_eligibility_does_not_claim_automatic_promotion():
    result = evaluate_promotion_gate(request())

    assert result.decision == PromotionDecision.ELIGIBLE
    assert any(
        "explicit" in reason.lower()
        for reason in result.reasons
    )
