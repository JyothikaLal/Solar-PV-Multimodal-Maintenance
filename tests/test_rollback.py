from __future__ import annotations

from src.mlops.retraining.rollback import (
    RollbackDecision,
    RollbackReason,
    RollbackRequest,
    evaluate_rollback_request,
)


def valid_request() -> RollbackRequest:
    return RollbackRequest(
        registered_model_name="SolarPV_RaptorMaps_ResNet18",
        current_production_version="2",
        rollback_target_version="1",
        rollback_target_metadata={
            "dataset": "RaptorMaps",
            "modality": "thermal",
            "task": "classification",
        },
        reason=RollbackReason.PRODUCTION_REGRESSION,
    )


def test_valid_rollback_is_eligible():
    result = evaluate_rollback_request(valid_request())

    assert result.decision == RollbackDecision.ELIGIBLE


def test_missing_current_production_is_blocked():
    request = valid_request()
    request = RollbackRequest(
        **{
            **request.__dict__,
            "current_production_version": "",
        }
    )

    result = evaluate_rollback_request(request)

    assert result.decision == RollbackDecision.BLOCKED


def test_missing_target_is_blocked():
    request = valid_request()
    request = RollbackRequest(
        **{
            **request.__dict__,
            "rollback_target_version": None,
        }
    )

    result = evaluate_rollback_request(request)

    assert result.decision == RollbackDecision.BLOCKED


def test_same_version_target_is_blocked():
    request = valid_request()
    request = RollbackRequest(
        **{
            **request.__dict__,
            "rollback_target_version": "2",
        }
    )

    result = evaluate_rollback_request(request)

    assert result.decision == RollbackDecision.BLOCKED


def test_missing_target_metadata_is_blocked():
    request = valid_request()
    request = RollbackRequest(
        **{
            **request.__dict__,
            "rollback_target_metadata": None,
        }
    )

    result = evaluate_rollback_request(request)

    assert result.decision == RollbackDecision.BLOCKED


def test_data_quality_failure_can_trigger_rollback():
    request = RollbackRequest(
        **{
            **valid_request().__dict__,
            "reason": RollbackReason.DATA_QUALITY_FAILURE,
        }
    )

    result = evaluate_rollback_request(request)

    assert result.decision == RollbackDecision.ELIGIBLE


def test_contract_violation_can_trigger_rollback():
    request = RollbackRequest(
        **{
            **valid_request().__dict__,
            "reason": RollbackReason.MODEL_CONTRACT_VIOLATION,
        }
    )

    result = evaluate_rollback_request(request)

    assert result.decision == RollbackDecision.ELIGIBLE


def test_regression_reason_can_trigger_rollback():
    request = RollbackRequest(
        **{
            **valid_request().__dict__,
            "reason": RollbackReason.PRODUCTION_REGRESSION,
        }
    )

    result = evaluate_rollback_request(request)

    assert result.decision == RollbackDecision.ELIGIBLE


def test_rollback_does_not_auto_promote_candidate():
    result = evaluate_rollback_request(valid_request())

    assert result.decision == RollbackDecision.ELIGIBLE
    assert any(
        "explicit" in reason.lower()
        for reason in result.reasons
    )
