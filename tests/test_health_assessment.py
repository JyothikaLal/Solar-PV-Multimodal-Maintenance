from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.fusion.health_assessment import (
    TELEMETRY_ELEVATED_THRESHOLD,
    TELEMETRY_EXTREME_THRESHOLD,
    TELEMETRY_STRONG_THRESHOLD,
    THERMAL_ANOMALY_THRESHOLD,
    ThermalAnomalyEvidence,
    TelemetryPerformanceEvidence,
    build_health_assessment,
    classify_telemetry_evidence,
    classify_thermal_evidence,
)


def make_telemetry(
    deviation: float,
) -> TelemetryPerformanceEvidence:
    actual = 0.80
    predicted = actual - deviation

    return TelemetryPerformanceEvidence(
        actual_normalized_pmpp=actual,
        predicted_normalized_pmpp=predicted,
        performance_deviation=deviation,
        evidence_level=classify_telemetry_evidence(deviation),
    )


def make_thermal(
    anomaly_evidence: float,
) -> ThermalAnomalyEvidence:
    no_anomaly_probability = 1.0 - anomaly_evidence

    return ThermalAnomalyEvidence(
        anomaly_class=(
            "Hot-Spot"
            if anomaly_evidence >= THERMAL_ANOMALY_THRESHOLD
            else "No-Anomaly"
        ),
        no_anomaly_probability=no_anomaly_probability,
        anomaly_evidence=anomaly_evidence,
        evidence_level=classify_thermal_evidence(anomaly_evidence),
    )


def test_telemetry_threshold_boundaries() -> None:
    assert (
        classify_telemetry_evidence(TELEMETRY_ELEVATED_THRESHOLD)
        == "elevated"
    )
    assert (
        classify_telemetry_evidence(TELEMETRY_STRONG_THRESHOLD)
        == "strong"
    )
    assert (
        classify_telemetry_evidence(TELEMETRY_EXTREME_THRESHOLD)
        == "extreme"
    )


def test_telemetry_above_elevated_threshold_is_none() -> None:
    assert classify_telemetry_evidence(-0.010) == "none"


def test_telemetry_evidence_requires_correct_deviation() -> None:
    with pytest.raises(
        ValidationError,
        match="performance_deviation must equal",
    ):
        TelemetryPerformanceEvidence(
            actual_normalized_pmpp=0.80,
            predicted_normalized_pmpp=0.90,
            performance_deviation=-0.05,
            evidence_level="strong",
        )


def test_thermal_threshold_boundary() -> None:
    assert (
        classify_thermal_evidence(THERMAL_ANOMALY_THRESHOLD)
        == "present"
    )
    assert classify_thermal_evidence(0.499999) == "none"


def test_thermal_evidence_is_one_minus_no_anomaly_probability() -> None:
    with pytest.raises(
        ValidationError,
        match="anomaly_evidence must equal",
    ):
        ThermalAnomalyEvidence(
            anomaly_class="Hot-Spot",
            no_anomaly_probability=0.20,
            anomaly_evidence=0.70,
            evidence_level="present",
        )


def test_no_evidence_is_insufficient() -> None:
    result = build_health_assessment(
        telemetry=None,
        thermal=None,
    )

    assert result.combined_evidence_state == "no_elevated_evidence"
    assert result.maintenance_priority == "insufficient_evidence"


def test_no_elevated_evidence_is_routine() -> None:
    result = build_health_assessment(
        telemetry=make_telemetry(-0.010),
        thermal=make_thermal(0.20),
    )

    assert result.combined_evidence_state == "no_elevated_evidence"
    assert result.maintenance_priority == "routine"


def test_telemetry_evidence_produces_monitor() -> None:
    result = build_health_assessment(
        telemetry=make_telemetry(-0.03),
        thermal=make_thermal(0.20),
    )

    assert result.combined_evidence_state == "telemetry_evidence"
    assert result.maintenance_priority == "monitor"


def test_thermal_evidence_produces_review() -> None:
    result = build_health_assessment(
        telemetry=make_telemetry(-0.010),
        thermal=make_thermal(0.50),
    )

    assert result.combined_evidence_state == "thermal_evidence"
    assert result.maintenance_priority == "review"


def test_elevated_telemetry_plus_thermal_is_review() -> None:
    result = build_health_assessment(
        telemetry=make_telemetry(-0.03),
        thermal=make_thermal(0.80),
    )

    assert result.combined_evidence_state == "multimodal_evidence"
    assert result.maintenance_priority == "review"


def test_strong_telemetry_without_thermal_is_review() -> None:
    result = build_health_assessment(
        telemetry=make_telemetry(-0.10),
        thermal=None,
    )

    assert result.combined_evidence_state == "telemetry_evidence"
    assert result.maintenance_priority == "review"


def test_strong_telemetry_plus_thermal_is_priority_review() -> None:
    result = build_health_assessment(
        telemetry=make_telemetry(-0.10),
        thermal=make_thermal(0.90),
    )

    assert result.combined_evidence_state == "multimodal_evidence"
    assert result.maintenance_priority == "priority_review"


def test_extreme_telemetry_plus_thermal_is_priority_review() -> None:
    result = build_health_assessment(
        telemetry=make_telemetry(-0.30),
        thermal=make_thermal(0.99),
    )

    assert result.maintenance_priority == "priority_review"


def test_health_assessment_rejects_inconsistent_combined_state() -> None:
    with pytest.raises(
        ValidationError,
        match="combined_evidence_state",
    ):
        from src.models.fusion.health_assessment import HealthAssessment

        HealthAssessment(
            telemetry=make_telemetry(-0.10),
            thermal=make_thermal(0.90),
            combined_evidence_state="telemetry_evidence",
            maintenance_priority="priority_review",
        )


def test_health_assessment_has_no_pairing_identifiers() -> None:
    result = build_health_assessment(
        telemetry=make_telemetry(-0.10),
        thermal=make_thermal(0.90),
    )

    assert not hasattr(result, "asset_id")
    assert not hasattr(result, "module_id")
    assert not hasattr(result, "image_id")
    assert not hasattr(result, "timestamp")


def test_extra_health_fields_are_rejected() -> None:
    from src.models.fusion.health_assessment import HealthAssessment

    with pytest.raises(ValidationError):
        HealthAssessment(
            telemetry=None,
            thermal=None,
            combined_evidence_state="no_elevated_evidence",
            maintenance_priority="insufficient_evidence",
            health_score=75,
        )
