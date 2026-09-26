from __future__ import annotations

from src.models.fusion.health_assessment import (
    ThermalAnomalyEvidence,
    TelemetryPerformanceEvidence,
    build_health_assessment,
    classify_telemetry_evidence,
    classify_thermal_evidence,
)


def make_telemetry(deviation: float) -> TelemetryPerformanceEvidence:
    actual = 0.80
    predicted = actual - deviation

    return TelemetryPerformanceEvidence(
        actual_normalized_pmpp=actual,
        predicted_normalized_pmpp=predicted,
        performance_deviation=deviation,
        evidence_level=classify_telemetry_evidence(deviation),
    )


def make_thermal(anomaly_evidence: float) -> ThermalAnomalyEvidence:
    return ThermalAnomalyEvidence(
        anomaly_class=(
            "Hot-Spot"
            if anomaly_evidence >= 0.50
            else "No-Anomaly"
        ),
        no_anomaly_probability=1.0 - anomaly_evidence,
        anomaly_evidence=anomaly_evidence,
        evidence_level=classify_thermal_evidence(anomaly_evidence),
    )


def test_telemetry_evidence_is_monotonic():
    priorities = [
        build_health_assessment(
            telemetry=make_telemetry(-0.010),
            thermal=make_thermal(0.20),
        ).maintenance_priority,
        build_health_assessment(
            telemetry=make_telemetry(-0.030),
            thermal=make_thermal(0.20),
        ).maintenance_priority,
        build_health_assessment(
            telemetry=make_telemetry(-0.100),
            thermal=make_thermal(0.20),
        ).maintenance_priority,
        build_health_assessment(
            telemetry=make_telemetry(-0.300),
            thermal=make_thermal(0.20),
        ).maintenance_priority,
    ]

    expected = [
        "routine",
        "monitor",
        "review",
        "review",
    ]

    assert priorities == expected


def test_thermal_evidence_escalates_review():
    routine = build_health_assessment(
        telemetry=make_telemetry(-0.010),
        thermal=make_thermal(0.20),
    )

    thermal_present = build_health_assessment(
        telemetry=make_telemetry(-0.010),
        thermal=make_thermal(0.90),
    )

    assert routine.maintenance_priority == "routine"
    assert thermal_present.maintenance_priority == "review"


def test_weak_telemetry_does_not_override_thermal_evidence():
    assessment = build_health_assessment(
        telemetry=make_telemetry(-0.030),
        thermal=make_thermal(0.90),
    )

    assert assessment.combined_evidence_state == "multimodal_evidence"
    assert assessment.maintenance_priority == "review"


def test_strong_telemetry_without_thermal_remains_review():
    assessment = build_health_assessment(
        telemetry=make_telemetry(-0.100),
        thermal=make_thermal(0.20),
    )

    assert assessment.combined_evidence_state == "telemetry_evidence"
    assert assessment.maintenance_priority == "review"


def test_strong_multimodal_evidence_escalates_priority():
    assessment = build_health_assessment(
        telemetry=make_telemetry(-0.100),
        thermal=make_thermal(0.90),
    )

    assert assessment.combined_evidence_state == "multimodal_evidence"
    assert assessment.maintenance_priority == "priority_review"


def test_extreme_telemetry_does_not_create_higher_priority_without_thermal():
    strong = build_health_assessment(
        telemetry=make_telemetry(-0.100),
        thermal=make_thermal(0.20),
    )

    extreme = build_health_assessment(
        telemetry=make_telemetry(-0.300),
        thermal=make_thermal(0.20),
    )

    assert strong.maintenance_priority == "review"
    assert extreme.maintenance_priority == "review"


def test_thermal_threshold_transition_is_monotonic():
    below = build_health_assessment(
        telemetry=make_telemetry(-0.010),
        thermal=make_thermal(0.49),
    )

    above = build_health_assessment(
        telemetry=make_telemetry(-0.010),
        thermal=make_thermal(0.51),
    )

    assert below.maintenance_priority == "routine"
    assert above.maintenance_priority == "review"


def test_all_eight_controlled_combinations_are_stable():
    expected = {
        ("none", "none"): ("no_elevated_evidence", "routine"),
        ("none", "present"): ("thermal_evidence", "review"),
        ("elevated", "none"): ("telemetry_evidence", "monitor"),
        ("elevated", "present"): ("multimodal_evidence", "review"),
        ("strong", "none"): ("telemetry_evidence", "review"),
        ("strong", "present"): (
            "multimodal_evidence",
            "priority_review",
        ),
        ("extreme", "none"): ("telemetry_evidence", "review"),
        ("extreme", "present"): (
            "multimodal_evidence",
            "priority_review",
        ),
    }

    telemetry_values = {
        "none": -0.010,
        "elevated": -0.030,
        "strong": -0.100,
        "extreme": -0.300,
    }

    thermal_values = {
        "none": 0.20,
        "present": 0.90,
    }

    for (telemetry_state, thermal_state), expected_result in expected.items():
        assessment = build_health_assessment(
            telemetry=make_telemetry(
                telemetry_values[telemetry_state]
            ),
            thermal=make_thermal(
                thermal_values[thermal_state]
            ),
        )

        assert (
            assessment.combined_evidence_state,
            assessment.maintenance_priority,
        ) == expected_result
