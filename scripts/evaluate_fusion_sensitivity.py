from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


TELEMETRY_TEST_VALUES = {
    "none": -0.010,
    "elevated": -0.030,
    "strong": -0.100,
    "extreme": -0.300,
}

THERMAL_TEST_VALUES = {
    "none": 0.20,
    "present": 0.90,
}


def make_telemetry(
    evidence_level: str,
) -> TelemetryPerformanceEvidence:
    deviation = TELEMETRY_TEST_VALUES[evidence_level]

    actual = 0.80
    predicted = actual - deviation

    return TelemetryPerformanceEvidence(
        actual_normalized_pmpp=actual,
        predicted_normalized_pmpp=predicted,
        performance_deviation=deviation,
        evidence_level=classify_telemetry_evidence(deviation),
    )


def make_thermal(
    evidence_level: str,
) -> ThermalAnomalyEvidence:
    anomaly_evidence = THERMAL_TEST_VALUES[evidence_level]

    return ThermalAnomalyEvidence(
        anomaly_class=(
            "Hot-Spot"
            if evidence_level == "present"
            else "No-Anomaly"
        ),
        no_anomaly_probability=1.0 - anomaly_evidence,
        anomaly_evidence=anomaly_evidence,
        evidence_level=classify_thermal_evidence(
            anomaly_evidence
        ),
    )


print("===== TASK 34 — CONTROLLED FUSION STATE MATRIX =====")

telemetry_states = ["none", "elevated", "strong", "extreme"]
thermal_states = ["none", "present"]

results = []

for telemetry_state, thermal_state in product(
    telemetry_states,
    thermal_states,
):
    assessment = build_health_assessment(
        telemetry=make_telemetry(telemetry_state),
        thermal=make_thermal(thermal_state),
    )

    results.append(
        {
            "telemetry": telemetry_state,
            "thermal": thermal_state,
            "combined_state": assessment.combined_evidence_state,
            "priority": assessment.maintenance_priority,
        }
    )

for result in results:
    print(
        f"telemetry={result['telemetry']:>8} | "
        f"thermal={result['thermal']:>7} | "
        f"state={result['combined_state']:<22} | "
        f"priority={result['priority']}"
    )


print("\n===== TELEMETRY MONOTONICITY =====")

telemetry_sequence = [
    "none",
    "elevated",
    "strong",
    "extreme",
]

previous = None

for state in telemetry_sequence:
    assessment = build_health_assessment(
        telemetry=make_telemetry(state),
        thermal=make_thermal("none"),
    )

    print(
        f"{state:>8} -> "
        f"{assessment.maintenance_priority}"
    )

    if previous is not None:
        print(
            f"         transition from "
            f"{previous} -> "
            f"{assessment.maintenance_priority}"
        )

    previous = assessment.maintenance_priority


print("\n===== THERMAL MONOTONICITY =====")

for state in thermal_states:
    assessment = build_health_assessment(
        telemetry=make_telemetry("none"),
        thermal=make_thermal(state),
    )

    print(
        f"{state:>7} -> "
        f"{assessment.maintenance_priority}"
    )


print("\n===== CROSS-MODAL DOMINANCE CHECKS =====")

checks = [
    (
        "weak telemetry + thermal",
        "elevated",
        "present",
    ),
    (
        "strong telemetry + no thermal",
        "strong",
        "none",
    ),
    (
        "none telemetry + thermal",
        "none",
        "present",
    ),
    (
        "strong telemetry + thermal",
        "strong",
        "present",
    ),
]

for name, telemetry_state, thermal_state in checks:
    assessment = build_health_assessment(
        telemetry=make_telemetry(telemetry_state),
        thermal=make_thermal(thermal_state),
    )

    print(
        f"{name:<34} -> "
        f"{assessment.maintenance_priority}"
    )


print("\n===== THRESHOLD CONSTANTS =====")
print(
    f"Telemetry elevated: "
    f"{TELEMETRY_ELEVATED_THRESHOLD}"
)
print(
    f"Telemetry strong:   "
    f"{TELEMETRY_STRONG_THRESHOLD}"
)
print(
    f"Telemetry extreme:  "
    f"{TELEMETRY_EXTREME_THRESHOLD}"
)
print(
    f"Thermal threshold:  "
    f"{THERMAL_ANOMALY_THRESHOLD}"
)

print("\nTask 34 controlled sensitivity evaluation completed.")
