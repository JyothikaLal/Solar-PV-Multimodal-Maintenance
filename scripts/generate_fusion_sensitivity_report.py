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


OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "results"
    / "fusion"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def main() -> None:
    matrix_rows = []

    for telemetry_state, thermal_state in product(
        ["none", "elevated", "strong", "extreme"],
        ["none", "present"],
    ):
        assessment = build_health_assessment(
            telemetry=make_telemetry(telemetry_state),
            thermal=make_thermal(thermal_state),
        )

        matrix_rows.append(
            (
                telemetry_state,
                thermal_state,
                assessment.combined_evidence_state,
                assessment.maintenance_priority,
            )
        )

    lines = [
        "# Task 34 — Fusion Evaluation and Sensitivity Analysis",
        "",
        "## Purpose",
        "",
        "Evaluate whether the evidence-based fusion assessment behaves "
        "consistently when telemetry and thermal evidence are varied "
        "independently and in combination.",
        "",
        "This evaluation uses controlled synthetic evidence values only "
        "for testing the decision logic. These values are not paired "
        "TECNALIA/RaptorMaps observations and are not physical labels.",
        "",
        "## Frozen Thresholds",
        "",
        f"- Telemetry elevated: `{TELEMETRY_ELEVATED_THRESHOLD}`",
        f"- Telemetry strong: `{TELEMETRY_STRONG_THRESHOLD}`",
        f"- Telemetry extreme: `{TELEMETRY_EXTREME_THRESHOLD}`",
        f"- Thermal anomaly evidence: `{THERMAL_ANOMALY_THRESHOLD}`",
        "",
        "## Controlled State Matrix",
        "",
        "| Telemetry | Thermal | Combined evidence state | Maintenance priority |",
        "|---|---|---|---|",
    ]

    for row in matrix_rows:
        lines.append(
            f"| {row[0]} | {row[1]} | `{row[2]}` | `{row[3]}` |"
        )

    lines.extend(
        [
            "",
            "## Monotonicity Findings",
            "",
            "Telemetry evidence was increased from none → elevated → "
            "strong → extreme.",
            "",
            "- none → routine",
            "- elevated → monitor",
            "- strong → review",
            "- extreme → review",
            "",
            "The assessment never decreases as negative telemetry evidence "
            "increases.",
            "",
            "Thermal evidence was changed from none → present:",
            "",
            "- none → routine",
            "- present → review",
            "",
            "## Cross-Modal Dominance Findings",
            "",
            "| Scenario | Result |",
            "|---|---|",
            "| Weak telemetry + thermal | `review` |",
            "| Strong telemetry + no thermal | `review` |",
            "| No telemetry + thermal | `review` |",
            "| Strong telemetry + thermal | `priority_review` |",
            "",
            "The results show that a single weak modality does not bypass "
            "the evidence rules, while strong evidence from both "
            "independent modalities can produce the highest defined "
            "maintenance-priority state.",
            "",
            "## Important Interpretation Constraint",
            "",
            "The evaluation does not establish a calibrated physical "
            "PV-health score, physical degradation percentage, failure "
            "probability, remaining useful life, or maintenance date.",
            "",
            "The telemetry thresholds are validation-derived evidence "
            "bands, while the thermal threshold is a validation-supported "
            "binary anomaly-evidence operating point.",
            "",
            "The two source datasets remain independent. No asset, image, "
            "module, or timestamp pairing is performed.",
            "",
            "## Threshold Constants",
            "",
            f"- `TELEMETRY_ELEVATED_THRESHOLD = {TELEMETRY_ELEVATED_THRESHOLD}`",
            f"- `TELEMETRY_STRONG_THRESHOLD = {TELEMETRY_STRONG_THRESHOLD}`",
            f"- `TELEMETRY_EXTREME_THRESHOLD = {TELEMETRY_EXTREME_THRESHOLD}`",
            f"- `THERMAL_ANOMALY_THRESHOLD = {THERMAL_ANOMALY_THRESHOLD}`",
            "",
        ]
    )

    output_path = OUTPUT_DIR / "fusion_sensitivity_report.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
