from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TelemetryEvidenceLevel = Literal[
    "none",
    "elevated",
    "strong",
    "extreme",
]

ThermalEvidenceLevel = Literal[
    "none",
    "present",
]

MaintenancePriority = Literal[
    "insufficient_evidence",
    "routine",
    "monitor",
    "review",
    "priority_review",
]


TELEMETRY_ELEVATED_THRESHOLD = -0.02075990540437562
TELEMETRY_STRONG_THRESHOLD = -0.06827571496472343
TELEMETRY_EXTREME_THRESHOLD = -0.23838252371512167

THERMAL_ANOMALY_THRESHOLD = 0.50


class TelemetryPerformanceEvidence(BaseModel):
    """
    Observation-derived TECNALIA performance evidence.

    performance_deviation is:
        actual_normalized_pmpp - predicted_normalized_pmpp

    These thresholds describe validation-distribution evidence bands.
    They are not physical degradation thresholds.
    """

    model_config = ConfigDict(extra="forbid")

    actual_normalized_pmpp: float = Field(ge=0.0)
    predicted_normalized_pmpp: float = Field(ge=0.0)
    performance_deviation: float
    evidence_level: TelemetryEvidenceLevel

    @field_validator(
        "actual_normalized_pmpp",
        "predicted_normalized_pmpp",
        "performance_deviation",
    )
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not float("-inf") < value < float("inf"):
            raise ValueError("Telemetry evidence values must be finite.")
        return value

    @model_validator(mode="after")
    def validate_deviation(self) -> "TelemetryPerformanceEvidence":
        expected_deviation = (
            self.actual_normalized_pmpp
            - self.predicted_normalized_pmpp
        )

        if abs(self.performance_deviation - expected_deviation) > 1e-6:
            raise ValueError(
                "performance_deviation must equal "
                "actual_normalized_pmpp - predicted_normalized_pmpp."
            )

        expected_level = classify_telemetry_evidence(
            self.performance_deviation
        )

        if self.evidence_level != expected_level:
            raise ValueError(
                "evidence_level does not match the validation-derived "
                "performance-deviation thresholds."
            )

        return self


class ThermalAnomalyEvidence(BaseModel):
    """
    Observation/model-output-derived RaptorMaps anomaly evidence.

    anomaly_evidence = 1 - P(No-Anomaly)

    The 0.50 threshold is a validation-supported binary anomaly
    operating point, not a physical severity threshold.
    """

    model_config = ConfigDict(extra="forbid")

    anomaly_class: str
    no_anomaly_probability: float = Field(ge=0.0, le=1.0)
    anomaly_evidence: float = Field(ge=0.0, le=1.0)
    evidence_level: ThermalEvidenceLevel

    @field_validator(
        "no_anomaly_probability",
        "anomaly_evidence",
    )
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not float("-inf") < value < float("inf"):
            raise ValueError("Thermal evidence values must be finite.")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> "ThermalAnomalyEvidence":
        expected_evidence = 1.0 - self.no_anomaly_probability

        if abs(self.anomaly_evidence - expected_evidence) > 1e-6:
            raise ValueError(
                "anomaly_evidence must equal "
                "1 - no_anomaly_probability."
            )

        expected_level: ThermalEvidenceLevel = (
            "present"
            if self.anomaly_evidence >= THERMAL_ANOMALY_THRESHOLD
            else "none"
        )

        if self.evidence_level != expected_level:
            raise ValueError(
                "evidence_level does not match the validation-supported "
                "thermal anomaly threshold."
            )

        return self


class HealthAssessment(BaseModel):
    """
    Evidence-based maintenance decision-support assessment.

    This is NOT:
      - a calibrated physical PV health percentage,
      - a failure probability,
      - a degradation percentage,
      - an RUL estimate.

    Telemetry and thermal evidence remain independently generated.
    """

    model_config = ConfigDict(extra="forbid")

    telemetry: TelemetryPerformanceEvidence | None = None
    thermal: ThermalAnomalyEvidence | None = None

    combined_evidence_state: Literal[
        "no_elevated_evidence",
        "telemetry_evidence",
        "thermal_evidence",
        "multimodal_evidence",
    ]

    maintenance_priority: MaintenancePriority

    @model_validator(mode="after")
    def validate_assessment(self) -> "HealthAssessment":
        if self.telemetry is None and self.thermal is None:
            if self.combined_evidence_state != "no_elevated_evidence":
                raise ValueError(
                    "No modality evidence requires "
                    "'no_elevated_evidence'."
                )

            if self.maintenance_priority != "insufficient_evidence":
                raise ValueError(
                    "No modality evidence requires "
                    "'insufficient_evidence' priority."
                )

            return self

        telemetry_present = (
            self.telemetry is not None
            and self.telemetry.evidence_level != "none"
        )

        thermal_present = (
            self.thermal is not None
            and self.thermal.evidence_level == "present"
        )

        if telemetry_present and thermal_present:
            expected_state = "multimodal_evidence"
        elif telemetry_present:
            expected_state = "telemetry_evidence"
        elif thermal_present:
            expected_state = "thermal_evidence"
        else:
            expected_state = "no_elevated_evidence"

        if self.combined_evidence_state != expected_state:
            raise ValueError(
                "combined_evidence_state does not match the "
                "independent modality evidence."
            )

        return self


def classify_telemetry_evidence(
    performance_deviation: float,
) -> TelemetryEvidenceLevel:
    """
    Classify TECNALIA performance deviation using frozen validation
    distribution thresholds.
    """
    if performance_deviation <= TELEMETRY_EXTREME_THRESHOLD:
        return "extreme"

    if performance_deviation <= TELEMETRY_STRONG_THRESHOLD:
        return "strong"

    if performance_deviation <= TELEMETRY_ELEVATED_THRESHOLD:
        return "elevated"

    return "none"


def classify_thermal_evidence(
    anomaly_evidence: float,
) -> ThermalEvidenceLevel:
    """Classify binary thermal anomaly evidence."""
    return (
        "present"
        if anomaly_evidence >= THERMAL_ANOMALY_THRESHOLD
        else "none"
    )


def determine_maintenance_priority(
    *,
    telemetry: TelemetryPerformanceEvidence | None,
    thermal: ThermalAnomalyEvidence | None,
) -> MaintenancePriority:
    """
    Map independently generated evidence into a maintenance-oriented
    workflow category.

    These categories are decision-support labels, not physical failure
    severity or calibrated failure probabilities.
    """

    telemetry_level = telemetry.evidence_level if telemetry else "none"
    thermal_level = thermal.evidence_level if thermal else "none"

    if telemetry is None and thermal is None:
        return "insufficient_evidence"

    if telemetry_level == "none" and thermal_level == "none":
        return "routine"

    if telemetry_level == "elevated" and thermal_level == "none":
        return "monitor"

    if telemetry_level == "none" and thermal_level == "present":
        return "review"

    if telemetry_level == "elevated" and thermal_level == "present":
        return "review"

    if telemetry_level in {"strong", "extreme"}:
        if thermal_level == "present":
            return "priority_review"

        return "review"

    return "monitor"


def build_health_assessment(
    *,
    telemetry: TelemetryPerformanceEvidence | None,
    thermal: ThermalAnomalyEvidence | None,
) -> HealthAssessment:
    """
    Build the evidence-preserving maintenance assessment.
    """

    telemetry_present = (
        telemetry is not None
        and telemetry.evidence_level != "none"
    )

    thermal_present = (
        thermal is not None
        and thermal.evidence_level == "present"
    )

    if telemetry_present and thermal_present:
        combined_state = "multimodal_evidence"
    elif telemetry_present:
        combined_state = "telemetry_evidence"
    elif thermal_present:
        combined_state = "thermal_evidence"
    else:
        combined_state = "no_elevated_evidence"

    return HealthAssessment(
        telemetry=telemetry,
        thermal=thermal,
        combined_evidence_state=combined_state,
        maintenance_priority=determine_maintenance_priority(
            telemetry=telemetry,
            thermal=thermal,
        ),
    )
