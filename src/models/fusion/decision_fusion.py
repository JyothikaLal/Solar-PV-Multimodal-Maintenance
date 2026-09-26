from __future__ import annotations

from src.models.fusion.contracts import FusionInput, FusionOutput


def fuse_decision_outputs(fusion_input: FusionInput) -> FusionOutput:
    """
    Combine independently generated telemetry and thermal outputs
    at the decision level.

    This function intentionally performs evidence-preserving fusion.
    It does not create a health score, maintenance priority, shared
    training label, or cross-modal probability.
    """

    thermal_anomaly_detected = (
        fusion_input.thermal.anomaly_class != "No-Anomaly"
    )

    evidence_state = (
        "thermal_anomaly_detected"
        if thermal_anomaly_detected
        else "no_thermal_anomaly"
    )

    return FusionOutput(
        telemetry=fusion_input.telemetry,
        thermal=fusion_input.thermal,
        thermal_anomaly_detected=thermal_anomaly_detected,
        evidence_state=evidence_state,
        embedding_indicators=fusion_input.embedding_indicators,
    )
