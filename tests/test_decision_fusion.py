from __future__ import annotations

from src.models.fusion.contracts import (
    FusionInput,
    ModelProvenance,
    RAPTORMAPS_CLASS_NAMES,
    TelemetryPrediction,
    ThermalPrediction,
)
from src.models.fusion.decision_fusion import fuse_decision_outputs


def make_provenance() -> ModelProvenance:
    return ModelProvenance(
        model_name="test_model",
        model_version="test",
        checkpoint_path="test/checkpoint.pt",
        checkpoint_epoch=10,
        validation_metric_name="macro_f1",
        validation_metric_value=0.65,
    )


def make_telemetry() -> TelemetryPrediction:
    return TelemetryPrediction(
        predicted_normalized_pmpp=0.82,
        provenance=make_provenance(),
    )


def make_thermal(
    anomaly_class: str,
    confidence: float,
) -> ThermalPrediction:
    probabilities = {
        class_name: 0.0
        for class_name in RAPTORMAPS_CLASS_NAMES
    }
    probabilities[anomaly_class] = confidence

    return ThermalPrediction(
        anomaly_class=anomaly_class,
        anomaly_confidence=confidence,
        class_probabilities=probabilities,
        provenance=make_provenance(),
    )


def test_no_anomaly_produces_no_thermal_anomaly_state() -> None:
    fusion_input = FusionInput(
        telemetry=make_telemetry(),
        thermal=make_thermal("No-Anomaly", 1.0),
    )

    result = fuse_decision_outputs(fusion_input)

    assert result.thermal_anomaly_detected is False
    assert result.evidence_state == "no_thermal_anomaly"


def test_anomaly_produces_thermal_anomaly_state() -> None:
    fusion_input = FusionInput(
        telemetry=make_telemetry(),
        thermal=make_thermal("Hot-Spot", 1.0),
    )

    result = fuse_decision_outputs(fusion_input)

    assert result.thermal_anomaly_detected is True
    assert result.evidence_state == "thermal_anomaly_detected"


def test_fusion_preserves_independent_branch_outputs() -> None:
    telemetry = make_telemetry()
    thermal = make_thermal("Soiling", 1.0)

    fusion_input = FusionInput(
        telemetry=telemetry,
        thermal=thermal,
        embedding_indicators={
            "embedding_distance": 0.42,
        },
    )

    result = fuse_decision_outputs(fusion_input)

    assert result.telemetry == telemetry
    assert result.thermal == thermal
    assert result.embedding_indicators == {
        "embedding_distance": 0.42,
    }


def test_fusion_does_not_create_pairing_identifiers() -> None:
    fusion_input = FusionInput(
        telemetry=make_telemetry(),
        thermal=make_thermal("Cracking", 1.0),
    )

    result = fuse_decision_outputs(fusion_input)

    assert not hasattr(result, "asset_id")
    assert not hasattr(result, "image_id")
    assert not hasattr(result, "timestamp")
    assert not hasattr(result, "module_id")


def test_fusion_does_not_create_health_or_priority_fields() -> None:
    fusion_input = FusionInput(
        telemetry=make_telemetry(),
        thermal=make_thermal("Diode", 1.0),
    )

    result = fuse_decision_outputs(fusion_input)

    assert not hasattr(result, "health_score")
    assert not hasattr(result, "maintenance_priority")
