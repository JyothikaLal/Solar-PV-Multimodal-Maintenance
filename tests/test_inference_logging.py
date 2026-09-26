from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models.fusion.contracts import (
    FusionInput,
    ModelProvenance,
    RAPTORMAPS_CLASS_NAMES,
    TelemetryPrediction,
    ThermalPrediction,
)
from src.models.fusion.decision_fusion import fuse_decision_outputs
from src.models.fusion.health_assessment import (
    ThermalAnomalyEvidence,
    TelemetryPerformanceEvidence,
    build_health_assessment,
    classify_telemetry_evidence,
    classify_thermal_evidence,
)
from src.mlops.inference_logging import InferenceLogStore


def make_telemetry() -> TelemetryPrediction:
    return TelemetryPrediction(
        predicted_normalized_pmpp=0.82,
        provenance=ModelProvenance(
            model_name="tecnalia_gradient_boosting",
            model_version="candidate-v1",
            checkpoint_path="models/tecnalia/final_model.joblib",
            checkpoint_epoch=None,
            validation_metric_name="mae",
            validation_metric_value=0.029539,
        ),
    )


def make_thermal() -> ThermalPrediction:
    probabilities = {
        class_name: 0.0
        for class_name in RAPTORMAPS_CLASS_NAMES
    }

    probabilities["Hot-Spot"] = 0.80
    probabilities["No-Anomaly"] = 0.20

    return ThermalPrediction(
        anomaly_class="Hot-Spot",
        anomaly_confidence=0.80,
        class_probabilities=probabilities,
        provenance=ModelProvenance(
            model_name="SolarPV_RaptorMaps_ResNet18",
            model_version="1",
            checkpoint_path=(
                "models/raptormaps/resnet18_finetune/"
                "best_model.pt"
            ),
            checkpoint_epoch=14,
            validation_metric_name="macro_f1",
            validation_metric_value=0.6507379837195436,
        ),
    )


def make_health():
    telemetry_deviation = -0.10
    anomaly_evidence = 0.80

    telemetry = TelemetryPerformanceEvidence(
        actual_normalized_pmpp=0.72,
        predicted_normalized_pmpp=0.82,
        performance_deviation=telemetry_deviation,
        evidence_level=classify_telemetry_evidence(
            telemetry_deviation
        ),
    )

    thermal = ThermalAnomalyEvidence(
        anomaly_class="Hot-Spot",
        no_anomaly_probability=0.20,
        anomaly_evidence=anomaly_evidence,
        evidence_level=classify_thermal_evidence(
            anomaly_evidence
        ),
    )

    return build_health_assessment(
        telemetry=telemetry,
        thermal=thermal,
    )


def make_fusion():
    telemetry = make_telemetry()
    thermal = make_thermal()

    return fuse_decision_outputs(
        FusionInput(
            telemetry=telemetry,
            thermal=thermal,
            embedding_indicators={
                "embedding_distance": 0.42,
            },
        )
    )


def test_inference_record_is_persisted_and_retrieved(tmp_path):
    database_path = tmp_path / "inference_logs.db"

    store = InferenceLogStore(database_path)

    telemetry = make_telemetry()
    thermal = make_thermal()
    fusion = make_fusion()
    health = make_health()

    timestamp = datetime(
        2026,
        9,
        26,
        14,
        30,
        0,
        tzinfo=timezone.utc,
    )

    inference_id = store.log_inference(
        telemetry=telemetry,
        thermal=thermal,
        fusion=fusion,
        health=health,
        prediction_timestamp=timestamp,
        inference_id="test-inference-001",
    )

    assert inference_id == "test-inference-001"
    assert store.count() == 1

    record = store.get_inference(
        "test-inference-001"
    )

    assert record is not None

    assert record["prediction_timestamp"] == (
        "2026-09-26T14:30:00+00:00"
    )

    assert record["telemetry_model_name"] == (
        "tecnalia_gradient_boosting"
    )
    assert record["telemetry_model_version"] == "candidate-v1"

    assert record["thermal_model_name"] == (
        "SolarPV_RaptorMaps_ResNet18"
    )
    assert record["thermal_model_version"] == "1"

    assert (
        record["telemetry_predicted_normalized_pmpp"]
        == 0.82
    )

    assert record["thermal_anomaly_class"] == "Hot-Spot"
    assert record["thermal_anomaly_confidence"] == 0.80

    assert (
        record["thermal_class_probabilities"]["Hot-Spot"]
        == 0.80
    )

    assert (
        record["thermal_class_probabilities"]["No-Anomaly"]
        == 0.20
    )

    assert (
        len(record["thermal_class_probabilities"])
        == 12
    )

    assert record["thermal_anomaly_detected"] is True
    assert (
        record["thermal_evidence_state"]
        == "thermal_anomaly_detected"
    )

    assert record["embedding_indicators"] == {
        "embedding_distance": 0.42,
    }

    assert (
        record["health_telemetry_evidence_level"]
        == "strong"
    )
    assert (
        record["health_thermal_evidence_level"]
        == "present"
    )

    assert (
        record["combined_evidence_state"]
        == "multimodal_evidence"
    )

    assert (
        record["maintenance_priority"]
        == "priority_review"
    )


def test_multiple_inference_records_are_independent(tmp_path):
    store = InferenceLogStore(
        tmp_path / "inference_logs.db"
    )

    telemetry = make_telemetry()
    thermal = make_thermal()
    fusion = make_fusion()
    health = make_health()

    first = store.log_inference(
        telemetry=telemetry,
        thermal=thermal,
        fusion=fusion,
        health=health,
        inference_id="first",
    )

    second = store.log_inference(
        telemetry=telemetry,
        thermal=thermal,
        fusion=fusion,
        health=health,
        inference_id="second",
    )

    assert first == "first"
    assert second == "second"
    assert store.count() == 2

    assert store.get_inference("first") is not None
    assert store.get_inference("second") is not None


def test_naive_prediction_timestamp_is_rejected(tmp_path):
    store = InferenceLogStore(
        tmp_path / "inference_logs.db"
    )

    naive_timestamp = datetime(
        2026,
        9,
        26,
        14,
        30,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        store.log_inference(
            telemetry=make_telemetry(),
            thermal=make_thermal(),
            fusion=make_fusion(),
            health=make_health(),
            prediction_timestamp=naive_timestamp,
        )


def test_mismatched_telemetry_output_is_rejected(tmp_path):
    store = InferenceLogStore(
        tmp_path / "inference_logs.db"
    )

    telemetry = make_telemetry()
    thermal = make_thermal()
    fusion = make_fusion()
    health = make_health()

    mismatched = TelemetryPrediction(
        predicted_normalized_pmpp=0.50,
        provenance=telemetry.provenance,
    )

    with pytest.raises(
        ValueError,
        match="Fusion telemetry output does not match",
    ):
        store.log_inference(
            telemetry=mismatched,
            thermal=thermal,
            fusion=fusion,
            health=health,
        )


def test_mismatched_thermal_output_is_rejected(tmp_path):
    store = InferenceLogStore(
        tmp_path / "inference_logs.db"
    )

    telemetry = make_telemetry()
    fusion = make_fusion()
    health = make_health()

    probabilities = {
        class_name: 0.0
        for class_name in RAPTORMAPS_CLASS_NAMES
    }
    probabilities["Soiling"] = 1.0

    mismatched = ThermalPrediction(
        anomaly_class="Soiling",
        anomaly_confidence=1.0,
        class_probabilities=probabilities,
        provenance=fusion.thermal.provenance,
    )

    with pytest.raises(
        ValueError,
        match="Fusion thermal output does not match",
    ):
        store.log_inference(
            telemetry=telemetry,
            thermal=mismatched,
            fusion=fusion,
            health=health,
        )


def test_optional_embedding_indicators_are_logged_as_none(
    tmp_path,
):
    store = InferenceLogStore(
        tmp_path / "inference_logs.db"
    )

    telemetry = make_telemetry()
    thermal = make_thermal()

    fusion = fuse_decision_outputs(
        FusionInput(
            telemetry=telemetry,
            thermal=thermal,
        )
    )

    health = make_health()

    inference_id = store.log_inference(
        telemetry=telemetry,
        thermal=thermal,
        fusion=fusion,
        health=health,
        inference_id="no-embedding",
    )

    record = store.get_inference(inference_id)

    assert record is not None
    assert record["embedding_indicators"] is None


def test_generated_inference_id_is_unique(tmp_path):
    store = InferenceLogStore(
        tmp_path / "inference_logs.db"
    )

    telemetry = make_telemetry()
    thermal = make_thermal()
    fusion = make_fusion()
    health = make_health()

    first = store.log_inference(
        telemetry=telemetry,
        thermal=thermal,
        fusion=fusion,
        health=health,
    )

    second = store.log_inference(
        telemetry=telemetry,
        thermal=thermal,
        fusion=fusion,
        health=health,
    )

    assert first != second
    assert store.count() == 2