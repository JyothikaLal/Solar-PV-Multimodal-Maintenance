from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from src.models.fusion.contracts import (
    FusionInput,
    ModelProvenance,
    RAPTORMAPS_CLASS_NAMES,
    TelemetryPrediction,
    ThermalPrediction,
)


def make_provenance() -> ModelProvenance:
    return ModelProvenance(
        model_name="test_model",
        model_version="test",
        validation_metric_name="test_metric",
        validation_metric_value=0.5,
    )


def make_thermal(
    *,
    anomaly_class: str = "No-Anomaly",
    confidence: float = 1.0,
    probabilities: dict[str, float] | None = None,
) -> ThermalPrediction:
    if probabilities is None:
        probabilities = {
            class_name: 0.0
            for class_name in RAPTORMAPS_CLASS_NAMES
        }
        probabilities[anomaly_class] = 1.0

    return ThermalPrediction(
        anomaly_class=anomaly_class,
        anomaly_confidence=confidence,
        class_probabilities=probabilities,
        provenance=make_provenance(),
    )


def make_telemetry() -> TelemetryPrediction:
    return TelemetryPrediction(
        predicted_normalized_pmpp=0.82,
        provenance=make_provenance(),
    )


def test_valid_fusion_input_is_accepted() -> None:
    fusion_input = FusionInput(
        telemetry=make_telemetry(),
        thermal=make_thermal(),
    )

    assert fusion_input.telemetry.predicted_normalized_pmpp == 0.82
    assert fusion_input.thermal.anomaly_class == "No-Anomaly"
    assert fusion_input.embedding_indicators is None


def test_embedding_indicators_are_optional() -> None:
    fusion_input = FusionInput(
        telemetry=make_telemetry(),
        thermal=make_thermal(),
        embedding_indicators={
            "example_indicator": 0.25,
        },
    )

    assert fusion_input.embedding_indicators == {
        "example_indicator": 0.25,
    }


def test_thermal_requires_exact_rp_class_set() -> None:
    probabilities = {
        class_name: 0.0
        for class_name in RAPTORMAPS_CLASS_NAMES
    }
    probabilities["No-Anomaly"] = 1.0
    probabilities.pop("Vegetation")

    with pytest.raises(ValidationError, match="exactly the RaptorMaps classes"):
        make_thermal(probabilities=probabilities)


def test_thermal_probabilities_must_sum_to_one() -> None:
    probabilities = {
        class_name: 0.0
        for class_name in RAPTORMAPS_CLASS_NAMES
    }
    probabilities["No-Anomaly"] = 0.9

    with pytest.raises(
        ValidationError,
        match="must sum to 1",
    ):
        make_thermal(
            confidence=0.9,
            probabilities=probabilities,
        )


def test_thermal_confidence_must_match_predicted_class_probability() -> None:
    probabilities = {
        class_name: 0.0
        for class_name in RAPTORMAPS_CLASS_NAMES
    }
    probabilities["No-Anomaly"] = 0.8
    probabilities["Soiling"] = 0.2

    with pytest.raises(
        ValidationError,
        match="anomaly_confidence must match",
    ):
        make_thermal(
            confidence=0.7,
            probabilities=probabilities,
        )


def test_invalid_thermal_class_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="Unknown RaptorMaps anomaly class",
    ):
        make_thermal(anomaly_class="Unknown-Class")


def test_probability_bounds_are_enforced() -> None:
    probabilities = {
        class_name: 0.0
        for class_name in RAPTORMAPS_CLASS_NAMES
    }
    probabilities["No-Anomaly"] = 1.0

    with pytest.raises(ValidationError):
        make_thermal(
            confidence=1.1,
            probabilities=probabilities,
        )


def test_telemetry_prediction_must_be_finite() -> None:
    with pytest.raises(ValidationError):
        TelemetryPrediction(
            predicted_normalized_pmpp=math.nan,
            provenance=make_provenance(),
        )

    with pytest.raises(ValidationError):
        TelemetryPrediction(
            predicted_normalized_pmpp=math.inf,
            provenance=make_provenance(),
        )


def test_embedding_indicators_must_be_finite() -> None:
    with pytest.raises(ValidationError):
        FusionInput(
            telemetry=make_telemetry(),
            thermal=make_thermal(),
            embedding_indicators={
                "bad_indicator": math.nan,
            },
        )


def test_extra_fusion_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        FusionInput(
            telemetry=make_telemetry(),
            thermal=make_thermal(),
            asset_id="forbidden_pairing_identifier",
        )


def test_no_shared_pairing_identifier_is_required() -> None:
    fusion_input = FusionInput(
        telemetry=make_telemetry(),
        thermal=make_thermal(),
    )

    assert hasattr(fusion_input, "telemetry")
    assert hasattr(fusion_input, "thermal")
    assert not hasattr(fusion_input, "asset_id")
    assert not hasattr(fusion_input, "image_id")
    assert not hasattr(fusion_input, "timestamp")
