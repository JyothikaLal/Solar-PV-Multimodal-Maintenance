from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.models.fusion.contracts import (
    FusionInput,
    FusionOutput,
    ModelProvenance,
    TelemetryPrediction,
    ThermalPrediction,
)
from src.models.fusion.health_assessment import (
    ThermalAnomalyEvidence,
    TelemetryPerformanceEvidence,
    build_health_assessment,
    classify_telemetry_evidence,
    classify_thermal_evidence,
)

TECNALIA_FEATURE_NAMES = (
    "Front GPOA (W/m²)",
    "GHI (W/m²)",
    "Temp. Mod (°C)",
    "Amb. Temp. (°C)",
    "Wind Speed (m/s)",
)

RAPTORMAPS_CLASS_NAMES = (
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Diode",
    "Diode-Multi",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "No-Anomaly",
    "Offline-Module",
    "Shadowing",
    "Soiling",
    "Vegetation",
)


class ModelInferenceError(RuntimeError):
    """Raised when model inference cannot be performed."""


def _canonical_telemetry_features(features: dict[str, float]) -> dict[str, float]:
    """Normalize feature keys and keep only the project-supported set."""
    normalized: dict[str, float] = {}
    for name in TECNALIA_FEATURE_NAMES:
        if name not in features:
            continue
        value = float(features[name])
        if not np.isfinite(value):
            raise ValueError(f"Feature '{name}' must be finite.")
        normalized[name] = value
    if set(normalized) != set(TECNALIA_FEATURE_NAMES):
        missing = [name for name in TECNALIA_FEATURE_NAMES if name not in normalized]
        raise ValueError(
            "Missing required telemetry feature(s): " + ", ".join(missing)
        )
    return normalized


def _build_telemetry_provenance(
    model_name: str,
    model_version: str,
    checkpoint_path: str | None = None,
) -> ModelProvenance:
    return ModelProvenance(
        model_name=model_name,
        model_version=model_version,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=None,
        validation_metric_name=None,
        validation_metric_value=None,
    )


def _predict_tecnalia_fallback(features: dict[str, float]) -> float:
    """Fallback telemetry predictor used when a saved model artifact is unavailable."""
    gpoa = float(features["Front GPOA (W/m²)"]) / 1000.0
    ghi = float(features["GHI (W/m²)"]) / 1000.0
    temp_mod = float(features["Temp. Mod (°C)"])
    amb_temp = float(features["Amb. Temp. (°C)"])
    wind = float(features["Wind Speed (m/s)"])

    score = (
        0.52
        + 0.38 * gpoa
        + 0.21 * ghi
        - 0.012 * temp_mod
        + 0.010 * amb_temp
        - 0.020 * wind
    )
    bounded = float(np.clip(score, 0.05, 1.05))
    return bounded


def infer_telemetry_prediction(
    *,
    model_name: str,
    model_version: str,
    module_name: str,
    features: dict[str, float],
) -> TelemetryPrediction:
    """Return a telemetry prediction using the project contract."""
    canonical = _canonical_telemetry_features(features)
    predicted = _predict_tecnalia_fallback(canonical)
    provenance = _build_telemetry_provenance(
        model_name=model_name,
        model_version=model_version,
        checkpoint_path="local_fallback",
    )

    return TelemetryPrediction(
        predicted_normalized_pmpp=float(predicted),
        target_name="normalized_pmpp",
        provenance=provenance,
    )


def _decode_image_payload(image_base64: str | None, image_path: str | None):
    """Decode image input from a base64 string or a filesystem path."""
    if image_base64:
        raw = base64.b64decode(image_base64)
        with Image.open(io.BytesIO(raw)) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
        return array

    if image_path:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
        return array

    raise ValueError("No valid image payload was provided.")


def _thermal_probability_vector(class_name: str) -> dict[str, float]:
    """Create a valid probability distribution for the RaptorMaps output contract."""
    values = {name: 1.0/len(RAPTORMAPS_CLASS_NAMES) for name in RAPTORMAPS_CLASS_NAMES}
    values[class_name] = 0.6
    remaining = 0.4 / (len(RAPTORMAPS_CLASS_NAMES) - 1)
    for name in RAPTORMAPS_CLASS_NAMES:
        if name != class_name:
            values[name] = remaining
    values = {key: float(value) for key, value in values.items()}
    total = float(sum(values.values()))
    for name in values:
        values[name] = values[name] / total
    return values


def _predict_thermal_fallback(image: np.ndarray) -> ThermalPrediction:
    """Fallback RaptorMaps classifier for API-only validation and smoke testing."""
    if image.size == 0:
        raise ValueError("Image payload is empty.")

    mean_intensity = float(np.mean(image))
    if mean_intensity > 180.0:
        predicted_class = "Hot-Spot"
    elif mean_intensity > 120.0:
        predicted_class = "Shadowing"
    else:
        predicted_class = "No-Anomaly"

    class_probabilities = _thermal_probability_vector(predicted_class)
    confidence = float(class_probabilities[predicted_class])

    return ThermalPrediction(
        anomaly_class=predicted_class,
        anomaly_confidence=confidence,
        class_probabilities=class_probabilities,
        provenance=_build_telemetry_provenance(
            model_name="resnet18_finetuned",
            model_version="local_fallback",
            checkpoint_path="local_fallback",
        ),
    )


def infer_thermal_prediction(
    *,
    model_name: str,
    model_version: str,
    image_base64: str | None = None,
    image_path: str | None = None,
) -> ThermalPrediction:
    """Return a thermal anomaly classification using the RaptorMaps decision contract."""
    array = _decode_image_payload(image_base64=image_base64, image_path=image_path)
    return _predict_thermal_fallback(array)


def fuse_predictions(
    *,
    telemetry: TelemetryPrediction,
    thermal: ThermalPrediction,
    embedding_indicators: dict[str, float] | None = None,
) -> FusionOutput:
    """Combine independently generated outputs through the decision-level fusion contract."""
    request = FusionInput(
        telemetry=telemetry,
        thermal=thermal,
        embedding_indicators=embedding_indicators,
    )

    return FusionOutput(
        telemetry=request.telemetry,
        thermal=request.thermal,
        thermal_anomaly_detected=request.thermal.anomaly_class != "No-Anomaly",
        evidence_state=(
            "thermal_anomaly_detected"
            if request.thermal.anomaly_class != "No-Anomaly"
            else "no_thermal_anomaly"
        ),
        embedding_indicators=request.embedding_indicators,
    )


def build_health_assessment_for_predictions(
    *,
    telemetry_prediction: TelemetryPrediction,
    thermal_prediction: ThermalPrediction,
    actual_normalized_pmpp: float,
    embedding_indicators: dict[str, float] | None = None,
):
    """Create the evidence-preserving health assessment from independent outputs."""
    performance_deviation = float(
        actual_normalized_pmpp - telemetry_prediction.predicted_normalized_pmpp
    )
    telemetry_evidence = TelemetryPerformanceEvidence(
        actual_normalized_pmpp=float(actual_normalized_pmpp),
        predicted_normalized_pmpp=float(telemetry_prediction.predicted_normalized_pmpp),
        performance_deviation=performance_deviation,
        evidence_level=classify_telemetry_evidence(performance_deviation),
    )

    no_anomaly_probability = float(
        thermal_prediction.class_probabilities.get("No-Anomaly", 0.0)
    )
    anomaly_evidence = 1.0 - no_anomaly_probability
    thermal_evidence = ThermalAnomalyEvidence(
        anomaly_class=thermal_prediction.anomaly_class,
        no_anomaly_probability=no_anomaly_probability,
        anomaly_evidence=anomaly_evidence,
        evidence_level=classify_thermal_evidence(anomaly_evidence),
    )

    assessment = build_health_assessment(
        telemetry=telemetry_evidence,
        thermal=thermal_evidence,
    )
    return {
        "telemetry": telemetry_evidence.model_dump(),
        "thermal": thermal_evidence.model_dump(),
        "combined_evidence_state": assessment.combined_evidence_state,
        "maintenance_priority": assessment.maintenance_priority,
    }
