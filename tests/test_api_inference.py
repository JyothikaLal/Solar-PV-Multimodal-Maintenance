from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app

client = TestClient(app)


def _make_image_payload() -> str:
    image = np.full((24, 40), 200, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(image, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_health_endpoint() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "# HELP" in response.text


def test_telemetry_prediction_endpoint() -> None:
    payload = {
        "model_name": "gradient_boosting_tuned",
        "model_version": "latest",
        "module_name": "Atersa",
        "features": {
            "Front GPOA (W/m²)": 650.0,
            "GHI (W/m²)": 540.0,
            "Temp. Mod (°C)": 42.5,
            "Amb. Temp. (°C)": 28.0,
            "Wind Speed (m/s)": 3.2,
        },
    }

    response = client.post("/api/v1/tecnalia/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["predicted_normalized_pmpp"] <= 1.5
    assert body["target_name"] == "normalized_pmpp"


def test_thermal_prediction_endpoint_with_base64() -> None:
    response = client.post(
        "/api/v1/raptormaps/predict",
        json={
            "model_name": "resnet18_finetuned",
            "model_version": "latest",
            "image_base64": _make_image_payload(),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["anomaly_class"] in {
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
    }
    assert 0.0 <= body["anomaly_confidence"] <= 1.0


def test_fusion_health_endpoint() -> None:
    telemetry = {
        "predicted_normalized_pmpp": 0.76,
        "target_name": "normalized_pmpp",
        "provenance": {
            "model_name": "gradient_boosting_tuned",
            "model_version": "latest",
            "checkpoint_path": "local_fallback",
            "checkpoint_epoch": None,
            "validation_metric_name": None,
            "validation_metric_value": None,
        },
    }
    thermal = {
        "anomaly_class": "Hot-Spot",
        "anomaly_confidence": 0.63,
        "class_probabilities": {
            "Cell": 0.02,
            "Cell-Multi": 0.01,
            "Cracking": 0.02,
            "Diode": 0.01,
            "Diode-Multi": 0.01,
            "Hot-Spot": 0.63,
            "Hot-Spot-Multi": 0.06,
            "No-Anomaly": 0.12,
            "Offline-Module": 0.02,
            "Shadowing": 0.04,
            "Soiling": 0.03,
            "Vegetation": 0.03,
        },
        "provenance": {
            "model_name": "resnet18_finetuned",
            "model_version": "latest",
            "checkpoint_path": "local_fallback",
            "checkpoint_epoch": None,
            "validation_metric_name": None,
            "validation_metric_value": None,
        },
    }

    response = client.post(
        "/api/v1/fusion/health",
        json={
            "telemetry_prediction": telemetry,
            "thermal_prediction": thermal,
            "actual_normalized_pmpp": 0.68,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["combined_evidence_state"] in {
        "no_elevated_evidence",
        "telemetry_evidence",
        "thermal_evidence",
        "multimodal_evidence",
    }
    assert body["maintenance_priority"] in {
        "insufficient_evidence",
        "routine",
        "monitor",
        "review",
        "priority_review",
    }


def test_invalid_telemetry_payload_is_rejected() -> None:
    response = client.post(
        "/api/v1/tecnalia/predict",
        json={
            "model_name": "gradient_boosting_tuned",
            "model_version": "latest",
            "module_name": "Atersa",
            "features": {
                "Front GPOA (W/m²)": 650.0,
                "GHI (W/m²)": 540.0,
            },
        },
    )
    assert response.status_code == 422
    assert "Missing required telemetry feature" in response.json()["detail"]
