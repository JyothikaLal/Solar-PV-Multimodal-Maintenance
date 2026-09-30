from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app
from src.data.tecnalia_preprocessing import preprocess_tecnalia_module
from src.inference.services import (
    build_health_assessment_for_predictions,
    infer_telemetry_prediction,
    infer_thermal_prediction,
)


def _make_test_tecnalia_csv(path) -> None:
    frame = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(
                [
                    "2025-01-01 10:00:00",
                    "2025-01-01 10:10:00",
                    "2025-01-01 10:20:00",
                ]
            ),
            "Vmpp(V)": [30.0, 31.0, 32.0],
            "Impp(A)": [5.0, 5.1, 5.2],
            "Temp. Mod (°C)": [20.0, 21.0, 22.0],
            "GHI (W/m²)": [500.0, 600.0, 700.0],
            "Front GPOA (W/m²)": [550.0, 650.0, 750.0],
            "Wind Speed (m/s)": [1.0, 1.5, 2.0],
            "Pmpp (W)": [150.0, 158.1, 166.4],
        }
    )
    frame.to_csv(path, sep=";", index=False)


def _make_thermal_payload() -> str:
    array = np.full((24, 40), 200, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_preprocessing_and_end_to_end_inference_contract(tmp_path) -> None:
    csv_path = tmp_path / "data_Atersa.csv"
    _make_test_tecnalia_csv(csv_path)

    processed, report = preprocess_tecnalia_module(csv_path, "Atersa")

    assert len(processed) == 3
    assert report["module_name"] == "Atersa"
    assert processed["normalized_pmpp"].notna().all()

    features = {
        "Front GPOA (W/m²)": float(processed["Front GPOA (W/m²)"].mean()),
        "GHI (W/m²)": float(processed["GHI (W/m²)"].mean()),
        "Temp. Mod (°C)": float(processed["Temp. Mod (°C)"].mean()),
        "Amb. Temp. (°C)": 28.0,
        "Wind Speed (m/s)": float(processed["Wind Speed (m/s)"].mean()),
    }

    telemetry = infer_telemetry_prediction(
        model_name="gradient_boosting_tuned",
        model_version="latest",
        module_name="Atersa",
        features=features,
    )
    thermal = infer_thermal_prediction(
        model_name="resnet18_finetuned",
        model_version="latest",
        image_base64=_make_thermal_payload(),
    )
    assessment = build_health_assessment_for_predictions(
        telemetry_prediction=telemetry,
        thermal_prediction=thermal,
        actual_normalized_pmpp=0.68,
    )

    assert 0.0 <= telemetry.predicted_normalized_pmpp <= 1.5
    assert thermal.anomaly_class in {
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
    assert assessment["combined_evidence_state"] in {
        "no_elevated_evidence",
        "telemetry_evidence",
        "thermal_evidence",
        "multimodal_evidence",
    }
    assert assessment["maintenance_priority"] in {
        "insufficient_evidence",
        "routine",
        "monitor",
        "review",
        "priority_review",
    }


def test_metrics_endpoint_and_api_contract_integration() -> None:
    client = TestClient(app)

    health_response = client.get("/api/v1/health")
    assert health_response.status_code == 200

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert "solar_pv_inference_requests_total" in metrics_response.text
    assert "# HELP" in metrics_response.text
