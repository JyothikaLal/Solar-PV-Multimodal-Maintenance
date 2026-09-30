from __future__ import annotations

import base64
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas.inference import (
    FusionRequest,
    HealthAssessmentRequest,
    TelemetryPredictionRequest,
    ThermalPredictionRequest,
)
from src.inference.services import (
    ModelInferenceError,
    build_health_assessment_for_predictions,
    fuse_predictions,
    infer_telemetry_prediction,
    infer_thermal_prediction,
)

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "solar-pv-multimodal-maintenance",
        "version": "0.1.0",
    }


@router.post("/tecnalia/predict")
async def predict_tecnalia(request: TelemetryPredictionRequest) -> dict[str, Any]:
    """Run the TECNALIA regression inference contract."""
    try:
        prediction = infer_telemetry_prediction(
            model_name=request.model_name,
            model_version=request.model_version,
            module_name=request.module_name,
            features=request.features,
        )
        return prediction.model_dump()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive layer
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Telemetry prediction failed: {exc}",
        ) from exc


@router.post("/raptormaps/predict")
async def predict_thermal(request: Request) -> dict[str, Any]:
    """Run the RaptorMaps thermal-inference contract.

    The endpoint accepts either a JSON payload containing a base64-encoded image
    or a multipart upload file. This keeps the API usable for both notebook and
    browser-based integration.
    """
    payload: ThermalPredictionRequest | None = None
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        payload_data = form.get("payload")

        if uploaded_file is not None:
            contents = await uploaded_file.read()
            image_base64 = base64.b64encode(contents).decode("ascii")
            payload = ThermalPredictionRequest(
                model_name="resnet18_finetuned",
                model_version="latest",
                image_base64=image_base64,
                image_path=None,
            )
        elif payload_data is not None:
            payload = ThermalPredictionRequest.model_validate_json(payload_data)

    else:
        raw_body = await request.body()
        if raw_body:
            body_data = json.loads(raw_body)
            payload = ThermalPredictionRequest.model_validate(body_data)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No thermal inference payload was supplied.",
        )

    try:
        prediction = infer_thermal_prediction(
            model_name=payload.model_name,
            model_version=payload.model_version,
            image_base64=payload.image_base64,
            image_path=payload.image_path,
        )
        return prediction.model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive layer
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Thermal prediction failed: {exc}",
        ) from exc


@router.post("/fusion")
async def fuse_telemetry_and_thermal(request: FusionRequest) -> dict[str, Any]:
    """Fuse independent telemetry and thermal outputs at the decision layer."""
    try:
        fused = fuse_predictions(
            telemetry=request.telemetry,
            thermal=request.thermal,
            embedding_indicators=request.embedding_indicators,
        )
        return fused.model_dump()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive layer
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fusion inference failed: {exc}",
        ) from exc


@router.post("/fusion/health")
async def health_assessment(request: HealthAssessmentRequest) -> dict[str, Any]:
    """Translate independent model outputs into an evidence-based health assessment."""
    try:
        result = build_health_assessment_for_predictions(
            telemetry_prediction=request.telemetry_prediction,
            thermal_prediction=request.thermal_prediction,
            actual_normalized_pmpp=request.actual_normalized_pmpp,
            embedding_indicators=request.embedding_indicators,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive layer
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health assessment failed: {exc}",
        ) from exc
