from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.routes.inference import router as inference_router
from src.inference.services import ModelInferenceError

REQUEST_COUNTER = Counter(
    "solar_pv_inference_requests_total",
    "Total number of API requests received by the Solar PV inference service.",
    labelnames=("method", "route"),
)
REQUEST_LATENCY = Histogram(
    "solar_pv_inference_request_seconds",
    "Time spent processing requests in the Solar PV inference service.",
    labelnames=("method", "route"),
)

app = FastAPI(
    title="Solar PV Multimodal Predictive Maintenance API",
    version="0.1.0",
    description=(
        "Inference API for the independent TECNALIA telemetry branch, "
        "RaptorMaps thermal-image branch, and decision-level fusion health assessment."
    ),
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUEST_COUNTER.labels(method=request.method, route=path).inc()
    REQUEST_LATENCY.labels(method=request.method, route=path).observe(elapsed)
    return response


app.include_router(inference_router)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    normalized_errors: list[dict[str, object]] = []
    for error in exc.errors():
        cleaned = {
            "loc": list(error.get("loc", [])),
            "msg": str(error.get("msg", "")),
            "type": str(error.get("type", "validation_error")),
        }
        ctx = error.get("ctx")
        if ctx:
            cleaned["ctx"] = {
                key: str(value) if isinstance(value, Exception) else value
                for key, value in ctx.items()
            }
        normalized_errors.append(cleaned)

    detail = "Request validation failed."
    if normalized_errors:
        detail = "; ".join(str(error["msg"]) for error in normalized_errors)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": detail,
            "error_code": "validation_error",
            "errors": normalized_errors,
        },
    )


@app.exception_handler(ModelInferenceError)
async def model_inference_exception_handler(
    request: Request,
    exc: ModelInferenceError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "error_code": "model_inference_error",
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "Solar PV Multimodal Predictive Maintenance API",
        "status": "online",
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
