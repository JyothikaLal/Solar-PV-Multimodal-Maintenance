from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.fusion.contracts import RAPTORMAPS_CLASS_NAMES, TelemetryPrediction, ThermalPrediction

TECNALIA_FEATURE_NAMES = (
    "Front GPOA (W/m²)",
    "GHI (W/m²)",
    "Temp. Mod (°C)",
    "Amb. Temp. (°C)",
    "Wind Speed (m/s)",
)


class TelemetryPredictionRequest(BaseModel):
    """Request schema for TECNALIA telemetry inference."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="gradient_boosting_tuned", min_length=1)
    model_version: str = Field(default="latest", min_length=1)
    module_name: str = Field(default="Atersa", min_length=1)
    features: dict[str, float] = Field(...)

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("features cannot be empty.")

        missing = [name for name in TECNALIA_FEATURE_NAMES if name not in value]
        if missing:
            raise ValueError(
                "Missing required telemetry feature(s): " + ", ".join(missing)
            )

        for key, item in value.items():
            if not isinstance(item, (int, float)):
                raise ValueError(f"Feature '{key}' must be numeric.")

        return value


class ThermalPredictionRequest(BaseModel):
    """Request schema for RaptorMaps thermal inference."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="resnet18_finetuned", min_length=1)
    model_version: str = Field(default="latest", min_length=1)
    image_base64: str | None = None
    image_path: str | None = None

    @model_validator(mode="after")
    def validate_image_source(self) -> "ThermalPredictionRequest":
        if not self.image_base64 and not self.image_path:
            raise ValueError(
                "Either 'image_base64' or 'image_path' must be provided."
            )
        return self


class FusionRequest(BaseModel):
    """Decision-level fusion request containing independent branch outputs."""

    model_config = ConfigDict(extra="forbid")

    telemetry: TelemetryPrediction
    thermal: ThermalPrediction
    embedding_indicators: dict[str, float] | None = None


class HealthAssessmentRequest(BaseModel):
    """Health-assessment request using the fused modality outputs."""

    model_config = ConfigDict(extra="forbid")

    telemetry_prediction: TelemetryPrediction
    thermal_prediction: ThermalPrediction
    actual_normalized_pmpp: float = Field(ge=0.0, le=1.5)
    embedding_indicators: dict[str, float] | None = None


class ErrorResponse(BaseModel):
    """Consistent API error payload."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    error_code: str | None = None


TelemetryPredictionResponse = TelemetryPrediction
ThermalPredictionResponse = ThermalPrediction
