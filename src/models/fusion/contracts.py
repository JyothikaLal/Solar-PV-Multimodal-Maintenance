from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class ModelProvenance(BaseModel):
    """Traceability information for one independently trained model."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    checkpoint_path: str | None = None
    checkpoint_epoch: int | None = Field(default=None, ge=0)
    validation_metric_name: str | None = None
    validation_metric_value: float | None = None


class TelemetryPrediction(BaseModel):
    """
    Decision-level output from the TECNALIA telemetry branch.

    The current locked telemetry model predicts normalized Pmpp.
    No unsupported telemetry failure probability is included.
    """

    model_config = ConfigDict(extra="forbid")

    predicted_normalized_pmpp: float = Field(ge=0.0)
    target_name: str = "normalized_pmpp"
    provenance: ModelProvenance

    @field_validator("predicted_normalized_pmpp")
    @classmethod
    def validate_finite_prediction(cls, value: float) -> float:
        if not float("-inf") < value < float("inf"):
            raise ValueError(
                "predicted_normalized_pmpp must be finite."
            )
        return value


class ThermalPrediction(BaseModel):
    """
    Decision-level output from the RaptorMaps thermal branch.

    The branch exposes the predicted anomaly class, its confidence,
    and the complete 12-class probability distribution.
    """

    model_config = ConfigDict(extra="forbid")

    anomaly_class: str
    anomaly_confidence: Probability
    class_probabilities: dict[str, Probability]
    provenance: ModelProvenance

    @field_validator("anomaly_class")
    @classmethod
    def validate_anomaly_class(cls, value: str) -> str:
        if value not in RAPTORMAPS_CLASS_NAMES:
            raise ValueError(
                f"Unknown RaptorMaps anomaly class: {value!r}."
            )
        return value

    @model_validator(mode="after")
    def validate_probability_distribution(
        self,
    ) -> "ThermalPrediction":
        expected = set(RAPTORMAPS_CLASS_NAMES)
        actual = set(self.class_probabilities)

        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "class_probabilities must contain exactly the "
                f"RaptorMaps classes. Missing={missing}, Extra={extra}."
            )

        probability_sum = sum(self.class_probabilities.values())

        if abs(probability_sum - 1.0) > 1e-5:
            raise ValueError(
                "class_probabilities must sum to 1 within tolerance "
                f"(received {probability_sum:.8f})."
            )

        predicted_probability = self.class_probabilities[
            self.anomaly_class
        ]

        if abs(
            predicted_probability - self.anomaly_confidence
        ) > 1e-5:
            raise ValueError(
                "anomaly_confidence must match the probability of "
                "anomaly_class."
            )

        return self


class FusionInput(BaseModel):
    """
    Input contract for decision-level fusion.

    Telemetry and thermal outputs are intentionally independent.
    No shared asset ID, timestamp, image ID, or module ID is required.
    """

    model_config = ConfigDict(extra="forbid")

    telemetry: TelemetryPrediction
    thermal: ThermalPrediction
    embedding_indicators: dict[str, float] | None = None

    @field_validator("embedding_indicators")
    @classmethod
    def validate_embedding_indicators(
        cls,
        value: dict[str, float] | None,
    ) -> dict[str, float] | None:
        if value is None:
            return None

        for name, indicator in value.items():
            if not name.strip():
                raise ValueError(
                    "Embedding indicator names must not be empty."
                )

            if not float("-inf") < indicator < float("inf"):
                raise ValueError(
                    f"Embedding indicator {name!r} must be finite."
                )

        return value
