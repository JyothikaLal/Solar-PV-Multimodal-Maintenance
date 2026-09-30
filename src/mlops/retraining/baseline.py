from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.mlops.model_registry import (
    LIFECYCLE_PRODUCTION,
    REGISTERED_MODEL_EFFICIENTNET_B0,
    REGISTERED_MODEL_RESNET18,
    configure_registry,
)


TECNALIA_BASELINE_ARTIFACT = Path(
    "reports/results/tecnalia/final_models/model_selection_summary.json"
)

RAPTORMAPS_PRODUCTION_METRIC = "best_validation_macro_f1"


@dataclass(frozen=True)
class ProductionBaseline:
    dataset: str
    model_family: str
    validation_metrics: Mapping[str, float]
    source: str


class ProductionBaselineUnavailable(RuntimeError):
    """Raised when no valid production baseline can be resolved."""


def resolve_tecnalia_production_baseline(
    *,
    artifact_path: Path = TECNALIA_BASELINE_ARTIFACT,
) -> ProductionBaseline:
    """Resolve the locked TECNALIA production/reference baseline.

    The model-selection artifact is authoritative for the validation
    acceptance baseline. Final test metrics are intentionally ignored.
    """
    if not artifact_path.exists():
        raise ProductionBaselineUnavailable(
            f"TECNALIA baseline artifact does not exist: {artifact_path}"
        )

    try:
        payload = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionBaselineUnavailable(
            f"Unable to read TECNALIA baseline artifact: {artifact_path}"
        ) from exc

    if payload.get("selected_model") != "gradient_boosting_tuned":
        raise ProductionBaselineUnavailable(
            "TECNALIA baseline artifact does not identify "
            "gradient_boosting_tuned as the selected model."
        )

    validation = payload.get("validation_results")
    if not isinstance(validation, dict):
        raise ProductionBaselineUnavailable(
            "TECNALIA baseline artifact is missing validation_results."
        )

    required = ("mae", "rmse", "r2")
    if any(metric not in validation for metric in required):
        raise ProductionBaselineUnavailable(
            "TECNALIA baseline artifact is missing required "
            "validation metrics."
        )

    return ProductionBaseline(
        dataset="TECNALIA",
        model_family="gradient_boosting_tuned",
        validation_metrics={
            "validation_mae": float(validation["mae"]),
            "validation_rmse": float(validation["rmse"]),
            "validation_r2": float(validation["r2"]),
        },
        source=str(artifact_path),
    )


def _find_raptormaps_production_versions(
    *,
    tracking_uri: str,
) -> list[tuple[str, object]]:
    """Return explicitly production-tagged RaptorMaps model versions."""
    client = configure_registry(tracking_uri)

    production_versions: list[tuple[str, object]] = []

    for model_name in (
        REGISTERED_MODEL_RESNET18,
        REGISTERED_MODEL_EFFICIENTNET_B0,
    ):
        try:
            versions = client.search_model_versions(
                f"name='{model_name}'"
            )
        except Exception:
            continue

        for version in versions:
            if (
                version.tags.get("lifecycle_stage")
                == LIFECYCLE_PRODUCTION
            ):
                production_versions.append((model_name, version))

    return production_versions


def resolve_raptormaps_production_baseline(
    *,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> ProductionBaseline:
    """Resolve the explicitly registered RaptorMaps production baseline.

    Candidate and archived versions never qualify. If no production model
    exists, fail closed rather than inventing a baseline.
    """
    production_versions = _find_raptormaps_production_versions(
        tracking_uri=tracking_uri,
    )

    if not production_versions:
        raise ProductionBaselineUnavailable(
            "No production RaptorMaps model is registered."
        )

    if len(production_versions) > 1:
        names = ", ".join(
            f"{name}:v{version.version}"
            for name, version in production_versions
        )
        raise ProductionBaselineUnavailable(
            "Multiple production RaptorMaps model versions exist; "
            f"baseline resolution is ambiguous: {names}"
        )

    model_name, version = production_versions[0]
    tags = dict(version.tags)

    metric_value = tags.get(RAPTORMAPS_PRODUCTION_METRIC)
    if metric_value is None:
        raise ProductionBaselineUnavailable(
            "Production RaptorMaps model is missing "
            "best_validation_macro_f1 metadata."
        )

    return ProductionBaseline(
        dataset="RaptorMaps",
        model_family=tags.get("model_family", model_name),
        validation_metrics={
            "best_validation_macro_f1": float(metric_value),
        },
        source=f"{model_name}:v{version.version}",
    )


def resolve_production_baseline(
    *,
    dataset: str,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> ProductionBaseline:
    """Resolve a production baseline for a supported dataset."""
    if dataset == "TECNALIA":
        return resolve_tecnalia_production_baseline()

    if dataset == "RaptorMaps":
        return resolve_raptormaps_production_baseline(
            tracking_uri=tracking_uri,
        )

    raise ValueError(f"Unsupported retraining dataset: {dataset}")
