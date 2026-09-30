from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


@dataclass(frozen=True)
class RetrainingDatasetConfig:
    raw_dir: Path
    split_manifest: Path


@dataclass(frozen=True)
class RetrainingModelConfig:
    model_family: str


@dataclass(frozen=True)
class RetrainingConfig:
    seed: int
    tracking_uri: str
    dry_run: bool
    register_candidate: bool
    allow_promotion: bool
    tecnalia: RetrainingDatasetConfig
    raptormaps: RetrainingDatasetConfig
    tecnalia_model: RetrainingModelConfig
    raptormaps_resnet18: RetrainingModelConfig
    raptormaps_efficientnet_b0: RetrainingModelConfig


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _resolve_path(value: Any, name: str, project_root: Path) -> Path:
    raw = Path(_require_string(value, name))
    return raw if raw.is_absolute() else project_root / raw


def load_retraining_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: Path | None = None,
) -> RetrainingConfig:
    """Load and validate the reproducible retraining configuration."""

    config_path = Path(config_path)

    if project_root is None:
        project_root = config_path.resolve().parent.parent

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Retraining configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    root = _require_mapping(raw, "configuration")

    reproducibility = _require_mapping(
        root.get("reproducibility"),
        "reproducibility",
    )
    runtime = _require_mapping(root.get("runtime"), "runtime")
    mlflow = _require_mapping(root.get("mlflow"), "mlflow")
    retraining = _require_mapping(
        root.get("retraining"),
        "retraining",
    )

    seed = _require_int(
        reproducibility.get("seed"),
        "reproducibility.seed",
    )

    # Runtime device is intentionally validated here even though Task 40's
    # classical TECNALIA path does not consume it yet. This keeps the shared
    # project configuration structurally valid.
    _require_string(runtime.get("device"), "runtime.device")

    tracking_uri = _require_string(
        mlflow.get("tracking_uri"),
        "mlflow.tracking_uri",
    )

    dry_run = _require_bool(
        retraining.get("dry_run", False),
        "retraining.dry_run",
    )
    register_candidate = _require_bool(
        retraining.get("register_candidate", True),
        "retraining.register_candidate",
    )
    allow_promotion = _require_bool(
        retraining.get("allow_promotion", False),
        "retraining.allow_promotion",
    )

    data = _require_mapping(root.get("data"), "data")
    tecnalia_data = _require_mapping(
        data.get("tecnalia"),
        "data.tecnalia",
    )
    raptormaps_data = _require_mapping(
        data.get("raptormaps"),
        "data.raptormaps",
    )

    retraining_tecnalia = _require_mapping(
        retraining.get("tecnalia"),
        "retraining.tecnalia",
    )
    retraining_raptormaps = _require_mapping(
        retraining.get("raptormaps"),
        "retraining.raptormaps",
    )

    tecnalia_model = RetrainingModelConfig(
        model_family=_require_string(
            retraining_tecnalia.get("model_family"),
            "retraining.tecnalia.model_family",
        )
    )

    raptormaps_resnet18 = RetrainingModelConfig(
        model_family=_require_string(
            retraining_raptormaps.get("resnet18_model_family"),
            "retraining.raptormaps.resnet18_model_family",
        )
    )

    raptormaps_efficientnet_b0 = RetrainingModelConfig(
        model_family=_require_string(
            retraining_raptormaps.get("efficientnet_b0_model_family"),
            "retraining.raptormaps.efficientnet_b0_model_family",
        )
    )


    return RetrainingConfig(
        seed=seed,
        tracking_uri=tracking_uri,
        dry_run=dry_run,
        register_candidate=register_candidate,
        allow_promotion=allow_promotion,
        tecnalia=RetrainingDatasetConfig(
            raw_dir=_resolve_path(
                tecnalia_data.get("raw_dir"),
                "data.tecnalia.raw_dir",
                project_root,
            ),
            split_manifest=_resolve_path(
                tecnalia_data.get("split_manifest"),
                "data.tecnalia.split_manifest",
                project_root,
            ),
        ),
        raptormaps=RetrainingDatasetConfig(
            raw_dir=_resolve_path(
                raptormaps_data.get("raw_dir"),
                "data.raptormaps.raw_dir",
                project_root,
            ),
            split_manifest=_resolve_path(
                raptormaps_data.get("split_manifest"),
                "data.raptormaps.split_manifest",
                project_root,
            ),
        ),
        tecnalia_model=tecnalia_model,
        raptormaps_resnet18=raptormaps_resnet18,
        raptormaps_efficientnet_b0=raptormaps_efficientnet_b0,
    )
