from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.raptormaps_classification import (
    load_raptormaps_manifest,
)
from src.data.raptormaps_torch import create_raptormaps_dataloaders
from src.data.tecnalia_regression import (
    attach_frozen_tecnalia_split,
    build_tecnalia_regression_frame,
    engineer_tecnalia_regression_modules,
    load_tecnalia_regression_modules,
    validate_regression_feature_contract,
)
from src.data.tecnalia_split import (
    validate_tecnalia_temporal_split,
)
from src.data.validation import (
    validate_raptormaps,
    validate_tecnalia_missingness,
    validate_tecnalia_ranges,
    validate_tecnalia_schema,
    validate_tecnalia_sentinels,
    validate_tecnalia_timestamps,
)


class RetrainingDataValidationError(ValueError):
    """Raised when retraining input data fails a required contract."""


def _require_tecnalia_raw_validation(
    raw_root: Path,
) -> dict[str, Any]:
    """Run existing TECNALIA raw-data validators."""

    results = {
        "schema": validate_tecnalia_schema(raw_root),
        "timestamps": validate_tecnalia_timestamps(raw_root),
        "missingness": validate_tecnalia_missingness(raw_root),
        "sentinels": validate_tecnalia_sentinels(raw_root),
        "ranges": validate_tecnalia_ranges(raw_root),
    }

    blocking_sections = {
        "schema",
        "timestamps",
    }

    failures: dict[str, Any] = {}

    for name in blocking_sections:
        section = results[name]
        failed_modules = {
            module: details
            for module, details in section.items()
            if details.get("status") == "FAIL"
        }

        if failed_modules:
            failures[name] = failed_modules

    if failures:
        raise RetrainingDataValidationError(
            "TECNALIA raw-data validation failed: "
            f"{failures}"
        )

    return results


def validate_tecnalia_retraining_data(
    *,
    raw_root: str | Path,
    split_manifest_path: str | Path,
) -> dict[str, Any]:
    """
    Validate TECNALIA data using the existing preprocessing,
    feature-engineering, frozen-split, and leakage contracts.
    """

    raw_root = Path(raw_root)
    split_manifest_path = Path(split_manifest_path)

    raw_validation = _require_tecnalia_raw_validation(
        raw_root
    )

    if not split_manifest_path.exists():
        raise RetrainingDataValidationError(
            f"TECNALIA split manifest not found: "
            f"{split_manifest_path}"
        )

    modules = load_tecnalia_regression_modules(
        raw_root=raw_root / "TECNALIA",
        module_paths={
            "Atersa": Path("data_Atersa/data_Atersa.csv"),
            "JaSolar3": Path("data_JaSolar3/data_JaSolar3.csv"),
            "NingboSolar": Path("data_NingboSolar/data_NingboSolar.csv"),
            "Photowatt": Path("data_Photowatt/data_Photowatt.csv"),
            "TrinaSolar": Path("data_TrinaSolar/data_TrinaSolar.csv"),
        },
    )

    engineered_modules = engineer_tecnalia_regression_modules(
        modules
    )

    regression_frame = build_tecnalia_regression_frame(
        engineered_modules
    )

    split_manifest = pd.read_csv(
        split_manifest_path
    )

    split_frame = attach_frozen_tecnalia_split(
        regression_frame,
        split_manifest,
    )

    validate_regression_feature_contract(
        split_frame
    )

    module_split_validation: dict[str, Any] = {}

    for module_name, module_df in split_frame.groupby(
        "module_name",
        sort=True,
    ):
        module_split_validation[module_name] = (
            validate_tecnalia_temporal_split(module_df)
        )

    split_labels = set(
        split_frame["split"].dropna().unique()
    )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    if split_labels != expected_splits:
        raise RetrainingDataValidationError(
            "Unexpected TECNALIA split labels: "
            f"{sorted(split_labels)}"
        )

    return {
        "status": "PASS",
        "dataset": "TECNALIA",
        "raw_validation": raw_validation,
        "module_count": len(engineered_modules),
        "regression_rows": len(split_frame),
        "split_counts": (
            split_frame["split"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "module_split_validation": (
            module_split_validation
        ),
        "feature_contract": {
            "status": "PASS",
        },
    }


def _require_raptormaps_raw_validation(
    raw_root: Path,
) -> dict[str, Any]:
    """Run the existing RaptorMaps integrity validators."""

    results = validate_raptormaps(raw_root)

    failures = {
        name: result
        for name, result in results.items()
        if result.get("status") == "FAIL"
    }

    if failures:
        raise RetrainingDataValidationError(
            "RaptorMaps raw-data validation failed: "
            f"{failures}"
        )

    return results


def validate_raptormaps_retraining_data(
    *,
    raw_root: str | Path,
    split_manifest_path: str | Path,
) -> dict[str, Any]:
    """
    Validate RaptorMaps data using the existing integrity,
    manifest, duplicate, label, split, and PyTorch loading contracts.
    """

    raw_root = Path(raw_root)
    split_manifest_path = Path(split_manifest_path)

    raw_validation = _require_raptormaps_raw_validation(
        raw_root
    )

    if not split_manifest_path.exists():
        raise RetrainingDataValidationError(
            f"RaptorMaps split manifest not found: "
            f"{split_manifest_path}"
        )

    manifest = load_raptormaps_manifest(
        manifest_path=split_manifest_path
    )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        manifest["split"].unique()
    )

    if not expected_splits.issubset(actual_splits):
        raise RetrainingDataValidationError(
            "RaptorMaps manifest is missing one or more "
            "supervised splits: "
            f"{sorted(expected_splits - actual_splits)}"
        )

    split_counts = (
        manifest[
            manifest["split"].isin(expected_splits)
        ]["split"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if any(count <= 0 for count in split_counts.values()):
        raise RetrainingDataValidationError(
            "RaptorMaps contains an empty supervised split: "
            f"{split_counts}"
        )

    image_root = raw_root / "InfraredSolarModules"

    train_loader, validation_loader, test_loader = (
        create_raptormaps_dataloaders(
            batch_size=1,
            num_workers=0,
            manifest_path=split_manifest_path,
            image_root=image_root,
            use_train_augmentation=False,
            pin_memory=False,
        )
    )

    loader_counts = {
        "train": len(train_loader.dataset),
        "validation": len(validation_loader.dataset),
        "test": len(test_loader.dataset),
    }

    if loader_counts != split_counts:
        raise RetrainingDataValidationError(
            "RaptorMaps loader counts do not match the "
            f"frozen manifest: loaders={loader_counts}, "
            f"manifest={split_counts}"
        )

    return {
        "status": "PASS",
        "dataset": "RaptorMaps",
        "raw_validation": raw_validation,
        "manifest_records": len(manifest),
        "split_counts": split_counts,
        "loader_counts": loader_counts,
        "excluded_records": int(
            (
                manifest["split"]
                == "excluded_conflicting_duplicate"
            ).sum()
        ),
    }


def validate_retraining_data(
    *,
    tecnalia_raw_root: str | Path,
    tecnalia_split_manifest_path: str | Path,
    raptormaps_raw_root: str | Path,
    raptormaps_split_manifest_path: str | Path,
) -> dict[str, Any]:
    """
    Validate both independent retraining datasets.

    TECNALIA and RaptorMaps remain independent datasets.
    This function performs no cross-dataset pairing or fusion.
    """

    return {
        "status": "PASS",
        "tecnalia": validate_tecnalia_retraining_data(
            raw_root=tecnalia_raw_root,
            split_manifest_path=(
                tecnalia_split_manifest_path
            ),
        ),
        "raptormaps": validate_raptormaps_retraining_data(
            raw_root=raptormaps_raw_root,
            split_manifest_path=(
                raptormaps_split_manifest_path
            ),
        ),
    }
