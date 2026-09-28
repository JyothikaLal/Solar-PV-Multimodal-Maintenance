from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_ROOT = Path(
    "data/raw/raptormaps/InfraredSolarModules"
)

EXPECTED_IMAGE_SIZE = (24, 40)

EXPECTED_CLASSES = [
    "No-Anomaly",
    "Cell",
    "Vegetation",
    "Diode",
    "Cell-Multi",
    "Shadowing",
    "Cracking",
    "Offline-Module",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Soiling",
    "Diode-Multi",
]

MODEL_DIMENSIONS = {
    "resnet18_finetuned": 512,
    "efficientnet_b0_finetuned": 1280,
}

REQUIRED_METADATA_COLUMNS = {
    "row_index",
    "metadata_id",
    "image_filepath",
    "anomaly_class",
    "split",
    "group_hash",
    "embedding_model",
    "embedding_dimension",
    "checkpoint_path",
    "checkpoint_epoch",
    "checkpoint_best_validation_macro_f1",
}


def validate_raptormaps_metadata(
    metadata: pd.DataFrame,
    model_name: str,
) -> dict[str, object]:
    """Validate the RaptorMaps metadata contract."""

    if model_name not in MODEL_DIMENSIONS:
        raise ValueError(
            f"Unsupported RaptorMaps model: {model_name}"
        )

    missing_columns = sorted(
        REQUIRED_METADATA_COLUMNS
        - set(metadata.columns)
    )

    if missing_columns:
        return {
            "row_count": int(len(metadata)),
            "required_columns_valid": False,
            "missing_required_columns": missing_columns,
            "metadata_quality_valid": False,
        }

    null_counts = metadata[
        sorted(REQUIRED_METADATA_COLUMNS)
    ].isna().sum()

    null_fields = {
        column: int(count)
        for column, count in null_counts.items()
        if count > 0
    }

    expected_dimension = MODEL_DIMENSIONS[model_name]

    dimension_values = pd.to_numeric(
        metadata["embedding_dimension"],
        errors="coerce",
    )

    invalid_dimensions = int(
        (
            dimension_values.isna()
            | (dimension_values != expected_dimension)
        ).sum()
    )

    invalid_classes = sorted(
        set(
            metadata["anomaly_class"]
            .dropna()
            .astype(str)
        )
        - set(EXPECTED_CLASSES)
    )

    invalid_splits = sorted(
        set(
            metadata["split"]
            .dropna()
            .astype(str)
        )
        - {"train", "validation", "test"}
    )

    return {
        "row_count": int(len(metadata)),
        "required_columns_valid": True,
        "missing_required_columns": [],
        "null_fields": null_fields,
        "embedding_dimension": expected_dimension,
        "invalid_embedding_dimension_count": (
            invalid_dimensions
        ),
        "unexpected_classes": invalid_classes,
        "invalid_split_values": invalid_splits,
        "metadata_quality_valid": (
            not null_fields
            and invalid_dimensions == 0
            and not invalid_classes
            and not invalid_splits
        ),
    }


def validate_raptormaps_images(
    metadata: pd.DataFrame,
    image_root: str | Path = IMAGE_ROOT,
    max_images: int | None = None,
) -> dict[str, object]:
    """Validate RaptorMaps image availability and readability."""

    root = Path(image_root)

    paths = metadata["image_filepath"].astype(str)

    if max_images is not None:
        paths = paths.head(max_images)

    missing_count = 0
    unreadable_count = 0
    unexpected_dimension_count = 0

    dimensions: dict[str, int] = {}

    for relative_path in paths:
        image_path = root / relative_path

        if not image_path.exists():
            missing_count += 1
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                size = image.size

            dimension_key = f"{size[0]}x{size[1]}"

            dimensions[dimension_key] = (
                dimensions.get(dimension_key, 0) + 1
            )

            if size != EXPECTED_IMAGE_SIZE:
                unexpected_dimension_count += 1

        except Exception:
            unreadable_count += 1

    checked_count = len(paths)

    return {
        "checked_count": checked_count,
        "missing_count": missing_count,
        "missing_rate": (
            float(missing_count / checked_count)
            if checked_count
            else 0.0
        ),
        "unreadable_count": unreadable_count,
        "unreadable_rate": (
            float(unreadable_count / checked_count)
            if checked_count
            else 0.0
        ),
        "unexpected_dimension_count": (
            unexpected_dimension_count
        ),
        "dimensions": dimensions,
        "image_quality_valid": (
            missing_count == 0
            and unreadable_count == 0
            and unexpected_dimension_count == 0
        ),
    }


def validate_raptormaps_embeddings(
    embeddings: np.ndarray,
    model_name: str,
) -> dict[str, object]:
    """Validate RaptorMaps embedding shape and numerical quality."""

    if model_name not in MODEL_DIMENSIONS:
        raise ValueError(
            f"Unsupported RaptorMaps model: {model_name}"
        )

    expected_dimension = MODEL_DIMENSIONS[model_name]

    array = np.asarray(embeddings)

    two_dimensional = array.ndim == 2

    actual_dimension = (
        int(array.shape[1])
        if two_dimensional
        else None
    )

    dimension_valid = (
        two_dimensional
        and actual_dimension == expected_dimension
    )

    finite_count = int(
        np.isfinite(array).sum()
    )

    total_values = int(array.size)

    non_finite_count = (
        total_values - finite_count
    )

    return {
        "row_count": (
            int(array.shape[0])
            if array.ndim >= 1
            else 0
        ),
        "expected_dimension": expected_dimension,
        "actual_dimension": actual_dimension,
        "two_dimensional": two_dimensional,
        "dimension_valid": dimension_valid,
        "non_finite_count": non_finite_count,
        "non_finite_rate": (
            float(non_finite_count / total_values)
            if total_values
            else 0.0
        ),
        "embedding_quality_valid": (
            dimension_valid
            and total_values > 0
            and non_finite_count == 0
        ),
    }


def validate_thermal_probabilities(
    probabilities: Iterable[float],
) -> dict[str, object]:
    """Validate one 12-class RaptorMaps probability vector."""

    values = np.asarray(
        list(probabilities),
        dtype=float,
    )

    expected_count = len(EXPECTED_CLASSES)

    correct_length = (
        values.size == expected_count
    )

    finite = (
        bool(np.isfinite(values).all())
        if values.size
        else False
    )

    non_negative = (
        bool((values >= 0).all())
        if values.size
        else False
    )

    total = (
        float(values.sum())
        if values.size
        else 0.0
    )

    probability_mass_valid = bool(
        finite
        and non_negative
        and np.isclose(
            total,
            1.0,
            rtol=1e-5,
            atol=1e-6,
        )
    )

    return {
        "class_count": int(values.size),
        "expected_class_count": expected_count,
        "class_count_valid": correct_length,
        "finite": finite,
        "non_negative": non_negative,
        "probability_sum": total,
        "probability_mass_valid": (
            probability_mass_valid
        ),
        "probability_quality_valid": (
            correct_length
            and probability_mass_valid
        ),
    }
