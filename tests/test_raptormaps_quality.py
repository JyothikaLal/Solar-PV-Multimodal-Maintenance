from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.mlops.monitoring.raptormaps_quality import (
    EXPECTED_CLASSES,
    REQUIRED_METADATA_COLUMNS,
    validate_raptormaps_embeddings,
    validate_raptormaps_images,
    validate_raptormaps_metadata,
    validate_thermal_probabilities,
)


def make_metadata(
    *,
    embedding_dimension: int = 512,
    image_count: int = 2,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": range(image_count),
            "metadata_id": [
                f"id-{index}"
                for index in range(image_count)
            ],
            "image_filepath": [
                f"images/{index}.jpg"
                for index in range(image_count)
            ],
            "anomaly_class": [
                EXPECTED_CLASSES[index]
                for index in range(image_count)
            ],
            "split": ["train"] * image_count,
            "group_hash": [
                f"group-{index}"
                for index in range(image_count)
            ],
            "embedding_model": [
                "resnet18_finetuned"
            ] * image_count,
            "embedding_dimension": [
                embedding_dimension
            ] * image_count,
            "checkpoint_path": [
                "models/test.pt"
            ] * image_count,
            "checkpoint_epoch": [1] * image_count,
            "checkpoint_best_validation_macro_f1": [
                0.65
            ] * image_count,
        }
    )


def test_valid_metadata_passes():
    metadata = make_metadata()

    result = validate_raptormaps_metadata(
        metadata,
        "resnet18_finetuned",
    )

    assert result["metadata_quality_valid"] is True
    assert result["row_count"] == 2
    assert result["required_columns_valid"] is True
    assert result["null_fields"] == {}
    assert result[
        "invalid_embedding_dimension_count"
    ] == 0
    assert result["unexpected_classes"] == []
    assert result["invalid_split_values"] == []


def test_missing_required_metadata_column_is_rejected():
    metadata = make_metadata().drop(
        columns=["group_hash"]
    )

    result = validate_raptormaps_metadata(
        metadata,
        "resnet18_finetuned",
    )

    assert result["required_columns_valid"] is False
    assert "group_hash" in result["missing_required_columns"]
    assert result["metadata_quality_valid"] is False


def test_unexpected_class_is_rejected():
    metadata = make_metadata()

    metadata.loc[0, "anomaly_class"] = (
        "Unknown-Class"
    )

    result = validate_raptormaps_metadata(
        metadata,
        "resnet18_finetuned",
    )

    assert result["unexpected_classes"] == [
        "Unknown-Class"
    ]
    assert result["metadata_quality_valid"] is False


def test_invalid_embedding_dimension_is_rejected():
    metadata = make_metadata(
        embedding_dimension=1280
    )

    result = validate_raptormaps_metadata(
        metadata,
        "resnet18_finetuned",
    )

    assert (
        result["invalid_embedding_dimension_count"]
        == 2
    )
    assert result["metadata_quality_valid"] is False


def test_invalid_split_is_rejected():
    metadata = make_metadata()

    metadata.loc[0, "split"] = "production"

    result = validate_raptormaps_metadata(
        metadata,
        "resnet18_finetuned",
    )

    assert result["invalid_split_values"] == [
        "production"
    ]
    assert result["metadata_quality_valid"] is False


def test_valid_images_pass(tmp_path):
    image_root = tmp_path

    image_directory = image_root / "images"
    image_directory.mkdir()

    for index in range(2):
        image = Image.new(
            "L",
            (24, 40),
            color=index,
        )

        image.save(
            image_directory / f"{index}.jpg"
        )

    metadata = make_metadata()

    result = validate_raptormaps_images(
        metadata,
        image_root=image_root,
    )

    assert result["image_quality_valid"] is True
    assert result["checked_count"] == 2
    assert result["missing_count"] == 0
    assert result["unreadable_count"] == 0
    assert result["unexpected_dimension_count"] == 0


def test_missing_image_is_detected(tmp_path):
    image_root = tmp_path
    metadata = make_metadata()

    result = validate_raptormaps_images(
        metadata,
        image_root=image_root,
    )

    assert result["missing_count"] == 2
    assert result["image_quality_valid"] is False


def test_wrong_image_dimensions_are_detected(
    tmp_path,
):
    image_root = tmp_path
    image_directory = image_root / "images"
    image_directory.mkdir()

    image = Image.new(
        "L",
        (40, 24),
        color=0,
    )

    image.save(
        image_directory / "0.jpg"
    )

    image = Image.new(
        "L",
        (24, 40),
        color=0,
    )

    image.save(
        image_directory / "1.jpg"
    )

    metadata = make_metadata()

    result = validate_raptormaps_images(
        metadata,
        image_root=image_root,
    )

    assert result[
        "unexpected_dimension_count"
    ] == 1
    assert result["image_quality_valid"] is False


def test_valid_embeddings_pass():
    embeddings = np.zeros(
        (10, 512),
        dtype=np.float32,
    )

    result = validate_raptormaps_embeddings(
        embeddings,
        "resnet18_finetuned",
    )

    assert result["embedding_quality_valid"] is True
    assert result["row_count"] == 10
    assert result["actual_dimension"] == 512
    assert result["non_finite_count"] == 0


def test_wrong_embedding_dimension_is_rejected():
    embeddings = np.zeros(
        (10, 1280),
        dtype=np.float32,
    )

    result = validate_raptormaps_embeddings(
        embeddings,
        "resnet18_finetuned",
    )

    assert result["embedding_quality_valid"] is False
    assert result["actual_dimension"] == 1280


def test_non_finite_embeddings_are_rejected():
    embeddings = np.zeros(
        (10, 512),
        dtype=np.float32,
    )

    embeddings[0, 0] = np.nan

    result = validate_raptormaps_embeddings(
        embeddings,
        "resnet18_finetuned",
    )

    assert result["non_finite_count"] == 1
    assert result["embedding_quality_valid"] is False


def test_valid_thermal_probabilities_pass():
    probabilities = np.full(
        12,
        1.0 / 12.0,
    )

    result = validate_thermal_probabilities(
        probabilities
    )

    assert result[
        "probability_quality_valid"
    ] is True
    assert result["class_count"] == 12
    assert result["finite"] is True
    assert result["non_negative"] is True
    assert result["probability_sum"] == pytest.approx(
        1.0
    )


def test_wrong_probability_class_count_is_rejected():
    probabilities = np.full(
        11,
        1.0 / 11.0,
    )

    result = validate_thermal_probabilities(
        probabilities
    )

    assert result[
        "probability_quality_valid"
    ] is False
    assert result["class_count"] == 11


def test_non_finite_probability_is_rejected():
    probabilities = np.full(
        12,
        1.0 / 12.0,
    )

    probabilities[0] = np.nan

    result = validate_thermal_probabilities(
        probabilities
    )

    assert result[
        "probability_quality_valid"
    ] is False
    assert result["finite"] is False
