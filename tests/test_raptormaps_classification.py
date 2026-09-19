import numpy as np
import pytest

from src.data.raptormaps_classification import (
    flatten_raptormaps_images,
    load_raptormaps_image,
    load_raptormaps_manifest,
    load_raptormaps_split,
    resolve_image_path,
)


def test_load_manifest():
    manifest = load_raptormaps_manifest()

    assert len(manifest) == 20_000

    assert {
        "metadata_id",
        "image_filepath",
        "anomaly_class",
        "split",
    }.issubset(manifest.columns)


def test_manifest_has_no_duplicate_metadata_ids():
    manifest = load_raptormaps_manifest()

    assert not manifest["metadata_id"].duplicated().any()


def test_manifest_has_expected_splits():
    manifest = load_raptormaps_manifest()

    assert set(manifest["split"].unique()) == {
        "train",
        "validation",
        "test",
        "excluded_conflicting_duplicate",
    }


def test_manifest_split_counts():
    manifest = load_raptormaps_manifest()

    split_counts = manifest["split"].value_counts()

    assert split_counts["train"] == 13_992
    assert split_counts["validation"] == 2_997
    assert split_counts["test"] == 2_999
    assert split_counts["excluded_conflicting_duplicate"] == 12


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError):
        load_raptormaps_split("invalid")


def test_image_is_loaded_as_normalized_grayscale():
    manifest = load_raptormaps_manifest()

    image_path = manifest.iloc[0]["image_filepath"]

    resolved_path = resolve_image_path(
        image_path
    )

    image = load_raptormaps_image(
        resolved_path
    )

    assert image.shape == (40, 24)
    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_split_loader_returns_expected_train_shape():
    X, y, metadata = load_raptormaps_split(
        "train"
    )

    assert X.shape == (
        13_992,
        40,
        24,
    )

    assert y.shape == (
        13_992,
    )

    assert len(metadata) == 13_992


def test_split_loader_returns_matching_labels():
    X, y, metadata = load_raptormaps_split(
        "validation"
    )

    assert len(X) == len(y)
    assert len(y) == len(metadata)

    assert (
        metadata["class_index"].to_numpy()
        == y
    ).all()


def test_supervised_splits_exclude_conflicting_duplicates():
    manifest = load_raptormaps_manifest()

    supervised = manifest[
        manifest["split"].isin(
            ("train", "validation", "test")
        )
    ]

    assert not (
        supervised["split"]
        == "excluded_conflicting_duplicate"
    ).any()

    assert len(supervised) == 19_988


def test_flatten_images_returns_expected_shape():
    X, _, _ = load_raptormaps_split(
        "validation"
    )

    flattened = flatten_raptormaps_images(X)

    assert flattened.shape == (
        2_997,
        960,
    )

    assert flattened.dtype == np.float32


def test_flatten_images_preserves_values():
    X, _, _ = load_raptormaps_split(
        "validation"
    )

    flattened = flatten_raptormaps_images(X)

    reconstructed = flattened.reshape(
        2_997,
        40,
        24,
    )

    np.testing.assert_array_equal(
        reconstructed,
        X,
    )


def test_flatten_images_rejects_invalid_dimensions():
    invalid_X = np.zeros(
        (10, 40),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        flatten_raptormaps_images(invalid_X)


def test_flatten_images_rejects_wrong_image_shape():
    invalid_X = np.zeros(
        (10, 24, 40),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        flatten_raptormaps_images(invalid_X)