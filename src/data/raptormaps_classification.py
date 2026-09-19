from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


MANIFEST_PATH = Path(
    "data/processed/raptormaps/split_manifest.csv"
)

IMAGE_ROOT = Path(
    "data/raw/raptormaps/InfraredSolarModules"
)

IMAGE_WIDTH = 24
IMAGE_HEIGHT = 40

SPLITS = (
    "train",
    "validation",
    "test",
)

EXCLUDED_SPLITS = (
    "excluded_conflicting_duplicate",
)

VALID_MANIFEST_SPLITS = SPLITS + EXCLUDED_SPLITS


def load_raptormaps_manifest(
    manifest_path: Path = MANIFEST_PATH,
) -> pd.DataFrame:
    """
    Load and validate the RaptorMaps split manifest.

    The manifest is treated as the source of truth for:
    - metadata identifiers
    - image paths
    - anomaly labels
    - split assignments

    Records marked as excluded_conflicting_duplicate are retained
    in the manifest for traceability but are not used by supervised
    train, validation, or test loading.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    required_columns = {
        "metadata_id",
        "image_filepath",
        "anomaly_class",
        "split",
    }

    missing_columns = (
        required_columns - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if manifest.empty:
        raise ValueError(
            "RaptorMaps manifest is empty."
        )

    if manifest["metadata_id"].duplicated().any():
        raise ValueError(
            "Manifest contains duplicate metadata IDs."
        )

    if manifest["anomaly_class"].isna().any():
        raise ValueError(
            "Manifest contains missing anomaly labels."
        )

    if manifest["split"].isna().any():
        raise ValueError(
            "Manifest contains missing split assignments."
        )

    unexpected_splits = (
        set(manifest["split"].unique())
        - set(VALID_MANIFEST_SPLITS)
    )

    if unexpected_splits:
        raise ValueError(
            "Manifest contains unexpected split labels: "
            + ", ".join(sorted(unexpected_splits))
        )

    return manifest


def resolve_image_path(
    image_path: str,
    image_root: Path = IMAGE_ROOT,
) -> Path:
    """
    Resolve an image path from the manifest against the dataset root.
    """
    relative_path = Path(image_path)

    if relative_path.is_absolute():
        resolved = relative_path
    else:
        resolved = image_root / relative_path

    if not resolved.exists():
        raise FileNotFoundError(
            f"Image not found: {resolved}"
        )

    return resolved

def flatten_raptormaps_images(
    X: np.ndarray,
) -> np.ndarray:
    """
    Flatten normalized RaptorMaps grayscale images into
    fixed-length pixel feature vectors.

    Args:
        X:
            Image array with shape (n_samples, 40, 24).

    Returns:
        NumPy array with shape (n_samples, 960)
        and dtype float32.
    """
    if X.ndim != 3:
        raise ValueError(
            "Expected image array with 3 dimensions "
            "(n_samples, height, width). "
            f"Received shape: {X.shape}."
        )

    expected_shape = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    if X.shape[1:] != expected_shape:
        raise ValueError(
            "Unexpected image dimensions: "
            f"{X.shape[1:]}. "
            f"Expected {expected_shape}."
        )

    flattened = X.reshape(
        X.shape[0],
        IMAGE_HEIGHT * IMAGE_WIDTH,
    )

    return flattened.astype(
        np.float32,
        copy=False,
    )


def load_raptormaps_image(
    image_path: Path,
) -> np.ndarray:
    """
    Load one RaptorMaps image as a normalized grayscale array.

    Returns:
        NumPy array with shape (24, 40) and dtype float32.
        Pixel values are normalized to [0, 1].
    """
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("L")
        array = np.asarray(
            image,
            dtype=np.float32,
        )

    expected_shape = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    if array.shape != expected_shape:
        raise ValueError(
            f"Unexpected image shape for {image_path}: "
            f"{array.shape}. Expected {expected_shape}."
        )

    array /= 255.0

    return array


def load_raptormaps_split(
    split: str,
    manifest_path: Path = MANIFEST_PATH,
    image_root: Path = IMAGE_ROOT,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load all images and labels for one supervised dataset split.

    Supported splits:
        - train
        - validation
        - test

    Records marked as excluded_conflicting_duplicate are never
    loaded into a supervised split.

    Returns:
        X:
            Image arrays with shape
            (n_samples, 24, 40).

        y:
            Integer-encoded class labels.

        metadata:
            Manifest rows corresponding to the returned samples.
    """
    if split not in SPLITS:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Expected one of: {SPLITS}."
        )

    manifest = load_raptormaps_manifest(
        manifest_path
    )

    split_manifest = (
        manifest[
            manifest["split"] == split
        ]
        .copy()
        .reset_index(drop=True)
    )

    if split_manifest.empty:
        raise ValueError(
            f"No samples found for split: {split}"
        )

    class_names = sorted(
        manifest["anomaly_class"].unique()
    )

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    images = []
    labels = []

    for row in split_manifest.itertuples(
        index=False
    ):
        image_path = resolve_image_path(
            row.image_filepath,
            image_root,
        )

        image = load_raptormaps_image(
            image_path
        )

        images.append(image)

        labels.append(
            class_to_index[row.anomaly_class]
        )

    X = np.stack(images).astype(
        np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    split_manifest["class_index"] = y

    return X, y, split_manifest