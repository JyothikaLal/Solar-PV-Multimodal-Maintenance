from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


EXPECTED_IMAGE_WIDTH = 24
EXPECTED_IMAGE_HEIGHT = 40
EXPECTED_CHANNELS = 1

RAPTOR_MAPS_MEAN = 0.61973207
RAPTOR_MAPS_STD = 0.15437313


def normalize_raptormaps_tensor(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Standardize a [0,1] RaptorMaps image tensor."""

    if tensor.dtype != torch.float32:
        raise ValueError(
            f"Expected float32 tensor, got {tensor.dtype}."
        )

    if tensor.ndim != 3:
        raise ValueError(
            f"Expected [C,H,W] tensor, got shape "
            f"{tuple(tensor.shape)}."
        )

    if tensor.shape != (
        EXPECTED_CHANNELS,
        EXPECTED_IMAGE_HEIGHT,
        EXPECTED_IMAGE_WIDTH,
    ):
        raise ValueError(
            f"Unexpected tensor shape: {tuple(tensor.shape)}."
        )

    if not torch.isfinite(tensor).all():
        raise ValueError(
            "Input tensor contains non-finite values."
        )

    if tensor.min() < 0 or tensor.max() > 1:
        raise ValueError(
            "Expected input tensor values in [0,1]."
        )

    return (
        tensor - RAPTOR_MAPS_MEAN
    ) / RAPTOR_MAPS_STD


def load_raptormaps_image(
    image_path: Path,
) -> torch.Tensor:
    """Load and preprocess one RaptorMaps infrared image.

    Returns:
        Float32 tensor with shape [1, 40, 24].
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"RaptorMaps image not found: {image_path}"
        )

    try:
        with Image.open(image_path) as image:
            image = image.convert("L")

            if image.size != (
                EXPECTED_IMAGE_WIDTH,
                EXPECTED_IMAGE_HEIGHT,
            ):
                raise ValueError(
                    "Unexpected RaptorMaps image size: "
                    f"{image.size}. Expected "
                    f"({EXPECTED_IMAGE_WIDTH}, "
                    f"{EXPECTED_IMAGE_HEIGHT})."
                )

            array = np.asarray(
                image,
                dtype=np.float32,
            )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Failed to load RaptorMaps image: "
            f"{image_path}"
        ) from exc

    if array.shape != (
        EXPECTED_IMAGE_HEIGHT,
        EXPECTED_IMAGE_WIDTH,
    ):
        raise ValueError(
            "Unexpected image array shape: "
            f"{array.shape}. Expected "
            f"({EXPECTED_IMAGE_HEIGHT}, "
            f"{EXPECTED_IMAGE_WIDTH})."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"Non-finite pixel values found in: "
            f"{image_path}"
        )

    if array.min() < 0 or array.max() > 255:
        raise ValueError(
            f"Unexpected pixel range in: "
            f"{image_path}. "
            f"Observed [{array.min()}, {array.max()}]."
        )

    array /= 255.0

    tensor = torch.from_numpy(array).unsqueeze(0)

    if tensor.shape != (
        EXPECTED_CHANNELS,
        EXPECTED_IMAGE_HEIGHT,
        EXPECTED_IMAGE_WIDTH,
    ):
        raise ValueError(
            "Unexpected tensor shape: "
            f"{tuple(tensor.shape)}."
        )

    if tensor.dtype != torch.float32:
        raise ValueError(
            f"Unexpected tensor dtype: {tensor.dtype}."
        )

    if tensor.min() < 0 or tensor.max() > 1:
        raise ValueError(
            "Normalized tensor values are outside [0, 1]."
        )

    return normalize_raptormaps_tensor(tensor)
