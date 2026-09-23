from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.raptormaps_preprocessing import (
    EXPECTED_IMAGE_HEIGHT,
    EXPECTED_IMAGE_WIDTH,
    load_raptormaps_image,
)


RAW_SAMPLE = Path(
    "data/raw/raptormaps/"
    "InfraredSolarModules/images/0.jpg"
)


def test_valid_image_shape_and_dtype() -> None:
    tensor = load_raptormaps_image(RAW_SAMPLE)

    assert tensor.shape == (
        1,
        EXPECTED_IMAGE_HEIGHT,
        EXPECTED_IMAGE_WIDTH,
    )
    assert tensor.dtype == torch.float32


def test_valid_image_is_standardized() -> None:
    from src.data.raptormaps_preprocessing import (
        RAPTOR_MAPS_MEAN,
        RAPTOR_MAPS_STD,
    )

    tensor = load_raptormaps_image(RAW_SAMPLE)

    assert torch.isfinite(tensor).all()

    restored = (
        tensor * RAPTOR_MAPS_STD
        + RAPTOR_MAPS_MEAN
    )

    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_preprocessing_is_deterministic() -> None:
    first = load_raptormaps_image(RAW_SAMPLE)
    second = load_raptormaps_image(RAW_SAMPLE)

    assert torch.equal(first, second)


def test_missing_image_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_raptormaps_image(
            Path("does_not_exist.jpg")
        )


def test_unexpected_size_raises(tmp_path: Path) -> None:
    image_path = tmp_path / "wrong_size.jpg"

    image = Image.new(
        "L",
        (32, 32),
        color=100,
    )
    image.save(image_path)

    with pytest.raises(ValueError, match="Unexpected"):
        load_raptormaps_image(image_path)


def test_normalization_parameters_are_frozen() -> None:
    from src.data.raptormaps_preprocessing import (
        RAPTOR_MAPS_MEAN,
        RAPTOR_MAPS_STD,
    )

    assert RAPTOR_MAPS_MEAN == pytest.approx(
        0.61973207
    )
    assert RAPTOR_MAPS_STD == pytest.approx(
        0.15437313
    )


def test_loaded_tensor_is_standardized() -> None:
    from src.data.raptormaps_preprocessing import (
        RAPTOR_MAPS_MEAN,
        RAPTOR_MAPS_STD,
    )

    tensor = load_raptormaps_image(RAW_SAMPLE)

    assert tensor.shape == (1, 40, 24)

    # Reverse the standardization and verify that the
    # underlying representation remains in [0,1].
    restored = (
        tensor * RAPTOR_MAPS_STD
        + RAPTOR_MAPS_MEAN
    )

    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_normalization_is_deterministic() -> None:
    first = load_raptormaps_image(RAW_SAMPLE)
    second = load_raptormaps_image(RAW_SAMPLE)

    assert torch.equal(first, second)
