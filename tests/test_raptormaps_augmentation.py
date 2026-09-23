import torch
from torch import nn

from src.data.raptormaps_augmentation import (
    RaptorMapsTrainAugmentation,
)


def test_augmentation_preserves_shape_and_dtype() -> None:
    image = torch.arange(
        40 * 24,
        dtype=torch.float32,
    ).reshape(1, 40, 24)

    augmentation = RaptorMapsTrainAugmentation(
        horizontal_flip_probability=1.0,
    )

    output = augmentation(image)

    assert output.shape == (1, 40, 24)
    assert output.dtype == torch.float32


def test_horizontal_flip_probability_zero_is_identity() -> None:
    image = torch.arange(
        40 * 24,
        dtype=torch.float32,
    ).reshape(1, 40, 24)

    augmentation = RaptorMapsTrainAugmentation(
        horizontal_flip_probability=0.0,
    )

    output = augmentation(image)

    assert torch.equal(output, image)


def test_horizontal_flip_probability_one_flips_width() -> None:
    image = torch.arange(
        40 * 24,
        dtype=torch.float32,
    ).reshape(1, 40, 24)

    augmentation = RaptorMapsTrainAugmentation(
        horizontal_flip_probability=1.0,
    )

    output = augmentation(image)

    expected = torch.flip(
        image,
        dims=[-1],
    )

    assert torch.equal(output, expected)


def test_invalid_probability_is_rejected() -> None:
    for probability in (-0.1, 1.1):
        try:
            RaptorMapsTrainAugmentation(probability)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Invalid probability was accepted."
            )


def test_augmentation_keeps_values_unchanged() -> None:
    image = torch.tensor(
        [[
            [-1.0, 0.2, 1.5],
            [0.7, -0.3, 2.1],
        ]],
        dtype=torch.float32,
    )

    augmentation = RaptorMapsTrainAugmentation(
        horizontal_flip_probability=1.0,
    )

    output = augmentation(image)

    assert torch.equal(
        torch.sort(output.flatten()).values,
        torch.sort(image.flatten()).values,
    )
