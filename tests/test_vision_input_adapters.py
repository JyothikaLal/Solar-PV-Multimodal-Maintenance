import pytest
import torch

from src.models.vision.input_adapters import (
    GrayscaleToRGB,
)


def test_grayscale_to_rgb_shape():
    adapter = GrayscaleToRGB()

    x = torch.randn(
        4,
        1,
        40,
        24,
    )

    output = adapter(x)

    assert output.shape == (
        4,
        3,
        40,
        24,
    )


def test_grayscale_to_rgb_repeats_values():
    adapter = GrayscaleToRGB()

    x = torch.tensor(
        [
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ]
        ]
    )

    output = adapter(x)

    assert torch.equal(
        output[:, 0],
        x[:, 0],
    )

    assert torch.equal(
        output[:, 1],
        x[:, 0],
    )

    assert torch.equal(
        output[:, 2],
        x[:, 0],
    )


def test_grayscale_to_rgb_preserves_dtype():
    adapter = GrayscaleToRGB()

    x = torch.randn(
        2,
        1,
        40,
        24,
        dtype=torch.float32,
    )

    output = adapter(x)

    assert output.dtype == torch.float32


def test_grayscale_to_rgb_preserves_values():
    adapter = GrayscaleToRGB()

    x = torch.randn(
        2,
        1,
        40,
        24,
    )

    output = adapter(x)

    assert torch.equal(
        output[:, 0],
        x[:, 0],
    )

    assert torch.equal(
        output[:, 1],
        x[:, 0],
    )

    assert torch.equal(
        output[:, 2],
        x[:, 0],
    )


def test_grayscale_to_rgb_rejects_non_4d_input():
    adapter = GrayscaleToRGB()

    with pytest.raises(ValueError):
        adapter(
            torch.randn(
                1,
                40,
                24,
            )
        )


def test_grayscale_to_rgb_rejects_non_grayscale_input():
    adapter = GrayscaleToRGB()

    with pytest.raises(ValueError):
        adapter(
            torch.randn(
                2,
                2,
                40,
                24,
            )
        )


def test_grayscale_to_rgb_rejects_non_finite_input():
    adapter = GrayscaleToRGB()

    x = torch.randn(
        2,
        1,
        40,
        24,
    )

    x[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        adapter(x)
