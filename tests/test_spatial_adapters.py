import pytest
import torch

from src.models.vision.spatial_adapters import (
    resize_for_transfer_learning,
)


def test_native_size_preserves_shape():
    x = torch.randn(
        2,
        3,
        40,
        24,
    )

    output = resize_for_transfer_learning(
        x,
        (40, 24),
    )

    assert output.shape == (
        2,
        3,
        40,
        24,
    )


def test_2x_size_preserves_aspect_ratio():
    x = torch.randn(
        2,
        3,
        40,
        24,
    )

    output = resize_for_transfer_learning(
        x,
        (80, 48),
    )

    assert output.shape == (
        2,
        3,
        80,
        48,
    )


def test_4x_size_preserves_aspect_ratio():
    x = torch.randn(
        2,
        3,
        40,
        24,
    )

    output = resize_for_transfer_learning(
        x,
        (160, 96),
    )

    assert output.shape == (
        2,
        3,
        160,
        96,
    )


def test_interpolation_preserves_finite_values():
    x = torch.randn(
        2,
        3,
        40,
        24,
    )

    output = resize_for_transfer_learning(
        x,
        (160, 96),
    )

    assert torch.isfinite(output).all()


def test_native_resize_is_identity_for_same_size():
    x = torch.randn(
        2,
        3,
        40,
        24,
    )

    output = resize_for_transfer_learning(
        x,
        (40, 24),
    )

    assert torch.equal(
        output,
        x,
    )


def test_rejects_wrong_channel_count():
    x = torch.randn(
        2,
        1,
        40,
        24,
    )

    with pytest.raises(ValueError):
        resize_for_transfer_learning(
            x,
            (80, 48),
        )


def test_rejects_invalid_target_size():
    x = torch.randn(
        2,
        3,
        40,
        24,
    )

    with pytest.raises(ValueError):
        resize_for_transfer_learning(
            x,
            (0, 48),
        )
