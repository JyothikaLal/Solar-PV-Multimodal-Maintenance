import torch

from src.models.vision.pretrained_backbones import (
    load_pretrained_efficientnet_b0,
    load_pretrained_resnet18,
    pretrained_parameters_are_finite,
)


def test_pretrained_resnet18_metadata():
    result = load_pretrained_resnet18()

    assert result.weights_name == (
        "ResNet18_Weights.IMAGENET1K_V1"
    )

    assert result.num_pretrained_classes == 1000
    assert result.input_channels == 3
    assert result.parameter_count == 11689512


def test_pretrained_resnet18_weights_are_finite():
    result = load_pretrained_resnet18()

    assert pretrained_parameters_are_finite(
        result.model
    )


def test_pretrained_efficientnet_b0_metadata():
    result = load_pretrained_efficientnet_b0()

    assert result.weights_name == (
        "EfficientNet_B0_Weights.IMAGENET1K_V1"
    )

    assert result.num_pretrained_classes == 1000
    assert result.input_channels == 3
    assert result.parameter_count == 5288548


def test_pretrained_efficientnet_b0_weights_are_finite():
    result = load_pretrained_efficientnet_b0()

    assert pretrained_parameters_are_finite(
        result.model
    )


def test_pretrained_resnet18_produces_logits():
    result = load_pretrained_resnet18()

    x = torch.randn(
        2,
        3,
        40,
        24,
    )

    result.model.eval()

    with torch.no_grad():
        logits = result.model(x)

    assert logits.shape == (
        2,
        1000,
    )

    assert torch.isfinite(logits).all()


def test_pretrained_efficientnet_b0_produces_logits():
    result = load_pretrained_efficientnet_b0()

    x = torch.randn(
        2,
        3,
        40,
        24,
    )

    result.model.eval()

    with torch.no_grad():
        logits = result.model(x)

    assert logits.shape == (
        2,
        1000,
    )

    assert torch.isfinite(logits).all()
