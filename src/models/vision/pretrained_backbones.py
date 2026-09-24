from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)


@dataclass(frozen=True)
class PretrainedBackbone:
    """Loaded pretrained backbone and its weight metadata."""

    model: nn.Module
    weights_name: str
    num_pretrained_classes: int
    input_channels: int
    parameter_count: int


def _count_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def load_pretrained_resnet18() -> PretrainedBackbone:
    """Load ImageNet-pretrained ResNet-18."""

    weights = ResNet18_Weights.IMAGENET1K_V1

    model = resnet18(
        weights=weights,
    )

    parameter_count = _count_parameters(
        model
    )

    if model.conv1.in_channels != 3:
        raise ValueError(
            "Unexpected ResNet-18 input channel count: "
            f"{model.conv1.in_channels}."
        )

    if model.fc.out_features != 1000:
        raise ValueError(
            "Unexpected ResNet-18 classifier output count: "
            f"{model.fc.out_features}."
        )

    return PretrainedBackbone(
        model=model,
        weights_name=str(weights),
        num_pretrained_classes=model.fc.out_features,
        input_channels=model.conv1.in_channels,
        parameter_count=parameter_count,
    )


def load_pretrained_efficientnet_b0() -> PretrainedBackbone:
    """Load ImageNet-pretrained EfficientNet-B0."""

    weights = (
        EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    model = efficientnet_b0(
        weights=weights,
    )

    parameter_count = _count_parameters(
        model
    )

    first_conv = model.features[0][0]

    classifier = model.classifier[-1]

    if first_conv.in_channels != 3:
        raise ValueError(
            "Unexpected EfficientNet-B0 input "
            f"channel count: {first_conv.in_channels}."
        )

    if classifier.out_features != 1000:
        raise ValueError(
            "Unexpected EfficientNet-B0 classifier "
            f"output count: {classifier.out_features}."
        )

    return PretrainedBackbone(
        model=model,
        weights_name=str(weights),
        num_pretrained_classes=classifier.out_features,
        input_channels=first_conv.in_channels,
        parameter_count=parameter_count,
    )


def pretrained_parameters_are_finite(
    model: nn.Module,
) -> bool:
    """Check all floating-point model parameters."""

    return all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
        if torch.is_floating_point(parameter)
    )
