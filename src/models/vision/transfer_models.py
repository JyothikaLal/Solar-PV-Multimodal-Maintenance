from __future__ import annotations

import torch
from torch import nn

from src.models.vision.pretrained_backbones import (
    load_pretrained_efficientnet_b0,
    load_pretrained_resnet18,
)


NUM_CLASSES = 12


def replace_resnet18_classifier(
    model: nn.Module,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Replace the ResNet-18 ImageNet classifier."""

    if num_classes <= 1:
        raise ValueError(
            "num_classes must be greater than 1."
        )

    if not hasattr(model, "fc"):
        raise ValueError(
            "Expected a ResNet-style model with an fc layer."
        )

    if not isinstance(model.fc, nn.Linear):
        raise ValueError(
            "Expected ResNet classifier to be nn.Linear."
        )

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes,
    )

    return model


def replace_efficientnet_b0_classifier(
    model: nn.Module,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Replace the EfficientNet-B0 ImageNet classifier."""

    if num_classes <= 1:
        raise ValueError(
            "num_classes must be greater than 1."
        )

    if not hasattr(model, "classifier"):
        raise ValueError(
            "Expected an EfficientNet-style model "
            "with a classifier."
        )

    if len(model.classifier) == 0:
        raise ValueError(
            "EfficientNet classifier is empty."
        )

    final_layer = model.classifier[-1]

    if not isinstance(final_layer, nn.Linear):
        raise ValueError(
            "Expected EfficientNet final classifier "
            "layer to be nn.Linear."
        )

    in_features = final_layer.in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


def build_pretrained_resnet18(
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Build ImageNet-pretrained ResNet-18 for RaptorMaps."""

    result = load_pretrained_resnet18()

    model = replace_resnet18_classifier(
        result.model,
        num_classes=num_classes,
    )

    return model


def build_pretrained_efficientnet_b0(
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Build ImageNet-pretrained EfficientNet-B0 for RaptorMaps."""

    result = load_pretrained_efficientnet_b0()

    model = replace_efficientnet_b0_classifier(
        result.model,
        num_classes=num_classes,
    )

    return model


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Count trainable model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_parameters(
    model: nn.Module,
) -> int:
    """Count all model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def freeze_resnet18_backbone(model) -> None:
    """Freeze the pretrained ResNet-18 backbone and keep the classifier trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def freeze_efficientnet_b0_backbone(model) -> None:
    """Freeze the pretrained EfficientNet-B0 backbone and keep the classifier trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_resnet18_layer4(model) -> None:
    """Unfreeze ResNet-18 layer4 and classifier for selective fine-tuning."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_efficientnet_b0_features8(model) -> None:
    """Unfreeze EfficientNet-B0 features[8] and classifier for selective fine-tuning."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.features[8].parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

def build_transfer_optimizer(
    model,
    *,
    classifier_module,
    classifier_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    weight_decay: float = 1e-4,
):
    """Build an AdamW optimizer using explicit trainable parameter groups."""
    if classifier_lr <= 0:
        raise ValueError("classifier_lr must be positive.")
    if backbone_lr <= 0:
        raise ValueError("backbone_lr must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")

    classifier_ids = {
        id(parameter)
        for parameter in classifier_module.parameters()
    }

    classifier_parameters = [
        parameter
        for parameter in classifier_module.parameters()
        if parameter.requires_grad
    ]

    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in classifier_ids
    ]

    parameter_groups = []

    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
            }
        )

    if classifier_parameters:
        parameter_groups.append(
            {
                "params": classifier_parameters,
                "lr": classifier_lr,
            }
        )

    if not parameter_groups:
        raise ValueError("No trainable parameters were found.")

    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=weight_decay,
    )


def load_transfer_checkpoint(
    model: nn.Module,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> None:
    """Load current or legacy transfer-learning checkpoints.

    Older versions of the transfer adapter registered the classifier
    twice, including redundant ``classifier_module.*`` entries.
    Those aliases are safely ignored because the canonical classifier
    parameters already exist under the backbone.
    """
    canonical_state_dict = {
        key: value
        for key, value in checkpoint_state_dict.items()
        if not key.startswith("classifier_module.")
    }

    model.load_state_dict(
        canonical_state_dict,
        strict=True,
    )
