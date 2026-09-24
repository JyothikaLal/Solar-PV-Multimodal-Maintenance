from src.models.vision.transfer_models import freeze_efficientnet_b0_backbone, freeze_resnet18_backbone
import torch
from torch import nn

from src.models.vision.transfer_models import (
    build_transfer_optimizer,
    build_pretrained_efficientnet_b0,
    build_pretrained_resnet18,
    count_parameters,
    count_trainable_parameters,
    replace_efficientnet_b0_classifier,
    replace_resnet18_classifier,
    unfreeze_efficientnet_b0_features8,
    unfreeze_resnet18_layer4,
)

def test_resnet18_classifier_is_replaced():
    from src.models.vision.pretrained_backbones import (
        load_pretrained_resnet18,
    )

    result = load_pretrained_resnet18()

    original_backbone_weight = (
        result.model.layer1[0].conv1.weight
        .detach()
        .clone()
    )

    model = replace_resnet18_classifier(
        result.model,
        num_classes=12,
    )

    assert model.fc.in_features == 512
    assert model.fc.out_features == 12

    assert torch.equal(
        original_backbone_weight,
        model.layer1[0].conv1.weight,
    )


def test_efficientnet_classifier_is_replaced():
    from src.models.vision.pretrained_backbones import (
        load_pretrained_efficientnet_b0,
    )

    result = load_pretrained_efficientnet_b0()

    original_backbone_weight = (
        result.model.features[0][0].weight
        .detach()
        .clone()
    )

    model = replace_efficientnet_b0_classifier(
        result.model,
        num_classes=12,
    )

    assert isinstance(
        model.classifier[0],
        nn.Dropout,
    )

    assert model.classifier[-1].in_features == 1280
    assert model.classifier[-1].out_features == 12

    assert torch.equal(
        original_backbone_weight,
        model.features[0][0].weight,
    )


def test_resnet18_transfer_model_output_shape():
    model = build_pretrained_resnet18(
        num_classes=12,
    )

    model.eval()

    x = torch.randn(
        2,
        3,
        40,
        24,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (
        2,
        12,
    )

    assert torch.isfinite(output).all()


def test_efficientnet_b0_transfer_model_output_shape():
    model = build_pretrained_efficientnet_b0(
        num_classes=12,
    )

    model.eval()

    x = torch.randn(
        2,
        3,
        40,
        24,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (
        2,
        12,
    )

    assert torch.isfinite(output).all()


def test_resnet18_transfer_model_parameter_count():
    model = build_pretrained_resnet18(
        num_classes=12,
    )

    assert count_parameters(model) == 11182668
    assert count_trainable_parameters(model) == 11182668


def test_efficientnet_b0_transfer_model_parameter_count():
    model = build_pretrained_efficientnet_b0(
        num_classes=12,
    )

    assert count_parameters(model) == 4022920
    assert count_trainable_parameters(model) == 4022920


def test_resnet18_backbone_freezing():
    model = build_pretrained_resnet18(num_classes=12)

    freeze_resnet18_backbone(model)

    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    ]

    classifier_parameters = list(model.fc.parameters())

    assert backbone_parameters
    assert classifier_parameters
    assert all(not parameter.requires_grad for parameter in backbone_parameters)
    assert all(parameter.requires_grad for parameter in classifier_parameters)
    assert count_trainable_parameters(model) == 6156


def test_efficientnet_b0_backbone_freezing():
    model = build_pretrained_efficientnet_b0(num_classes=12)

    freeze_efficientnet_b0_backbone(model)

    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.")
    ]

    classifier_parameters = list(model.classifier.parameters())

    assert backbone_parameters
    assert classifier_parameters
    assert all(not parameter.requires_grad for parameter in backbone_parameters)
    assert all(parameter.requires_grad for parameter in classifier_parameters)
    assert count_trainable_parameters(model) == 15372


def _optimizer_parameter_ids(optimizer):
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_resnet18_optimizer_excludes_frozen_backbone():
    model = build_pretrained_resnet18(num_classes=12)
    freeze_resnet18_backbone(model)

    optimizer = build_transfer_optimizer(
        model,
        classifier_module=model.fc,
        classifier_lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-4,
    )

    optimizer_ids = _optimizer_parameter_ids(optimizer)
    classifier_ids = {id(parameter) for parameter in model.fc.parameters()}

    assert len(optimizer.param_groups) == 1
    assert optimizer_ids == classifier_ids
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_efficientnet_b0_optimizer_excludes_frozen_backbone():
    model = build_pretrained_efficientnet_b0(num_classes=12)
    freeze_efficientnet_b0_backbone(model)

    optimizer = build_transfer_optimizer(
        model,
        classifier_module=model.classifier,
        classifier_lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-4,
    )

    optimizer_ids = _optimizer_parameter_ids(optimizer)
    classifier_ids = {
        id(parameter)
        for parameter in model.classifier.parameters()
    }

    assert len(optimizer.param_groups) == 1
    assert optimizer_ids == classifier_ids
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_transfer_optimizer_uses_two_groups_when_backbone_is_trainable():
    model = build_pretrained_resnet18(num_classes=12)

    optimizer = build_transfer_optimizer(
        model,
        classifier_module=model.fc,
        classifier_lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-4,
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 1e-3

    optimized_ids = _optimizer_parameter_ids(optimizer)
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    assert optimized_ids == trainable_ids

def test_resnet18_selective_fine_tuning_unfreezes_only_layer4_and_classifier():
    model = build_pretrained_resnet18(num_classes=12)

    unfreeze_resnet18_layer4(model)

    for name, parameter in model.named_parameters():
        if name.startswith("layer4.") or name.startswith("fc."):
            assert parameter.requires_grad
        else:
            assert not parameter.requires_grad

    assert count_trainable_parameters(model) == (
        sum(
            parameter.numel()
            for parameter in model.layer4.parameters()
        )
        + sum(
            parameter.numel()
            for parameter in model.fc.parameters()
        )
    )


def test_efficientnet_b0_selective_fine_tuning_unfreezes_only_features8_and_classifier():
    model = build_pretrained_efficientnet_b0(num_classes=12)

    unfreeze_efficientnet_b0_features8(model)

    for name, parameter in model.named_parameters():
        if (
            name.startswith("features.8.")
            or name.startswith("classifier.")
        ):
            assert parameter.requires_grad
        else:
            assert not parameter.requires_grad

    assert count_trainable_parameters(model) == (
        sum(
            parameter.numel()
            for parameter in model.features[8].parameters()
        )
        + sum(
            parameter.numel()
            for parameter in model.classifier.parameters()
        )
    )


def test_resnet18_fine_tuning_optimizer_has_backbone_and_classifier_groups():
    model = build_pretrained_resnet18(num_classes=12)

    unfreeze_resnet18_layer4(model)

    optimizer = build_transfer_optimizer(
        model,
        classifier_module=model.fc,
        classifier_lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-4,
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 1e-3

    optimized_ids = _optimizer_parameter_ids(optimizer)

    expected_ids = {
        id(parameter)
        for parameter in model.layer4.parameters()
    } | {
        id(parameter)
        for parameter in model.fc.parameters()
    }

    assert optimized_ids == expected_ids


def test_efficientnet_b0_fine_tuning_optimizer_has_backbone_and_classifier_groups():
    model = build_pretrained_efficientnet_b0(num_classes=12)

    unfreeze_efficientnet_b0_features8(model)

    optimizer = build_transfer_optimizer(
        model,
        classifier_module=model.classifier,
        classifier_lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-4,
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 1e-3

    optimized_ids = _optimizer_parameter_ids(optimizer)

    expected_ids = {
        id(parameter)
        for parameter in model.features[8].parameters()
    } | {
        id(parameter)
        for parameter in model.classifier.parameters()
    }

    assert optimized_ids == expected_ids
