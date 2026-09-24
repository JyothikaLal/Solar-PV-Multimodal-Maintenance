import torch
from torch import nn

from src.models.vision.raptormaps_transfer import (
    RaptorMapsTransferAdapter,
)

from src.models.vision.transfer_models import (
    build_pretrained_efficientnet_b0,
    build_pretrained_resnet18,
    freeze_efficientnet_b0_backbone,
    freeze_resnet18_backbone,
    unfreeze_efficientnet_b0_features8,
    unfreeze_resnet18_layer4,
)


def test_transfer_preprocessing_output_shape():
    backbone = build_pretrained_resnet18(
        num_classes=12,
    )
    freeze_resnet18_backbone(backbone)

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.fc,
    )

    inputs = torch.rand(
        4,
        1,
        40,
        24,
    )

    outputs = model.preprocess(inputs)

    assert outputs.shape == (
        4,
        3,
        160,
        96,
    )

    assert torch.isfinite(outputs).all()


def test_transfer_model_output_shape():
    backbone = build_pretrained_resnet18(
        num_classes=12,
    )
    freeze_resnet18_backbone(backbone)

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.fc,
    )

    inputs = torch.rand(
        2,
        1,
        40,
        24,
    )

    outputs = model(inputs)

    assert outputs.shape == (
        2,
        12,
    )

    assert torch.isfinite(outputs).all()


def test_transfer_model_keeps_frozen_backbone_in_eval_mode():
    backbone = build_pretrained_resnet18(
        num_classes=12,
    )
    freeze_resnet18_backbone(backbone)

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.fc,
    )

    model.train()

    assert not model.backbone.training
    assert model.backbone.fc.training


def test_transfer_model_classifier_remains_trainable():
    backbone = build_pretrained_resnet18(
        num_classes=12,
    )
    freeze_resnet18_backbone(backbone)

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.fc,
    )

    model.train()

    backbone_parameters = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if not name.startswith("fc.")
    ]

    classifier_parameters = list(
        model.backbone.fc.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter in backbone_parameters
    )

    assert all(
        parameter.requires_grad
        for parameter in classifier_parameters
    )


def test_efficientnet_transfer_model_output_shape():
    from src.models.vision.transfer_models import (
        build_pretrained_efficientnet_b0,
        freeze_efficientnet_b0_backbone,
    )

    backbone = build_pretrained_efficientnet_b0(
        num_classes=12,
    )

    freeze_efficientnet_b0_backbone(
        backbone
    )

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.classifier,
    )

    outputs = model(
        torch.rand(
            2,
            1,
            40,
            24,
        )
    )

    assert outputs.shape == (
        2,
        12,
    )

    assert torch.isfinite(outputs).all()


def test_efficientnet_frozen_backbone_train_mode():
    from src.models.vision.transfer_models import (
        build_pretrained_efficientnet_b0,
        freeze_efficientnet_b0_backbone,
    )

    backbone = build_pretrained_efficientnet_b0(
        num_classes=12,
    )

    freeze_efficientnet_b0_backbone(
        backbone
    )

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.classifier,
    )

    model.train()

    assert not model.backbone.training
    assert model.classifier_module.training


def test_resnet_adapter_does_not_duplicate_classifier_in_state_dict():
    from src.models.vision.transfer_models import (
        build_pretrained_resnet18,
        freeze_resnet18_backbone,
    )

    backbone = build_pretrained_resnet18(
        num_classes=12,
    )

    freeze_resnet18_backbone(
        backbone
    )

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.fc,
    )

    keys = set(
        model.state_dict().keys()
    )

    assert "backbone.fc.weight" in keys
    assert "backbone.fc.bias" in keys
    assert "classifier_module.weight" not in keys
    assert "classifier_module.bias" not in keys


def test_efficientnet_adapter_does_not_duplicate_classifier_in_state_dict():
    from src.models.vision.transfer_models import (
        build_pretrained_efficientnet_b0,
        freeze_efficientnet_b0_backbone,
    )

    backbone = build_pretrained_efficientnet_b0(
        num_classes=12,
    )

    freeze_efficientnet_b0_backbone(
        backbone
    )

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.classifier,
    )

    keys = set(
        model.state_dict().keys()
    )

    assert (
        "backbone.classifier.1.weight"
        in keys
    )

    assert (
        "backbone.classifier.1.bias"
        in keys
    )

    assert (
        "classifier_module.1.weight"
        not in keys
    )

    assert (
        "classifier_module.1.bias"
        not in keys
    )

def test_resnet18_fine_tuning_keeps_frozen_batchnorm_in_eval_mode():
    backbone = build_pretrained_resnet18(num_classes=12)

    unfreeze_resnet18_layer4(backbone)

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.fc,
    )

    model.train()

    assert model.backbone.layer4.training
    assert model.backbone.fc.training

    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if name.startswith("layer4."):
                assert module.training
            else:
                assert not module.training


def test_efficientnet_b0_fine_tuning_keeps_frozen_batchnorm_in_eval_mode():
    backbone = build_pretrained_efficientnet_b0(num_classes=12)

    unfreeze_efficientnet_b0_features8(backbone)

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.classifier,
    )

    model.train()

    assert model.backbone.features[8].training
    assert model.classifier_module.training

    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if name.startswith("features.8."):
                assert module.training
            else:
                assert not module.training
