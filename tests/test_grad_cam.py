import torch
import torch.nn as nn
import pytest

from src.models.vision.custom_cnn import RaptorMapsCustomCNN
from src.models.vision.grad_cam import GradCAM
from src.models.vision.transfer_models import (
    build_pretrained_efficientnet_b0,
    build_pretrained_resnet18,
)


def test_grad_cam_custom_cnn_output():
    model = RaptorMapsCustomCNN(num_classes=12)
    model.eval()

    target_layer = model.features[8]

    grad_cam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    inputs = torch.rand(
        2,
        1,
        40,
        24,
    )

    result = grad_cam(
        inputs,
        output_size=(40, 24),
    )

    assert result["heatmaps"].shape == (
        2,
        40,
        24,
    )

    assert result["logits"].shape == (
        2,
        12,
    )

    assert result["predicted_classes"].shape == (2,)
    assert result["target_classes"].shape == (2,)
    assert result["target_scores"].shape == (2,)

    assert torch.isfinite(result["heatmaps"]).all()
    assert torch.isfinite(result["logits"]).all()

    assert torch.all(result["heatmaps"] >= 0.0)
    assert torch.all(result["heatmaps"] <= 1.0)

    grad_cam.remove_hooks()


@pytest.mark.parametrize(
    "builder,target_getter",
    [
        (
            build_pretrained_resnet18,
            lambda model: model.layer4[1].conv2,
        ),
        (
            build_pretrained_efficientnet_b0,
            lambda model: model.features[8][0],
        ),
    ],
)
def test_grad_cam_transfer_models(builder, target_getter):
    model = builder(num_classes=12)
    model.eval()

    grad_cam = GradCAM(
        model=model,
        target_layer=target_getter(model),
    )

    inputs = torch.rand(
        2,
        3,
        96,
        160,
    )

    result = grad_cam(
        inputs,
        output_size=(96, 160),
    )

    assert result["heatmaps"].shape == (
        2,
        96,
        160,
    )

    assert result["logits"].shape == (
        2,
        12,
    )

    assert torch.isfinite(result["heatmaps"]).all()
    assert torch.isfinite(result["logits"]).all()

    assert torch.all(result["heatmaps"] >= 0.0)
    assert torch.all(result["heatmaps"] <= 1.0)

    grad_cam.remove_hooks()


def test_grad_cam_explicit_target_classes():
    model = RaptorMapsCustomCNN(num_classes=12)
    model.eval()

    grad_cam = GradCAM(
        model=model,
        target_layer=model.features[8],
    )

    inputs = torch.rand(
        2,
        1,
        40,
        24,
    )

    target_classes = torch.tensor(
        [2, 7],
        dtype=torch.long,
    )

    result = grad_cam(
        inputs,
        target_classes=target_classes,
        output_size=(40, 24),
    )

    assert torch.equal(
        result["target_classes"],
        target_classes,
    )

    assert result["heatmaps"].shape == (
        2,
        40,
        24,
    )

    assert torch.isfinite(result["heatmaps"]).all()

    grad_cam.remove_hooks()


def test_grad_cam_rejects_invalid_target_class():
    model = RaptorMapsCustomCNN(num_classes=12)
    model.eval()

    grad_cam = GradCAM(
        model=model,
        target_layer=model.features[8],
    )

    inputs = torch.rand(
        1,
        1,
        40,
        24,
    )

    with pytest.raises(ValueError):
        grad_cam(
            inputs,
            target_classes=torch.tensor([12]),
        )

    grad_cam.remove_hooks()
