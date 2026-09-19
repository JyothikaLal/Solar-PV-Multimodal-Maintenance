import torch

from src.data.raptormaps_torch import RaptorMapsTorchDataset
from src.models.vision.losses import (
    compute_raptormaps_class_weights,
    create_weighted_cross_entropy,
)


def test_class_weights_have_expected_shape():
    dataset = RaptorMapsTorchDataset("train")

    weights = compute_raptormaps_class_weights(
        dataset
    )

    assert weights.shape == (
        len(dataset.class_names),
    )

    assert weights.dtype == torch.float32


def test_class_weights_are_positive_and_finite():
    dataset = RaptorMapsTorchDataset("train")

    weights = compute_raptormaps_class_weights(
        dataset
    )

    assert torch.isfinite(weights).all()
    assert torch.all(weights > 0)


def test_minority_classes_receive_larger_weights():
    dataset = RaptorMapsTorchDataset("train")

    weights = compute_raptormaps_class_weights(
        dataset
    )

    class_to_index = dataset.class_to_index

    assert (
        weights[class_to_index["Hot-Spot"]]
        > weights[class_to_index["No-Anomaly"]]
    )

    assert (
        weights[class_to_index["Diode-Multi"]]
        > weights[class_to_index["Diode"]]
    )


def test_weighted_cross_entropy_creation():
    dataset = RaptorMapsTorchDataset("train")

    weights = compute_raptormaps_class_weights(
        dataset
    )

    loss_function = create_weighted_cross_entropy(
        weights
    )

    assert isinstance(
        loss_function,
        torch.nn.CrossEntropyLoss,
    )


def test_weighted_cross_entropy_produces_finite_loss():
    dataset = RaptorMapsTorchDataset("train")

    weights = compute_raptormaps_class_weights(
        dataset
    )

    loss_function = create_weighted_cross_entropy(
        weights
    )

    logits = torch.randn(
        8,
        len(dataset.class_names),
    )

    targets = torch.randint(
        0,
        len(dataset.class_names),
        (8,),
    )

    loss = loss_function(
        logits,
        targets,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0
