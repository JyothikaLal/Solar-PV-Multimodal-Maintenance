from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.data.raptormaps_torch import RaptorMapsTorchDataset
from src.data.raptormaps_classification import (
    IMAGE_ROOT,
    MANIFEST_PATH,
)


def compute_raptormaps_class_weights(
    dataset: RaptorMapsTorchDataset,
) -> torch.Tensor:
    """Compute balanced class weights from one dataset split."""

    labels = [
        dataset.class_to_index[anomaly_class]
        for anomaly_class in dataset.manifest["anomaly_class"]
    ]

    labels_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    num_classes = len(dataset.class_names)

    counts = np.bincount(
        labels_array,
        minlength=num_classes,
    )

    if np.any(counts == 0):
        raise ValueError(
            "Every class must have at least one training sample."
        )

    total_samples = len(labels_array)

    weights = total_samples / (
        num_classes * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def create_weighted_cross_entropy(
    class_weights: torch.Tensor,
) -> nn.CrossEntropyLoss:
    """Create weighted CrossEntropyLoss."""

    if class_weights.ndim != 1:
        raise ValueError(
            "class_weights must be a 1D tensor."
        )

    if len(class_weights) <= 1:
        raise ValueError(
            "class_weights must contain multiple classes."
        )

    if not torch.isfinite(class_weights).all():
        raise ValueError(
            "class_weights must contain only finite values."
        )

    if torch.any(class_weights <= 0):
        raise ValueError(
            "class_weights must be strictly positive."
        )

    return nn.CrossEntropyLoss(
        weight=class_weights,
    )


def create_raptormaps_weighted_loss() -> tuple[
    torch.Tensor,
    nn.CrossEntropyLoss,
]:
    """Create RaptorMaps training class weights and loss."""

    train_dataset = RaptorMapsTorchDataset(
        split="train",
        manifest_path=MANIFEST_PATH,
        image_root=IMAGE_ROOT,
    )

    class_weights = compute_raptormaps_class_weights(
        train_dataset,
    )

    loss_function = create_weighted_cross_entropy(
        class_weights,
    )

    return class_weights, loss_function
