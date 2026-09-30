from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.mlops.retraining.training import _evaluate_raptormaps_test_set


def test_raptormaps_test_evaluation_returns_complete_registry_metrics():
    class_names = ["class_a", "class_b", "class_c"]

    images = torch.randn(6, 1, 4, 4)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])

    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros((x.shape[0], 3))
            logits[:, 0] = 3.0
            logits[:, 1] = 2.0
            logits[:, 2] = 1.0
            return logits

    metrics = _evaluate_raptormaps_test_set(
        model=DummyModel(),
        loader=loader,
        device=torch.device("cpu"),
        class_names=class_names,
    )

    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "pr_auc_macro",
    }

    assert all(np.isfinite(value) for value in metrics.values())
