from __future__ import annotations

import torch
from torch import nn


NUM_CLASSES = 12
EMBEDDING_DIM = 128


class RaptorMapsCustomCNN(nn.Module):
    """Small custom CNN for RaptorMaps grayscale images."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1."
            )

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.3)

        self.classifier = nn.Linear(
            EMBEDDING_DIM,
            num_classes,
        )

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Return the penultimate image representation."""

        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Return classification logits."""

        embeddings = self.forward_features(x)
        embeddings = self.dropout(embeddings)

        return self.classifier(embeddings)
