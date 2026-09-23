from __future__ import annotations

import torch
from torch import nn


class RaptorMapsTrainAugmentation(nn.Module):
    """Conservative train-only augmentation for RaptorMaps."""

    def __init__(
        self,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        super().__init__()

        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError(
                "horizontal_flip_probability must be "
                "between 0 and 1."
            )

        self.horizontal_flip_probability = (
            horizontal_flip_probability
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Apply training augmentation to [C,H,W] image."""

        if image.ndim != 3:
            raise ValueError(
                "Expected image shape [C,H,W], "
                f"got {tuple(image.shape)}."
            )

        if not torch.isfinite(image).all():
            raise ValueError(
                "Input image contains non-finite values."
            )

        if torch.rand(
            1,
            device=image.device,
        ).item() < self.horizontal_flip_probability:
            image = torch.flip(
                image,
                dims=[-1],
            )

        return image
