from __future__ import annotations

import torch
from torch import nn


class GrayscaleToRGB(nn.Module):
    """Adapt single-channel RaptorMaps tensors to 3 channels."""

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Convert [B,1,H,W] grayscale to [B,3,H,W]."""

        if x.ndim != 4:
            raise ValueError(
                "Expected a 4D tensor [B,C,H,W], "
                f"got shape {tuple(x.shape)}."
            )

        if x.shape[1] != 1:
            raise ValueError(
                "GrayscaleToRGB expects exactly one "
                f"input channel, got {x.shape[1]}."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "Input contains non-finite values."
            )

        return x.repeat(1, 3, 1, 1)
