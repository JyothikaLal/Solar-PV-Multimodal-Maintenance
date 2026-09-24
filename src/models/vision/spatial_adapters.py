from __future__ import annotations

import torch
import torch.nn.functional as F


SUPPORTED_SIZES = {
    "native": (40, 24),
    "2x": (80, 48),
    "4x": (160, 96),
}


def resize_for_transfer_learning(
    x: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Resize BCHW tensors while preserving explicit target size."""

    if x.ndim != 4:
        raise ValueError(
            "Expected [B,C,H,W] tensor, "
            f"got {tuple(x.shape)}."
        )

    if x.shape[1] != 3:
        raise ValueError(
            "Transfer-learning spatial adapter expects "
            f"3 channels, got {x.shape[1]}."
        )

    if not torch.isfinite(x).all():
        raise ValueError(
            "Input contains non-finite values."
        )

    height, width = size

    if height <= 0 or width <= 0:
        raise ValueError(
            "Target height and width must be positive."
        )

    return F.interpolate(
        x,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
