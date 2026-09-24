from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


RAPTOR_MAPS_MEAN = 0.61973207
RAPTOR_MAPS_STD = 0.15437313

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


def _set_frozen_batchnorm_eval(module: nn.Module) -> None:
    """Keep BatchNorm layers with no trainable parameters in eval mode."""
    for child in module.modules():
        if isinstance(
            child,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.SyncBatchNorm,
            ),
        ):
            parameters = list(child.parameters())

            if parameters and not any(
                parameter.requires_grad
                for parameter in parameters
            ):
                child.eval()


class RaptorMapsTransferAdapter(nn.Module):
    """Architecture-independent input adapter for RaptorMaps transfer learning."""

    def __init__(
        self,
        backbone: nn.Module,
        classifier_module: nn.Module,
        spatial_size: tuple[int, int] = (160, 96),
    ) -> None:
        super().__init__()

        if spatial_size[0] <= 0 or spatial_size[1] <= 0:
            raise ValueError(
                "spatial_size dimensions must be positive."
            )

        self.backbone = backbone

        # Keep a non-registered reference to the classifier.
        # The classifier is already owned by ``backbone`` and must
        # not appear twice in the module state_dict.
        object.__setattr__(
            self,
            "classifier_module",
            classifier_module,
        )

        self.spatial_size = spatial_size

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
        )

    def preprocess(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Convert standardized grayscale RaptorMaps tensors to transfer input."""

        if x.ndim != 4:
            raise ValueError(
                "Expected input shape [B, 1, H, W]."
            )

        if x.shape[1] != 1:
            raise ValueError(
                "Expected exactly one grayscale input channel."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "Input contains non-finite values."
            )

        # Undo RaptorMaps train-set standardization.
        x = (
            x * RAPTOR_MAPS_STD
            + RAPTOR_MAPS_MEAN
        )

        x = torch.clamp(
            x,
            min=0.0,
            max=1.0,
        )

        # Replicate grayscale into three channels so the
        # pretrained first convolution remains unchanged.
        x = x.repeat(1, 3, 1, 1)

        x = F.interpolate(
            x,
            size=self.spatial_size,
            mode="bilinear",
            align_corners=False,
        )

        # Apply ImageNet normalization expected by pretrained weights.
        x = (
            x - self.imagenet_mean
        ) / self.imagenet_std

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(
            self.preprocess(x)
        )

    def train(
        self,
        mode: bool = True,
    ) -> RaptorMapsTransferAdapter:
        """Set training mode while preserving frozen-backbone behavior."""

        super().train(mode)

        if not mode:
            self.backbone.eval()
            self.classifier_module.eval()
            return self

        classifier_ids = {
            id(parameter)
            for parameter in self.classifier_module.parameters()
        }

        backbone_feature_parameters = [
            parameter
            for parameter in self.backbone.parameters()
            if id(parameter) not in classifier_ids
        ]

        backbone_is_trainable = any(
            parameter.requires_grad
            for parameter in backbone_feature_parameters
        )

        if backbone_is_trainable:
            self.backbone.train()
            _set_frozen_batchnorm_eval(self.backbone)
        else:
            self.backbone.eval()

        self.classifier_module.train()

        return self
