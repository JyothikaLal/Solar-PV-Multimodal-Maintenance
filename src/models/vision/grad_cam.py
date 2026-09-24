"""Grad-CAM utilities for RaptorMaps vision models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM maps for a classification model."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module.")

        if not isinstance(target_layer, nn.Module):
            raise TypeError("target_layer must be an nn.Module.")

        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_handle = target_layer.register_forward_hook(
            self._forward_hook,
        )

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if output.ndim != 4:
            raise ValueError(
                "Grad-CAM target layer must produce a 4-D tensor "
                "[batch, channels, height, width]."
            )

        output.retain_grad()

        self.activations = output

    def _validate_target_classes(
        self,
        target_classes: torch.Tensor,
        batch_size: int,
        num_classes: int,
    ) -> torch.Tensor:
        if target_classes.ndim != 1:
            raise ValueError(
                "target_classes must have shape [batch]."
            )

        if target_classes.shape[0] != batch_size:
            raise ValueError(
                "target_classes batch size must match model output."
            )

        target_classes = target_classes.to(
            device=self.activations.device,
            dtype=torch.long,
        )

        if torch.any(target_classes < 0):
            raise ValueError(
                "target_classes must contain non-negative class indices."
            )

        if torch.any(target_classes >= num_classes):
            raise ValueError(
                "target_classes contains an index outside "
                "the model output range."
            )

        return target_classes

    @staticmethod
    def _normalize(
        cams: torch.Tensor,
    ) -> torch.Tensor:
        if cams.ndim != 3:
            raise ValueError(
                "cams must have shape [batch, height, width]."
            )

        batch_size = cams.shape[0]

        flat = cams.reshape(batch_size, -1)

        minimum = flat.min(dim=1).values.view(
            batch_size,
            1,
            1,
        )

        maximum = flat.max(dim=1).values.view(
            batch_size,
            1,
            1,
        )

        denominator = maximum - minimum

        normalized = (cams - minimum) / denominator.clamp_min(
            torch.finfo(cams.dtype).eps,
        )

        constant_mask = denominator <= torch.finfo(cams.dtype).eps

        if constant_mask.any():
            normalized = torch.where(
                constant_mask,
                torch.zeros_like(normalized),
                normalized,
            )

        return normalized

    def __call__(
        self,
        inputs: torch.Tensor,
        *,
        target_classes: Optional[torch.Tensor] = None,
        output_size: Optional[tuple[int, int]] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute Grad-CAM maps for a batch of inputs."""

        if inputs.ndim != 4:
            raise ValueError(
                "inputs must have shape [batch, channels, height, width]."
            )

        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        logits = self.model(inputs)

        if logits.ndim != 2:
            raise ValueError(
                "Model output must have shape [batch, num_classes]."
            )

        batch_size, num_classes = logits.shape

        if target_classes is None:
            selected_classes = logits.argmax(dim=1)
        else:
            if self.activations is None:
                raise RuntimeError(
                    "Target-layer activations were not captured."
                )

            selected_classes = self._validate_target_classes(
                target_classes,
                batch_size,
                num_classes,
            )

        if self.activations is None:
            raise RuntimeError(
                "Target-layer activations were not captured."
            )

        selected_scores = logits[
            torch.arange(
                batch_size,
                device=logits.device,
            ),
            selected_classes,
        ]

        selected_scores.sum().backward()

        self.gradients = self.activations.grad

        if self.gradients is None:
            raise RuntimeError(
                "Target-layer gradients were not captured."
            )

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cams = (
            weights * activations
        ).sum(dim=1)

        cams = F.relu(cams)
        cams = self._normalize(cams)

        if output_size is None:
            output_size = (
                inputs.shape[2],
                inputs.shape[3],
            )

        cams = F.interpolate(
            cams.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        cams = cams.clamp(0.0, 1.0)

        return {
            "heatmaps": cams.detach(),
            "logits": logits.detach(),
            "predicted_classes": logits.argmax(dim=1).detach(),
            "target_classes": selected_classes.detach(),
            "target_scores": selected_scores.detach(),
        }

    def remove_hooks(self) -> None:
        """Remove registered forward hooks."""

        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None

    def __del__(self) -> None:
        self.remove_hooks()
