from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.raptormaps_torch import (
    create_raptormaps_dataloaders,
)
from src.models.vision.raptormaps_transfer import (
    RaptorMapsTransferAdapter,
)
from src.models.vision.transfer_models import (
    build_pretrained_efficientnet_b0,
    build_pretrained_resnet18,
    load_transfer_checkpoint,
)


BATCH_SIZE = 64
NUM_WORKERS = 0

ROOT = Path(
    "reports/results/raptormaps"
)

MODEL_SPECS = {
    "resnet18_transfer": {
        "checkpoint": (
            Path(
                "models/raptormaps/"
                "resnet18_transfer/"
                "best_model.pt"
            )
        ),
        "output": (
            ROOT
            / "resnet18_transfer"
            / "test_predictions.csv"
        ),
        "builder": "resnet18",
    },
    "efficientnet_b0_transfer": {
        "checkpoint": (
            Path(
                "models/raptormaps/"
                "efficientnet_b0_transfer/"
                "best_model.pt"
            )
        ),
        "output": (
            ROOT
            / "efficientnet_b0_transfer"
            / "test_predictions.csv"
        ),
        "builder": "efficientnet",
    },
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_model(
    model_type: str,
    num_classes: int,
) -> torch.nn.Module:
    if model_type == "resnet18":
        backbone = build_pretrained_resnet18(
            num_classes=num_classes,
        )

        return RaptorMapsTransferAdapter(
            backbone=backbone,
            classifier_module=backbone.fc,
            spatial_size=(160, 96),
        )

    if model_type == "efficientnet":
        backbone = build_pretrained_efficientnet_b0(
            num_classes=num_classes,
        )

        return RaptorMapsTransferAdapter(
            backbone=backbone,
            classifier_module=backbone.classifier,
            spatial_size=(160, 96),
        )

    raise ValueError(
        f"Unknown model type: {model_type}"
    )


@torch.no_grad()
def export_predictions(
    model: torch.nn.Module,
    test_loader,
    class_names: list[str],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()

    rows = []
    sample_index = 0

    for images, targets in test_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        if not torch.isfinite(logits).all():
            raise ValueError(
                "Non-finite logits encountered."
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        confidence = torch.max(
            probabilities,
            dim=1,
        ).values

        true_indices = (
            targets.cpu().numpy()
        )
        predicted_indices = (
            predictions.cpu().numpy()
        )
        confidence_values = (
            confidence.cpu().numpy()
        )

        for offset in range(
            len(true_indices)
        ):
            true_index = int(
                true_indices[offset]
            )

            predicted_index = int(
                predicted_indices[offset]
            )

            rows.append(
                {
                    "sample_index": (
                        sample_index
                        + offset
                    ),
                    "y_true_index": true_index,
                    "y_pred_index": predicted_index,
                    "y_true_class": (
                        class_names[
                            true_index
                        ]
                    ),
                    "y_pred_class": (
                        class_names[
                            predicted_index
                        ]
                    ),
                    "confidence": float(
                        confidence_values[
                            offset
                        ]
                    ),
                }
            )

        sample_index += len(
            true_indices
        )

    result = pd.DataFrame(rows)

    expected_columns = [
        "sample_index",
        "y_true_index",
        "y_pred_index",
        "y_true_class",
        "y_pred_class",
        "confidence",
    ]

    result = result[
        expected_columns
    ]

    if len(result) != 2999:
        raise ValueError(
            "Expected 2999 test predictions; "
            f"got {len(result)}."
        )

    return result


def main() -> None:
    device = get_device()

    print(
        f"Using device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    _, _, test_loader = (
        create_raptormaps_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    )

    class_names = (
        test_loader.dataset.class_names
    )

    for model_name, spec in (
        MODEL_SPECS.items()
    ):
        checkpoint_path = spec[
            "checkpoint"
        ]

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: "
                f"{checkpoint_path}"
            )

        print()
        print(
            f"Exporting: {model_name}"
        )

        model = build_model(
            spec["builder"],
            len(class_names),
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        load_transfer_checkpoint(
            model,
            checkpoint[
                "model_state_dict"
            ],
        )

        model.to(device)

        predictions = export_predictions(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
        )

        output_path = spec[
            "output"
        ]

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions.to_csv(
            output_path,
            index=False,
        )

        accuracy = (
            predictions["y_true_index"]
            == predictions["y_pred_index"]
        ).mean()

        print(
            f"Test samples: {len(predictions)}"
        )

        print(
            f"Accuracy: {accuracy:.6f}"
        )

        print(
            "Mean confidence: "
            f"{predictions['confidence'].mean():.6f}"
        )

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    main()
