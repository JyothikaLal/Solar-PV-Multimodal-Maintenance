from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.raptormaps_torch import (
    create_raptormaps_dataloaders,
)
from src.evaluation.classification import (
    evaluate_classification,
)
from src.models.vision.raptormaps_transfer import (
    RaptorMapsTransferAdapter,
)
from src.models.vision.transfer_models import (
    build_pretrained_resnet18,
)


BATCH_SIZE = 64
NUM_WORKERS = 0

MODEL_PATH = Path(
    "models/raptormaps/resnet18_transfer/best_model.pt"
)

RESULTS_DIR = Path(
    "reports/results/raptormaps/resnet18_transfer"
)

OVERALL_PATH = (
    RESULTS_DIR / "test_overall_metrics.csv"
)

PER_CLASS_PATH = (
    RESULTS_DIR / "test_per_class_metrics.csv"
)

CONFUSION_PATH = (
    RESULTS_DIR / "test_confusion_matrix.csv"
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


@torch.no_grad()
def predict_test_set(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

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

        all_targets.append(
            targets.numpy()
        )
        all_predictions.append(
            predictions.cpu().numpy()
        )
        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    y_proba = np.concatenate(all_probabilities)

    return y_true, y_pred, y_proba


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
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

    backbone = build_pretrained_resnet18(
        num_classes=len(class_names),
    )

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        spatial_size=(160, 96),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    print(
        "Loaded checkpoint epoch:",
        checkpoint["epoch"],
    )
    print(
        "Selected validation Macro F1:",
        f"{checkpoint['best_validation_macro_f1']:.6f}",
    )

    y_true, y_pred, y_proba = (
        predict_test_set(
            model=model,
            test_loader=test_loader,
            device=device,
        )
    )

    evaluation = evaluate_classification(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        y_proba=y_proba,
    )

    overall = evaluation["overall"]
    per_class = evaluation["per_class"]
    confusion_matrix = evaluation[
        "confusion_matrix"
    ]

    pd.DataFrame(
        [overall]
    ).to_csv(
        OVERALL_PATH,
        index=False,
    )

    per_class.to_csv(
        PER_CLASS_PATH,
        index=False,
    )

    pd.DataFrame(
        confusion_matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(
        CONFUSION_PATH
    )

    print()
    print("Test evaluation complete.")
    print(
        f"Test samples: {len(y_true)}"
    )

    print()
    print("Overall metrics:")
    for metric, value in overall.items():
        print(
            f"{metric}: {value:.6f}"
        )

    print()
    print("Per-class metrics:")
    print(
        per_class.to_string(
            index=False
        )
    )

    print()
    print(
        f"Overall: {OVERALL_PATH}"
    )
    print(
        f"Per-class: {PER_CLASS_PATH}"
    )
    print(
        f"Confusion: {CONFUSION_PATH}"
    )


if __name__ == "__main__":
    main()
