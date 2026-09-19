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
from src.models.vision.custom_cnn import (
    RaptorMapsCustomCNN,
)


CHECKPOINT_PATH = Path(
    "models/raptormaps/custom_cnn/best_model.pt"
)

RESULTS_DIR = Path(
    "reports/results/raptormaps/custom_cnn"
)

OVERALL_RESULTS_PATH = (
    RESULTS_DIR / "test_overall_metrics.csv"
)

PER_CLASS_RESULTS_PATH = (
    RESULTS_DIR / "test_per_class_metrics.csv"
)

CONFUSION_MATRIX_PATH = (
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
    """Generate test predictions and probabilities."""

    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

    for images, targets in test_loader:
        images = images.to(device)

        logits = model(images)

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

    y_true = np.concatenate(
        all_targets
    )

    y_pred = np.concatenate(
        all_predictions
    )

    y_proba = np.concatenate(
        all_probabilities
    )

    return y_true, y_pred, y_proba


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    _, _, test_loader = (
        create_raptormaps_dataloaders(
            batch_size=64,
            num_workers=0,
        )
    )

    class_names = (
        test_loader.dataset.class_names
    )

    model = RaptorMapsCustomCNN(
        num_classes=len(class_names),
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    print(
        "Loaded checkpoint from epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        "Validation Macro F1 at selection: "
        f"{checkpoint['best_validation_macro_f1']:.4f}"
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

    overall_df = pd.DataFrame(
        [
            overall
        ]
    )

    overall_df.to_csv(
        OVERALL_RESULTS_PATH,
        index=False,
    )

    per_class.to_csv(
        PER_CLASS_RESULTS_PATH,
        index=False,
    )

    confusion_df = pd.DataFrame(
        confusion_matrix,
        index=class_names,
        columns=class_names,
    )

    confusion_df.to_csv(
        CONFUSION_MATRIX_PATH
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
            f"{metric}: {value:.4f}"
        )

    print()
    print("Per-class metrics:")
    print(per_class.to_string(index=False))

    print()
    print(
        f"Overall results: "
        f"{OVERALL_RESULTS_PATH}"
    )

    print(
        f"Per-class results: "
        f"{PER_CLASS_RESULTS_PATH}"
    )

    print(
        f"Confusion matrix: "
        f"{CONFUSION_MATRIX_PATH}"
    )


if __name__ == "__main__":
    main()
