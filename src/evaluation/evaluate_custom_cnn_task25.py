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
    "models/raptormaps/custom_cnn_task25/"
    "best_model.pt"
)

RESULTS_DIR = Path(
    "reports/results/raptormaps/custom_cnn_task25"
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

PREDICTIONS_PATH = (
    RESULTS_DIR / "test_predictions.csv"
)


def get_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


@torch.no_grad()
def predict_test_set(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate predictions and probabilities."""

    model.eval()

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for images, targets in test_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

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
            f"Task 25 checkpoint not found: "
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
            use_train_augmentation=False,
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

    required_checkpoint_keys = {
        "model_state_dict",
        "best_validation_macro_f1",
        "epoch",
        "class_names",
        "config",
    }

    missing_keys = (
        required_checkpoint_keys
        - set(checkpoint.keys())
    )

    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    if checkpoint["class_names"] != class_names:
        raise ValueError(
            "Checkpoint class_names do not match "
            "the test dataset class mapping."
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

    expected_test_samples = (
        len(test_loader.dataset)
    )

    if len(y_true) != expected_test_samples:
        raise ValueError(
            "Unexpected number of test predictions: "
            f"{len(y_true)}. "
            f"Expected {expected_test_samples}."
        )

    if not np.isfinite(y_proba).all():
        raise ValueError(
            "Test probabilities contain non-finite values."
        )

    probability_sums = y_proba.sum(axis=1)

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Predicted class probabilities do not "
            "sum to 1."
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
        [overall]
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

    prediction_df = pd.DataFrame(
        {
            "sample_index": np.arange(
                len(y_true)
            ),
            "y_true_index": y_true,
            "y_pred_index": y_pred,
            "y_true_class": [
                class_names[index]
                for index in y_true
            ],
            "y_pred_class": [
                class_names[index]
                for index in y_pred
            ],
            "confidence": y_proba.max(
                axis=1
            ),
        }
    )

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print()
    print("Task 25 test evaluation complete.")
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
    print(
        per_class.to_string(
            index=False
        )
    )

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

    print(
        f"Predictions: "
        f"{PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()
