from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

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
    build_pretrained_resnet18,
)


BATCH_SIZE = 64
NUM_WORKERS = 0

MODEL_PATH = Path(
    "models/raptormaps/resnet18_finetune/best_model.pt"
)

OUTPUT_DIR = Path(
    "reports/results/raptormaps/resnet18_finetune/validation"
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


@torch.no_grad()
def predict_validation_set(
    model: torch.nn.Module,
    validation_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

    for images, targets in validation_loader:
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

    return (
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(f"Using device: {device}")

    (
        _,
        validation_loader,
        _,
    ) = create_raptormaps_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    class_names = (
        validation_loader.dataset.class_names
    )

    backbone = build_pretrained_resnet18(
        num_classes=len(class_names),
    )

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.fc,
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

    (
        y_true,
        y_pred,
        y_proba,
    ) = predict_validation_set(
        model=model,
        validation_loader=validation_loader,
        device=device,
    )

    if not np.isfinite(y_proba).all():
        raise ValueError(
            "Validation probabilities contain "
            "non-finite values."
        )

    probability_sums = y_proba.sum(axis=1)

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Validation probabilities do not "
            "sum to 1."
        )

    predicted_probability = y_proba[
        np.arange(len(y_pred)),
        y_pred,
    ]

    no_anomaly_index = class_names.index(
        "No-Anomaly"
    )

    no_anomaly_probability = y_proba[
        :,
        no_anomaly_index,
    ]

    anomaly_evidence = (
        1.0 - no_anomaly_probability
    )

    output = pd.DataFrame(
        {
            "sample_index": np.arange(
                len(y_true)
            ),
            "true_class_index": y_true,
            "true_class": [
                class_names[index]
                for index in y_true
            ],
            "predicted_class_index": y_pred,
            "predicted_class": [
                class_names[index]
                for index in y_pred
            ],
            "confidence": predicted_probability,
            "no_anomaly_probability": (
                no_anomaly_probability
            ),
            "anomaly_evidence": anomaly_evidence,
        }
    )

    for index, class_name in enumerate(
        class_names
    ):
        output[
            f"prob_{class_name}"
        ] = y_proba[:, index]

    output["model"] = "resnet18_finetuned"
    output["checkpoint_epoch"] = checkpoint[
        "epoch"
    ]
    output["best_validation_macro_f1"] = (
        checkpoint[
            "best_validation_macro_f1"
        ]
    )

    prediction_path = (
        OUTPUT_DIR
        / "validation_probabilities.csv"
    )

    output.to_csv(
        prediction_path,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "model": "resnet18_finetuned",
                "checkpoint_epoch": checkpoint[
                    "epoch"
                ],
                "best_validation_macro_f1": (
                    checkpoint[
                        "best_validation_macro_f1"
                    ]
                ),
                "n": len(output),
                "accuracy": float(
                    (
                        y_true == y_pred
                    ).mean()
                ),
                "mean_confidence": float(
                    predicted_probability.mean()
                ),
                "median_confidence": float(
                    np.median(
                        predicted_probability
                    )
                ),
                "mean_no_anomaly_probability": (
                    float(
                        no_anomaly_probability.mean()
                    )
                ),
                "mean_anomaly_evidence": float(
                    anomaly_evidence.mean()
                ),
                "q50_anomaly_evidence": float(
                    np.quantile(
                        anomaly_evidence,
                        0.50,
                    )
                ),
                "q75_anomaly_evidence": float(
                    np.quantile(
                        anomaly_evidence,
                        0.75,
                    )
                ),
                "q90_anomaly_evidence": float(
                    np.quantile(
                        anomaly_evidence,
                        0.90,
                    )
                ),
                "q95_anomaly_evidence": float(
                    np.quantile(
                        anomaly_evidence,
                        0.95,
                    )
                ),
                "q99_anomaly_evidence": float(
                    np.quantile(
                        anomaly_evidence,
                        0.99,
                    )
                ),
            }
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "validation_probability_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    class_summary = (
        output.groupby(
            "predicted_class",
            dropna=False,
        )
        .agg(
            count=(
                "predicted_class",
                "size",
            ),
            mean_confidence=(
                "confidence",
                "mean",
            ),
            median_confidence=(
                "confidence",
                "median",
            ),
            mean_anomaly_evidence=(
                "anomaly_evidence",
                "mean",
            ),
        )
        .reset_index()
    )

    class_summary.to_csv(
        OUTPUT_DIR
        / "validation_predicted_class_summary.csv",
        index=False,
    )

    print(
        "\nValidation probability export completed."
    )
    print(
        f"Validation rows: {len(output)}"
    )
    print(
        f"Number of classes: {len(class_names)}"
    )
    print(
        f"Accuracy: "
        f"{((y_true == y_pred).mean()):.6f}"
    )
    print(
        f"Mean confidence: "
        f"{predicted_probability.mean():.6f}"
    )
    print(
        f"Mean P(No-Anomaly): "
        f"{no_anomaly_probability.mean():.6f}"
    )
    print(
        f"Mean anomaly evidence: "
        f"{anomaly_evidence.mean():.6f}"
    )
    print(
        f"Predictions: {prediction_path}"
    )
    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
