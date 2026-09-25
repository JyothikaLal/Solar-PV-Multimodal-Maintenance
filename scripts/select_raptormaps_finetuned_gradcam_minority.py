"""Select Task-28 minority-class samples for fine-tuned Grad-CAM."""

from pathlib import Path

import pandas as pd
import torch

from src.data.raptormaps_torch import create_raptormaps_dataloaders
from src.models.vision.raptormaps_transfer import RaptorMapsTransferAdapter
from src.models.vision.transfer_models import (
    build_pretrained_efficientnet_b0,
    build_pretrained_resnet18,
)


BATCH_SIZE = 64
NUM_WORKERS = 0

MINORITY_CLASSES = {
    "Diode-Multi",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Soiling",
}

MODEL_SPECS = {
    "resnet18_finetune": {
        "checkpoint": Path(
            "models/raptormaps/resnet18_finetune/best_model.pt"
        ),
        "builder": build_pretrained_resnet18,
        "classifier_attr": "fc",
    },
    "efficientnet_b0_finetune": {
        "checkpoint": Path(
            "models/raptormaps/efficientnet_b0_finetune/best_model.pt"
        ),
        "builder": build_pretrained_efficientnet_b0,
        "classifier_attr": "classifier",
    },
}

OUTPUT_PATH = Path(
    "reports/results/raptormaps/grad_cam/"
    "finetuned_minority_selection.csv"
)


def load_model(spec):
    checkpoint = torch.load(
        spec["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    backbone = spec["builder"](num_classes=12)

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=getattr(
            backbone,
            spec["classifier_attr"],
        ),
        spatial_size=(160, 96),
    )

    result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch."
        )

    model.eval()
    return model


def collect_predictions(
    model,
    test_loader,
    class_names,
    model_name,
):
    rows = []

    sample_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted = logits.argmax(dim=1)
            confidence = probabilities.max(dim=1).values

            for batch_index in range(images.shape[0]):
                true_index = int(
                    labels[batch_index].item()
                )
                predicted_index = int(
                    predicted[batch_index].item()
                )

                true_class = class_names[true_index]

                if true_class in MINORITY_CLASSES:
                    rows.append(
                        {
                            "model": model_name,
                            "sample_index": sample_index,
                            "true_index": true_index,
                            "true_class": true_class,
                            "predicted_index": predicted_index,
                            "predicted_class": class_names[
                                predicted_index
                            ],
                            "confidence": float(
                                confidence[
                                    batch_index
                                ].item()
                            ),
                            "correct": (
                                true_index
                                == predicted_index
                            ),
                        }
                    )

                sample_index += 1

    return rows


def select_samples(predictions):
    selected = []

    df = pd.DataFrame(predictions)

    for model_name in sorted(
        df["model"].unique()
    ):
        model_df = df[
            df["model"] == model_name
        ]

        for true_class in sorted(
            MINORITY_CLASSES
        ):
            class_df = model_df[
                model_df["true_class"]
                == true_class
            ]

            for status in [True, False]:
                candidates = class_df[
                    class_df["correct"]
                    == status
                ]

                if candidates.empty:
                    continue

                # For correct cases, highest confidence.
                # For incorrect cases, also highest confidence:
                # these are the strongest examples of the
                # model's actual decision.
                chosen = candidates.sort_values(
                    "confidence",
                    ascending=False,
                ).iloc[0]

                row = chosen.to_dict()
                row["selection_reason"] = (
                    "highest_confidence_correct"
                    if status
                    else "highest_confidence_error"
                )

                selected.append(row)

    return pd.DataFrame(selected)


def main():
    _, _, test_loader = create_raptormaps_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_train_augmentation=False,
    )

    class_names = test_loader.dataset.class_names

    all_predictions = []

    for model_name, spec in MODEL_SPECS.items():
        print(
            f"\nCollecting predictions: {model_name}"
        )

        model = load_model(spec)

        predictions = collect_predictions(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            model_name=model_name,
        )

        all_predictions.extend(predictions)

        print(
            "Minority test samples:",
            len(predictions),
        )

    selected = select_samples(
        all_predictions
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSelection saved:", OUTPUT_PATH)
    print("Shape:", selected.shape)

    print("\nSelection counts:")
    print(
        selected.groupby(
            ["model", "true_class", "correct"]
        ).size()
    )

    print("\nSelected samples:")
    print(
        selected.to_string(index=False)
    )


if __name__ == "__main__":
    main()
