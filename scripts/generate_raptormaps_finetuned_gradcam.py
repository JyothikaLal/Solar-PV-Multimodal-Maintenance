"""Generate Grad-CAM explanations for Task 27 fine-tuned RaptorMaps models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.raptormaps_torch import create_raptormaps_dataloaders
from src.models.vision.grad_cam import GradCAM
from src.models.vision.raptormaps_transfer import RaptorMapsTransferAdapter
from src.models.vision.transfer_models import (
    build_pretrained_efficientnet_b0,
    build_pretrained_resnet18,
)


SEED = 42
NUM_CORRECT = 8
NUM_INCORRECT = 8
BATCH_SIZE = 64
NUM_WORKERS = 0

OUTPUT_ROOT = Path(
    "reports/figures/raptormaps/grad_cam"
)

RESULTS_ROOT = Path(
    "reports/results/raptormaps/grad_cam"
)

MODEL_SPECS = {
    "resnet18_finetune": {
        "checkpoint": Path(
            "models/raptormaps/resnet18_finetune/best_model.pt"
        ),
        "builder": build_pretrained_resnet18,
        "classifier_attr": "fc",
        "target_layer_name": "backbone.layer4[1].conv2",
        "target_layer_getter": (
            lambda model: model.backbone.layer4[1].conv2
        ),
    },
    "efficientnet_b0_finetune": {
        "checkpoint": Path(
            "models/raptormaps/efficientnet_b0_finetune/best_model.pt"
        ),
        "builder": build_pretrained_efficientnet_b0,
        "classifier_attr": "classifier",
        "target_layer_name": "backbone.features[8][0]",
        "target_layer_getter": (
            lambda model: model.backbone.features[8][0]
        ),
    },
}


def set_seed(seed: int) -> None:
    """Set deterministic seeds for sample selection."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(spec: dict) -> torch.nn.Module:
    """Build the Task 27 adapter and load its fine-tuned checkpoint."""
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
            "Checkpoint/model mismatch: "
            f"missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )

    model.eval()
    return model


def select_representative_samples(
    predictions: list[dict],
) -> list[dict]:
    """Select balanced correct/incorrect representative examples."""
    correct = [
        item for item in predictions
        if item["correct"]
    ]
    incorrect = [
        item for item in predictions
        if not item["correct"]
    ]

    if len(correct) < NUM_CORRECT:
        raise RuntimeError(
            f"Only {len(correct)} correct samples available; "
            f"need {NUM_CORRECT}."
        )

    if len(incorrect) < NUM_INCORRECT:
        raise RuntimeError(
            f"Only {len(incorrect)} incorrect samples available; "
            f"need {NUM_INCORRECT}."
        )

    rng = np.random.default_rng(SEED)

    selected_correct = rng.choice(
        correct,
        size=NUM_CORRECT,
        replace=False,
    ).tolist()

    selected_incorrect = rng.choice(
        incorrect,
        size=NUM_INCORRECT,
        replace=False,
    ).tolist()

    selected = selected_correct + selected_incorrect

    selected.sort(key=lambda item: item["sample_index"])

    return selected


def collect_predictions(
    model: torch.nn.Module,
    test_loader,
) -> list[dict]:
    """Run the model over the complete frozen test set."""
    predictions = []

    sample_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            predicted = logits.argmax(dim=1)
            confidence = probabilities.max(dim=1).values

            for batch_index in range(images.shape[0]):
                true_index = int(labels[batch_index].item())
                predicted_index = int(
                    predicted[batch_index].item()
                )

                predictions.append(
                    {
                        "sample_index": sample_index,
                        "true_index": true_index,
                        "predicted_index": predicted_index,
                        "confidence": float(
                            confidence[batch_index].item()
                        ),
                        "correct": (
                            true_index == predicted_index
                        ),
                    }
                )

                sample_index += 1

    return predictions


def create_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
) -> np.ndarray:
    """Create a thermal grayscale + Grad-CAM overlay."""
    image = np.clip(image, 0.0, 1.0)

    image_rgb = np.repeat(
        image[..., None],
        3,
        axis=2,
    )

    heatmap_rgb = plt.get_cmap("jet")(
        heatmap
    )[..., :3]

    overlay = (
        0.55 * image_rgb
        + 0.45 * heatmap_rgb
    )

    return np.clip(overlay, 0.0, 1.0)


def save_explanation(
    image: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save original image, Grad-CAM and overlay."""
    overlay = create_overlay(
        image,
        heatmap,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(10, 3.5),
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Original Thermal")

    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def generate_model_explanations(
    model_name: str,
    spec: dict,
    test_loader,
    class_names: list[str],
) -> pd.DataFrame:
    """Generate representative Grad-CAM explanations."""
    print(f"\nGenerating Grad-CAM: {model_name}")

    model = load_model(spec)

    predictions = collect_predictions(
        model,
        test_loader,
    )

    selected = select_representative_samples(
        predictions
    )

    selected_indices = {
        item["sample_index"]
        for item in selected
    }

    selected_metadata = {
        item["sample_index"]: item
        for item in selected
    }

    output_dir = (
        OUTPUT_ROOT
        / model_name
        / "representative"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    grad_cam = GradCAM(
        model=model,
        target_layer=spec["target_layer_getter"](model),
    )

    sample_index = 0

    for images, labels in test_loader:
        for batch_index in range(images.shape[0]):
            if sample_index not in selected_indices:
                sample_index += 1
                continue

            image = images[
                batch_index:batch_index + 1
            ]

            label = int(
                labels[batch_index].item()
            )

            prediction = selected_metadata[
                sample_index
            ]

            target_class = torch.tensor(
                [prediction["predicted_index"]],
                dtype=torch.long,
            )

            result = grad_cam(
                image,
                target_classes=target_class,
                output_size=(40, 24),
            )

            heatmap = (
                result["heatmaps"][0]
                .cpu()
                .numpy()
            )

            image_np = (
                image[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            predicted_index = int(
                result["predicted_classes"][0].item()
            )

            target_index = int(
                result["target_classes"][0].item()
            )

            confidence = float(
                torch.softmax(
                    result["logits"],
                    dim=1,
                )[0, predicted_index].item()
            )

            status = (
                "correct"
                if label == predicted_index
                else "incorrect"
            )

            filename = (
                f"sample_{sample_index:04d}"
                f"__true_{label}_{class_names[label]}"
                f"__pred_{predicted_index}_{class_names[predicted_index]}"
                f"__{status}.png"
            )

            filename = filename.replace(
                "/",
                "_",
            )

            output_path = (
                output_dir / filename
            )

            title = (
                f"{model_name} | "
                f"True: {class_names[label]} | "
                f"Pred: {class_names[predicted_index]} | "
                f"Confidence: {confidence:.3f} | "
                f"{status}"
            )

            save_explanation(
                image=image_np,
                heatmap=heatmap,
                output_path=output_path,
                title=title,
            )

            rows.append(
                {
                    "model": model_name,
                    "sample_index": sample_index,
                    "true_index": label,
                    "true_class": class_names[label],
                    "predicted_index": predicted_index,
                    "predicted_class": class_names[
                        predicted_index
                    ],
                    "target_index": target_index,
                    "target_class": class_names[
                        target_index
                    ],
                    "confidence": confidence,
                    "correct": label == predicted_index,
                    "heatmap_height": heatmap.shape[0],
                    "heatmap_width": heatmap.shape[1],
                    "heatmap_min": float(
                        heatmap.min()
                    ),
                    "heatmap_max": float(
                        heatmap.max()
                    ),
                    "heatmap_mean": float(
                        heatmap.mean()
                    ),
                    "heatmap_nonzero_fraction": float(
                        np.mean(heatmap > 0)
                    ),
                    "target_layer": spec[
                        "target_layer_name"
                    ],
                    "checkpoint": str(
                        spec["checkpoint"]
                    ),
                }
            )

            sample_index += 1

        if sample_index >= len(test_loader.dataset):
            break

    grad_cam.remove_hooks()

    return pd.DataFrame(rows)


def main() -> None:
    """Run representative Grad-CAM generation."""
    set_seed(SEED)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, _, test_loader = create_raptormaps_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_train_augmentation=False,
    )

    class_names = test_loader.dataset.class_names

    all_metadata = []

    for model_name, spec in MODEL_SPECS.items():
        metadata = generate_model_explanations(
            model_name=model_name,
            spec=spec,
            test_loader=test_loader,
            class_names=class_names,
        )

        all_metadata.append(metadata)

    combined = pd.concat(
        all_metadata,
        ignore_index=True,
    )

    output_path = (
        RESULTS_ROOT
        / "finetuned_representative_metadata.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    print("\nGenerated explanations:", len(combined))
    print("Output metadata:", output_path)

    print("\nCounts by model/status:")
    print(
        combined.groupby(
            ["model", "correct"]
        ).size()
    )


if __name__ == "__main__":
    main()
