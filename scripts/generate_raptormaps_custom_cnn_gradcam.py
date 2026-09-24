"""Generate Custom CNN Grad-CAM pilot explanations for RaptorMaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.raptormaps_torch import create_raptormaps_dataloaders
from src.models.vision.custom_cnn import RaptorMapsCustomCNN
from src.models.vision.grad_cam import GradCAM


CHECKPOINT_PATH = Path(
    "models/raptormaps/custom_cnn_task25/best_model.pt"
)

OUTPUT_DIR = Path(
    "reports/figures/raptormaps/grad_cam/custom_cnn/pilot"
)

METADATA_PATH = OUTPUT_DIR / "pilot_metadata.csv"

SEED = 42
NUM_PILOT_SAMPLES = 12

MEAN = 0.61973207
STD = 0.15437313

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 24


def get_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def denormalize_image(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Recover original [0,1] thermal intensity."""

    image = tensor * STD + MEAN

    return image.clamp(0.0, 1.0)


def create_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
) -> np.ndarray:
    """Create a simple thermal-image / Grad-CAM overlay."""

    heatmap_rgb = plt.get_cmap("jet")(heatmap)[..., :3]

    image_rgb = np.repeat(
        image[..., None],
        3,
        axis=2,
    )

    overlay = (
        0.55 * image_rgb
        + 0.45 * heatmap_rgb
    )

    return np.clip(
        overlay,
        0.0,
        1.0,
    )


def save_explanation(
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    *,
    sample_index: int,
    true_class: str,
    predicted_class: str,
    confidence: float,
    correct: bool,
    output_path: Path,
) -> None:
    """Save original image, Grad-CAM and overlay."""

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].imshow(
        image,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Original Thermal")
    axes[0].axis("off")

    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(
        overlay,
    )
    axes[2].set_title(
        f"Overlay\n"
        f"True: {true_class}\n"
        f"Pred: {predicted_class} "
        f"({confidence:.3f})"
    )
    axes[2].axis("off")

    figure.suptitle(
        f"RaptorMaps Custom CNN | "
        f"Sample {sample_index} | "
        f"{'Correct' if correct else 'Incorrect'}"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(f"Using device: {device}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT_PATH}"
        )

    _, _, test_loader = create_raptormaps_dataloaders(
        batch_size=64,
        num_workers=0,
        use_train_augmentation=False,
    )

    class_names = test_loader.dataset.class_names

    model = RaptorMapsCustomCNN(
        num_classes=len(class_names),
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    required_keys = {
        "model_state_dict",
        "best_validation_macro_f1",
        "epoch",
        "class_names",
        "config",
    }

    missing_keys = (
        required_keys
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
        checkpoint["model_state_dict"],
    )

    model.to(device)
    model.eval()

    grad_cam = GradCAM(
        model=model,
        target_layer=model.features[8],
    )

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        "Validation Macro F1 at selection: "
        f"{checkpoint['best_validation_macro_f1']:.6f}"
    )

    print(
        f"Test samples available: "
        f"{len(test_loader.dataset)}"
    )

    selected = []

    for batch_images, batch_targets in test_loader:
        for image, target in zip(
            batch_images,
            batch_targets,
        ):
            selected.append(
                (
                    image,
                    int(target.item()),
                )
            )

            if len(selected) >= NUM_PILOT_SAMPLES:
                break

        if len(selected) >= NUM_PILOT_SAMPLES:
            break

    if len(selected) != NUM_PILOT_SAMPLES:
        raise ValueError(
            f"Expected {NUM_PILOT_SAMPLES} pilot samples, "
            f"got {len(selected)}."
        )

    records = []

    for sample_index, (image, target) in enumerate(
        selected
    ):
        inputs = image.unsqueeze(0).to(device)

        result = grad_cam(
            inputs,
            output_size=(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            ),
        )

        heatmap = (
            result["heatmaps"][0]
            .cpu()
            .numpy()
        )

        logits = result["logits"][0]

        probabilities = torch.softmax(
            logits,
            dim=0,
        )

        predicted_index = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[predicted_index].item()
        )

        true_class = class_names[target]
        predicted_class = class_names[
            predicted_index
        ]

        correct = (
            target == predicted_index
        )

        original = (
            denormalize_image(image)
            .squeeze(0)
            .cpu()
            .numpy()
        )

        overlay = create_overlay(
            original,
            heatmap,
        )

        output_path = (
            OUTPUT_DIR
            / (
                f"sample_{sample_index:04d}"
                f"_{true_class}"
                f"_pred-{predicted_class}"
                f"_{'correct' if correct else 'incorrect'}"
                ".png"
            )
        )

        save_explanation(
            original,
            heatmap,
            overlay,
            sample_index=sample_index,
            true_class=true_class,
            predicted_class=predicted_class,
            confidence=confidence,
            correct=correct,
            output_path=output_path,
        )

        records.append(
            {
                "sample_index": sample_index,
                "true_index": target,
                "true_class": true_class,
                "predicted_index": predicted_index,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "correct": correct,
                "heatmap_height": heatmap.shape[0],
                "heatmap_width": heatmap.shape[1],
                "heatmap_min": float(heatmap.min()),
                "heatmap_max": float(heatmap.max()),
                "heatmap_mean": float(heatmap.mean()),
            }
        )

        print(
            f"[{sample_index + 1:02d}/{NUM_PILOT_SAMPLES}] "
            f"true={true_class} | "
            f"pred={predicted_class} | "
            f"confidence={confidence:.4f} | "
            f"{'correct' if correct else 'incorrect'}"
        )

    metadata = pd.DataFrame(records)

    metadata.to_csv(
        METADATA_PATH,
        index=False,
    )

    grad_cam.remove_hooks()

    print()
    print("=" * 80)
    print("CUSTOM CNN GRAD-CAM PILOT COMPLETED")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata:         {METADATA_PATH}")
    print(f"Samples:          {len(metadata)}")
    print(
        "Heatmap shape:    "
        f"{IMAGE_HEIGHT} x {IMAGE_WIDTH}"
    )


if __name__ == "__main__":
    main()
