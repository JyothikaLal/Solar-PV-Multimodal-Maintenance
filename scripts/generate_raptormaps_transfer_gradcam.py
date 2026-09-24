"""Generate transfer-model Grad-CAM pilot explanations for RaptorMaps."""

from __future__ import annotations

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
    load_transfer_checkpoint,
)


SEED = 42
NUM_PILOT_SAMPLES = 12

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 24

ROOT = Path("models/raptormaps")

MODEL_SPECS = {
    "resnet18": {
        "checkpoint": ROOT / "resnet18_transfer" / "best_model.pt",
        "output_dir": Path(
            "reports/figures/raptormaps/grad_cam/resnet18_transfer/pilot"
        ),
        "backbone_builder": build_pretrained_resnet18,
        "classifier_name": "fc",
    },
    "efficientnet_b0": {
        "checkpoint": ROOT / "efficientnet_b0_transfer" / "best_model.pt",
        "output_dir": Path(
            "reports/figures/raptormaps/grad_cam/efficientnet_b0_transfer/pilot"
        ),
        "backbone_builder": build_pretrained_efficientnet_b0,
        "classifier_name": "classifier",
    },
}


def get_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_model(
    model_name: str,
    num_classes: int,
) -> torch.nn.Module:
    """Build the exact Task 26 transfer wrapper."""

    spec = MODEL_SPECS[model_name]

    backbone = spec["backbone_builder"](
        num_classes=num_classes,
    )

    classifier_module = getattr(
        backbone,
        spec["classifier_name"],
    )

    return RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=classifier_module,
        spatial_size=(160, 96),
    )


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
    model_name: str,
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

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\n"
        f"True: {true_class}\n"
        f"Pred: {predicted_class} "
        f"({confidence:.3f})"
    )
    axes[2].axis("off")

    figure.suptitle(
        f"RaptorMaps {model_name} | "
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

    device = get_device()

    print(f"Using device: {device}")

    _, _, test_loader = create_raptormaps_dataloaders(
        batch_size=64,
        num_workers=0,
        use_train_augmentation=False,
    )

    class_names = test_loader.dataset.class_names

    print(f"Test samples available: {len(test_loader.dataset)}")
    print(f"Classes: {len(class_names)}")

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

    for model_name, spec in MODEL_SPECS.items():
        print("\n" + "=" * 72)
        print(f"{model_name.upper()} TRANSFER GRAD-CAM PILOT")
        print("=" * 72)

        checkpoint_path = spec["checkpoint"]

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint_path}"
            )

        output_dir = spec["output_dir"]
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model = build_model(
            model_name=model_name,
            num_classes=len(class_names),
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        load_transfer_checkpoint(
            model,
            checkpoint["model_state_dict"],
        )

        model.to(device)
        model.eval()

        if model_name == "resnet18":
            target_layer = model.backbone.layer4[1].conv2
            target_layer_name = "backbone.layer4[1].conv2"
        else:
            target_layer = model.backbone.features[8][0]
            target_layer_name = "backbone.features[8][0]"

        grad_cam = GradCAM(
            model=model,
            target_layer=target_layer,
        )

        print("Checkpoint:", checkpoint_path)
        print("Epoch:", checkpoint["epoch"])
        print(
            "Validation Macro F1 at selection:",
            f"{checkpoint['best_validation_macro_f1']:.6f}",
        )
        print("Target layer:", target_layer_name)
        print("Pilot samples:", len(selected))

        records = []

        for sample_index, (image, target) in enumerate(selected):
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
            predicted_index = int(
                result["predicted_classes"][0].item()
            )

            probabilities = torch.softmax(
                logits,
                dim=0,
            )

            confidence = float(
                probabilities[predicted_index].item()
            )

            true_index = int(target)

            true_class = class_names[true_index]
            predicted_class = class_names[predicted_index]

            correct = (
                true_index == predicted_index
            )

            image_np = (
                image.squeeze(0)
                .cpu()
                .numpy()
            )

            image_np = (
                image_np * 0.15437313
                + 0.61973207
            )

            image_np = np.clip(
                image_np,
                0.0,
                1.0,
            )

            overlay = create_overlay(
                image_np,
                heatmap,
            )

            output_path = (
                output_dir
                / f"sample_{sample_index:04d}.png"
            )

            save_explanation(
                image=image_np,
                heatmap=heatmap,
                overlay=overlay,
                model_name=model_name,
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
                    "true_index": true_index,
                    "predicted_index": predicted_index,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "correct": correct,
                    "target_layer": target_layer_name,
                    "heatmap_height": heatmap.shape[0],
                    "heatmap_width": heatmap.shape[1],
                    "heatmap_min": float(heatmap.min()),
                    "heatmap_max": float(heatmap.max()),
                    "heatmap_mean": float(heatmap.mean()),
                }
            )

        metadata = pd.DataFrame(records)

        metadata_path = (
            output_dir / "pilot_metadata.csv"
        )

        metadata.to_csv(
            metadata_path,
            index=False,
        )

        print(
            f"Generated explanations: "
            f"{len(records)}"
        )
        print(
            "Heatmap size:",
            f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}",
        )
        print(
            "Correct:",
            int(metadata["correct"].sum()),
            "/",
            len(metadata),
        )
        print(
            "Output:",
            output_dir,
        )
        print(
            "Metadata:",
            metadata_path,
        )

        grad_cam.remove_hooks()


if __name__ == "__main__":
    main()
