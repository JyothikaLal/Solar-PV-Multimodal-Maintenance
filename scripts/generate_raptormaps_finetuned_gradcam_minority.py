"""Generate Grad-CAM explanations for selected minority-class cases.

Uses the exact Task 27 fine-tuned checkpoints and the frozen RaptorMaps
test split. Selection is read from
reports/results/raptormaps/grad_cam/finetuned_minority_selection.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.raptormaps_torch import create_raptormaps_dataloaders
from src.models.vision.grad_cam import GradCAM
from src.models.vision.raptormaps_transfer import RaptorMapsTransferAdapter
from src.models.vision.transfer_models import (
    build_pretrained_efficientnet_b0,
    build_pretrained_resnet18,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SELECTION_PATH = (
    PROJECT_ROOT
    / "reports/results/raptormaps/grad_cam/finetuned_minority_selection.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "reports/figures/raptormaps/grad_cam"

METADATA_OUTPUT = (
    PROJECT_ROOT
    / "reports/results/raptormaps/grad_cam/finetuned_minority_metadata.csv"
)

NUM_CLASSES = 12
IMAGE_SIZE = (160, 96)

MODEL_CONFIGS = {
    "resnet18_finetune": {
        "checkpoint": (
            PROJECT_ROOT
            / "models/raptormaps/resnet18_finetune/best_model.pt"
        ),
        "builder": build_pretrained_resnet18,
        "classifier_attr": "fc",
        "target_layer_name": "backbone.layer4[1].conv2",
        "target_layer_getter": (
            lambda model: model.backbone.layer4[1].conv2
        ),
    },
    "efficientnet_b0_finetune": {
        "checkpoint": (
            PROJECT_ROOT
            / "models/raptormaps/efficientnet_b0_finetune/best_model.pt"
        ),
        "builder": build_pretrained_efficientnet_b0,
        "classifier_attr": "classifier",
        "target_layer_name": "backbone.features[8][0]",
        "target_layer_getter": (
            lambda model: model.backbone.features[8][0]
        ),
    },
}


def load_model(model_name: str, device: torch.device):
    config = MODEL_CONFIGS[model_name]

    backbone = config["builder"](num_classes=NUM_CLASSES)

    if model_name == "resnet18_finetune":
        classifier = backbone.fc
    else:
        classifier = backbone.classifier

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=classifier,
        spatial_size=IMAGE_SIZE,
    )

    checkpoint = torch.load(
        config["checkpoint"],
        map_location=device,
    )

    state_dict = checkpoint["model_state_dict"]

    result = model.load_state_dict(state_dict, strict=False)

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint loading mismatch for {model_name}: "
            f"missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    return model


def collect_test_samples():
    loaders = create_raptormaps_dataloaders(
        batch_size=1,
        num_workers=0,
        use_train_augmentation=False,
    )

    test_loader = loaders[1]

    samples = {}

    for sample_index, (image, label) in enumerate(test_loader):
        samples[sample_index] = (
            image,
            int(label.item()),
        )

    return samples


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


def main():
    if not SELECTION_PATH.exists():
        raise FileNotFoundError(
            f"Selection file not found: {SELECTION_PATH}"
        )

    selection = pd.read_csv(SELECTION_PATH)

    required_columns = {
        "model",
        "sample_index",
        "true_index",
        "true_class",
        "predicted_index",
        "predicted_class",
        "confidence",
        "correct",
        "selection_reason",
    }

    missing = required_columns - set(selection.columns)

    if missing:
        raise ValueError(
            f"Selection file is missing columns: {sorted(missing)}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Selected samples: {len(selection)}")

    test_samples = collect_test_samples()

    metadata_rows = []

    for model_name in MODEL_CONFIGS:
        model_selection = selection[
            selection["model"] == model_name
        ].copy()

        print(
            f"\nGenerating Grad-CAM: {model_name} "
            f"({len(model_selection)} samples)"
        )

        model = load_model(model_name, device)

        target_layer_name = MODEL_CONFIGS[model_name][
            "target_layer_name"
        ]

        target_layer = MODEL_CONFIGS[model_name][
            "target_layer_getter"
        ](model)

        grad_cam = GradCAM(
            model=model,
            target_layer=target_layer,
        )

        model_output_dir = (
            OUTPUT_ROOT
            / model_name
            / "minority"
        )
        model_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for _, row in model_selection.iterrows():
            sample_index = int(row["sample_index"])

            if sample_index not in test_samples:
                raise IndexError(
                    f"Test sample index {sample_index} "
                    f"not found for {model_name}"
                )

            image, true_index = test_samples[sample_index]

            image = image.to(device)

            target_class = torch.tensor(
                [int(row["predicted_index"])],
                dtype=torch.long,
                device=device,
            )

            result = grad_cam(
                image,
                target_classes=target_class,
                output_size=(40, 24),
            )

            heatmap = result["heatmaps"][0]
            predicted_index = int(
                result["predicted_classes"][0].item()
            )
            target_index = int(
                result["target_classes"][0].item()
            )
            target_score = float(
                result["target_scores"][0].item()
            )
            logits = result["logits"][0]

            probabilities = torch.softmax(
                logits,
                dim=0,
            )

            confidence = float(
                probabilities[predicted_index].item()
            )

            heatmap_min = float(heatmap.min().item())
            heatmap_max = float(heatmap.max().item())
            heatmap_mean = float(heatmap.mean().item())
            heatmap_nonzero_fraction = float(
                (heatmap > 0).float().mean().item()
            )

            status = (
                "correct"
                if bool(row["correct"])
                else "incorrect"
            )

            true_class = str(row["true_class"])
            predicted_class = str(row["predicted_class"])

            filename = (
                f"{model_name}"
                f"__idx_{sample_index}"
                f"__true_{true_class}"
                f"__pred_{predicted_class}"
                f"__{status}.png"
            )

            output_path = model_output_dir / filename

            image_np = (
                image[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            title = (
                f"{model_name} | "
                f"True: {true_class} | "
                f"Pred: {predicted_class} | "
                f"Confidence: {confidence:.3f} | "
                f"{status}"
            )

            save_explanation(
                image=image_np,
                heatmap=heatmap.detach().cpu().numpy(),
                output_path=output_path,
                title=title,
            )

            metadata_rows.append(
                {
                    "model": model_name,
                    "sample_index": sample_index,
                    "true_index": true_index,
                    "true_class": true_class,
                    "predicted_index": predicted_index,
                    "predicted_class": predicted_class,
                    "target_index": target_index,
                    "target_class": predicted_class,
                    "confidence": confidence,
                    "selection_correct": bool(row["correct"]),
                    "selection_reason": str(
                        row["selection_reason"]
                    ),
                    "target_score": target_score,
                    "heatmap_height": int(heatmap.shape[0]),
                    "heatmap_width": int(heatmap.shape[1]),
                    "heatmap_min": heatmap_min,
                    "heatmap_max": heatmap_max,
                    "heatmap_mean": heatmap_mean,
                    "heatmap_nonzero_fraction": (
                        heatmap_nonzero_fraction
                    ),
                    "target_layer": target_layer_name,
                    "checkpoint": str(
                        MODEL_CONFIGS[model_name]["checkpoint"]
                        .relative_to(PROJECT_ROOT)
                    ),
                    "output_path": str(
                        output_path.relative_to(PROJECT_ROOT)
                    ),
                }
            )

        print(
            f"Generated {len(model_selection)} explanations."
        )

    metadata = pd.DataFrame(metadata_rows)

    METADATA_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        METADATA_OUTPUT,
        index=False,
    )

    print(
        f"\nMetadata saved: {METADATA_OUTPUT.relative_to(PROJECT_ROOT)}"
    )
    print(f"Shape: {metadata.shape}")

    print("\nGeneration counts:")
    print(
        metadata.groupby(
            ["model", "selection_correct"]
        ).size()
    )


if __name__ == "__main__":
    main()
