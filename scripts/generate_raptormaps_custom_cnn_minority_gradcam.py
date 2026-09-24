from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.raptormaps_torch import create_raptormaps_dataloaders
from src.models.vision.custom_cnn import RaptorMapsCustomCNN
from src.models.vision.grad_cam import GradCAM


CHECKPOINT = Path("models/raptormaps/custom_cnn_task25/best_model.pt")

SELECTION_MANIFEST = Path(
    "reports/results/raptormaps/grad_cam/minority_class_selection.csv"
)

OUTPUT_DIR = Path(
    "reports/figures/raptormaps/grad_cam/custom_cnn/minority_classes"
)

BATCH_SIZE = 64
NUM_WORKERS = 0

MEAN = 0.61973207
STD = 0.15437313


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize_image(tensor):
    image = tensor.detach().cpu().numpy()

    if image.ndim == 3:
        image = image[0]

    image = image * STD + MEAN
    return np.clip(image, 0.0, 1.0)


def create_overlay(image, heatmap):
    base = np.repeat(image[..., None], 3, axis=2)

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[..., :3]

    alpha = np.clip(heatmap[..., None], 0.0, 1.0) * 0.65

    overlay = (
        base * (1.0 - alpha)
        + colored * alpha
    )

    return np.clip(overlay, 0.0, 1.0)


def save_explanation(
    image,
    heatmap,
    output_path,
    true_class,
    predicted_class,
    confidence,
    correct,
):
    overlay = create_overlay(image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Thermal Input")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    status = "correct" if correct else "incorrect"

    fig.suptitle(
        f"True: {true_class} | "
        f"Pred: {predicted_class} | "
        f"Confidence: {confidence:.4f} | "
        f"{status}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    device = get_device()

    print(f"Using device: {device}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    if not SELECTION_MANIFEST.exists():
        raise FileNotFoundError(
            f"Selection manifest not found: {SELECTION_MANIFEST}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection_df = pd.read_csv(SELECTION_MANIFEST)

    required_columns = [
        "sample_index",
        "true_index",
        "true_class",
    ]

    missing = [
        column
        for column in required_columns
        if column not in selection_df.columns
    ]

    if missing:
        raise ValueError(
            f"Selection manifest missing columns: {missing}"
        )

    _, _, test_loader = create_raptormaps_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_train_augmentation=False,
    )

    class_names = test_loader.dataset.class_names

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    checkpoint_class_names = checkpoint["class_names"]

    if checkpoint_class_names != class_names:
        raise ValueError(
            "Checkpoint class mapping does not match dataset class mapping."
        )

    model = RaptorMapsCustomCNN(
        num_classes=len(class_names)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    target_layer = model.features[8]

    grad_cam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    # Collect the exact test-set samples in loader order.
    all_images = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            all_images.append(images)
            all_targets.append(targets)

    all_images = torch.cat(all_images, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if len(all_images) != len(test_loader.dataset):
        raise RuntimeError(
            "Collected test samples do not match dataset length."
        )

    metadata = []

    print(f"Selected samples: {len(selection_df)}")

    for position, row in selection_df.iterrows():
        sample_index = int(row["sample_index"])
        expected_true_index = int(row["true_index"])
        expected_true_class = row["true_class"]

        image = all_images[sample_index:sample_index + 1].to(device)
        target = all_targets[sample_index:sample_index + 1].to(device)

        actual_true_index = int(target.item())
        actual_true_class = class_names[actual_true_index]

        if actual_true_index != expected_true_index:
            raise RuntimeError(
                f"Manifest mismatch at sample {sample_index}: "
                f"manifest index={expected_true_index}, "
                f"dataset index={actual_true_index}"
            )

        if actual_true_class != expected_true_class:
            raise RuntimeError(
                f"Manifest mismatch at sample {sample_index}: "
                f"manifest class={expected_true_class}, "
                f"dataset class={actual_true_class}"
            )

        result = grad_cam(
            image,
            target_classes=None,
            output_size=(40, 24),
        )

        logits = result["logits"]
        probabilities = torch.softmax(logits, dim=1)

        predicted_index = int(
            result["predicted_classes"][0].item()
        )

        predicted_class = class_names[predicted_index]

        confidence = float(
            probabilities[0, predicted_index].item()
        )

        correct = predicted_index == actual_true_index

        heatmap = result["heatmaps"][0].detach().cpu().numpy()

        thermal_image = denormalize_image(image[0])

        status = "correct" if correct else "incorrect"

        output_name = (
            f"sample_{sample_index:04d}_"
            f"{actual_true_class}_"
            f"pred-{predicted_class}_"
            f"{status}.png"
        )

        output_path = OUTPUT_DIR / output_name

        save_explanation(
            image=thermal_image,
            heatmap=heatmap,
            output_path=output_path,
            true_class=actual_true_class,
            predicted_class=predicted_class,
            confidence=confidence,
            correct=correct,
        )

        metadata.append(
            {
                "sample_index": sample_index,
                "true_index": actual_true_index,
                "true_class": actual_true_class,
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
            f"[{position + 1:02d}/{len(selection_df):02d}] "
            f"true={actual_true_class} | "
            f"pred={predicted_class} | "
            f"confidence={confidence:.4f} | "
            f"{status}"
        )

    metadata_df = pd.DataFrame(metadata)

    metadata_path = OUTPUT_DIR / "minority_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    grad_cam.remove_hooks()

    print()
    print("=" * 72)
    print("CUSTOM CNN MINORITY-CLASS GRAD-CAM COMPLETED")
    print("=" * 72)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata:         {metadata_path}")
    print(f"Samples:          {len(metadata_df)}")
    print("Heatmap shape:    40 x 24")


if __name__ == "__main__":
    main()
