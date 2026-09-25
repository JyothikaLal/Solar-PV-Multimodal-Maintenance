from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.raptormaps_torch import (
    RaptorMapsTorchDataset,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from src.models.vision.raptormaps_transfer import (
    RaptorMapsTransferAdapter,
)
from src.models.vision.transfer_models import (
    build_pretrained_efficientnet_b0,
    build_pretrained_resnet18,
)


BATCH_SIZE = 64
NUM_WORKERS = 0

OUTPUT_DIR = Path(
    "reports/results/raptormaps/embeddings"
)

RESNET_CHECKPOINT = Path(
    "models/raptormaps/resnet18_finetune/best_model.pt"
)

EFFICIENTNET_CHECKPOINT = Path(
    "models/raptormaps/efficientnet_b0_finetune/best_model.pt"
)

SPLITS = (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_resnet18_model(
    device: torch.device,
) -> RaptorMapsTransferAdapter:
    backbone = build_pretrained_resnet18(
        num_classes=12,
    )

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.fc,
        spatial_size=(160, 96),
    )

    checkpoint = torch.load(
        RESNET_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def load_efficientnet_b0_model(
    device: torch.device,
) -> RaptorMapsTransferAdapter:
    backbone = build_pretrained_efficientnet_b0(
        num_classes=12,
    )

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=backbone.classifier,
        spatial_size=(160, 96),
    )

    checkpoint = torch.load(
        EFFICIENTNET_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def extract_resnet18_features(
    model: RaptorMapsTransferAdapter,
    images: torch.Tensor,
) -> torch.Tensor:
    x = model.preprocess(images)

    backbone = model.backbone

    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)

    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)

    x = backbone.avgpool(x)
    x = torch.flatten(x, 1)

    if x.ndim != 2 or x.shape[1] != 512:
        raise RuntimeError(
            "Unexpected ResNet-18 embedding shape: "
            f"{tuple(x.shape)}"
        )

    return x


def extract_efficientnet_b0_features(
    model: RaptorMapsTransferAdapter,
    images: torch.Tensor,
) -> torch.Tensor:
    x = model.preprocess(images)

    backbone = model.backbone

    x = backbone.features(x)
    x = backbone.avgpool(x)
    x = torch.flatten(x, 1)

    if x.ndim != 2 or x.shape[1] != 1280:
        raise RuntimeError(
            "Unexpected EfficientNet-B0 embedding shape: "
            f"{tuple(x.shape)}"
        )

    return x


@torch.no_grad()
def extract_model_embeddings(
    model: RaptorMapsTransferAdapter,
    feature_extractor,
    device: torch.device,
    model_name: str,
    embedding_dimension: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    embedding_batches: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []

    row_offset = 0

    for split in SPLITS:
        dataset = RaptorMapsTorchDataset(
            split=split,
            transform=None,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        split_embeddings: list[np.ndarray] = []

        for images, _targets in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            embeddings = feature_extractor(
                model,
                images,
            )

            if embeddings.shape[1] != embedding_dimension:
                raise RuntimeError(
                    f"{model_name}: unexpected embedding "
                    f"dimension {embeddings.shape[1]}."
                )

            if not torch.isfinite(embeddings).all():
                raise RuntimeError(
                    f"{model_name}: non-finite embeddings encountered."
                )

            batch_embeddings = (
                embeddings.detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

            split_embeddings.append(batch_embeddings)

        split_matrix = np.concatenate(
            split_embeddings,
            axis=0,
        )

        split_manifest = dataset.manifest.copy()

        if len(split_manifest) != len(split_matrix):
            raise RuntimeError(
                f"{model_name}: metadata/embedding count "
                f"mismatch for split '{split}'."
            )

        split_manifest = split_manifest[
            [
                "metadata_id",
                "image_filepath",
                "anomaly_class",
                "split",
                "group_hash",
            ]
        ].copy()

        split_manifest.insert(
            0,
            "row_index",
            np.arange(
                row_offset,
                row_offset + len(split_manifest),
            ),
        )

        split_manifest["embedding_model"] = model_name
        split_manifest["embedding_dimension"] = (
            embedding_dimension
        )

        embedding_batches.append(split_matrix)
        metadata_frames.append(split_manifest)

        row_offset += len(split_manifest)

        print(
            f"{model_name}: {split} "
            f"{len(split_manifest)} samples "
            f"→ {split_matrix.shape}"
        )

    embeddings = np.concatenate(
        embedding_batches,
        axis=0,
    )

    metadata = pd.concat(
        metadata_frames,
        ignore_index=True,
    )

    if len(embeddings) != len(metadata):
        raise RuntimeError(
            f"{model_name}: final embedding/metadata "
            "count mismatch."
        )

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"{model_name}: embeddings must be 2-D."
        )

    if embeddings.shape[1] != embedding_dimension:
        raise RuntimeError(
            f"{model_name}: final embedding dimension "
            f"{embeddings.shape[1]} != {embedding_dimension}."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            f"{model_name}: final embeddings contain "
            "non-finite values."
        )

    expected_indices = np.arange(len(metadata))

    if not np.array_equal(
        metadata["row_index"].to_numpy(),
        expected_indices,
    ):
        raise RuntimeError(
            f"{model_name}: row_index is not contiguous."
        )

    return embeddings, metadata


def save_embeddings(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    model_name: str,
    checkpoint_path: Path,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    metadata = metadata.copy()

    metadata["checkpoint_path"] = str(
        checkpoint_path
    )

    metadata["checkpoint_epoch"] = int(
        checkpoint["epoch"]
    )

    metadata["checkpoint_best_validation_macro_f1"] = float(
        checkpoint["best_validation_macro_f1"]
    )

    stem = model_name.lower().replace("-", "_")

    embedding_path = (
        OUTPUT_DIR / f"{stem}_embeddings.npz"
    )

    metadata_path = (
        OUTPUT_DIR / f"{stem}_metadata.csv"
    )

    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
    )

    metadata.to_csv(
        metadata_path,
        index=False,
    )

    print()
    print(f"Saved embeddings: {embedding_path}")
    print(f"Saved metadata:   {metadata_path}")
    print(f"Shape:            {embeddings.shape}")
    print(f"Rows:             {len(metadata)}")


def main() -> None:
    device = get_device()

    print(f"Using device: {device}")

    if not RESNET_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {RESNET_CHECKPOINT}"
        )

    if not EFFICIENTNET_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {EFFICIENTNET_CHECKPOINT}"
        )

    print("\nLoading fine-tuned ResNet-18...")
    resnet = load_resnet18_model(device)

    resnet_embeddings, resnet_metadata = (
        extract_model_embeddings(
            model=resnet,
            feature_extractor=extract_resnet18_features,
            device=device,
            model_name="resnet18_finetuned",
            embedding_dimension=512,
        )
    )

    save_embeddings(
        embeddings=resnet_embeddings,
        metadata=resnet_metadata,
        model_name="resnet18_finetuned",
        checkpoint_path=RESNET_CHECKPOINT,
    )

    del resnet

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nLoading fine-tuned EfficientNet-B0...")
    efficientnet = load_efficientnet_b0_model(device)

    efficientnet_embeddings, efficientnet_metadata = (
        extract_model_embeddings(
            model=efficientnet,
            feature_extractor=extract_efficientnet_b0_features,
            device=device,
            model_name="efficientnet_b0_finetuned",
            embedding_dimension=1280,
        )
    )

    save_embeddings(
        embeddings=efficientnet_embeddings,
        metadata=efficientnet_metadata,
        model_name="efficientnet_b0_finetuned",
        checkpoint_path=EFFICIENTNET_CHECKPOINT,
    )

    print("\nTask 30 embedding extraction complete.")


if __name__ == "__main__":
    main()
