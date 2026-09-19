from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.raptormaps_classification import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_ROOT,
    MANIFEST_PATH,
    load_raptormaps_manifest,
    resolve_image_path,
)


TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

DEFAULT_BATCH_SIZE = 64


class RaptorMapsTorchDataset(Dataset):
    """PyTorch dataset for the frozen RaptorMaps split."""

    def __init__(
        self,
        split: str,
        manifest_path: Path = MANIFEST_PATH,
        image_root: Path = IMAGE_ROOT,
    ) -> None:
        if split not in {
            TRAIN_SPLIT,
            VALIDATION_SPLIT,
            TEST_SPLIT,
        }:
            raise ValueError(f"Unsupported split: {split}")

        manifest = load_raptormaps_manifest(
            manifest_path=manifest_path,
        )

        split_manifest = manifest[
            manifest["split"] == split
        ].reset_index(drop=True)

        if split_manifest.empty:
            raise ValueError(
                f"No samples found for split: {split}"
            )

        class_names = sorted(
            manifest[
                manifest["split"].isin(
                    [
                        TRAIN_SPLIT,
                        VALIDATION_SPLIT,
                        TEST_SPLIT,
                    ]
                )
            ]["anomaly_class"].unique()
        )

        self.class_names = class_names

        self.class_to_index = {
            name: index
            for index, name in enumerate(class_names)
        }

        self.manifest = split_manifest
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[index]

        image_path = resolve_image_path(
            row["image_filepath"],
            image_root=self.image_root,
        )

        with Image.open(image_path) as image:
            image = image.convert("L")

            if image.size != (
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
            ):
                raise ValueError(
                    "Unexpected image size: "
                    f"{image.size}. "
                    f"Expected "
                    f"({IMAGE_WIDTH}, {IMAGE_HEIGHT})."
                )

            array = np.asarray(
                image,
                dtype=np.float32,
            )

        array /= 255.0

        tensor = torch.from_numpy(
            array
        ).unsqueeze(0)

        label = self.class_to_index[
            row["anomaly_class"]
        ]

        target = torch.tensor(
            label,
            dtype=torch.long,
        )

        return tensor, target


def create_raptormaps_dataloaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    manifest_path: Path = MANIFEST_PATH,
    image_root: Path = IMAGE_ROOT,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """Create train, validation and test DataLoaders."""

    train_dataset = RaptorMapsTorchDataset(
        split=TRAIN_SPLIT,
        manifest_path=manifest_path,
        image_root=image_root,
    )

    validation_dataset = RaptorMapsTorchDataset(
        split=VALIDATION_SPLIT,
        manifest_path=manifest_path,
        image_root=image_root,
    )

    test_dataset = RaptorMapsTorchDataset(
        split=TEST_SPLIT,
        manifest_path=manifest_path,
        image_root=image_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )
