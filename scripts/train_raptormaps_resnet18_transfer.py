from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.raptormaps_torch import (
    create_raptormaps_dataloaders,
)
from src.models.vision.raptormaps_transfer import (
    RaptorMapsTransferAdapter,
)
from src.models.vision.transfer_models import (
    build_pretrained_resnet18,
    build_transfer_optimizer,
    freeze_resnet18_backbone,
)
from src.models.vision.losses import (
    create_raptormaps_weighted_loss,
)
from src.training.transfer_training import (
    TransferTrainingConfig,
    fit_transfer_model,
)


SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 0

SPATIAL_SIZE = (160, 96)

MODEL_DIR = Path(
    "models/raptormaps/resnet18_transfer"
)

RESULTS_DIR = Path(
    "reports/results/raptormaps/resnet18_transfer"
)

CHECKPOINT_PATH = (
    MODEL_DIR / "best_model.pt"
)

HISTORY_PATH = (
    RESULTS_DIR / "training_history.csv"
)

CONFIG_PATH = (
    RESULTS_DIR / "training_config.json"
)

CLASS_WEIGHTS_PATH = (
    RESULTS_DIR / "class_weights.csv"
)


class TransferDataset(Dataset):
    """Wrap the existing RaptorMaps dataset without changing its split."""

    def __init__(
        self,
        base_dataset,
    ) -> None:
        self.base_dataset = base_dataset
        self.class_names = base_dataset.class_names

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        return self.base_dataset[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_transfer_loaders():
    train_loader, validation_loader, test_loader = (
        create_raptormaps_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    )

    train_dataset = TransferDataset(
        train_loader.dataset
    )
    validation_dataset = TransferDataset(
        validation_loader.dataset
    )
    test_dataset = TransferDataset(
        test_loader.dataset
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }

    transfer_train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    transfer_validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    transfer_test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return (
        transfer_train_loader,
        transfer_validation_loader,
        transfer_test_loader,
    )


def main() -> None:
    set_seed(SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    (
        train_loader,
        validation_loader,
        _,
    ) = create_transfer_loaders()

    class_weights, loss_function = (
        create_raptormaps_weighted_loss()
    )

    class_names = (
        train_loader.dataset.class_names
    )

    pd.DataFrame(
        {
            "class_name": class_names,
            "class_index": range(
                len(class_names)
            ),
            "class_weight": class_weights.numpy(),
        }
    ).to_csv(
        CLASS_WEIGHTS_PATH,
        index=False,
    )

    backbone = build_pretrained_resnet18(
        num_classes=len(class_names),
    )

    freeze_resnet18_backbone(
        backbone
    )

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        spatial_size=SPATIAL_SIZE,
    )

    optimizer = build_transfer_optimizer(
        backbone,
        classifier_module=backbone.fc,
        classifier_lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-4,
    )

    config = TransferTrainingConfig(
        max_epochs=30,
        early_stopping_patience=5,
        scheduler_factor=0.5,
        scheduler_patience=2,
        min_learning_rate=1e-6,
    )

    history = fit_transfer_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        class_names=class_names,
        optimizer=optimizer,
        device=device,
        config=config,
        checkpoint_path=CHECKPOINT_PATH,
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
    )

    best_row = history_df.loc[
        history_df[
            "validation_macro_f1"
        ].idxmax()
    ]

    payload = {
        "model": "resnet18",
        "pretrained": True,
        "backbone_frozen": True,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "device": str(device),
        "spatial_size": list(
            SPATIAL_SIZE
        ),
        "class_names": class_names,
        "classifier_learning_rate": 1e-3,
        "backbone_learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "max_epochs": 30,
        "early_stopping_patience": 5,
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "min_learning_rate": 1e-6,
        "best_epoch": int(
            best_row["epoch"]
        ),
        "best_validation_macro_f1": float(
            best_row[
                "validation_macro_f1"
            ]
        ),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print()
    print("ResNet-18 transfer training complete.")
    print(
        f"Best epoch: {payload['best_epoch']}"
    )
    print(
        "Best validation Macro F1: "
        f"{payload['best_validation_macro_f1']:.4f}"
    )
    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )
    print(
        f"History: {HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()
