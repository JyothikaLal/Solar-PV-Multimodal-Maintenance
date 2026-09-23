from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.raptormaps_torch import (
    create_raptormaps_dataloaders,
)
from src.models.vision.custom_cnn import (
    RaptorMapsCustomCNN,
)
from src.models.vision.losses import (
    create_raptormaps_weighted_loss,
)
from src.training.vision_training import (
    TrainingConfig,
    fit_model,
)


SEED = 42

BATCH_SIZE = 64
NUM_WORKERS = 0

USE_TRAIN_AUGMENTATION = False

MODEL_DIR = Path(
    "models/raptormaps/custom_cnn_task25"
)

RESULTS_DIR = Path(
    "reports/results/raptormaps/custom_cnn_task25"
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


def set_seed(seed: int) -> None:
    """Set deterministic random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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

    train_loader, validation_loader, _ = (
        create_raptormaps_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            use_train_augmentation=(
                USE_TRAIN_AUGMENTATION
            ),
        )
    )

    class_weights, loss_function = (
        create_raptormaps_weighted_loss()
    )

    class_names = (
        train_loader.dataset.class_names
    )

    class_weights_df = pd.DataFrame(
        {
            "class_name": class_names,
            "class_index": range(
                len(class_names)
            ),
            "class_weight": (
                class_weights.numpy()
            ),
        }
    )

    class_weights_df.to_csv(
        CLASS_WEIGHTS_PATH,
        index=False,
    )

    model = RaptorMapsCustomCNN(
        num_classes=len(class_names),
    )

    config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=30,
        early_stopping_patience=5,
        scheduler_factor=0.5,
        scheduler_patience=2,
        min_learning_rate=1e-6,
    )

    history = fit_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        class_names=class_names,
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

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    config_payload = {
        "task": "25",
        "experiment": "custom_cnn_baseline",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "device": str(device),
        "class_names": class_names,
        "input_shape": [1, 40, 24],
        "native_resolution": [40, 24],
        "normalization": {
            "method": "training_set_standardization",
            "mean": 0.61973207,
            "std": 0.15437313,
        },
        "augmentation": {
            "enabled": USE_TRAIN_AUGMENTATION,
            "horizontal_flip_probability": 0.5,
        },
        "class_imbalance": {
            "strategy": "weighted_cross_entropy",
            "sampling": "natural",
        },
        "model": {
            "name": "RaptorMapsCustomCNN",
            "num_classes": len(class_names),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "embedding_dimension": 128,
        },
        "training_config": config.__dict__,
        "best_checkpoint": str(
            CHECKPOINT_PATH
        ),
        "best_validation_macro_f1": float(
            best_row["validation_macro_f1"]
        ),
        "best_epoch": int(
            best_row["epoch"]
        ),
    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config_payload,
            file,
            indent=2,
        )

    print()
    print("Task 25 baseline training complete.")
    print(
        f"Epochs completed: {len(history_df)}"
    )
    print(
        "Best validation Macro F1: "
        f"{config_payload['best_validation_macro_f1']:.4f}"
    )
    print(
        f"Best epoch: "
        f"{config_payload['best_epoch']}"
    )
    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )
    print(
        f"History: {HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()
