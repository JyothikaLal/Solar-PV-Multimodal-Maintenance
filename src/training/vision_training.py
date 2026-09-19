from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.evaluation.classification import evaluate_classification


@dataclass
class EpochResult:
    loss: float
    macro_f1: float
    accuracy: float
    balanced_accuracy: float


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    early_stopping_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    min_learning_rate: float = 1e-6


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_function(
            logits,
            targets,
        )

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Training loader contains no samples."
        )

    return total_loss / total_samples


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> EpochResult:
    """Evaluate the model on a validation or test loader."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)

        loss = loss_function(
            logits,
            targets,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(
            targets.cpu().numpy()
        )
        all_predictions.append(
            predictions.cpu().numpy()
        )
        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    if total_samples == 0:
        raise ValueError(
            "Evaluation loader contains no samples."
        )

    y_true = np.concatenate(
        all_targets
    )

    y_pred = np.concatenate(
        all_predictions
    )

    y_proba = np.concatenate(
        all_probabilities
    )

    evaluation = evaluate_classification(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        y_proba=y_proba,
    )

    metrics = evaluation["overall"]

    return EpochResult(
        loss=total_loss / total_samples,
        macro_f1=float(metrics["macro_f1"]),
        accuracy=float(metrics["accuracy"]),
        balanced_accuracy=float(
            metrics["balanced_accuracy"]
        ),
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: nn.Module,
    class_names: list[str],
    device: torch.device,
    config: TrainingConfig | None = None,
    checkpoint_path: Path | None = None,
) -> list[dict[str, float]]:
    """Train with validation-based checkpointing and early stopping."""

    if config is None:
        config = TrainingConfig()

    if config.max_epochs <= 0:
        raise ValueError(
            "max_epochs must be greater than zero."
        )

    if config.early_stopping_patience < 0:
        raise ValueError(
            "early_stopping_patience cannot be negative."
        )

    model.to(device)
    loss_function.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )

    best_macro_f1 = -float("inf")
    epochs_without_improvement = 0

    history: list[dict[str, float]] = []

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        validation = evaluate_one_epoch(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device,
            class_names=class_names,
        )

        scheduler.step(
            validation.macro_f1
        )

        learning_rate = optimizer.param_groups[0][
            "lr"
        ]

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "validation_loss": float(
                validation.loss
            ),
            "validation_macro_f1": float(
                validation.macro_f1
            ),
            "validation_accuracy": float(
                validation.accuracy
            ),
            "validation_balanced_accuracy": float(
                validation.balanced_accuracy
            ),
            "learning_rate": float(
                learning_rate
            ),
        }

        history.append(epoch_record)

        if validation.macro_f1 > best_macro_f1:
            best_macro_f1 = validation.macro_f1
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "best_validation_macro_f1": (
                            best_macro_f1
                        ),
                        "epoch": epoch,
                        "class_names": class_names,
                        "config": config.__dict__,
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= config.early_stopping_patience
        ):
            break

    return history
