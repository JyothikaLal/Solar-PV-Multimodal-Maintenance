import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.vision_training import (
    TrainingConfig,
    evaluate_one_epoch,
    train_one_epoch,
)


def _create_toy_loader():
    torch.manual_seed(42)

    images = torch.randn(
        16,
        1,
        40,
        24,
    )

    labels = torch.randint(
        0,
        3,
        (16,),
    )

    dataset = TensorDataset(
        images,
        labels,
    )

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                40 * 24,
                3,
            ),
        )

    def forward(self, x):
        return self.network(x)


def test_training_config_defaults():
    config = TrainingConfig()

    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.max_epochs == 30
    assert config.early_stopping_patience == 5


def test_train_one_epoch_returns_finite_loss():
    loader = _create_toy_loader()

    model = TinyClassifier()

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    loss = train_one_epoch(
        model=model,
        loader=loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert np_is_finite(loss)
    assert loss > 0


def test_evaluate_one_epoch_returns_metrics():
    loader = _create_toy_loader()

    model = TinyClassifier()

    loss_function = nn.CrossEntropyLoss()

    result = evaluate_one_epoch(
        model=model,
        loader=loader,
        loss_function=loss_function,
        device=torch.device("cpu"),
        class_names=[
            "class_0",
            "class_1",
            "class_2",
        ],
    )

    assert np_is_finite(result.loss)
    assert np_is_finite(result.macro_f1)
    assert np_is_finite(result.accuracy)
    assert np_is_finite(
        result.balanced_accuracy
    )


def np_is_finite(value):
    return torch.isfinite(
        torch.tensor(value)
    ).item()
