import pytest
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


def test_training_step_performs_forward_backward_and_update():
    import copy

    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.vision.custom_cnn import (
        RaptorMapsCustomCNN,
    )

    torch.manual_seed(42)

    images = torch.randn(
        4,
        1,
        40,
        24,
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=False,
    )

    model = RaptorMapsCustomCNN()

    loss_function = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    model.train()

    batch_images, batch_labels = next(iter(loader))

    optimizer.zero_grad(set_to_none=True)

    logits = model(batch_images)

    assert logits.shape == (4, 12)
    assert torch.isfinite(logits).all()

    loss = loss_function(
        logits,
        batch_labels,
    )

    assert torch.isfinite(loss)

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )

    optimizer.step()

    changed = any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter in model.named_parameters()
    )

    assert changed


def test_fit_model_checkpoints_best_validation_macro_f1(
    tmp_path,
    monkeypatch,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import src.training.vision_training as vision_training

    torch.manual_seed(42)

    images = torch.randn(
        4,
        1,
        40,
        24,
    )

    labels = torch.tensor(
        [0, 1, 2, 0],
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=False,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            40 * 24,
            3,
        ),
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer_config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=3,
        early_stopping_patience=10,
        scheduler_factor=0.5,
        scheduler_patience=2,
        min_learning_rate=1e-6,
    )

    validation_results = iter(
        [
            vision_training.EpochResult(
                loss=1.0,
                macro_f1=0.30,
                accuracy=0.30,
                balanced_accuracy=0.30,
            ),
            vision_training.EpochResult(
                loss=0.8,
                macro_f1=0.45,
                accuracy=0.45,
                balanced_accuracy=0.45,
            ),
            vision_training.EpochResult(
                loss=0.7,
                macro_f1=0.40,
                accuracy=0.40,
                balanced_accuracy=0.40,
            ),
        ]
    )

    def fake_evaluate_one_epoch(*args, **kwargs):
        return next(validation_results)

    monkeypatch.setattr(
        vision_training,
        "evaluate_one_epoch",
        fake_evaluate_one_epoch,
    )

    checkpoint_path = (
        tmp_path / "best_model.pt"
    )

    history = vision_training.fit_model(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        loss_function=loss_function,
        class_names=[
            "class_0",
            "class_1",
            "class_2",
        ],
        device=torch.device("cpu"),
        config=optimizer_config,
        checkpoint_path=checkpoint_path,
    )

    assert len(history) == 3

    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert checkpoint["epoch"] == 2

    assert checkpoint[
        "best_validation_macro_f1"
    ] == pytest.approx(0.45)

    assert "model_state_dict" in checkpoint
    assert "class_names" in checkpoint
    assert "config" in checkpoint

    assert checkpoint["class_names"] == [
        "class_0",
        "class_1",
        "class_2",
    ]


def test_fit_model_early_stopping_uses_validation_macro_f1(
    tmp_path,
    monkeypatch,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import src.training.vision_training as vision_training

    torch.manual_seed(42)

    images = torch.randn(
        4,
        1,
        40,
        24,
    )

    labels = torch.tensor(
        [0, 1, 2, 0],
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=False,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            40 * 24,
            3,
        ),
    )

    loss_function = nn.CrossEntropyLoss()

    config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=10,
        early_stopping_patience=2,
        scheduler_factor=0.5,
        scheduler_patience=10,
        min_learning_rate=1e-6,
    )

    validation_results = iter(
        [
            vision_training.EpochResult(
                loss=1.0,
                macro_f1=0.50,
                accuracy=0.50,
                balanced_accuracy=0.50,
            ),
            vision_training.EpochResult(
                loss=1.1,
                macro_f1=0.40,
                accuracy=0.40,
                balanced_accuracy=0.40,
            ),
            vision_training.EpochResult(
                loss=1.2,
                macro_f1=0.30,
                accuracy=0.30,
                balanced_accuracy=0.30,
            ),
        ]
    )

    def fake_evaluate_one_epoch(*args, **kwargs):
        return next(validation_results)

    monkeypatch.setattr(
        vision_training,
        "evaluate_one_epoch",
        fake_evaluate_one_epoch,
    )

    history = vision_training.fit_model(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        loss_function=loss_function,
        class_names=[
            "class_0",
            "class_1",
            "class_2",
        ],
        device=torch.device("cpu"),
        config=config,
        checkpoint_path=tmp_path / "best_model.pt",
    )

    assert len(history) == 3

    assert [
        record["validation_macro_f1"]
        for record in history
    ] == pytest.approx(
        [0.50, 0.40, 0.30]
    )


def test_fit_model_reduces_learning_rate_on_plateau(
    tmp_path,
    monkeypatch,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import src.training.vision_training as vision_training

    torch.manual_seed(42)

    images = torch.randn(
        4,
        1,
        40,
        24,
    )

    labels = torch.tensor(
        [0, 1, 2, 0],
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=False,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            40 * 24,
            3,
        ),
    )

    loss_function = nn.CrossEntropyLoss()

    config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=4,
        early_stopping_patience=10,
        scheduler_factor=0.5,
        scheduler_patience=1,
        min_learning_rate=1e-6,
    )

    validation_results = iter(
        [
            vision_training.EpochResult(
                loss=1.0,
                macro_f1=0.50,
                accuracy=0.50,
                balanced_accuracy=0.50,
            ),
            vision_training.EpochResult(
                loss=1.1,
                macro_f1=0.40,
                accuracy=0.40,
                balanced_accuracy=0.40,
            ),
            vision_training.EpochResult(
                loss=1.2,
                macro_f1=0.30,
                accuracy=0.30,
                balanced_accuracy=0.30,
            ),
            vision_training.EpochResult(
                loss=1.3,
                macro_f1=0.20,
                accuracy=0.20,
                balanced_accuracy=0.20,
            ),
        ]
    )

    def fake_evaluate_one_epoch(*args, **kwargs):
        return next(validation_results)

    monkeypatch.setattr(
        vision_training,
        "evaluate_one_epoch",
        fake_evaluate_one_epoch,
    )

    history = vision_training.fit_model(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        loss_function=loss_function,
        class_names=[
            "class_0",
            "class_1",
            "class_2",
        ],
        device=torch.device("cpu"),
        config=config,
        checkpoint_path=tmp_path / "best_model.pt",
    )

    learning_rates = [
        record["learning_rate"]
        for record in history
    ]

    assert learning_rates[0] == pytest.approx(
        1e-3
    )

    assert learning_rates[1] == pytest.approx(
        1e-3
    )

    assert learning_rates[2] == pytest.approx(
        5e-4
    )

    assert learning_rates[3] == pytest.approx(
        5e-4
    )
