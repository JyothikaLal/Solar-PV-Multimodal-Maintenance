import torch

from src.data.raptormaps_torch import (
    RaptorMapsTorchDataset,
    create_raptormaps_dataloaders,
)


def test_train_dataset_shape_and_dtype():
    dataset = RaptorMapsTorchDataset("train")

    image, label = dataset[0]

    assert image.shape == (1, 40, 24)
    assert image.dtype == torch.float32
    assert label.dtype == torch.long
    assert torch.isfinite(image).all()


def test_dataset_class_mapping_is_consistent():
    train_dataset = RaptorMapsTorchDataset("train")
    validation_dataset = RaptorMapsTorchDataset("validation")
    test_dataset = RaptorMapsTorchDataset("test")

    assert (
        train_dataset.class_to_index
        == validation_dataset.class_to_index
    )

    assert (
        train_dataset.class_to_index
        == test_dataset.class_to_index
    )

    assert len(train_dataset.class_names) == 12


def test_dataloaders_have_expected_sizes():
    train_loader, validation_loader, test_loader = (
        create_raptormaps_dataloaders(
            batch_size=64,
            num_workers=0,
        )
    )

    assert len(train_loader.dataset) == 13992
    assert len(validation_loader.dataset) == 2997
    assert len(test_loader.dataset) == 2999


def test_training_batch_shape():
    train_loader, _, _ = create_raptormaps_dataloaders(
        batch_size=64,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    assert images.ndim == 4
    assert images.shape[1:] == (1, 40, 24)
    assert labels.ndim == 1
    assert images.shape[0] == labels.shape[0]

def test_dataloaders_have_expected_configuration():
    train_loader, validation_loader, test_loader = (
        create_raptormaps_dataloaders(
            batch_size=64,
            num_workers=0,
            pin_memory=True,
        )
    )

    assert train_loader.batch_size == 64
    assert validation_loader.batch_size == 64
    assert test_loader.batch_size == 64

    assert train_loader.num_workers == 0
    assert validation_loader.num_workers == 0
    assert test_loader.num_workers == 0

    assert train_loader.pin_memory is True
    assert validation_loader.pin_memory is True
    assert test_loader.pin_memory is True

    assert train_loader.drop_last is False
    assert validation_loader.drop_last is False
    assert test_loader.drop_last is False


def test_dataloaders_can_disable_pinned_memory():
    train_loader, validation_loader, test_loader = (
        create_raptormaps_dataloaders(
            batch_size=32,
            num_workers=0,
            pin_memory=False,
        )
    )

    assert train_loader.pin_memory is False
    assert validation_loader.pin_memory is False
    assert test_loader.pin_memory is False


def test_invalid_batch_size_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        create_raptormaps_dataloaders(
            batch_size=0,
        )


def test_invalid_num_workers_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        create_raptormaps_dataloaders(
            num_workers=-1,
        )
