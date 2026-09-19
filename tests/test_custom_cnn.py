import torch

from src.models.vision.custom_cnn import (
    EMBEDDING_DIM,
    NUM_CLASSES,
    RaptorMapsCustomCNN,
)


def test_custom_cnn_output_shape():
    model = RaptorMapsCustomCNN()

    inputs = torch.randn(
        4,
        1,
        40,
        24,
    )

    outputs = model(inputs)

    assert outputs.shape == (
        4,
        NUM_CLASSES,
    )


def test_custom_cnn_embedding_shape():
    model = RaptorMapsCustomCNN()

    inputs = torch.randn(
        4,
        1,
        40,
        24,
    )

    embeddings = model.forward_features(inputs)

    assert embeddings.shape == (
        4,
        EMBEDDING_DIM,
    )


def test_custom_cnn_supports_custom_class_count():
    model = RaptorMapsCustomCNN(
        num_classes=5,
    )

    inputs = torch.randn(
        2,
        1,
        40,
        24,
    )

    outputs = model(inputs)

    assert outputs.shape == (
        2,
        5,
    )


def test_custom_cnn_rejects_invalid_class_count():
    try:
        RaptorMapsCustomCNN(num_classes=1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for invalid class count."
        )


def test_custom_cnn_produces_finite_outputs():
    model = RaptorMapsCustomCNN()

    inputs = torch.randn(
        4,
        1,
        40,
        24,
    )

    outputs = model(inputs)

    assert torch.isfinite(outputs).all()
