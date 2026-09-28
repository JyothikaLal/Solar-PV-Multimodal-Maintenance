from __future__ import annotations

import numpy as np
import pytest

from src.mlops.monitoring.embedding_drift import (
    MODEL_DIMENSIONS,
    load_embedding_reference,
    monitor_embedding_drift,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet18_finetuned",
        "efficientnet_b0_finetuned",
    ],
)
def test_embedding_reference_is_valid(model_name):
    reference = load_embedding_reference(
        model_name
    )

    expected_dimension = MODEL_DIMENSIONS[
        model_name
    ]

    assert reference["centroid"].shape == (
        expected_dimension,
    )

    assert reference["embedding_sample"].ndim == 2
    assert (
        reference["embedding_sample"].shape[1]
        == expected_dimension
    )

    assert reference["distance_sample"].ndim == 1

    assert np.isfinite(
        reference["centroid"]
    ).all()

    assert np.isfinite(
        reference["embedding_sample"]
    ).all()

    assert np.isfinite(
        reference["distance_sample"]
    ).all()

    assert (
        reference["summary"]["reference_split"].iloc[0]
        == "train"
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet18_finetuned",
        "efficientnet_b0_finetuned",
    ],
)
def test_embedding_self_reference_has_negligible_drift(
    model_name,
):
    reference = load_embedding_reference(
        model_name
    )

    embeddings = reference[
        "embedding_sample"
    ]

    result = monitor_embedding_drift(
        model_name,
        embeddings,
    )

    assert result["model_name"] == model_name
    assert result["reference_split"] == "train"
    assert result["current_count"] == (
        embeddings.shape[0]
    )
    assert result["embedding_dimension"] == (
        embeddings.shape[1]
    )

    assert result["centroid_shift"] < 1.0

    assert result["distance_psi"] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert result["distance_wasserstein"] < 1e-5


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet18_finetuned",
        "efficientnet_b0_finetuned",
    ],
)
def test_embedding_shift_produces_measurable_centroid_drift(
    model_name,
):
    reference = load_embedding_reference(
        model_name
    )

    embeddings = np.asarray(
        reference["embedding_sample"],
        dtype=np.float64,
    )

    shifted = embeddings.copy()

    # Apply a controlled shift to one representation
    # dimension across the current population.
    shifted[:, 0] += 5.0

    result = monitor_embedding_drift(
        model_name,
        shifted,
    )

    assert result["centroid_shift"] == pytest.approx(
        5.0,
        abs=0.01,
    )

    assert result["centroid_shift"] > 1.0
    assert result["distance_psi"] > 0.0
    assert result["distance_wasserstein"] > 0.0


def test_unknown_embedding_model_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unknown embedding model",
    ):
        load_embedding_reference(
            "unknown_model"
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet18_finetuned",
        "efficientnet_b0_finetuned",
    ],
)
def test_wrong_embedding_dimension_is_rejected(
    model_name,
):
    expected_dimension = MODEL_DIMENSIONS[
        model_name
    ]

    current = np.zeros(
        (10, expected_dimension + 1),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="dimension",
    ):
        monitor_embedding_drift(
            model_name,
            current,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet18_finetuned",
        "efficientnet_b0_finetuned",
    ],
)
def test_empty_embedding_population_is_rejected(
    model_name,
):
    expected_dimension = MODEL_DIMENSIONS[
        model_name
    ]

    current = np.empty(
        (0, expected_dimension),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        monitor_embedding_drift(
            model_name,
            current,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet18_finetuned",
        "efficientnet_b0_finetuned",
    ],
)
def test_non_finite_embeddings_are_rejected(
    model_name,
):
    expected_dimension = MODEL_DIMENSIONS[
        model_name
    ]

    current = np.zeros(
        (10, expected_dimension),
        dtype=np.float64,
    )

    current[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite values",
    ):
        monitor_embedding_drift(
            model_name,
            current,
        )


def test_reference_model_dimensions_are_locked():
    assert MODEL_DIMENSIONS == {
        "resnet18_finetuned": 512,
        "efficientnet_b0_finetuned": 1280,
    }
