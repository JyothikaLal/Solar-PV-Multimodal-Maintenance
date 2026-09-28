from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.mlops.monitoring.drift_metrics import (
    population_stability_index,
    wasserstein_distance,
)


REFERENCE_BASE_PATH = Path(
    "reports/results/raptormaps/monitoring"
)


MODEL_DIMENSIONS = {
    "resnet18_finetuned": 512,
    "efficientnet_b0_finetuned": 1280,
}


def load_embedding_reference(
    model_name: str,
) -> dict[str, np.ndarray | pd.DataFrame]:
    if model_name not in MODEL_DIMENSIONS:
        raise ValueError(
            f"Unknown embedding model: {model_name}"
        )

    npz_path = (
        REFERENCE_BASE_PATH
        / f"{model_name}_reference.npz"
    )

    summary_path = (
        REFERENCE_BASE_PATH
        / f"{model_name}_reference.csv"
    )

    data = np.load(npz_path)
    summary = pd.read_csv(summary_path)

    required_arrays = {
        "centroid",
        "embedding_sample",
        "distance_sample",
    }

    missing_arrays = required_arrays - set(data.files)

    if missing_arrays:
        raise ValueError(
            f"Embedding reference is missing arrays: "
            f"{sorted(missing_arrays)}"
        )

    if len(summary) != 1:
        raise ValueError(
            "Embedding reference summary must contain "
            "exactly one row."
        )

    expected_dimension = MODEL_DIMENSIONS[model_name]

    centroid = data["centroid"]
    embedding_sample = data["embedding_sample"]
    distance_sample = data["distance_sample"]

    if centroid.shape != (expected_dimension,):
        raise ValueError(
            f"{model_name}: invalid centroid shape "
            f"{centroid.shape}."
        )

    if embedding_sample.ndim != 2:
        raise ValueError(
            f"{model_name}: reference embeddings must be 2-D."
        )

    if embedding_sample.shape[1] != expected_dimension:
        raise ValueError(
            f"{model_name}: invalid embedding dimension."
        )

    if distance_sample.ndim != 1:
        raise ValueError(
            f"{model_name}: reference distances must be 1-D."
        )

    if not (
        np.isfinite(centroid).all()
        and np.isfinite(embedding_sample).all()
        and np.isfinite(distance_sample).all()
    ):
        raise ValueError(
            f"{model_name}: reference contains non-finite values."
        )

    if summary["reference_split"].iloc[0] != "train":
        raise ValueError(
            f"{model_name}: embedding reference must use train."
        )

    return {
        "centroid": centroid.astype(np.float64),
        "embedding_sample": embedding_sample,
        "distance_sample": distance_sample.astype(
            np.float64
        ),
        "summary": summary,
    }


def monitor_embedding_drift(
    model_name: str,
    current_embeddings: np.ndarray,
) -> dict[str, float | int | str]:
    reference = load_embedding_reference(model_name)

    expected_dimension = MODEL_DIMENSIONS[model_name]

    current = np.asarray(
        current_embeddings,
        dtype=np.float64,
    )

    if current.ndim != 2:
        raise ValueError(
            "Current embeddings must be a 2-D array."
        )

    if current.shape[1] != expected_dimension:
        raise ValueError(
            f"{model_name}: current embeddings have dimension "
            f"{current.shape[1]}, expected {expected_dimension}."
        )

    if current.shape[0] == 0:
        raise ValueError(
            "Current embedding population must not be empty."
        )

    if not np.isfinite(current).all():
        raise ValueError(
            "Current embeddings contain non-finite values."
        )

    reference_centroid = reference["centroid"]

    current_centroid = current.mean(
        axis=0,
        dtype=np.float64,
    )

    centroid_shift = float(
        np.linalg.norm(
            current_centroid - reference_centroid
        )
    )

    current_distances = np.linalg.norm(
        current - reference_centroid,
        axis=1,
    )

    reference_distances = reference["distance_sample"]

    return {
        "model_name": model_name,
        "reference_split": "train",
        "reference_count": int(
            reference["summary"]["reference_count"].iloc[0]
        ),
        "current_count": int(current.shape[0]),
        "embedding_dimension": expected_dimension,
        "reference_centroid_norm": float(
            np.linalg.norm(reference_centroid)
        ),
        "current_centroid_norm": float(
            np.linalg.norm(current_centroid)
        ),
        "centroid_shift": centroid_shift,
        "reference_distance_mean": float(
            reference_distances.mean()
        ),
        "current_distance_mean": float(
            current_distances.mean()
        ),
        "reference_distance_median": float(
            np.median(reference_distances)
        ),
        "current_distance_median": float(
            np.median(current_distances)
        ),
        "distance_psi": population_stability_index(
            reference_distances,
            current_distances,
        ),
        "distance_wasserstein": wasserstein_distance(
            reference_distances,
            current_distances,
        ),
    }
