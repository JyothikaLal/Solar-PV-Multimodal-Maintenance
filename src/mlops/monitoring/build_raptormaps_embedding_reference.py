from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_PATH = Path(
    "reports/results/raptormaps/embeddings"
)

OUTPUT_PATH = Path(
    "reports/results/raptormaps/monitoring"
)

REFERENCE_SAMPLE_SIZE = 10_000
REFERENCE_RANDOM_STATE = 42


MODELS = {
    "resnet18_finetuned": 512,
    "efficientnet_b0_finetuned": 1280,
}


def build_reference(
    model_name: str,
    expected_dimension: int,
) -> None:
    embedding_path = (
        BASE_PATH / f"{model_name}_embeddings.npz"
    )
    metadata_path = (
        BASE_PATH / f"{model_name}_metadata.csv"
    )

    embeddings = np.load(
        embedding_path
    )["embeddings"]

    metadata = pd.read_csv(metadata_path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"{model_name}: embeddings must be 2-D."
        )

    if embeddings.shape[1] != expected_dimension:
        raise ValueError(
            f"{model_name}: expected {expected_dimension} dimensions, "
            f"got {embeddings.shape[1]}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{model_name}: embeddings contain non-finite values."
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"{model_name}: embedding/metadata row counts differ."
        )

    train_mask = metadata["split"].eq("train")

    train_embeddings = embeddings[train_mask.to_numpy()]

    if train_embeddings.size == 0:
        raise ValueError(
            f"{model_name}: training embedding set is empty."
        )

    centroid = train_embeddings.mean(
        axis=0,
        dtype=np.float64,
    )

    distances = np.linalg.norm(
        train_embeddings.astype(np.float64)
        - centroid,
        axis=1,
    )

    rng = np.random.default_rng(
        REFERENCE_RANDOM_STATE
    )

    sample_size = min(
        REFERENCE_SAMPLE_SIZE,
        len(train_embeddings),
    )

    sample_indices = rng.choice(
        len(train_embeddings),
        size=sample_size,
        replace=False,
    )

    sample_embeddings = train_embeddings[
        sample_indices
    ]

    sample_distances = distances[
        sample_indices
    ]

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_PATH / f"{model_name}_reference.npz",
        centroid=centroid.astype(np.float32),
        embedding_sample=sample_embeddings.astype(
            np.float32
        ),
        distance_sample=sample_distances.astype(
            np.float32
        ),
    )

    summary = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "reference_split": "train",
                "reference_count": int(
                    len(train_embeddings)
                ),
                "sample_count": int(sample_size),
                "embedding_dimension": int(
                    expected_dimension
                ),
                "centroid_norm": float(
                    np.linalg.norm(centroid)
                ),
                "distance_mean": float(
                    distances.mean()
                ),
                "distance_std": float(
                    distances.std()
                ),
                "distance_min": float(
                    distances.min()
                ),
                "distance_p01": float(
                    np.quantile(distances, 0.01)
                ),
                "distance_p05": float(
                    np.quantile(distances, 0.05)
                ),
                "distance_p25": float(
                    np.quantile(distances, 0.25)
                ),
                "distance_median": float(
                    np.median(distances)
                ),
                "distance_p75": float(
                    np.quantile(distances, 0.75)
                ),
                "distance_p95": float(
                    np.quantile(distances, 0.95)
                ),
                "distance_p99": float(
                    np.quantile(distances, 0.99)
                ),
                "distance_max": float(
                    distances.max()
                ),
            }
        ]
    )

    summary.to_csv(
        OUTPUT_PATH / f"{model_name}_reference.csv",
        index=False,
    )

    print(f"\n{model_name}")
    print("Reference split: train")
    print("Reference embeddings:", len(train_embeddings))
    print("Embedding dimension:", expected_dimension)
    print("Reference sample:", sample_size)
    print("Centroid norm:", float(np.linalg.norm(centroid)))
    print(
        "Distance mean:",
        float(distances.mean()),
    )
    print(
        "Distance median:",
        float(np.median(distances)),
    )


if __name__ == "__main__":
    for model_name, expected_dimension in MODELS.items():
        build_reference(
            model_name=model_name,
            expected_dimension=expected_dimension,
        )
