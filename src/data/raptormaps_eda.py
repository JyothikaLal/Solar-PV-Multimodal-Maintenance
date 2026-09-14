from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt


RAPTORMAPS_RAW_DIR = Path("data/raw/raptormaps/InfraredSolarModules")
RAPTORMAPS_MANIFEST = Path("data/processed/raptormaps/split_manifest.csv")


def load_raptormaps_manifest(
    manifest_path: str | Path = RAPTORMAPS_MANIFEST,
) -> pd.DataFrame:
    """Load the validated RaptorMaps split manifest."""

    return pd.read_csv(manifest_path)


def analyze_image_statistics(
    manifest: pd.DataFrame,
    raw_dir: str | Path = RAPTORMAPS_RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate per-image and per-class pixel statistics."""

    raw_dir = Path(raw_dir)
    records = []

    for row in manifest.itertuples(index=False):
        image_path = raw_dir / row.image_filepath

        with Image.open(image_path) as image:
            array = np.asarray(image, dtype=np.float32)

        records.append(
            {
                "metadata_id": row.metadata_id,
                "anomaly_class": row.anomaly_class,
                "split": row.split,
                "width": image.width,
                "height": image.height,
                "pixel_min": float(array.min()),
                "pixel_max": float(array.max()),
                "pixel_mean": float(array.mean()),
                "pixel_std": float(array.std()),
            }
        )

    per_image = pd.DataFrame(records)

    per_class = (
        per_image.groupby("anomaly_class")
        .agg(
            image_count=("metadata_id", "count"),
            mean_pixel_min=("pixel_min", "mean"),
            mean_pixel_max=("pixel_max", "mean"),
            mean_pixel_mean=("pixel_mean", "mean"),
            mean_pixel_std=("pixel_std", "mean"),
        )
        .sort_values("image_count", ascending=False)
        .round(4)
    )

    return per_image, per_class


def summarize_dataset(per_image: pd.DataFrame) -> pd.DataFrame:
    """Create a dataset-level image-statistics summary."""

    return pd.DataFrame(
        [
            {
                "image_count": len(per_image),
                "unique_widths": per_image["width"].nunique(),
                "unique_heights": per_image["height"].nunique(),
                "min_pixel_value": per_image["pixel_min"].min(),
                "max_pixel_value": per_image["pixel_max"].max(),
                "mean_pixel_value": per_image["pixel_mean"].mean(),
                "mean_image_std": per_image["pixel_std"].mean(),
            }
        ]
    ).round(4)

def analyze_class_distribution(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze overall and split-wise class distribution."""

    overall_counts = (
        manifest.groupby("anomaly_class")
        .size()
        .rename("image_count")
        .to_frame()
    )

    overall_counts["percentage"] = (
        overall_counts["image_count"]
        / len(manifest)
        * 100
    )

    split_counts = (
        manifest.groupby(["anomaly_class", "split"])
        .size()
        .unstack(fill_value=0)
    )

    for split in ["train", "validation", "test"]:
        if split not in split_counts.columns:
            split_counts[split] = 0

    split_counts = split_counts[
        ["train", "validation", "test"]
    ]

    split_counts["total_included"] = split_counts.sum(axis=1)

    split_counts["minority_ratio_vs_no_anomaly"] = (
        split_counts["total_included"]
        / split_counts.loc["No-Anomaly", "total_included"]
    )

    return (
        overall_counts.sort_values(
            "image_count",
            ascending=False,
        ).round(4),
        split_counts.sort_values(
            "total_included",
            ascending=False,
        ).round(4),
    )

def analyze_intensity_features(
    manifest: pd.DataFrame,
    raw_dir: str | Path = RAPTORMAPS_RAW_DIR,
) -> pd.DataFrame:
    """Calculate detailed pixel-intensity statistics for every image."""

    raw_dir = Path(raw_dir)
    records = []

    for row in manifest.itertuples(index=False):
        image_path = raw_dir / row.image_filepath

        with Image.open(image_path) as image:
            array = np.asarray(image, dtype=np.float32)

        records.append(
            {
                "metadata_id": row.metadata_id,
                "anomaly_class": row.anomaly_class,
                "split": row.split,
                "pixel_mean": float(array.mean()),
                "pixel_std": float(array.std()),
                "pixel_min": float(array.min()),
                "pixel_max": float(array.max()),
                "pixel_range": float(array.max() - array.min()),
                "pixel_median": float(np.median(array)),
                "pixel_p25": float(np.percentile(array, 25)),
                "pixel_p75": float(np.percentile(array, 75)),
                "fraction_above_180": float((array > 180).mean()),
                "fraction_above_220": float((array > 220).mean()),
                "fraction_above_240": float((array > 240).mean()),
            }
        )

    return pd.DataFrame(records)

def summarize_intensity_by_class(
    intensity_features: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize intensity features by anomaly class."""

    feature_columns = [
        "pixel_mean",
        "pixel_std",
        "pixel_min",
        "pixel_max",
        "pixel_range",
        "pixel_median",
        "pixel_p25",
        "pixel_p75",
        "fraction_above_180",
        "fraction_above_220",
        "fraction_above_240",
    ]

    return (
        intensity_features.groupby("anomaly_class")[feature_columns]
        .mean()
        .round(4)
    )

def plot_class_examples(
    manifest: pd.DataFrame,
    raw_dir: str | Path = RAPTORMAPS_RAW_DIR,
    output_path: str | Path = (
        "reports/figures/raptormaps/class_examples.png"
    ),
    samples_per_class: int = 4,
) -> None:
    """Create a representative image grid for every anomaly class."""

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = sorted(manifest["anomaly_class"].unique())

    fig, axes = plt.subplots(
        len(classes),
        samples_per_class,
        figsize=(12, 24),
    )

    for row_idx, anomaly_class in enumerate(classes):
        class_rows = (
            manifest[manifest["anomaly_class"] == anomaly_class]
            .sort_values("metadata_id")
            .head(samples_per_class)
        )

        for col_idx, (_, row) in enumerate(class_rows.iterrows()):
            image_path = raw_dir / row["image_filepath"]

            with Image.open(image_path) as image:
                image_array = np.asarray(image)

            axes[row_idx, col_idx].imshow(
                image_array,
                cmap="gray",
            )
            axes[row_idx, col_idx].axis("off")

            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(
                    anomaly_class,
                    rotation=0,
                    labelpad=65,
                    va="center",
                )

    fig.suptitle(
        "RaptorMaps Representative Images by Anomaly Class"
    )
    fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def analyze_class_imbalance(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify class imbalance and split support."""

    total_counts = (
        manifest.groupby("anomaly_class")
        .size()
        .rename("total_count")
    )

    percentages = (
        total_counts / len(manifest) * 100
    ).rename("percentage")

    split_counts = (
        manifest.groupby(["anomaly_class", "split"])
        .size()
        .unstack(fill_value=0)
    )

    for split in ["train", "validation", "test"]:
        if split not in split_counts.columns:
            split_counts[split] = 0

    split_counts = split_counts[
        ["train", "validation", "test"]
    ]

    largest_class_count = total_counts.max()

    result = pd.concat(
        [
            total_counts,
            percentages,
            split_counts,
        ],
        axis=1,
    )

    result["ratio_vs_largest_class"] = (
        result["total_count"] / largest_class_count
    )

    result["ratio_vs_no_anomaly"] = (
        result["total_count"]
        / result.loc["No-Anomaly", "total_count"]
    )

    result["minority_class"] = (
        result["percentage"] < 2.0
    )

    return result.sort_values(
        "total_count",
        ascending=False,
    ).round(4)

def summarize_preprocessing_requirements(
    manifest: pd.DataFrame,
    raw_dir: str | Path = RAPTORMAPS_RAW_DIR,
) -> pd.DataFrame:
    """Summarize image properties relevant to preprocessing decisions."""

    raw_dir = Path(raw_dir)

    widths = []
    heights = []
    modes = []
    min_values = []
    max_values = []

    for row in manifest.itertuples(index=False):
        image_path = raw_dir / row.image_filepath

        with Image.open(image_path) as image:
            array = np.asarray(image)

            widths.append(image.width)
            heights.append(image.height)
            modes.append(image.mode)
            min_values.append(float(array.min()))
            max_values.append(float(array.max()))

    return pd.DataFrame(
        [
            {
                "image_count": len(manifest),
                "unique_widths": len(set(widths)),
                "unique_heights": len(set(heights)),
                "unique_modes": len(set(modes)),
                "modes": ",".join(sorted(set(modes))),
                "width": widths[0],
                "height": heights[0],
                "tensor_channels": 1,
                "tensor_height": heights[0],
                "tensor_width": widths[0],
                "global_min_pixel": min(min_values),
                "global_max_pixel": max(max_values),
                "native_aspect_ratio": round(
                    heights[0] / widths[0], 4
                ),
                "recommended_scaling": "divide_by_255",
                "per_image_normalization": False,
                "native_resolution_preserved": True,
            }
        ]
    )
