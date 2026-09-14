from pathlib import Path

from src.data.raptormaps_eda import (
    analyze_class_distribution,
    analyze_class_imbalance,
    analyze_image_statistics,
    analyze_intensity_features,
    load_raptormaps_manifest,
    plot_class_examples,
    summarize_dataset,
    summarize_intensity_by_class,
    summarize_preprocessing_requirements,
)


OUTPUT_DIR = Path("reports/results/raptormaps")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_raptormaps_manifest()

    per_image, per_class = analyze_image_statistics(manifest)
    dataset_summary = summarize_dataset(per_image)

    class_distribution, split_distribution = analyze_class_distribution(
        manifest
    )

    class_distribution.to_csv(
        OUTPUT_DIR / "class_distribution.csv"
    )

    split_distribution.to_csv(
        OUTPUT_DIR / "class_split_distribution.csv"
    )

    per_image.to_csv(
        OUTPUT_DIR / "image_statistics.csv",
        index=False,
    )

    per_class.to_csv(
        OUTPUT_DIR / "class_image_statistics.csv",
    )

    dataset_summary.to_csv(
        OUTPUT_DIR / "dataset_image_summary.csv",
        index=False,
    )

    print("=== Dataset Summary ===")
    print(dataset_summary.to_string(index=False))

    print("\n=== Per-Class Image Statistics ===")
    print(per_class.to_string())

    print("\nRaptorMaps 12.1 EDA completed.")

    print("\n=== Class Distribution ===")
    print(class_distribution.to_string())

    print("\n=== Class Distribution by Split ===")
    print(split_distribution.to_string())

    intensity_features = analyze_intensity_features(manifest)

    intensity_class_summary = summarize_intensity_by_class(
        intensity_features
    )
    intensity_features.to_csv(
        OUTPUT_DIR / "intensity_features.csv",
        index=False,
    )

    intensity_class_summary.to_csv(
        OUTPUT_DIR / "intensity_class_statistics.csv"
    )

    print("\n=== Intensity Statistics by Class ===")
    print(intensity_class_summary.to_string())

    plot_class_examples(manifest)

    print("\nRaptorMaps class visualization completed.")  

    imbalance_summary = analyze_class_imbalance(manifest)

    imbalance_summary.to_csv(
        OUTPUT_DIR / "class_imbalance_summary.csv"
    )

    print("\n=== Class Imbalance Summary ===")
    print(imbalance_summary.to_string()) 

    preprocessing_summary = summarize_preprocessing_requirements(
        manifest
    )

    preprocessing_summary.to_csv(
        OUTPUT_DIR / "preprocessing_summary.csv",
        index=False,
    )

    print("\n=== Preprocessing Summary ===")
    print(preprocessing_summary.to_string(index=False)) 

if __name__ == "__main__":
    main()
