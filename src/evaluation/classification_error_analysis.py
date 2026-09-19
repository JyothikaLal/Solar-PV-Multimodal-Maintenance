from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(
    "reports/results/raptormaps/classification_baselines"
)

MINORITY_CLASSES = (
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Soiling",
    "Diode-Multi",
)


def load_per_class_metrics(model_name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{model_name}_test_per_class.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def analyze_minority_classes(
    model_name: str,
) -> pd.DataFrame:
    df = load_per_class_metrics(model_name)

    result = df[df["class_name"].isin(MINORITY_CLASSES)].copy()

    result["model"] = model_name

    return result[
        [
            "model",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
        ]
    ]


def extract_top_confusions(
    model_name: str,
    top_n: int = 20,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    path = (
        results_dir
        / f"{model_name}_test_confusion_matrix.csv"
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    matrix_df = pd.read_csv(path, index_col=0)

    classes = list(matrix_df.index)
    matrix = matrix_df.to_numpy()

    rows = []

    for i, true_class in enumerate(classes):
        for j, predicted_class in enumerate(classes):
            if i == j:
                continue

            count = int(matrix[i, j])

            if count > 0:
                rows.append(
                    {
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "count": count,
                    }
                )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        by="count",
        ascending=False,
    ).head(top_n)


def create_baseline_comparison() -> pd.DataFrame:
    path = RESULTS_DIR / "baseline_test_summary.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    summary = pd.read_csv(path)

    return summary.sort_values(
        by="macro_f1",
        ascending=False,
    )


def run_error_analysis() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = create_baseline_comparison()

    comparison.to_csv(
        RESULTS_DIR / "baseline_comparison.csv",
        index=False,
    )

    all_minority = []

    for model_name in comparison["model"]:
        minority = analyze_minority_classes(model_name)
        all_minority.append(minority)

    minority_df = pd.concat(
        all_minority,
        ignore_index=True,
    )

    minority_df.to_csv(
        RESULTS_DIR / "minority_class_error_analysis.csv",
        index=False,
    )

    random_forest_confusions = extract_top_confusions(
        "random_forest",
        top_n=20,
    )

    random_forest_confusions.to_csv(
        RESULTS_DIR / "random_forest_top_confusions.csv",
        index=False,
    )

    print("\nBaseline comparison:")
    print(comparison.to_string(index=False))

    print("\nMinority-class test performance:")
    print(minority_df.to_string(index=False))

    print("\nRandom Forest top confusions:")
    print(random_forest_confusions.to_string(index=False))


if __name__ == "__main__":
    run_error_analysis()