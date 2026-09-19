from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(
    "reports/results/raptormaps"
)

OUTPUT_PATH = (
    RESULTS_DIR / "classification_model_comparison.csv"
)


BASELINE_PATH = (
    RESULTS_DIR
    / "classification_baselines"
    / "baseline_comparison.csv"
)

CNN_PATH = (
    RESULTS_DIR
    / "custom_cnn"
    / "test_overall_metrics.csv"
)


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = pd.read_csv(
        BASELINE_PATH
    )

    cnn = pd.read_csv(
        CNN_PATH
    )

    required_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "pr_auc_macro",
    ]

    rows = []

    for _, row in baseline.iterrows():
        rows.append(
            {
                "model": row["model"],
                **{
                    column: row[column]
                    for column in required_columns
                },
            }
        )

    rows.append(
        {
            "model": "custom_cnn",
            **{
                column: cnn.iloc[0][column]
                for column in required_columns
            },
        }
    )

    comparison = pd.DataFrame(
        rows
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(comparison.to_string(index=False))
    print()
    print(
        f"Saved comparison: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
