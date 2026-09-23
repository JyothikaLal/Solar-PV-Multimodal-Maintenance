from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TASK18_DIR = Path(
    "reports/results/raptormaps/custom_cnn"
)

TASK25_DIR = Path(
    "reports/results/raptormaps/custom_cnn_task25"
)

OUTPUT_DIR = TASK25_DIR / "baseline_comparison"

OVERALL_OUTPUT = (
    OUTPUT_DIR / "task18_vs_task25_overall.csv"
)

PER_CLASS_OUTPUT = (
    OUTPUT_DIR / "task18_vs_task25_per_class.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "comparison_summary.json"
)


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "roc_auc_ovr_macro",
    "pr_auc_macro",
]


def load_overall_metrics(
    path: Path,
) -> dict[str, float]:
    """Load one-row overall metrics CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    frame = pd.read_csv(path)

    if len(frame) != 1:
        raise ValueError(
            f"Expected one metrics row in {path}, "
            f"found {len(frame)}."
        )

    row = frame.iloc[0]

    missing = [
        metric
        for metric in METRICS
        if metric not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metrics in {path}: {missing}"
        )

    return {
        metric: float(row[metric])
        for metric in METRICS
    }


def load_per_class_metrics(
    path: Path,
) -> pd.DataFrame:
    """Load per-class metrics."""

    if not path.exists():
        raise FileNotFoundError(
            f"Per-class metrics not found: {path}"
        )

    frame = pd.read_csv(path)

    required = {
        "class_name",
        "precision",
        "recall",
        "f1",
        "support",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing per-class columns: {sorted(missing)}"
        )

    return frame[
        [
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
        ]
    ].copy()


def build_overall_comparison() -> pd.DataFrame:
    """Build overall Task 18 vs Task 25 comparison."""

    task18 = load_overall_metrics(
        TASK18_DIR / "test_overall_metrics.csv"
    )

    task25 = load_overall_metrics(
        TASK25_DIR / "test_overall_metrics.csv"
    )

    rows = []

    for metric in METRICS:
        value18 = task18[metric]
        value25 = task25[metric]

        absolute_delta = value25 - value18

        relative_delta = (
            absolute_delta / abs(value18) * 100.0
            if value18 != 0
            else float("nan")
        )

        rows.append(
            {
                "metric": metric,
                "task18": value18,
                "task25": value25,
                "absolute_delta_task25_minus_task18": (
                    absolute_delta
                ),
                "relative_delta_percent": (
                    relative_delta
                ),
            }
        )

    return pd.DataFrame(rows)


def build_per_class_comparison() -> pd.DataFrame:
    """Build per-class F1 comparison."""

    task18 = load_per_class_metrics(
        TASK18_DIR / "test_per_class_metrics.csv"
    )

    task25 = load_per_class_metrics(
        TASK25_DIR / "test_per_class_metrics.csv"
    )

    task18 = task18[
        ["class_name", "f1", "support"]
    ].rename(
        columns={
            "f1": "task18_f1",
            "support": "task18_support",
        }
    )

    task25 = task25[
        ["class_name", "f1", "support"]
    ].rename(
        columns={
            "f1": "task25_f1",
            "support": "task25_support",
        }
    )

    comparison = task18.merge(
        task25,
        on="class_name",
        how="outer",
        validate="one_to_one",
    )

    if comparison[
        ["task18_f1", "task25_f1"]
    ].isna().any().any():
        raise ValueError(
            "Class sets differ between Task 18 "
            "and Task 25."
        )

    comparison[
        "f1_delta_task25_minus_task18"
    ] = (
        comparison["task25_f1"]
        - comparison["task18_f1"]
    )

    comparison[
        "support_delta_task25_minus_task18"
    ] = (
        comparison["task25_support"]
        - comparison["task18_support"]
    )

    return comparison.sort_values(
        "class_name"
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall = build_overall_comparison()
    per_class = build_per_class_comparison()

    overall.to_csv(
        OVERALL_OUTPUT,
        index=False,
    )

    per_class.to_csv(
        PER_CLASS_OUTPUT,
        index=False,
    )

    macro_f1_row = overall[
        overall["metric"] == "macro_f1"
    ].iloc[0]

    summary = {
        "task18_result_source": str(
            TASK18_DIR / "test_overall_metrics.csv"
        ),
        "task25_result_source": str(
            TASK25_DIR / "test_overall_metrics.csv"
        ),
        "task18_macro_f1": float(
            macro_f1_row["task18"]
        ),
        "task25_macro_f1": float(
            macro_f1_row["task25"]
        ),
        "macro_f1_delta": float(
            macro_f1_row[
                "absolute_delta_task25_minus_task18"
            ]
        ),
        "comparison_note": (
            "Descriptive baseline comparison. "
            "Task 18 and Task 25 use the same "
            "frozen RaptorMaps split, compact CNN "
            "architecture and training configuration. "
            "Task 25 uses the finalized Task 24 "
            "training-set standardization contract. "
            "This comparison is not a formal causal ablation."
        ),
        "preprocessing_difference": {
            "task18": "range scaling by 255",
            "task25": (
                "training-set standardization; "
                "mean=0.61973207; std=0.15437313"
            ),
        },
    }

    with SUMMARY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("Baseline comparison complete.")
    print()
    print(overall.to_string(index=False))
    print()
    print(
        "Overall comparison:",
        OVERALL_OUTPUT,
    )
    print(
        "Per-class comparison:",
        PER_CLASS_OUTPUT,
    )
    print(
        "Summary:",
        SUMMARY_OUTPUT,
    )


if __name__ == "__main__":
    main()
