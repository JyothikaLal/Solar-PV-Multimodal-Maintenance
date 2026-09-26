"""Generate a task-aware comparison report from MLflow runs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops.mlflow_comparison import (
    get_comparison_runs,
    group_comparison_runs,
)


OUTPUT_DIR = PROJECT_ROOT / "reports" / "results" / "mlflow"
CSV_PATH = OUTPUT_DIR / "experiment_comparison.csv"
MD_PATH = OUTPUT_DIR / "experiment_comparison.md"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = get_comparison_runs()

    if not rows:
        raise RuntimeError("No MLflow comparison runs were found.")

    frame = pd.DataFrame(rows)

    frame.to_csv(CSV_PATH, index=False)

    groups = group_comparison_runs(rows)

    lines = [
        "# MLflow Experiment Comparison",
        "",
        "Generated from the local MLflow tracking backend.",
        "",
        "## Comparison policy",
        "",
        "- Runs are grouped by dataset, modality, and task.",
        "- Regression and classification metrics are not ranked against each other.",
        "- Only existing tracked runs are included.",
        "- This report does not retrain or modify any model.",
        "- TECNALIA and RaptorMaps remain independent datasets.",
        "",
        f"Total tracked runs: **{len(rows)}**",
        f"Comparison groups: **{len(groups)}**",
        "",
    ]

    for group_key, group_rows in groups.items():
        dataset, modality, task = group_key

        lines.extend(
            [
                f"## {dataset} — {modality} — {task}",
                "",
            ]
        )

        group_frame = pd.DataFrame(group_rows)

        if task == "regression":
            columns = [
                "model_family",
                "run_name",
                "validation_mae",
                "validation_rmse",
                "validation_r2",
                "test_mae",
                "test_rmse",
                "test_r2",
                "test_mape_percent",
            ]
        elif task == "classification":
            columns = [
                "model_family",
                "run_name",
                "best_validation_macro_f1",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
                "test_weighted_f1",
                "test_roc_auc_ovr_macro",
                "test_pr_auc_macro",
            ]
        else:
            columns = [
                "model_family",
                "run_name",
            ]

        available = [
            column for column in columns
            if column in group_frame.columns
        ]

        table = group_frame[available].copy()

        lines.append("| " + " | ".join(available) + " |")
        lines.append("| " + " | ".join(["---"] * len(available)) + " |")

        for _, row in table.iterrows():
            values = []
            for column in available:
                value = row[column]
                if pd.isna(value):
                    value = ""
                values.append(str(value))

            lines.append("| " + " | ".join(values) + " |")

        lines.append("")

    MD_PATH.write_text("\n".join(lines))

    print("MLflow comparison report generated.")
    print("CSV:", CSV_PATH)
    print("Markdown:", MD_PATH)
    print("Tracked runs:", len(rows))
    print("Comparison groups:", len(groups))


if __name__ == "__main__":
    main()
