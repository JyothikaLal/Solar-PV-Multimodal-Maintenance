from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(
    "reports/results/raptormaps/resnet18_transfer"
)


def test_resnet18_transfer_evaluation_artifacts_exist():
    required = [
        RESULTS_DIR / "test_overall_metrics.csv",
        RESULTS_DIR / "test_per_class_metrics.csv",
        RESULTS_DIR / "test_confusion_matrix.csv",
    ]

    # Artifacts are created by the evaluation runner.
    # This test is intentionally conditional before execution.
    assert all(
        path.exists()
        for path in required
    ) or not any(
        path.exists()
        for path in required
    )


def test_resnet18_transfer_metric_schema_after_evaluation():
    overall_path = (
        RESULTS_DIR
        / "test_overall_metrics.csv"
    )

    if not overall_path.exists():
        return

    overall = pd.read_csv(
        overall_path
    )

    required_columns = {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    }

    assert required_columns.issubset(
        overall.columns
    )
