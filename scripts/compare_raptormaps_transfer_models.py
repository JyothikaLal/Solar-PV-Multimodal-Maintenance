from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(
    "reports/results/raptormaps"
)

OUTPUT_DIR = ROOT / "model_comparison"

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "roc_auc_ovr_macro",
    "pr_auc_macro",
]


MODEL_PATHS = {
    "custom_cnn_task25": {
        "metrics": ROOT
        / "custom_cnn_task25"
        / "test_overall_metrics.csv",
        "per_class": ROOT
        / "custom_cnn_task25"
        / "test_per_class_metrics.csv",
        "config": ROOT
        / "custom_cnn_task25"
        / "training_config.json",
    },
    "resnet18_transfer_frozen": {
        "metrics": ROOT
        / "resnet18_transfer"
        / "test_overall_metrics.csv",
        "per_class": ROOT
        / "resnet18_transfer"
        / "test_per_class_metrics.csv",
        "config": ROOT
        / "resnet18_transfer"
        / "training_config.json",
    },
    "efficientnet_b0_transfer_frozen": {
        "metrics": ROOT
        / "efficientnet_b0_transfer"
        / "test_overall_metrics.csv",
        "per_class": ROOT
        / "efficientnet_b0_transfer"
        / "test_per_class_metrics.csv",
        "config": ROOT
        / "efficientnet_b0_transfer"
        / "training_config.json",
    },
}


PARAMETER_COUNTS = {
    "custom_cnn_task25": {
        "total_parameters": 94668,
        "trainable_parameters": 94668,
    },
    "resnet18_transfer_frozen": {
        "total_parameters": 11182668,
        "trainable_parameters": 6156,
    },
    "efficientnet_b0_transfer_frozen": {
        "total_parameters": 4022920,
        "trainable_parameters": 15372,
    },
}


def load_model_record(
    model_name: str,
    paths: dict[str, Path],
) -> tuple[dict, pd.DataFrame]:
    metrics_path = paths["metrics"]
    per_class_path = paths["per_class"]

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics file for {model_name}: "
            f"{metrics_path}"
        )

    if not per_class_path.exists():
        raise FileNotFoundError(
            f"Missing per-class file for {model_name}: "
            f"{per_class_path}"
        )

    metrics = pd.read_csv(metrics_path)

    if metrics.shape != (1, 8):
        raise ValueError(
            f"Unexpected metrics shape for "
            f"{model_name}: {metrics.shape}"
        )

    missing_metrics = set(METRIC_COLUMNS) - set(
        metrics.columns
    )

    if missing_metrics:
        raise ValueError(
            f"Missing metrics for {model_name}: "
            f"{sorted(missing_metrics)}"
        )

    per_class = pd.read_csv(per_class_path)

    if len(per_class) != 12:
        raise ValueError(
            f"Expected 12 classes for {model_name}; "
            f"got {len(per_class)}"
        )

    row = metrics.iloc[0].to_dict()

    row["model"] = model_name

    row["total_parameters"] = (
        PARAMETER_COUNTS[model_name][
            "total_parameters"
        ]
    )

    row["trainable_parameters"] = (
        PARAMETER_COUNTS[model_name][
            "trainable_parameters"
        ]
    )

    config_path = paths["config"]

    if config_path.exists():
        config = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

        row["best_epoch"] = config.get(
            "best_epoch"
        )

        row["best_validation_macro_f1"] = config.get(
            "best_validation_macro_f1"
        )

    return row, per_class


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_rows = []
    per_class_frames = []

    for model_name, paths in MODEL_PATHS.items():
        row, per_class = load_model_record(
            model_name,
            paths,
        )

        overall_rows.append(row)

        per_class = per_class.copy()
        per_class["model"] = model_name

        per_class_frames.append(
            per_class
        )

    overall = pd.DataFrame(
        overall_rows
    )

    preferred_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "pr_auc_macro",
        "total_parameters",
        "trainable_parameters",
        "best_epoch",
        "best_validation_macro_f1",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in overall.columns
    ]

    overall = overall[
        existing_columns
    ]

    overall.to_csv(
        OUTPUT_DIR
        / "overall_model_comparison.csv",
        index=False,
    )

    per_class_all = pd.concat(
        per_class_frames,
        ignore_index=True,
    )

    per_class_all.to_csv(
        OUTPUT_DIR
        / "per_class_model_comparison.csv",
        index=False,
    )

    baseline = overall[
        overall["model"]
        == "custom_cnn_task25"
    ].iloc[0]

    delta_rows = []

    for _, row in overall.iterrows():
        delta = {
            "model": row["model"],
        }

        for metric in METRIC_COLUMNS:
            delta[
                f"{metric}_delta_vs_custom_cnn"
            ] = (
                row[metric]
                - baseline[metric]
            )

        delta_rows.append(delta)

    deltas = pd.DataFrame(
        delta_rows
    )

    deltas.to_csv(
        OUTPUT_DIR
        / "metric_deltas_vs_custom_cnn.csv",
        index=False,
    )

    print()
    print("Overall model comparison:")
    print(
        overall.to_string(
            index=False
        )
    )

    print()
    print(
        "Metric deltas versus Custom CNN:"
    )
    print(
        deltas.to_string(
            index=False
        )
    )

    print()
    print(
        "Artifacts written to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
