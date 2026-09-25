from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("reports/results/raptormaps")
OUTPUT_DIR = ROOT / "dl_evaluation"

MODEL_NAMES = [
    "custom_cnn_task25",
    "resnet18_transfer_frozen",
    "efficientnet_b0_transfer_frozen",
    "resnet18_finetune",
    "efficientnet_b0_finetune",
]

MODEL_DIRS = {
    "custom_cnn_task25": ROOT / "custom_cnn_task25",
    "resnet18_transfer_frozen": ROOT / "resnet18_transfer",
    "efficientnet_b0_transfer_frozen": ROOT / "efficientnet_b0_transfer",
    "resnet18_finetune": ROOT / "resnet18_finetune",
    "efficientnet_b0_finetune": ROOT / "efficientnet_b0_finetune",
}

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

EXPECTED_CLASSES = [
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Diode",
    "Diode-Multi",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "No-Anomaly",
    "Offline-Module",
    "Shadowing",
    "Soiling",
    "Vegetation",
]

MINORITY_CLASSES = {
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Soiling",
    "Diode-Multi",
}


def load_model_evaluation(
    model_name: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    model_dir = MODEL_DIRS[model_name]

    overall_path = model_dir / "test_overall_metrics.csv"
    per_class_path = model_dir / "test_per_class_metrics.csv"
    confusion_path = model_dir / "test_confusion_matrix.csv"

    for path in (
        overall_path,
        per_class_path,
        confusion_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{model_name}: missing evaluation artifact: {path}"
            )

    overall = pd.read_csv(overall_path)
    per_class = pd.read_csv(per_class_path)
    confusion = pd.read_csv(
        confusion_path,
        index_col=0,
    )

    if overall.shape != (1, 8):
        raise ValueError(
            f"{model_name}: unexpected overall shape "
            f"{overall.shape}"
        )

    missing_metrics = [
        column
        for column in METRIC_COLUMNS
        if column not in overall.columns
    ]

    if missing_metrics:
        raise ValueError(
            f"{model_name}: missing metrics "
            f"{missing_metrics}"
        )

    if len(per_class) != 12:
        raise ValueError(
            f"{model_name}: expected 12 per-class rows; "
            f"got {len(per_class)}"
        )

    if list(per_class["class_name"]) != EXPECTED_CLASSES:
        raise ValueError(
            f"{model_name}: class ordering mismatch"
        )

    if list(confusion.index) != EXPECTED_CLASSES:
        raise ValueError(
            f"{model_name}: confusion-matrix row ordering mismatch"
        )

    if list(confusion.columns) != EXPECTED_CLASSES:
        raise ValueError(
            f"{model_name}: confusion-matrix column ordering mismatch"
        )

    row = overall.iloc[0].to_dict()
    row["model"] = model_name

    return row, per_class, confusion


def load_imbalance() -> pd.DataFrame:
    path = ROOT / "class_imbalance_summary.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing class imbalance artifact: {path}"
        )

    imbalance = pd.read_csv(path)

    required = {
        "anomaly_class",
        "total_count",
        "percentage",
        "train",
        "validation",
        "test",
        "minority_class",
    }

    missing = required - set(imbalance.columns)

    if missing:
        raise ValueError(
            f"Missing imbalance columns: {sorted(missing)}"
        )

    if len(imbalance) != 12:
        raise ValueError(
            f"Expected 12 imbalance rows; got {len(imbalance)}"
        )

    actual_minority = set(
        imbalance.loc[
            imbalance["minority_class"].astype(bool),
            "anomaly_class",
        ]
    )

    if actual_minority != MINORITY_CLASSES:
        raise ValueError(
            "Minority-class definition mismatch: "
            f"{actual_minority}"
        )

    return imbalance


def build_per_class_error_rates(
    per_class_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for model_name, per_class in per_class_frames.items():
        frame = per_class[
            [
                "class_name",
                "precision",
                "recall",
                "f1",
                "support",
            ]
        ].copy()

        frame["model"] = model_name
        frame["error_rate"] = 1.0 - frame["recall"]

        rows.append(frame)

    return pd.concat(
        rows,
        ignore_index=True,
    )[
        [
            "model",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
            "error_rate",
        ]
    ]


def build_minority_analysis(
    error_rates: pd.DataFrame,
    imbalance: pd.DataFrame,
) -> pd.DataFrame:
    minority = imbalance[
        imbalance["minority_class"].astype(bool)
    ][
        [
            "anomaly_class",
            "train",
            "validation",
            "test",
            "minority_class",
        ]
    ].rename(
        columns={
            "anomaly_class": "class_name",
            "train": "train_support",
            "validation": "validation_support",
            "test": "test_support",
        }
    )

    result = error_rates.merge(
        minority,
        on="class_name",
        how="inner",
        validate="many_to_one",
    )

    result["errors"] = (
        result["support"] * result["error_rate"]
    ).round().astype(int)

    return result[
        [
            "model",
            "class_name",
            "train_support",
            "validation_support",
            "test_support",
            "support",
            "precision",
            "recall",
            "f1",
            "error_rate",
            "errors",
            "minority_class",
        ]
    ]


def build_confusion_analysis(
    confusion_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for model_name, confusion in confusion_frames.items():
        matrix = confusion.to_numpy()

        for i, true_class in enumerate(EXPECTED_CLASSES):
            true_class_errors = int(
                matrix[i, :].sum() - matrix[i, i]
            )

            for j, predicted_class in enumerate(EXPECTED_CLASSES):
                if i == j:
                    continue

                count = int(matrix[i, j])

                if count <= 0:
                    continue

                error_share = (
                    count / true_class_errors
                    if true_class_errors > 0
                    else 0.0
                )

                rows.append(
                    {
                        "model": model_name,
                        "y_true_index": i,
                        "y_true_class": true_class,
                        "y_pred_index": j,
                        "y_pred_class": predicted_class,
                        "count": count,
                        "total_true_class_errors": true_class_errors,
                        "error_share_within_true_class": error_share,
                        "minority_true_class": (
                            true_class in MINORITY_CLASSES
                        ),
                    }
                )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        ["model", "count"],
        ascending=[True, False],
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    imbalance = load_imbalance()

    overall_rows = []
    per_class_frames = {}
    confusion_frames = {}

    for model_name in MODEL_NAMES:
        (
            overall_row,
            per_class,
            confusion,
        ) = load_model_evaluation(model_name)

        overall_rows.append(overall_row)
        per_class_frames[model_name] = per_class.copy()
        confusion_frames[model_name] = confusion.copy()

    overall = pd.DataFrame(overall_rows)

    preferred_columns = [
        "model",
        *METRIC_COLUMNS,
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in overall.columns
    ]

    overall = overall[available_columns]

    per_class = pd.concat(
        [
            frame.assign(model=model_name)
            for model_name, frame
            in per_class_frames.items()
        ],
        ignore_index=True,
    )

    per_class = per_class[
        [
            "model",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
        ]
    ]

    error_rates = build_per_class_error_rates(
        per_class_frames
    )

    minority_analysis = build_minority_analysis(
        error_rates,
        imbalance,
    )

    confusion_analysis = build_confusion_analysis(
        confusion_frames
    )

    # ---------------------------------------------------------------
    # Metric deltas versus the Custom CNN reference.
    # ---------------------------------------------------------------

    baseline = overall.loc[
        overall["model"] == "custom_cnn_task25"
    ].iloc[0]

    delta_rows = []

    for _, row in overall.iterrows():
        delta = {"model": row["model"]}

        for metric in METRIC_COLUMNS:
            delta[
                f"{metric}_delta_vs_custom_cnn"
            ] = (
                row[metric]
                - baseline[metric]
            )

        delta_rows.append(delta)

    deltas = pd.DataFrame(delta_rows)

    # ---------------------------------------------------------------
    # Write Task-28 artifacts.
    # ---------------------------------------------------------------

    overall.to_csv(
        OUTPUT_DIR / "overall_model_comparison.csv",
        index=False,
    )

    per_class.to_csv(
        OUTPUT_DIR / "per_class_model_comparison.csv",
        index=False,
    )

    deltas.to_csv(
        OUTPUT_DIR / "metric_deltas_vs_custom_cnn.csv",
        index=False,
    )

    error_rates.to_csv(
        OUTPUT_DIR / "per_class_error_rates.csv",
        index=False,
    )

    minority_analysis.to_csv(
        OUTPUT_DIR / "minority_class_analysis.csv",
        index=False,
    )

    confusion_analysis[
        [
            "model",
            "y_true_index",
            "y_true_class",
            "y_pred_index",
            "y_pred_class",
            "count",
        ]
    ].to_csv(
        OUTPUT_DIR / "top_confusions_all_models.csv",
        index=False,
    )

    confusion_analysis.to_csv(
        OUTPUT_DIR / "error_case_summary.csv",
        index=False,
    )

    print("\nTask-28 DL evaluation completed.")
    print("\nOverall model comparison:")
    print(overall.to_string(index=False))

    print("\nMinority-class analysis:")
    print(minority_analysis.to_string(index=False))

    print("\nTop confusion counts by model:")
    for model_name in MODEL_NAMES:
        model_confusions = confusion_analysis[
            confusion_analysis["model"] == model_name
        ].head(5)

        print(f"\n{model_name}")
        print(model_confusions.to_string(index=False))

    print("\nArtifacts written to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
