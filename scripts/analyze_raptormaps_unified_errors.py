from pathlib import Path

import pandas as pd


ROOT = Path("reports/results/raptormaps")

MODEL_PATHS = {
    "custom_cnn_task25": (
        ROOT / "custom_cnn_task25" / "test_predictions.csv"
    ),
    "resnet18_transfer": (
        ROOT / "resnet18_transfer" / "test_predictions.csv"
    ),
    "efficientnet_b0_transfer": (
        ROOT / "efficientnet_b0_transfer" / "test_predictions.csv"
    ),
}

OUT = ROOT / "error_analysis"
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "sample_index",
    "y_true_index",
    "y_pred_index",
    "y_true_class",
    "y_pred_class",
    "confidence",
]


def load_predictions(model_name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"{model_name}: unexpected columns: {list(df.columns)}"
        )

    if len(df) != 2999:
        raise ValueError(
            f"{model_name}: expected 2999 rows, got {len(df)}"
        )

    if df["sample_index"].duplicated().any():
        raise ValueError(f"{model_name}: duplicate sample_index values")

    if not df["confidence"].between(0.0, 1.0).all():
        raise ValueError(
            f"{model_name}: confidence outside [0,1]"
        )

    return df


predictions = {
    name: load_predictions(name, path)
    for name, path in MODEL_PATHS.items()
}

# ---------------------------------------------------------------------
# 1. Top confusion pairs
# ---------------------------------------------------------------------
confusion_rows = []

for model_name, df in predictions.items():
    errors = df[df["y_true_index"] != df["y_pred_index"]]

    confusion = (
        errors.groupby(
            [
                "y_true_index",
                "y_true_class",
                "y_pred_index",
                "y_pred_class",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    confusion.insert(0, "model", model_name)
    confusion_rows.append(confusion)

top_confusions = pd.concat(confusion_rows, ignore_index=True)

top_confusions.to_csv(
    OUT / "top_confusions_all_models.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 2. Per-class error rates
# ---------------------------------------------------------------------
per_class_rows = []

for model_name, df in predictions.items():
    grouped = (
        df.groupby(
            ["y_true_index", "y_true_class"],
            sort=True,
        )
        .agg(
            support=("sample_index", "size"),
            errors=(
                "y_true_index",
                lambda s: (
                    df.loc[s.index, "y_pred_index"] != s
                ).sum(),
            ),
        )
        .reset_index()
    )

    grouped["correct"] = grouped["support"] - grouped["errors"]
    grouped["error_rate"] = (
        grouped["errors"] / grouped["support"]
    )

    grouped.insert(0, "model", model_name)

    per_class_rows.append(grouped)


per_class = pd.concat(per_class_rows, ignore_index=True)

per_class.to_csv(
    OUT / "per_class_error_rates.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 3. Confidence summary
# ---------------------------------------------------------------------
confidence_rows = []

for model_name, df in predictions.items():
    correct = df["y_true_index"] == df["y_pred_index"]
    errors = ~correct

    confidence_rows.append(
        {
            "model": model_name,
            "test_samples": len(df),
            "accuracy": correct.mean(),
            "mean_confidence_all": df["confidence"].mean(),
            "mean_confidence_correct": df.loc[
                correct, "confidence"
            ].mean(),
            "mean_confidence_errors": df.loc[
                errors, "confidence"
            ].mean(),
            "max_error_confidence": df.loc[
                errors, "confidence"
            ].max(),
        }
    )

confidence_summary = pd.DataFrame(confidence_rows)

confidence_summary.to_csv(
    OUT / "confidence_summary.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 4. High-confidence errors
# ---------------------------------------------------------------------
HIGH_CONFIDENCE_THRESHOLD = 0.80

high_confidence_rows = []

for model_name, df in predictions.items():
    errors = df[
        (df["y_true_index"] != df["y_pred_index"])
        & (df["confidence"] >= HIGH_CONFIDENCE_THRESHOLD)
    ].copy()

    errors.insert(0, "model", model_name)
    high_confidence_rows.append(errors)

high_confidence = pd.concat(
    high_confidence_rows,
    ignore_index=True,
)

high_confidence = high_confidence.sort_values(
    ["model", "confidence"],
    ascending=[True, False],
)

high_confidence.to_csv(
    OUT / "high_confidence_errors.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 5. Minority-class analysis
# ---------------------------------------------------------------------
# Derive training support from the frozen split manifest.
# The manifest schema may store either class names or class indices.

manifest_path = (
    ROOT.parent.parent.parent
    / "data"
    / "processed"
    / "raptormaps"
    / "split_manifest.csv"
)

if not manifest_path.exists():
    raise FileNotFoundError(
        f"Missing split manifest: {manifest_path}"
    )

manifest = pd.read_csv(manifest_path)

train_manifest = manifest[
    manifest["split"] == "train"
].copy()

class_name_candidates = [
    "class_name",
    "class",
    "label_name",
    "anomaly_class",
    "category",
]

class_index_candidates = [
    "class_index",
    "label_index",
    "class_id",
    "label",
    "y_true_index",
]

class_name_col = next(
    (
        col
        for col in class_name_candidates
        if col in train_manifest.columns
    ),
    None,
)

if class_name_col is not None:
    train_manifest["resolved_class_name"] = (
        train_manifest[class_name_col].astype(str)
    )
else:
    class_index_col = next(
        (
            col
            for col in class_index_candidates
            if col in train_manifest.columns
        ),
        None,
    )

    if class_index_col is None:
        raise KeyError(
            "Could not find a class-name or class-index column "
            f"in split manifest. Available columns: "
            f"{train_manifest.columns.tolist()}"
        )

    reference_model = next(iter(predictions))
    label_map = (
        predictions[reference_model][
            ["y_true_index", "y_true_class"]
        ]
        .drop_duplicates()
        .set_index("y_true_index")["y_true_class"]
        .to_dict()
    )

    train_manifest["resolved_class_name"] = (
        pd.to_numeric(
            train_manifest[class_index_col],
            errors="coerce",
        )
        .map(label_map)
    )

    if train_manifest["resolved_class_name"].isna().any():
        missing_values = (
            train_manifest.loc[
                train_manifest["resolved_class_name"].isna(),
                class_index_col,
            ]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Could not map all manifest class indices to class names. "
            f"Unmapped values: {missing_values}"
        )

train_counts = (
    train_manifest["resolved_class_name"]
    .value_counts()
    .rename_axis("class_name")
    .reset_index(name="train_support")
)

minority_threshold = train_counts["train_support"].median()

train_counts["minority_class"] = (
    train_counts["train_support"] < minority_threshold
)

minority_rows = []

for model_name, df in predictions.items():
    tmp = (
        df.groupby(
            ["y_true_index", "y_true_class"],
            sort=True,
        )
        .agg(
            test_support=("sample_index", "size"),
            errors=(
                "y_true_index",
                lambda s: (
                    df.loc[s.index, "y_pred_index"] != s
                ).sum(),
            ),
        )
        .reset_index()
    )

    tmp["test_error_rate"] = (
        tmp["errors"] / tmp["test_support"]
    )

    tmp = tmp.rename(
        columns={"y_true_class": "class_name"}
    )

    tmp = tmp.merge(
        train_counts,
        on="class_name",
        how="left",
        validate="one_to_one",
    )

    if tmp["train_support"].isna().any():
        missing_classes = (
            tmp.loc[
                tmp["train_support"].isna(),
                "class_name",
            ]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Missing training support for classes: "
            f"{missing_classes}"
        )

    tmp.insert(0, "model", model_name)
    minority_rows.append(tmp)

minority_analysis = pd.concat(
    minority_rows,
    ignore_index=True,
)

minority_analysis = minority_analysis[
    [
        "model",
        "y_true_index",
        "class_name",
        "train_support",
        "minority_class",
        "test_support",
        "errors",
        "test_error_rate",
    ]
]

minority_analysis.to_csv(
    OUT / "minority_class_analysis.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 6. Model-by-class error comparison
# ---------------------------------------------------------------------
comparison = per_class[
    [
        "model",
        "y_true_index",
        "y_true_class",
        "support",
        "errors",
        "error_rate",
    ]
].copy()

comparison = comparison.rename(
    columns={"y_true_class": "class_name"}
)

pivot = comparison.pivot(
    index=["y_true_index", "class_name"],
    columns="model",
    values="error_rate",
).reset_index()

pivot.columns.name = None

pivot = pivot.rename(
    columns={
        "custom_cnn_task25": "custom_cnn_error_rate",
        "resnet18_transfer": "resnet18_error_rate",
        "efficientnet_b0_transfer": "efficientnet_b0_error_rate",
    }
)

pivot.to_csv(
    OUT / "model_per_class_error_comparison.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Validation output
# ---------------------------------------------------------------------
print("Unified RaptorMaps error analysis: PASSED")
print(f"Output directory: {OUT}")
print(
    f"Top confusion rows: {len(top_confusions)}"
)
print(
    f"Per-class rows: {len(per_class)}"
)
print(
    f"High-confidence errors: {len(high_confidence)}"
)
print(
    f"Minority-analysis rows: {len(minority_analysis)}"
)
print(
    f"Model/class comparison rows: {len(pivot)}"
)
