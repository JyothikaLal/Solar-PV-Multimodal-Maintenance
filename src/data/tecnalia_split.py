from __future__ import annotations

import pandas as pd


TIMESTAMP_COLUMN = "Fecha"
SPLIT_COLUMN = "split"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def _validate_split_ratios() -> None:
    """Validate that the configured split ratios sum to one."""
    total = TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO

    if not abs(total - 1.0) < 1e-12:
        raise ValueError(
            "Train, validation, and test ratios must sum to 1.0."
        )


def prepare_tecnalia_split_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and chronologically prepare a TECNALIA dataframe.

    The input dataframe is copied and therefore is not modified in place.

    Requirements:
    - Fecha column must exist.
    - Fecha must contain valid timestamps.
    - Duplicate timestamps are rejected.
    - Rows are sorted chronologically.
    """
    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required column: {TIMESTAMP_COLUMN}"
        )

    prepared = df.copy()

    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(
        prepared[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if prepared[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Found invalid timestamps.")

    if prepared[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError("Found duplicate timestamps.")

    prepared = prepared.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)

    return prepared


def create_tecnalia_temporal_split(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a deterministic chronological train/validation/test split.

    Split proportions:
    - earliest 70%: train
    - following 15%: validation
    - latest 15%: test

    No random sampling is used.

    The input dataframe is not modified in place.
    """
    _validate_split_ratios()

    prepared = prepare_tecnalia_split_input(df)

    n_rows = len(prepared)

    if n_rows < 3:
        raise ValueError(
            "At least 3 observations are required for a train/"
            "validation/test split."
        )

    train_end = int(n_rows * TRAIN_RATIO)
    validation_end = int(
        n_rows * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    if train_end <= 0:
        raise ValueError("Training split would be empty.")

    if validation_end <= train_end:
        raise ValueError("Validation split would be empty.")

    if validation_end >= n_rows:
        raise ValueError("Test split would be empty.")

    prepared[SPLIT_COLUMN] = "test"

    prepared.loc[
        prepared.index < train_end,
        SPLIT_COLUMN,
    ] = "train"

    prepared.loc[
        (prepared.index >= train_end)
        & (prepared.index < validation_end),
        SPLIT_COLUMN,
    ] = "validation"

    return prepared


def validate_tecnalia_temporal_split(
    df: pd.DataFrame,
) -> dict:
    """
    Validate temporal split integrity.

    Returns a dictionary containing:
    - total row count
    - split row counts
    - split timestamp boundaries
    - chronological ordering checks
    - timestamp overlap checks
    """
    required_columns = {
        TIMESTAMP_COLUMN,
        SPLIT_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    timestamps = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError("Found invalid timestamps.")

    if timestamps.duplicated().any():
        raise ValueError("Found duplicate timestamps.")

    split_values = set(df[SPLIT_COLUMN].dropna().unique())

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    if split_values != expected_splits:
        raise ValueError(
            "Unexpected split labels. "
            f"Expected {sorted(expected_splits)}, "
            f"found {sorted(split_values)}."
        )

    train = timestamps[df[SPLIT_COLUMN] == "train"]
    validation = timestamps[df[SPLIT_COLUMN] == "validation"]
    test = timestamps[df[SPLIT_COLUMN] == "test"]

    train_end = train.max()
    validation_start = validation.min()
    validation_end = validation.max()
    test_start = test.min()

    train_before_validation = train_end < validation_start
    validation_before_test = validation_end < test_start

    if not train_before_validation:
        raise ValueError(
            "Training and validation timestamps overlap."
        )

    if not validation_before_test:
        raise ValueError(
            "Validation and test timestamps overlap."
        )

    counts = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
    }

    if sum(counts.values()) != len(df):
        raise ValueError(
            "Split row counts do not equal total row count."
        )

    return {
        "total_rows": len(df),
        "train_rows": counts["train"],
        "validation_rows": counts["validation"],
        "test_rows": counts["test"],
        "train_start": train.min(),
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": test.max(),
        "train_before_validation": train_before_validation,
        "validation_before_test": validation_before_test,
        "timestamp_overlap": False,
    }

TECNALIA_MODULES = (
    "Atersa",
    "JaSolar3",
    "NingboSolar",
    "Photowatt",
    "TrinaSolar",
)


def build_tecnalia_split_manifest(
    module_dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build one split manifest from multiple TECNALIA module dataframes.

    Each dataframe must contain Fecha and module_name.
    The returned manifest contains:
    - module_name
    - Fecha
    - split

    No source dataframe is modified.
    """
    if not module_dataframes:
        raise ValueError("No module dataframes were provided.")

    unexpected_modules = set(module_dataframes) - set(TECNALIA_MODULES)

    if unexpected_modules:
        raise ValueError(
            "Unexpected module names: "
            + ", ".join(sorted(unexpected_modules))
        )

    manifest_parts = []

    for module_name in TECNALIA_MODULES:
        if module_name not in module_dataframes:
            raise ValueError(
                f"Missing required module: {module_name}"
            )

        df = module_dataframes[module_name]

        if "module_name" not in df.columns:
            raise ValueError(
                f"Missing module_name column for {module_name}."
            )

        prepared = create_tecnalia_temporal_split(df)

        if not (prepared["module_name"] == module_name).all():
            raise ValueError(
                f"module_name values do not match expected module "
                f"{module_name}."
            )

        manifest_parts.append(
            prepared[
                [
                    "module_name",
                    TIMESTAMP_COLUMN,
                    SPLIT_COLUMN,
                ]
            ].copy()
        )

    manifest = (
        pd.concat(manifest_parts, ignore_index=True)
        .sort_values(
            ["module_name", TIMESTAMP_COLUMN]
        )
        .reset_index(drop=True)
    )

    return manifest


def validate_tecnalia_split_manifest(
    manifest: pd.DataFrame,
) -> dict:
    """
    Validate a multi-module TECNALIA split manifest.
    """
    required_columns = {
        "module_name",
        TIMESTAMP_COLUMN,
        SPLIT_COLUMN,
    }

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            "Missing required manifest columns: "
            + ", ".join(sorted(missing_columns))
        )

    timestamps = pd.to_datetime(
        manifest[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError("Manifest contains invalid timestamps.")

    if manifest[["module_name", TIMESTAMP_COLUMN]].duplicated().any():
        raise ValueError(
            "Manifest contains duplicate module/timestamp pairs."
        )

    modules = set(manifest["module_name"].unique())

    expected_modules = set(TECNALIA_MODULES)

    if modules != expected_modules:
        raise ValueError(
            "Manifest module coverage mismatch."
        )

    module_reports = {}

    for module_name in TECNALIA_MODULES:
        module_df = manifest[
            manifest["module_name"] == module_name
        ].copy()

        report = validate_tecnalia_temporal_split(
            module_df
        )

        module_reports[module_name] = report

    total_rows = len(manifest)

    expected_total = sum(
        report["total_rows"]
        for report in module_reports.values()
    )

    if total_rows != expected_total:
        raise ValueError(
            "Manifest total row count does not match "
            "module totals."
        )

    return {
        "total_rows": total_rows,
        "module_count": len(module_reports),
        "module_reports": module_reports,
    }