from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.tecnalia_features import engineer_tecnalia_features
from src.data.tecnalia_preprocessing import preprocess_tecnalia_module


TIMESTAMP_COLUMN = "Fecha"
MODULE_COLUMN = "module_name"
TARGET_COLUMN = "normalized_pmpp"
OPERATING_COLUMN = "Front GPOA (W/m²)"
OPERATING_THRESHOLD = 200.0

RAW_ROOT = Path("data/raw/tecnalia/TECNALIA")

SPLIT_MANIFEST = Path(
    "reports/results/tecnalia/tecnalia_split_manifest.csv"
)

MODULE_PATHS = {
    "Atersa": Path("data_Atersa/data_Atersa.csv"),
    "JaSolar3": Path("data_JaSolar3/data_JaSolar3.csv"),
    "NingboSolar": Path("data_NingboSolar/data_NingboSolar.csv"),
    "Photowatt": Path("data_Photowatt/data_Photowatt.csv"),
    "TrinaSolar": Path("data_TrinaSolar/data_TrinaSolar.csv"),
}

BASE_NUMERIC_FEATURES = [
    "Front GPOA (W/m²)",
    "GHI (W/m²)",
    "Temp. Mod (°C)",
    "Amb. Temp. (°C)",
    "Wind Speed (m/s)",
]

BASE_CATEGORICAL_FEATURES = [
    MODULE_COLUMN,
]

LEAKAGE_COLUMNS = {
    "Pmpp",
    "Vmpp",
    "Impp",
    TARGET_COLUMN,
}


def load_tecnalia_regression_modules(
    raw_root: str | Path,
    module_paths: dict[str, str | Path],
) -> dict[str, pd.DataFrame]:
    """
    Load and preprocess all TECNALIA module datasets.

    The existing Task 14 preprocessing implementation is reused as the
    single preprocessing source of truth.
    """
    root = Path(raw_root)
    modules: dict[str, pd.DataFrame] = {}

    for module_name, relative_path in module_paths.items():
        file_path = root / relative_path
        processed, _ = preprocess_tecnalia_module(
            file_path=file_path,
            module_name=module_name,
        )
        modules[module_name] = processed

    return modules


def engineer_tecnalia_regression_modules(
    modules: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Apply the existing Task 15 feature-engineering pipeline."""
    return {
        module_name: engineer_tecnalia_features(df)
        for module_name, df in modules.items()
    }


def build_tecnalia_regression_frame(
    modules: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine module datasets and construct the regression target.

    Only daytime/operating observations with Front GPOA >= 200 W/m²
    are retained.
    """
    if not modules:
        raise ValueError("At least one TECNALIA module dataframe is required.")

    frames = []

    for module_name, df in modules.items():
        prepared = df.copy()

        if MODULE_COLUMN not in prepared.columns:
            prepared[MODULE_COLUMN] = module_name

        if TARGET_COLUMN not in prepared.columns:
            if "Pmpp" not in prepared.columns:
                raise ValueError(
                    f"{module_name}: Pmpp is required to construct "
                    f"{TARGET_COLUMN}."
                )

            if "rated_power" not in prepared.columns:
                raise ValueError(
                    f"{module_name}: rated_power is required to construct "
                    f"{TARGET_COLUMN}."
                )

            prepared[TARGET_COLUMN] = (
                prepared["Pmpp"] / prepared["rated_power"]
            )

        prepared = prepared.loc[
            prepared[OPERATING_COLUMN].ge(OPERATING_THRESHOLD)
        ].copy()

        frames.append(prepared)

    combined = pd.concat(frames, ignore_index=True)

    combined[TIMESTAMP_COLUMN] = pd.to_datetime(
        combined[TIMESTAMP_COLUMN],
        errors="raise",
    )

    combined = combined.sort_values(
        [TIMESTAMP_COLUMN, MODULE_COLUMN]
    ).reset_index(drop=True)

    return combined


def attach_frozen_tecnalia_split(
    df: pd.DataFrame,
    split_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the frozen Task 16 split using module/timestamp identity.
    """
    required_manifest_columns = {
        MODULE_COLUMN,
        TIMESTAMP_COLUMN,
        "split",
    }

    missing = required_manifest_columns - set(split_manifest.columns)
    if missing:
        raise ValueError(
            f"Split manifest is missing required columns: {sorted(missing)}"
        )

    prepared = df.copy()
    manifest = split_manifest[
        [MODULE_COLUMN, TIMESTAMP_COLUMN, "split"]
    ].copy()

    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(
        prepared[TIMESTAMP_COLUMN],
        errors="raise",
    )
    manifest[TIMESTAMP_COLUMN] = pd.to_datetime(
        manifest[TIMESTAMP_COLUMN],
        errors="raise",
    )

    if manifest.duplicated(
        [MODULE_COLUMN, TIMESTAMP_COLUMN]
    ).any():
        raise ValueError(
            "Split manifest contains duplicate module/timestamp pairs."
        )

    merged = prepared.merge(
        manifest,
        on=[MODULE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        validate="one_to_one",
    )

    if merged["split"].isna().any():
        raise ValueError(
            "Some regression observations do not have a frozen split assignment."
        )

    return merged


def get_regression_feature_columns() -> tuple[list[str], list[str]]:
    """Return the frozen baseline numeric and categorical feature contract."""
    return (
        BASE_NUMERIC_FEATURES.copy(),
        BASE_CATEGORICAL_FEATURES.copy(),
    )


def validate_regression_feature_contract(
    df: pd.DataFrame,
) -> None:
    """
    Validate that baseline predictors exist and target-derived electrical
    measurements are not included as predictors.
    """
    numeric_features, categorical_features = get_regression_feature_columns()
    required = set(numeric_features + categorical_features + [TARGET_COLUMN])

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Regression frame is missing required columns: {sorted(missing)}"
        )

    if set(numeric_features + categorical_features) & LEAKAGE_COLUMNS:
        raise ValueError("Regression feature contract contains target leakage.")


def split_tecnalia_regression_frame(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return the frozen train/validation/test regression subsets."""
    validate_regression_feature_contract(df)

    expected_splits = {"train", "validation", "test"}
    actual_splits = set(df["split"].dropna().unique())

    if actual_splits != expected_splits:
        raise ValueError(
            f"Unexpected split labels: {sorted(actual_splits)}"
        )

    return {
        split_name: (
            df.loc[df["split"] == split_name]
            .copy()
            .reset_index(drop=True)
        )
        for split_name in ("train", "validation", "test")
    }
