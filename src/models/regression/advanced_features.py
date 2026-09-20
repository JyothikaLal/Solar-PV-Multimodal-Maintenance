from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "normalized_pmpp"

CONTEXT_COLUMNS = {
    "Fecha",
    "split",
}

CATEGORICAL_FEATURES = {
    "module_name",
}

# Electrical measurements that directly describe module power/performance.
# These are excluded because the target is normalized Pmpp.
ELECTRICAL_LEAKAGE_FEATURES = {
    "Pmpp (W)",
    "Vmpp(V)",
    "Impp(A)",
    "Isc(A)",
    "Voc(A)",
    "FF",
    "normalized_pmpp",
}

# Target-derived temporal features created by Task 15.
TARGET_DERIVED_TEMPORAL_FEATURES = {
    "delta_pmpp",
    "delta_normalized_pmpp",
    "pmpp_1h_slope",
    "pmpp_3h_slope",
    "normalized_pmpp_1h_slope",
    "normalized_pmpp_3h_slope",
    "normalized_pmpp_7d_past_mean",
    "normalized_pmpp_30d_past_mean",
    "normalized_pmpp_7d_past_slope",
    "normalized_pmpp_30d_past_slope",
}

# Target-derived environmental normalization features.
TARGET_DERIVED_RATIO_FEATURES = {
    "pmpp_per_gpoa",
    "normalized_pmpp_per_gpoa",
}

# Features known to contain no observations in the TECNALIA dataset.
ALL_MISSING_FEATURES = {
    "DHI (W/m²)",
    "DNI (W/m²)",
    "Vac(V)",
    "Iac(A)",
    "Pac(W)",
    "Back GPOA (W/m²)",
    "SR (Wh/m²)",
    "Pressure (mmHg)",
    "Humidity(%)",
    "Rain Accumulation (mm)",
    "Intensity(mm/h)",
    "Direction of wind",
}

TARGET_DERIVED_FEATURES = (
    ELECTRICAL_LEAKAGE_FEATURES
    | TARGET_DERIVED_TEMPORAL_FEATURES
    | TARGET_DERIVED_RATIO_FEATURES
)


def get_advanced_regression_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Return leakage-safe features for advanced TECNALIA regression."""

    missing_required = CATEGORICAL_FEATURES - set(df.columns)
    if missing_required:
        raise ValueError(
            "Missing required categorical feature columns: "
            f"{sorted(missing_required)}"
        )

    excluded = (
        CONTEXT_COLUMNS
        | TARGET_DERIVED_FEATURES
        | ALL_MISSING_FEATURES
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError(
            "No valid advanced regression features found."
        )

    validate_advanced_regression_feature_contract(
        feature_columns
    )

    return feature_columns


def validate_advanced_regression_feature_contract(
    feature_columns: list[str] | tuple[str, ...],
) -> None:
    """Validate the frozen leakage-safe regression feature contract."""

    feature_set = set(feature_columns)

    leakage_violations = (
        feature_set & TARGET_DERIVED_FEATURES
    )

    if leakage_violations:
        raise ValueError(
            "Target-derived/electrical leakage features detected: "
            f"{sorted(leakage_violations)}"
        )

    context_violations = feature_set & CONTEXT_COLUMNS

    if context_violations:
        raise ValueError(
            "Context-only columns detected in regression contract: "
            f"{sorted(context_violations)}"
        )

    all_missing_violations = (
        feature_set & ALL_MISSING_FEATURES
    )

    if all_missing_violations:
        raise ValueError(
            "All-missing features detected in regression contract: "
            f"{sorted(all_missing_violations)}"
        )

    missing_categorical = CATEGORICAL_FEATURES - feature_set

    if missing_categorical:
        raise ValueError(
            "Required categorical features missing from "
            f"regression contract: {sorted(missing_categorical)}"
        )
