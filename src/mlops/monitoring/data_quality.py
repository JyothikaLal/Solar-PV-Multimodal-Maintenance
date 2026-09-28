from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.tecnalia_regression import (
    BASE_NUMERIC_FEATURES,
    MODULE_COLUMN,
    TIMESTAMP_COLUMN,
)


TECNALIA_EXPECTED_MODULES = {
    "Atersa",
    "JaSolar3",
    "NingboSolar",
    "Photowatt",
    "TrinaSolar",
}


TECNALIA_REFERENCE_SUMMARY_PATH = Path(
    "reports/results/tecnalia/monitoring/"
    "telemetry_feature_reference.csv"
)


def _validate_required_columns(
    current: pd.DataFrame,
    required_columns: set[str],
) -> list[str]:
    return sorted(required_columns - set(current.columns))


def monitor_tecnalia_data_quality(
    current: pd.DataFrame,
    reference_summary_path: str | Path = (
        TECNALIA_REFERENCE_SUMMARY_PATH
    ),
) -> dict[str, object]:
    """
    Monitor structural and numeric data quality for TECNALIA telemetry.

    This function reports individual quality signals. It does not collapse
    them into a single quality score or apply alert thresholds.
    """
    required_columns = {
        TIMESTAMP_COLUMN,
        MODULE_COLUMN,
        *BASE_NUMERIC_FEATURES,
    }

    missing_required_columns = _validate_required_columns(
        current,
        required_columns,
    )

    result: dict[str, object] = {
        "row_count": int(len(current)),
        "missing_required_columns": missing_required_columns,
        "required_columns_valid": not missing_required_columns,
    }

    if missing_required_columns:
        return result

    reference_summary = pd.read_csv(reference_summary_path)

    reference_missing_rates = (
        reference_summary.set_index("feature_name")[
            "missing_rate"
        ].to_dict()
    )

    feature_quality: dict[str, dict[str, object]] = {}

    for feature in BASE_NUMERIC_FEATURES:
        numeric = pd.to_numeric(
            current[feature],
            errors="coerce",
        )

        missing_mask = numeric.isna()
        non_finite_mask = np.isinf(
            numeric.fillna(0).to_numpy(dtype=float)
        )

        feature_quality[feature] = {
            "count": int(len(numeric)),
            "missing_count": int(missing_mask.sum()),
            "missing_rate": float(missing_mask.mean()),
            "reference_missing_rate": float(
                reference_missing_rates[feature]
            ),
            "missing_rate_delta": float(
                missing_mask.mean()
                - reference_missing_rates[feature]
            ),
            "non_finite_count": int(non_finite_mask.sum()),
            "non_finite_rate": float(
                non_finite_mask.mean()
            ),
        }

    timestamp = pd.to_datetime(
        current[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    invalid_timestamp_count = int(timestamp.isna().sum())

    duplicate_key_count = int(
        current.duplicated(
            subset=[MODULE_COLUMN, TIMESTAMP_COLUMN],
            keep=False,
        ).sum()
    )

    modules = current[MODULE_COLUMN].dropna().astype(str)

    unexpected_modules = sorted(
        set(modules.unique()) - TECNALIA_EXPECTED_MODULES
    )

    missing_modules = sorted(
        TECNALIA_EXPECTED_MODULES - set(modules.unique())
    )

    result.update(
        {
            "feature_quality": feature_quality,
            "invalid_timestamp_count": invalid_timestamp_count,
            "invalid_timestamp_rate": float(
                invalid_timestamp_count / len(current)
            )
            if len(current)
            else 0.0,
            "duplicate_key_count": duplicate_key_count,
            "duplicate_key_rate": float(
                duplicate_key_count / len(current)
            )
            if len(current)
            else 0.0,
            "unexpected_modules": unexpected_modules,
            "missing_expected_modules": missing_modules,
            "module_values_valid": not unexpected_modules,
        }
    )

    return result
