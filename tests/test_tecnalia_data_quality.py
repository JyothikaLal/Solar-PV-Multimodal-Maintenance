from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.tecnalia_regression import (
    BASE_NUMERIC_FEATURES,
    MODULE_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.mlops.monitoring.data_quality import (
    TECNALIA_EXPECTED_MODULES,
    monitor_tecnalia_data_quality,
)


def make_reference_summary(
    tmp_path,
) -> str:
    reference = pd.DataFrame(
        {
            "feature_name": BASE_NUMERIC_FEATURES,
            "missing_rate": [0.0] * len(BASE_NUMERIC_FEATURES),
        }
    )

    path = tmp_path / "reference_summary.csv"
    reference.to_csv(path, index=False)

    return str(path)


def make_clean_tecnalia_data() -> pd.DataFrame:
    modules = list(TECNALIA_EXPECTED_MODULES)

    rows = []

    for index, module in enumerate(modules):
        rows.append(
            {
                TIMESTAMP_COLUMN: (
                    f"2025-01-01 10:0{index}:00"
                ),
                MODULE_COLUMN: module,
                BASE_NUMERIC_FEATURES[0]: 500.0 + index,
                BASE_NUMERIC_FEATURES[1]: 400.0 + index,
                BASE_NUMERIC_FEATURES[2]: 30.0 + index,
                BASE_NUMERIC_FEATURES[3]: 20.0 + index,
                BASE_NUMERIC_FEATURES[4]: 1.0 + index,
            }
        )

    return pd.DataFrame(rows)


def test_clean_tecnalia_data_passes(tmp_path):
    current = make_clean_tecnalia_data()

    result = monitor_tecnalia_data_quality(
        current,
        make_reference_summary(tmp_path),
    )

    assert result["row_count"] == 5
    assert result["required_columns_valid"] is True
    assert result["missing_required_columns"] == []

    assert result["invalid_timestamp_count"] == 0
    assert result["duplicate_key_count"] == 0

    assert result["unexpected_modules"] == []
    assert result["missing_expected_modules"] == []
    assert result["module_values_valid"] is True

    for feature in BASE_NUMERIC_FEATURES:
        quality = result["feature_quality"][feature]

        assert quality["count"] == 5
        assert quality["missing_count"] == 0
        assert quality["missing_rate"] == 0.0
        assert quality["non_finite_count"] == 0
        assert quality["non_finite_rate"] == 0.0


def test_missing_required_column_is_detected(tmp_path):
    current = make_clean_tecnalia_data()

    current = current.drop(
        columns=[BASE_NUMERIC_FEATURES[0]]
    )

    result = monitor_tecnalia_data_quality(
        current,
        make_reference_summary(tmp_path),
    )

    assert result["required_columns_valid"] is False
    assert (
        BASE_NUMERIC_FEATURES[0]
        in result["missing_required_columns"]
    )


def test_missing_numeric_value_is_reported(tmp_path):
    current = make_clean_tecnalia_data()

    feature = BASE_NUMERIC_FEATURES[0]
    current[feature] = current[feature].astype(object)
    current.loc[0, feature] = np.nan

    result = monitor_tecnalia_data_quality(
        current,
        make_reference_summary(tmp_path),
    )

    quality = result["feature_quality"][feature]

    assert quality["missing_count"] == 1
    assert quality["missing_rate"] == 0.2
    assert quality["reference_missing_rate"] == 0.0
    assert quality["missing_rate_delta"] == 0.2


def test_non_finite_numeric_value_is_detected(tmp_path):
    current = make_clean_tecnalia_data()

    feature = BASE_NUMERIC_FEATURES[1]
    current.loc[0, feature] = np.inf

    result = monitor_tecnalia_data_quality(
        current,
        make_reference_summary(tmp_path),
    )

    quality = result["feature_quality"][feature]

    assert quality["non_finite_count"] == 1
    assert quality["non_finite_rate"] == 0.2


def test_invalid_timestamp_is_detected(tmp_path):
    current = make_clean_tecnalia_data()

    current[TIMESTAMP_COLUMN] = (
        current[TIMESTAMP_COLUMN].astype(object)
    )
    current.loc[0, TIMESTAMP_COLUMN] = "not-a-timestamp"

    result = monitor_tecnalia_data_quality(
        current,
        make_reference_summary(tmp_path),
    )

    assert result["invalid_timestamp_count"] == 1
    assert result["invalid_timestamp_rate"] == 0.2


def test_duplicate_module_timestamp_keys_are_detected(
    tmp_path,
):
    current = make_clean_tecnalia_data()

    duplicate = current.iloc[[0]].copy()

    current = pd.concat(
        [current, duplicate],
        ignore_index=True,
    )

    result = monitor_tecnalia_data_quality(
        current,
        make_reference_summary(tmp_path),
    )

    assert result["duplicate_key_count"] == 2
    assert result["duplicate_key_rate"] == 2 / 6


def test_unexpected_module_is_detected(tmp_path):
    current = make_clean_tecnalia_data()

    current.loc[0, MODULE_COLUMN] = "UnknownModule"

    result = monitor_tecnalia_data_quality(
        current,
        make_reference_summary(tmp_path),
    )

    assert result["unexpected_modules"] == [
        "UnknownModule"
    ]
    assert result["module_values_valid"] is False


def test_missing_expected_module_is_reported(tmp_path):
    current = make_clean_tecnalia_data()

    current = current[
        current[MODULE_COLUMN] != "Atersa"
    ].reset_index(drop=True)

    result = monitor_tecnalia_data_quality(
        current,
        make_reference_summary(tmp_path),
    )

    assert result["missing_expected_modules"] == [
        "Atersa"
    ]


def test_empty_dataframe_is_handled(tmp_path):
    current = make_clean_tecnalia_data(
    ).iloc[0:0].copy()

    result = monitor_tecnalia_data_quality(
        current,
        make_reference_summary(tmp_path),
    )

    assert result["row_count"] == 0
    assert result["required_columns_valid"] is True
    assert result["invalid_timestamp_count"] == 0
    assert result["invalid_timestamp_rate"] == 0.0
    assert result["duplicate_key_count"] == 0
    assert result["duplicate_key_rate"] == 0.0

    assert result["unexpected_modules"] == []
    assert result["missing_expected_modules"] == sorted(
        TECNALIA_EXPECTED_MODULES
    )


def test_reference_missing_rate_is_preserved(tmp_path):
    current = make_clean_tecnalia_data()

    reference = pd.DataFrame(
        {
            "feature_name": BASE_NUMERIC_FEATURES,
            "missing_rate": [0.01] * len(
                BASE_NUMERIC_FEATURES
            ),
        }
    )

    path = tmp_path / "reference_summary.csv"
    reference.to_csv(path, index=False)

    result = monitor_tecnalia_data_quality(
        current,
        path,
    )

    for feature in BASE_NUMERIC_FEATURES:
        quality = result["feature_quality"][feature]

        assert quality["reference_missing_rate"] == 0.01
        assert quality["missing_rate_delta"] == -0.01


def test_all_expected_modules_are_defined():
    assert TECNALIA_EXPECTED_MODULES == {
        "Atersa",
        "JaSolar3",
        "NingboSolar",
        "Photowatt",
        "TrinaSolar",
    }
