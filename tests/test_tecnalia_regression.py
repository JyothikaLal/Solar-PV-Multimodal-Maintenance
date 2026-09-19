import pandas as pd
import pytest

from src.data.tecnalia_regression import (
    TARGET_COLUMN,
    attach_frozen_tecnalia_split,
    build_tecnalia_regression_frame,
    get_regression_feature_columns,
    split_tecnalia_regression_frame,
    validate_regression_feature_contract,
)


def make_module_frame(module_name="Atersa"):
    return pd.DataFrame(
        {
            "Fecha": pd.to_datetime(
                [
                    "2025-01-01 10:00",
                    "2025-01-01 11:00",
                    "2025-01-01 12:00",
                ]
            ),
            "module_name": [module_name] * 3,
            "Front GPOA (W/m²)": [100.0, 250.0, 500.0],
            "GHI (W/m²)": [90.0, 230.0, 450.0],
            "Temp. Mod (°C)": [25.0, 30.0, 35.0],
            "Amb. Temp. (°C)": [20.0, 25.0, 30.0],
            "Wind Speed (m/s)": [1.0, 2.0, 3.0],
            "Pmpp": [50.0, 165.0, 330.0],
            "rated_power": [330.0] * 3,
        }
    )


def test_operating_filter_and_target():
    result = build_tecnalia_regression_frame(
        {"Atersa": make_module_frame()}
    )

    assert len(result) == 2
    assert result["Front GPOA (W/m²)"].min() >= 200
    assert result[TARGET_COLUMN].tolist() == pytest.approx(
        [0.5, 1.0]
    )


def test_baseline_feature_contract_excludes_target_columns():
    numeric, categorical = get_regression_feature_columns()
    features = set(numeric + categorical)

    assert features == {
        "Front GPOA (W/m²)",
        "GHI (W/m²)",
        "Temp. Mod (°C)",
        "Amb. Temp. (°C)",
        "Wind Speed (m/s)",
        "module_name",
    }

    assert not features.intersection(
        {"Pmpp", "Vmpp", "Impp", "normalized_pmpp"}
    )


def test_frozen_split_attaches_by_module_and_timestamp():
    frame = build_tecnalia_regression_frame(
        {"Atersa": make_module_frame()}
    )

    manifest = pd.DataFrame(
        {
            "module_name": ["Atersa", "Atersa"],
            "Fecha": pd.to_datetime(
                ["2025-01-01 11:00", "2025-01-01 12:00"]
            ),
            "split": ["train", "test"],
        }
    )

    result = attach_frozen_tecnalia_split(frame, manifest)

    assert result["split"].tolist() == ["train", "test"]


def test_split_function_returns_three_frozen_subsets():
    frame = build_tecnalia_regression_frame(
        {"Atersa": make_module_frame()}
    )

    manifest = pd.DataFrame(
        {
            "module_name": ["Atersa", "Atersa"],
            "Fecha": pd.to_datetime(
                ["2025-01-01 11:00", "2025-01-01 12:00"]
            ),
            "split": ["train", "test"],
        }
    )

    prepared = attach_frozen_tecnalia_split(frame, manifest)

    with pytest.raises(ValueError):
        split_tecnalia_regression_frame(prepared)


def test_feature_contract_accepts_valid_frame():
    frame = build_tecnalia_regression_frame(
        {"Atersa": make_module_frame()}
    )

    validate_regression_feature_contract(frame)


def test_duplicate_manifest_is_rejected():
    frame = build_tecnalia_regression_frame(
        {"Atersa": make_module_frame()}
    )

    manifest = pd.DataFrame(
        {
            "module_name": ["Atersa", "Atersa"],
            "Fecha": pd.to_datetime(
                ["2025-01-01 11:00", "2025-01-01 11:00"]
            ),
            "split": ["train", "test"],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        attach_frozen_tecnalia_split(frame, manifest)
