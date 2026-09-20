import pandas as pd
import pytest

from src.models.regression.advanced_features import (
    ALL_MISSING_FEATURES,
    CATEGORICAL_FEATURES,
    CONTEXT_COLUMNS,
    TARGET_DERIVED_FEATURES,
    get_advanced_regression_feature_columns,
    validate_advanced_regression_feature_contract,
)


def test_feature_selection_excludes_target_derived_features():
    df = pd.DataFrame(
        columns=[
            "Fecha",
            "split",
            "module_name",
            "Front GPOA (W/m²)",
            "GHI (W/m²)",
            "Temp. Mod (°C)",
            "delta_pmpp",
            "pmpp_1h_slope",
            "pmpp_3h_slope",
            "pmpp_per_gpoa",
            "normalized_pmpp",
            "Vmpp(V)",
            "Impp(A)",
            "Isc(A)",
            "Voc(A)",
            "FF",
        ]
    )

    features = get_advanced_regression_feature_columns(df)

    assert "module_name" in features
    assert "Front GPOA (W/m²)" in features
    assert "GHI (W/m²)" in features

    assert "Fecha" not in features
    assert "split" not in features

    assert not set(features) & TARGET_DERIVED_FEATURES


def test_feature_contract_rejects_target_derived_features():
    with pytest.raises(ValueError, match="leakage"):
        validate_advanced_regression_feature_contract(
            [
                "module_name",
                "Front GPOA (W/m²)",
                "delta_pmpp",
            ]
        )


def test_feature_contract_rejects_context_columns():
    with pytest.raises(ValueError, match="Context-only"):
        validate_advanced_regression_feature_contract(
            [
                "module_name",
                "Front GPOA (W/m²)",
                "Fecha",
            ]
        )


def test_feature_contract_rejects_all_missing_features():
    with pytest.raises(ValueError, match="All-missing"):
        validate_advanced_regression_feature_contract(
            [
                "module_name",
                "Front GPOA (W/m²)",
                "DHI (W/m²)",
            ]
        )


def test_feature_contract_requires_module_name():
    df = pd.DataFrame(
        columns=[
            "Fecha",
            "split",
            "Front GPOA (W/m²)",
            "GHI (W/m²)",
        ]
    )

    with pytest.raises(ValueError, match="categorical"):
        get_advanced_regression_feature_columns(df)


def test_feature_contract_contains_expected_protected_features():
    assert "module_name" in CATEGORICAL_FEATURES

    assert "Fecha" in CONTEXT_COLUMNS
    assert "split" in CONTEXT_COLUMNS

    assert "normalized_pmpp" in TARGET_DERIVED_FEATURES
    assert "Vmpp(V)" in TARGET_DERIVED_FEATURES
    assert "Impp(A)" in TARGET_DERIVED_FEATURES
    assert "Isc(A)" in TARGET_DERIVED_FEATURES
    assert "Voc(A)" in TARGET_DERIVED_FEATURES
    assert "FF" in TARGET_DERIVED_FEATURES

    assert "delta_pmpp" in TARGET_DERIVED_FEATURES
    assert "pmpp_1h_slope" in TARGET_DERIVED_FEATURES
    assert "pmpp_3h_slope" in TARGET_DERIVED_FEATURES

    assert "DHI (W/m²)" in ALL_MISSING_FEATURES
