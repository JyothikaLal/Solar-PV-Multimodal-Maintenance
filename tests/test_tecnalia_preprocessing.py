import pandas as pd
import pytest

from src.data.tecnalia_preprocessing import (
    add_module_metadata,
    analyze_tecnalia_missing_values,
    build_tecnalia_validation_report,
    convert_tecnalia_sentinels,
    count_tecnalia_infinite_values,
    load_tecnalia_module,
    preprocess_tecnalia_module,
    validate_tecnalia_duplicates,
    validate_tecnalia_physical_ranges,
)


def make_test_dataframe() -> pd.DataFrame:
    """Create a small valid TECNALIA-like dataframe for unit tests."""
    return pd.DataFrame(
        {
            "Fecha": pd.to_datetime(
                [
                    "2025-01-01 10:00:00",
                    "2025-01-01 10:10:00",
                    "2025-01-01 10:20:00",
                ]
            ),
            "Vmpp(V)": [30.0, 31.0, 32.0],
            "Impp(A)": [5.0, 5.1, 5.2],
            "Temp. Mod (°C)": [20.0, 21.0, 22.0],
            "GHI (W/m²)": [500.0, 600.0, 700.0],
            "Front GPOA (W/m²)": [550.0, 650.0, 750.0],
            "Wind Speed (m/s)": [1.0, 1.5, 2.0],
            "Pmpp (W)": [150.0, 158.1, 166.4],
        }
    )


def test_load_tecnalia_module(tmp_path):
    """Loader should parse the CSV and timestamps correctly."""
    df = make_test_dataframe()

    file_path = tmp_path / "data_Atersa.csv"
    df.to_csv(file_path, sep=";", index=False)

    loaded = load_tecnalia_module(file_path)

    assert len(loaded) == 3
    assert pd.api.types.is_datetime64_any_dtype(
        loaded["Fecha"]
)
    assert loaded["Fecha"].is_monotonic_increasing
    assert loaded["Fecha"].duplicated().sum() == 0


def test_convert_tecnalia_sentinels():
    """Known -9999 sentinel should become NaN."""
    df = make_test_dataframe()
    df.loc[1, "Temp. Mod (°C)"] = -9999

    processed = convert_tecnalia_sentinels(df)

    assert processed["Temp. Mod (°C)"].isna().sum() == 1
    assert (processed["Temp. Mod (°C)"] == -9999).sum() == 0

    # Original dataframe must remain unchanged.
    assert df.loc[1, "Temp. Mod (°C)"] == -9999


def test_missing_value_analysis():
    """Missing-value summary should correctly report NaNs."""
    df = make_test_dataframe()
    df.loc[1, "Temp. Mod (°C)"] = pd.NA

    summary = analyze_tecnalia_missing_values(df)

    row = summary[
        summary["column"] == "Temp. Mod (°C)"
    ].iloc[0]

    assert row["missing_count"] == 1
    assert row["missing_percentage"] == pytest.approx(
        100 / 3
    )
    assert not row["completely_missing"]


def test_physical_range_validation():
    """Non-negative physical columns should reject negative values."""
    df = make_test_dataframe()

    result = validate_tecnalia_physical_ranges(df)

    assert result["invalid_count"].sum() == 0

    df.loc[0, "Pmpp (W)"] = -1.0

    result = validate_tecnalia_physical_ranges(df)

    pmpp_row = result[
        result["column"] == "Pmpp (W)"
    ].iloc[0]

    assert pmpp_row["negative_count"] == 1
    assert pmpp_row["invalid_count"] == 1


def test_negative_temperature_is_allowed():
    """Negative module temperature should not be flagged as invalid."""
    df = make_test_dataframe()
    df.loc[0, "Temp. Mod (°C)"] = -5.0

    result = validate_tecnalia_physical_ranges(df)

    assert result["invalid_count"].sum() == 0


def test_infinite_values():
    """Infinite numeric values should be detected."""
    df = make_test_dataframe()
    df.loc[0, "Pmpp (W)"] = float("inf")

    assert count_tecnalia_infinite_values(df) == 1


def test_duplicate_validation():
    """Duplicate timestamps should raise an error."""
    df = make_test_dataframe()

    df.loc[2, "Fecha"] = df.loc[1, "Fecha"]

    with pytest.raises(
        ValueError,
        match="duplicate timestamps",
    ):
        validate_tecnalia_duplicates(df)


def test_duplicate_validation_passes_for_valid_data():
    """Unique, sorted timestamps should pass."""
    df = make_test_dataframe()

    result = validate_tecnalia_duplicates(df)

    assert result["duplicate_count"] == 0
    assert result["is_sorted"] is True


@pytest.mark.parametrize(
    "module_name,rated_power",
    [
        ("Atersa", 330.0),
        ("JaSolar3", 315.0),
        ("NingboSolar", 175.0),
        ("Photowatt", 155.0),
        ("TrinaSolar", 185.0),
    ],
)
def test_module_metadata(module_name, rated_power):
    """Module metadata should use the correct rated power."""
    df = make_test_dataframe()

    processed = add_module_metadata(
        df,
        module_name,
    )

    assert processed["module_name"].eq(
        module_name
    ).all()

    assert processed["rated_power"].eq(
        rated_power
    ).all()

    expected = processed["Pmpp (W)"] / rated_power

    pd.testing.assert_series_equal(
        processed["normalized_pmpp"],
        expected,
        check_names=False,
    )


def test_unknown_module_rejected():
    """Unknown module names should raise an error."""
    df = make_test_dataframe()

    with pytest.raises(
        ValueError,
        match="Unknown TECNALIA module",
    ):
        add_module_metadata(
            df,
            "UnknownModule",
        )


def test_complete_preprocessing_pipeline(tmp_path):
    """Complete pipeline should produce a valid processed dataframe."""
    df = make_test_dataframe()
    df.loc[1, "Temp. Mod (°C)"] = -9999

    file_path = tmp_path / "data_Atersa.csv"
    df.to_csv(file_path, sep=";", index=False)

    processed, report = preprocess_tecnalia_module(
        file_path,
        "Atersa",
    )

    assert len(processed) == 3
    assert len(processed.columns) == 11

    assert processed["module_name"].eq(
        "Atersa"
    ).all()

    assert processed["rated_power"].eq(
        330.0
    ).all()

    assert processed["normalized_pmpp"].notna().all()

    assert processed["Temp. Mod (°C)"].isna().sum() == 1

    assert report["module_name"] == "Atersa"
    assert report["rows"] == 3
    assert report["duplicate_timestamps"] == 0
    assert report["timestamp_sorted"] is True
    assert report["infinite_values"] == 0


def test_validation_report_builder():
    """Validation reports should be converted into summary tables."""
    df = make_test_dataframe()

    validation_report = {
        "file": "data_Atersa.csv",
        "module_name": "Atersa",
        "rows": 3,
        "columns": 11,
        "timestamp_start": df["Fecha"].min(),
        "timestamp_end": df["Fecha"].max(),
        "duplicate_timestamps": 0,
        "timestamp_sorted": True,
        "infinite_values": 0,
        "physical_ranges": pd.DataFrame(),
        "missing_values": analyze_tecnalia_missing_values(df),
    }

    module_summary, missing_values = (
        build_tecnalia_validation_report(
            [validation_report]
        )
    )

    assert len(module_summary) == 1
    assert module_summary.iloc[0]["module_name"] == "Atersa"

    assert len(missing_values) == len(df.columns)
    assert missing_values["module_name"].eq(
        "Atersa"
    ).all()