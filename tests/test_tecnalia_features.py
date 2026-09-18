import pandas as pd
import pytest
import numpy as np

from src.data.tecnalia_features import (
    prepare_tecnalia_time_series,
)


def make_test_dataframe() -> pd.DataFrame:
    """Create a small irregularly sampled time series."""
    return pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:10:00",
                "2025-01-01 10:00:00",
                "2025-01-01 10:20:00",
                "2025-01-01 10:30:00",
            ],
            "normalized_pmpp": [
                0.60,
                0.50,
                0.70,
                0.75,
            ],
        }
    )


def test_time_series_is_sorted():
    """Timestamps should be sorted chronologically."""
    df = make_test_dataframe()

    prepared = prepare_tecnalia_time_series(df)

    assert prepared["Fecha"].is_monotonic_increasing


def test_elapsed_minutes_uses_actual_timestamps():
    """Elapsed time should use actual timestamp differences."""
    df = make_test_dataframe()

    prepared = prepare_tecnalia_time_series(df)

    expected = [None, 10.0, 10.0, 10.0]

    pd.testing.assert_series_equal(
        prepared["elapsed_minutes"],
        pd.Series(
            expected,
            dtype="float64",
            name="elapsed_minutes",
        ),
        check_names=True,
    )


def test_irregular_sampling_is_preserved():
    """A 5-minute interval must not be converted into a fixed interval."""
    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:05:00",
                "2025-01-01 10:15:00",
            ],
        }
    )

    prepared = prepare_tecnalia_time_series(df)

    assert prepared["elapsed_minutes"].iloc[0] != 5
    assert prepared["elapsed_minutes"].iloc[1] == 5.0
    assert prepared["elapsed_minutes"].iloc[2] == 10.0


def test_invalid_timestamp_is_rejected():
    """Invalid timestamps should raise an error."""
    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "not-a-timestamp",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="invalid timestamps",
    ):
        prepare_tecnalia_time_series(df)


def test_duplicate_timestamp_is_rejected():
    """Duplicate timestamps should raise an error."""
    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:00:00",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="duplicate timestamps",
    ):
        prepare_tecnalia_time_series(df)


def test_missing_timestamp_column_is_rejected():
    """A missing timestamp column should raise an error."""
    df = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01 10:00:00",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required timestamp column",
    ):
        prepare_tecnalia_time_series(df)


def test_input_dataframe_is_not_modified():
    """The original dataframe should remain unchanged."""
    df = make_test_dataframe()
    original = df.copy(deep=True)

    prepare_tecnalia_time_series(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )

def test_environmental_rolling_features_use_time_windows():
    """Rolling features should use timestamp-based windows."""
    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:10:00",
                "2025-01-01 10:20:00",
                "2025-01-01 10:30:00",
            ],
            "Front GPOA (W/m²)": [
                100.0,
                200.0,
                300.0,
                400.0,
            ],
        }
    )

    from src.data.tecnalia_features import (
        add_environmental_rolling_features,
    )

    result = add_environmental_rolling_features(df)

    assert result.loc[0, "gpoa_30min_mean"] == pytest.approx(
        100.0
    )

    assert result.loc[1, "gpoa_30min_mean"] == pytest.approx(
        150.0
    )

    assert result.loc[2, "gpoa_30min_mean"] == pytest.approx(
        200.0
    )

    assert result.loc[3, "gpoa_30min_mean"] == pytest.approx(
        300.0
    )


def test_environmental_rolling_features_include_current_observation():
    """The current observation should be included."""
    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:10:00",
                "2025-01-01 10:20:00",
            ],
            "GHI (W/m²)": [
                100.0,
                200.0,
                1000.0,
            ],
        }
    )

    from src.data.tecnalia_features import (
        add_environmental_rolling_features,
    )

    result = add_environmental_rolling_features(df)

    assert result.loc[2, "ghi_30min_mean"] == pytest.approx(
        (100.0 + 200.0 + 1000.0) / 3
    )


def test_environmental_rolling_features_do_not_use_future_values():
    """Future observations must not affect earlier rows."""
    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:10:00",
                "2025-01-01 10:20:00",
            ],
            "GHI (W/m²)": [
                100.0,
                200.0,
                10000.0,
            ],
        }
    )

    from src.data.tecnalia_features import (
        add_environmental_rolling_features,
    )

    result = add_environmental_rolling_features(df)

    assert result.loc[0, "ghi_30min_mean"] == pytest.approx(
        100.0
    )

    assert result.loc[1, "ghi_30min_mean"] == pytest.approx(
        150.0
    )


def test_environmental_rolling_features_preserve_row_count():
    """Rolling feature generation must preserve the number of rows."""
    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:10:00",
                "2025-01-01 10:20:00",
            ],
            "Wind Speed (m/s)": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    from src.data.tecnalia_features import (
        add_environmental_rolling_features,
    )

    result = add_environmental_rolling_features(df)

    assert len(result) == len(df)


def test_environmental_rolling_features_do_not_modify_input():
    """The original dataframe must remain unchanged."""
    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:10:00",
            ],
            "GHI (W/m²)": [
                100.0,
                200.0,
            ],
        }
    )

    original = df.copy(deep=True)

    from src.data.tecnalia_features import (
        add_environmental_rolling_features,
    )

    add_environmental_rolling_features(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )

    def test_delta_features():
        """Delta features should equal current minus previous value."""
        from src.data.tecnalia_features import (
            add_delta_features,
        )

        df = pd.DataFrame(
            {
                "Fecha": [
                    "2025-01-01 10:00:00",
                    "2025-01-01 10:10:00",
                    "2025-01-01 10:20:00",
                ],
                "Front GPOA (W/m²)": [
                    100.0,
                    200.0,
                    150.0,
                ],
                "GHI (W/m²)": [
                    80.0,
                    120.0,
                    100.0,
                ],
                "normalized_pmpp": [
                    0.50,
                    0.60,
                    0.55,
                ],
            }
        )

        result = add_delta_features(df)

        assert pd.isna(
            result.loc[0, "delta_gpoa"]
        )

        assert result.loc[1, "delta_gpoa"] == pytest.approx(
            100.0
        )

        assert result.loc[2, "delta_gpoa"] == pytest.approx(
            -50.0
        )

        assert result.loc[1, "delta_ghi"] == pytest.approx(
            40.0
        )

        assert result.loc[2, "delta_normalized_pmpp"] == pytest.approx(
            -0.05
        )


def test_delta_features_are_chronological():
    """Deltas should be calculated after chronological ordering."""
    from src.data.tecnalia_features import (
        add_delta_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:10:00",
                "2025-01-01 10:00:00",
                "2025-01-01 10:20:00",
            ],
            "GHI (W/m²)": [
                200.0,
                100.0,
                300.0,
            ],
        }
    )

    result = add_delta_features(df)

    assert result["Fecha"].is_monotonic_increasing

    assert pd.isna(
        result.loc[0, "delta_ghi"]
    )

    assert result.loc[1, "delta_ghi"] == pytest.approx(
        100.0
    )

    assert result.loc[2, "delta_ghi"] == pytest.approx(
        100.0
    )


def test_delta_features_preserve_nan():
    """Missing source values should not be silently imputed."""
    from src.data.tecnalia_features import (
        add_delta_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:10:00",
                "2025-01-01 10:20:00",
            ],
            "GHI (W/m²)": [
                100.0,
                float("nan"),
                300.0,
            ],
        }
    )

    result = add_delta_features(df)

    assert pd.isna(
        result.loc[1, "delta_ghi"]
    )

    assert pd.isna(
        result.loc[2, "delta_ghi"]
    )


def test_delta_features_do_not_modify_input():
    """Delta generation must not modify the source dataframe."""
    from src.data.tecnalia_features import (
        add_delta_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:10:00",
            ],
            "GHI (W/m²)": [
                100.0,
                200.0,
            ],
        }
    )

    original = df.copy(deep=True)

    add_delta_features(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )

def test_slope_features_use_actual_elapsed_time():
    """Slope should use actual elapsed time between observations."""
    from src.data.tecnalia_features import (
        add_slope_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 11:00:00",
                "2025-01-01 12:00:00",
            ],
            "normalized_pmpp": [
                0.50,
                0.60,
                0.80,
            ],
        }
    )

    result = add_slope_features(df)

    assert pd.isna(
        result.loc[0, "normalized_pmpp_1h_slope"]
    )

    assert result.loc[1, "normalized_pmpp_1h_slope"] == pytest.approx(
        0.10
    )

    assert result.loc[2, "normalized_pmpp_1h_slope"] == pytest.approx(
        0.20
    )


def test_slope_features_use_actual_elapsed_time_for_irregular_sampling():
    """
    Slope should use actual elapsed time rather than assuming
    an exact one-hour interval.
    """
    from src.data.tecnalia_features import (
        add_slope_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 11:05:00",
                "2025-01-01 12:00:00",
            ],
            "normalized_pmpp": [
                0.50,
                0.65,
                0.80,
            ],
        }
    )

    result = add_slope_features(df)

    # At 11:05, the latest observation at or before
    # 10:05 is the 10:00 observation.
    #
    # Actual elapsed time = 65 minutes.
    expected_slope = (
        (0.65 - 0.50) / (65 / 60)
    )

    assert result.loc[
        1,
        "normalized_pmpp_1h_slope",
    ] == pytest.approx(
        expected_slope
    )


def test_slope_features_do_not_use_future_values():
    """Future observations must never affect historical slopes."""
    from src.data.tecnalia_features import (
        add_slope_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 11:00:00",
                "2025-01-01 12:00:00",
            ],
            "normalized_pmpp": [
                0.50,
                0.60,
                100.0,
            ],
        }
    )

    result = add_slope_features(df)

    assert result.loc[
        1,
        "normalized_pmpp_1h_slope",
    ] == pytest.approx(0.10)


def test_slope_features_handle_missing_values():
    """Missing values should result in NaN slopes."""
    from src.data.tecnalia_features import (
        add_slope_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 11:00:00",
            ],
            "normalized_pmpp": [
                0.50,
                float("nan"),
            ],
        }
    )

    result = add_slope_features(df)

    assert pd.isna(
        result.loc[
            1,
            "normalized_pmpp_1h_slope",
        ]
    )


def test_slope_features_preserve_row_count():
    """Slope generation should preserve the number of rows."""
    from src.data.tecnalia_features import (
        add_slope_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 11:00:00",
                "2025-01-01 12:00:00",
            ],
            "Pmpp (W)": [
                100.0,
                120.0,
                150.0,
            ],
        }
    )

    result = add_slope_features(df)

    assert len(result) == len(df)

def test_environmental_normalized_features():
    """Ratios should be calculated only above the GPOA threshold."""
    from src.data.tecnalia_features import (
        add_environmental_normalized_features,
    )

    df = pd.DataFrame(
        {
            "Front GPOA (W/m²)": [
                100.0,
                199.0,
                200.0,
                400.0,
            ],
            "Pmpp (W)": [
                10.0,
                20.0,
                50.0,
                100.0,
            ],
            "normalized_pmpp": [
                0.03,
                0.06,
                0.15,
                0.30,
            ],
        }
    )

    result = add_environmental_normalized_features(df)

    assert pd.isna(
        result.loc[0, "pmpp_per_gpoa"]
    )

    assert pd.isna(
        result.loc[1, "pmpp_per_gpoa"]
    )

    assert result.loc[
        2,
        "pmpp_per_gpoa",
    ] == pytest.approx(0.25)

    assert result.loc[
        3,
        "pmpp_per_gpoa",
    ] == pytest.approx(0.25)

    assert result.loc[
        2,
        "normalized_pmpp_per_gpoa",
    ] == pytest.approx(0.15 / 200)


def test_environmental_normalized_features_handle_missing_values():
    """Missing source values should remain missing."""
    from src.data.tecnalia_features import (
        add_environmental_normalized_features,
    )

    df = pd.DataFrame(
        {
            "Front GPOA (W/m²)": [
                300.0,
                400.0,
            ],
            "Pmpp (W)": [
                float("nan"),
                100.0,
            ],
            "normalized_pmpp": [
                float("nan"),
                0.30,
            ],
        }
    )

    result = add_environmental_normalized_features(df)

    assert pd.isna(
        result.loc[0, "pmpp_per_gpoa"]
    )

    assert pd.isna(
        result.loc[0, "normalized_pmpp_per_gpoa"]
    )

    assert result.loc[
        1,
        "pmpp_per_gpoa",
    ] == pytest.approx(0.25)


def test_environmental_normalized_features_do_not_create_infinity():
    """Guarded ratios should never produce infinite values."""
    from src.data.tecnalia_features import (
        add_environmental_normalized_features,
    )

    df = pd.DataFrame(
        {
            "Front GPOA (W/m²)": [
                0.0,
                100.0,
                200.0,
            ],
            "Pmpp (W)": [
                0.0,
                10.0,
                50.0,
            ],
            "normalized_pmpp": [
                0.0,
                0.03,
                0.15,
            ],
        }
    )

    result = add_environmental_normalized_features(df)

    assert not result[
        "pmpp_per_gpoa"
    ].isin(
        [float("inf"), float("-inf")]
    ).any()

    assert not result[
        "normalized_pmpp_per_gpoa"
    ].isin(
        [float("inf"), float("-inf")]
    ).any()


def test_environmental_normalized_features_preserve_rows():
    """Ratio generation should preserve all telemetry rows."""
    from src.data.tecnalia_features import (
        add_environmental_normalized_features,
    )

    df = pd.DataFrame(
        {
            "Front GPOA (W/m²)": [
                0.0,
                100.0,
                200.0,
                500.0,
            ],
            "Pmpp (W)": [
                0.0,
                20.0,
                50.0,
                125.0,
            ],
            "normalized_pmpp": [
                0.0,
                0.06,
                0.15,
                0.375,
            ],
        }
    )

    result = add_environmental_normalized_features(df)

    assert len(result) == len(df)

def test_performance_trend_indicators_are_historical_only():
    """Historical means must exclude the current target."""
    from src.data.tecnalia_features import (
        add_performance_trend_indicators,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:10:00",
                "2025-01-01 10:20:00",
            ],
            "normalized_pmpp": [
                0.50,
                0.60,
                100.0,
            ],
        }
    )

    result = add_performance_trend_indicators(df)

    # At 10:10, only the 10:00 observation can contribute.
    assert result.loc[
        1,
        "normalized_pmpp_7d_past_mean",
    ] == pytest.approx(0.50)

    # At 10:20, the 100.0 current target must not contribute.
    assert result.loc[
        2,
        "normalized_pmpp_7d_past_mean",
    ] == pytest.approx((0.50 + 0.60) / 2)


def test_performance_trend_indicators_do_not_use_future_values():
    """Future performance values must not affect earlier features."""
    from src.data.tecnalia_features import (
        add_performance_trend_indicators,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:10:00",
                "2025-01-01 10:20:00",
            ],
            "normalized_pmpp": [
                0.50,
                0.60,
                1000.0,
            ],
        }
    )

    result = add_performance_trend_indicators(df)

    assert result.loc[
        1,
        "normalized_pmpp_7d_past_mean",
    ] == pytest.approx(0.50)


def test_performance_trend_indicators_handle_missing_values():
    """Missing historical performance should remain missing."""
    from src.data.tecnalia_features import (
        add_performance_trend_indicators,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:10:00",
                "2025-01-01 10:20:00",
            ],
            "normalized_pmpp": [
                0.50,
                float("nan"),
                0.70,
            ],
        }
    )

    result = add_performance_trend_indicators(df)

    assert result.loc[
        1,
        "normalized_pmpp_7d_past_mean",
    ] == pytest.approx(0.50)

    assert result.loc[
        2,
        "normalized_pmpp_7d_past_mean",
    ] == pytest.approx(0.50)


def test_performance_trend_slopes_use_actual_time():
    """Historical performance slopes should use elapsed time."""
    from src.data.tecnalia_features import (
        add_performance_trend_indicators,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-02 10:00:00",
                "2025-01-03 10:00:00",
                "2025-01-04 10:00:00",
            ],
            "normalized_pmpp": [
                0.50,
                0.60,
                0.70,
                0.80,
            ],
        }
    )

    result = add_performance_trend_indicators(df)

    # At Jan 3, the 7-day historical slope uses Jan 1
    # through Jan 2 as historical observations.
    #
    # The current Jan 3 value is excluded.
    expected = (0.60 - 0.50) / 24

    assert result.loc[
        2,
        "normalized_pmpp_7d_past_slope",
    ] == pytest.approx(expected)


def test_performance_trend_indicators_preserve_rows():
    """Trend indicators should preserve the number of rows."""
    from src.data.tecnalia_features import (
        add_performance_trend_indicators,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 10:00:00",
                "2025-01-02 10:00:00",
            ],
            "normalized_pmpp": [
                0.50,
                0.60,
            ],
        }
    )

    result = add_performance_trend_indicators(df)

    assert len(result) == len(df)

def test_engineer_tecnalia_features_runs_complete_pipeline():
    """The complete pipeline should create all expected feature groups."""
    from src.data.tecnalia_features import (
        engineer_tecnalia_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": pd.date_range(
                "2025-01-01 10:00:00",
                periods=20,
                freq="10min",
            ),
            "Front GPOA (W/m²)": [500.0] * 20,
            "GHI (W/m²)": [450.0] * 20,
            "Temp. Mod (°C)": [25.0] * 20,
            "Amb. Temp. (°C)": [20.0] * 20,
            "Wind Speed (m/s)": [1.0] * 20,
            "Pmpp (W)": [100.0 + i for i in range(20)],
            "normalized_pmpp": [
                0.50 + i * 0.01
                for i in range(20)
            ],
        }
    )

    result = engineer_tecnalia_features(df)

    expected_columns = [
        "elapsed_minutes",
        "gpoa_30min_mean",
        "gpoa_1h_mean",
        "gpoa_3h_mean",
        "ghi_30min_mean",
        "module_temp_1h_mean",
        "ambient_temp_1h_mean",
        "wind_speed_3h_mean",
        "delta_gpoa",
        "delta_ghi",
        "delta_module_temp",
        "delta_ambient_temp",
        "delta_wind_speed",
        "delta_pmpp",
        "delta_normalized_pmpp",
        "normalized_pmpp_1h_slope",
        "normalized_pmpp_3h_slope",
        "pmpp_1h_slope",
        "pmpp_3h_slope",
        "gpoa_1h_slope",
        "gpoa_3h_slope",
        "pmpp_per_gpoa",
        "normalized_pmpp_per_gpoa",
        "normalized_pmpp_7d_past_mean",
        "normalized_pmpp_30d_past_mean",
        "normalized_pmpp_7d_past_slope",
        "normalized_pmpp_30d_past_slope",
        "hour",
        "day_of_week",
        "day_of_year",
        "month",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]

    for column in expected_columns:
        assert column in result.columns

    assert len(result) == len(df)


def test_engineer_tecnalia_features_preserves_input_dataframe():
    """The complete pipeline must not mutate the input dataframe."""
    from src.data.tecnalia_features import (
        engineer_tecnalia_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": pd.date_range(
                "2025-01-01 10:00:00",
                periods=5,
                freq="10min",
            ),
            "Front GPOA (W/m²)": [500.0] * 5,
            "GHI (W/m²)": [450.0] * 5,
            "Temp. Mod (°C)": [25.0] * 5,
            "Amb. Temp. (°C)": [20.0] * 5,
            "Wind Speed (m/s)": [1.0] * 5,
            "Pmpp (W)": [100.0] * 5,
            "normalized_pmpp": [0.5] * 5,
        }
    )

    original = df.copy(deep=True)

    engineer_tecnalia_features(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_engineer_tecnalia_features_preserves_low_gpoa_rows():
    """General feature engineering must not remove low-GPOA rows."""
    from src.data.tecnalia_features import (
        engineer_tecnalia_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": pd.date_range(
                "2025-01-01 00:00:00",
                periods=4,
                freq="10min",
            ),
            "Front GPOA (W/m²)": [
                0.0,
                100.0,
                200.0,
                500.0,
            ],
            "GHI (W/m²)": [0.0, 80.0, 180.0, 450.0],
            "Temp. Mod (°C)": [15.0] * 4,
            "Amb. Temp. (°C)": [14.0] * 4,
            "Wind Speed (m/s)": [1.0] * 4,
            "Pmpp (W)": [0.0, 10.0, 40.0, 100.0],
            "normalized_pmpp": [0.0, 0.03, 0.12, 0.30],
        }
    )

    result = engineer_tecnalia_features(df)

    assert len(result) == 4

    assert pd.isna(
        result.loc[0, "pmpp_per_gpoa"]
    )

    assert pd.isna(
        result.loc[1, "pmpp_per_gpoa"]
    )

    assert not pd.isna(
        result.loc[2, "pmpp_per_gpoa"]
    )

def test_add_time_features_calendar_values():
    """Calendar features should correctly represent timestamps."""
    from src.data.tecnalia_features import (
        add_time_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 00:00:00",
                "2025-01-05 13:30:00",
            ],
        }
    )

    result = add_time_features(df)

    assert result.loc[0, "hour"] == 0
    assert result.loc[0, "day_of_week"] == 2
    assert result.loc[0, "day_of_year"] == 1
    assert result.loc[0, "month"] == 1

    assert result.loc[1, "hour"] == 13
    assert result.loc[1, "day_of_week"] == 6
    assert result.loc[1, "day_of_year"] == 5
    assert result.loc[1, "month"] == 1


def test_add_time_features_hour_is_cyclical():
    """Hour encoding should preserve the 23:00 -> 00:00 cycle."""
    from src.data.tecnalia_features import (
        add_time_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": [
                "2025-01-01 00:00:00",
                "2025-01-01 23:00:00",
            ],
        }
    )

    result = add_time_features(df)

    # 00:00 and 23:00 should be close in cyclical space.
    distance = np.sqrt(
        (
            result.loc[0, "hour_sin"]
            - result.loc[1, "hour_sin"]
        ) ** 2
        +
        (
            result.loc[0, "hour_cos"]
            - result.loc[1, "hour_cos"]
        ) ** 2
    )

    assert distance < 0.30


def test_add_time_features_cyclical_values_are_bounded():
    """Sine/cosine features must remain within [-1, 1]."""
    from src.data.tecnalia_features import (
        add_time_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": pd.date_range(
                "2025-01-01",
                periods=48,
                freq="h",
            ),
        }
    )

    result = add_time_features(df)

    cyclical_columns = [
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]

    for column in cyclical_columns:
        assert result[column].between(
            -1,
            1,
        ).all()


def test_add_time_features_preserves_rows():
    """Time feature engineering must not change row count."""
    from src.data.tecnalia_features import (
        add_time_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": pd.date_range(
                "2025-01-01",
                periods=10,
                freq="10min",
            ),
        }
    )

    result = add_time_features(df)

    assert len(result) == len(df)


def test_add_time_features_does_not_modify_input():
    """Time feature engineering must not mutate the input."""
    from src.data.tecnalia_features import (
        add_time_features,
    )

    df = pd.DataFrame(
        {
            "Fecha": pd.date_range(
                "2025-01-01",
                periods=3,
                freq="10min",
            ),
        }
    )

    original = df.copy(deep=True)

    add_time_features(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )