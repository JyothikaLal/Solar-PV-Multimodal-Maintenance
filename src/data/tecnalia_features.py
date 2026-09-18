from pathlib import Path

import pandas as pd
import numpy as np


TIMESTAMP_COLUMN = "Fecha"


def prepare_tecnalia_time_series(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare a TECNALIA dataframe for time-aware feature engineering.

    The input dataframe is copied and therefore is not modified in place.

    Operations:
    - validate the timestamp column
    - convert timestamps to datetime
    - reject invalid timestamps
    - sort chronologically
    - reject duplicate timestamps
    - calculate elapsed time between observations in minutes
    """
    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required timestamp column: {TIMESTAMP_COLUMN}"
        )

    prepared = df.copy()

    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(
        prepared[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    invalid_timestamps = int(
        prepared[TIMESTAMP_COLUMN].isna().sum()
    )

    if invalid_timestamps > 0:
        raise ValueError(
            f"Found {invalid_timestamps} invalid timestamps."
        )

    prepared = prepared.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)

    duplicate_timestamps = int(
        prepared[TIMESTAMP_COLUMN].duplicated().sum()
    )

    if duplicate_timestamps > 0:
        raise ValueError(
            f"Found {duplicate_timestamps} duplicate timestamps."
        )

    prepared["elapsed_minutes"] = (
        prepared[TIMESTAMP_COLUMN]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    return prepared

ENVIRONMENTAL_FEATURES = {
    "Front GPOA (W/m²)": "gpoa",
    "GHI (W/m²)": "ghi",
    "Temp. Mod (°C)": "module_temp",
    "Amb. Temp. (°C)": "ambient_temp",
    "Wind Speed (m/s)": "wind_speed",
}

ROLLING_WINDOWS = {
    "30min": "30min",
    "1h": "1h",
    "3h": "3h",
}


def add_environmental_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add time-based rolling mean and standard deviation features
    for TECNALIA environmental variables.

    Rolling windows are based on elapsed timestamp time rather
    than row counts.

    The current observation is included in the rolling window.
    Future observations are never used.

    The input dataframe is copied and is not modified in place.
    """
    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required timestamp column: {TIMESTAMP_COLUMN}"
        )

    prepared = df.copy()

    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(
        prepared[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if prepared[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "Found invalid timestamps."
        )

    prepared = prepared.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)

    if prepared[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(
            "Found duplicate timestamps."
        )

    time_indexed = prepared.set_index(
        TIMESTAMP_COLUMN
    )

    for column, prefix in ENVIRONMENTAL_FEATURES.items():
        if column not in time_indexed.columns:
            continue

        for window_name, window in ROLLING_WINDOWS.items():
            rolling = time_indexed[column].rolling(
                window=window,
                closed="right",
                min_periods=1,
            )

            time_indexed[
                f"{prefix}_{window_name}_mean"
            ] = rolling.mean()

            time_indexed[
                f"{prefix}_{window_name}_std"
            ] = rolling.std()

    return time_indexed.reset_index()

DELTA_FEATURES = {
    "Front GPOA (W/m²)": "gpoa",
    "GHI (W/m²)": "ghi",
    "Temp. Mod (°C)": "module_temp",
    "Amb. Temp. (°C)": "ambient_temp",
    "Wind Speed (m/s)": "wind_speed",
    "Pmpp (W)": "pmpp",
    "normalized_pmpp": "normalized_pmpp",
}


def add_delta_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add change-from-previous-observation features.

    Each delta is calculated as:

        current_value - previous_value

    The first observation naturally receives NaN because there
    is no previous observation.

    The input dataframe is copied and is not modified in place.
    """
    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required timestamp column: {TIMESTAMP_COLUMN}"
        )

    prepared = df.copy()

    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(
        prepared[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if prepared[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "Found invalid timestamps."
        )

    prepared = prepared.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)

    if prepared[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(
            "Found duplicate timestamps."
        )

    for column, prefix in DELTA_FEATURES.items():
        if column not in prepared.columns:
            continue

        prepared[f"delta_{prefix}"] = (
            prepared[column].diff()
        )

    return prepared

SLOPE_FEATURES = {
    "normalized_pmpp": "normalized_pmpp",
    "Pmpp (W)": "pmpp",
    "Front GPOA (W/m²)": "gpoa",
}

SLOPE_WINDOWS = {
    "1h": pd.Timedelta(hours=1),
    "3h": pd.Timedelta(hours=3),
}


def add_slope_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add timestamp-aware historical slope features.

    For each requested horizon, the function finds the latest
    observation at or before the historical target timestamp.

    The slope is calculated using the actual elapsed time between
    the historical observation and the current observation.

    No interpolation is performed.

    Future observations are never used.
    """
    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required timestamp column: {TIMESTAMP_COLUMN}"
        )

    prepared = df.copy()

    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(
        prepared[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if prepared[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "Found invalid timestamps."
        )

    prepared = prepared.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)

    if prepared[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(
            "Found duplicate timestamps."
        )

    timestamps = prepared[TIMESTAMP_COLUMN]

    for column, prefix in SLOPE_FEATURES.items():
        if column not in prepared.columns:
            continue

        for window_name, window in SLOPE_WINDOWS.items():
            historical_targets = timestamps - window

            historical_positions = (
                timestamps.searchsorted(
                    historical_targets,
                    side="right",
                )
                - 1
            )

            slopes = []

            for current_position, historical_position in enumerate(
                historical_positions
            ):
                if historical_position < 0:
                    slopes.append(float("nan"))
                    continue

                if historical_position == current_position:
                    slopes.append(float("nan"))
                    continue

                current_value = prepared.loc[
                    current_position,
                    column,
                ]

                historical_value = prepared.loc[
                    historical_position,
                    column,
                ]

                if pd.isna(current_value) or pd.isna(
                    historical_value
                ):
                    slopes.append(float("nan"))
                    continue

                elapsed_hours = (
                    timestamps.iloc[current_position]
                    - timestamps.iloc[historical_position]
                ).total_seconds() / 3600

                if elapsed_hours <= 0:
                    slopes.append(float("nan"))
                    continue

                slope = (
                    current_value - historical_value
                ) / elapsed_hours

                slopes.append(slope)

            prepared[
                f"{prefix}_{window_name}_slope"
            ] = slopes

    return prepared

MIN_GPOA_FOR_NORMALIZED_INDICATORS = 200.0


def add_environmental_normalized_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add guarded performance-to-irradiance indicators.

    Features:
    - pmpp_per_gpoa
    - normalized_pmpp_per_gpoa

    Ratios are calculated only when Front GPOA is at least
    MIN_GPOA_FOR_NORMALIZED_INDICATORS.

    Rows below the threshold are preserved and receive NaN
    for these ratio features.

    The input dataframe is copied and is not modified in place.
    """
    required_columns = [
        "Front GPOA (W/m²)",
        "Pmpp (W)",
        "normalized_pmpp",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    prepared = df.copy()

    gpoa = prepared["Front GPOA (W/m²)"]

    valid_gpoa = (
        gpoa >= MIN_GPOA_FOR_NORMALIZED_INDICATORS
    )

    prepared["pmpp_per_gpoa"] = (
        prepared["Pmpp (W)"]
        .div(gpoa)
        .where(valid_gpoa)
    )

    prepared["normalized_pmpp_per_gpoa"] = (
        prepared["normalized_pmpp"]
        .div(gpoa)
        .where(valid_gpoa)
    )

    return prepared

PERFORMANCE_TREND_WINDOWS = {
    "7d": pd.Timedelta(days=7),
    "30d": pd.Timedelta(days=30),
}


def add_performance_trend_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical normalized-performance trend indicators.

    Features:
    - 7-day past rolling mean
    - 30-day past rolling mean
    - 7-day past slope
    - 30-day past slope

    Only observations strictly before the current timestamp are
    allowed to contribute to these features.

    These are historical performance indicators, not degradation
    labels or degradation ground truth.

    No interpolation is performed.
    """
    required_columns = [
        TIMESTAMP_COLUMN,
        "normalized_pmpp",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    prepared = df.copy()

    # ------------------------------------------------------------------
    # Validate and prepare timestamps
    # ------------------------------------------------------------------
    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(
        prepared[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if prepared[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "Found invalid timestamps."
        )

    prepared = (
        prepared
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    if prepared[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(
            "Found duplicate timestamps."
        )

    time_indexed = prepared.set_index(
        TIMESTAMP_COLUMN
    )

    # ------------------------------------------------------------------
    # Historical performance series
    #
    # Shift by one observation so the current normalized_pmpp value
    # can never contribute to its own historical features.
    # ------------------------------------------------------------------
    historical_performance = (
        time_indexed["normalized_pmpp"].shift(1)
    )

    # ------------------------------------------------------------------
    # Historical rolling means and timestamp-aware slopes
    # ------------------------------------------------------------------
    timestamps = time_indexed.index
    performance_values = time_indexed["normalized_pmpp"]

    for window_name, window in PERFORMANCE_TREND_WINDOWS.items():

        # ==============================================================
        # Past-only rolling mean
        # ==============================================================

        rolling = historical_performance.rolling(
            window=window,
            closed="right",
            min_periods=1,
        )

        time_indexed[
            f"normalized_pmpp_{window_name}_past_mean"
        ] = rolling.mean()

        # ==============================================================
        # Timestamp-aware historical slope
        #
        # For each current timestamp:
        #   1. Exclude the current observation.
        #   2. Find the earliest available historical observation
        #      within the requested time window.
        #   3. Use the latest available observation before the current
        #      timestamp as the second point.
        #   4. Calculate slope using the actual elapsed time.
        #
        # No interpolation is performed.
        # ==============================================================

        slopes = []

        for current_position, current_timestamp in enumerate(
            timestamps
        ):
            # Current observation must never be used.
            latest_past_position = current_position - 1

            if latest_past_position < 0:
                slopes.append(float("nan"))
                continue

            # Beginning of the historical window.
            window_start = current_timestamp - window

            # Only timestamps up to the latest past observation
            # are eligible.
            historical_candidates = timestamps[
                :latest_past_position + 1
            ]

            # Find the first observation at or after the beginning
            # of the historical window.
            historical_position = (
                historical_candidates.searchsorted(
                    window_start,
                    side="left",
                )
            )

            # No historical observation exists in the window.
            if (
                historical_position
                >= latest_past_position + 1
            ):
                slopes.append(float("nan"))
                continue

            historical_value = performance_values.iloc[
                historical_position
            ]

            latest_past_value = performance_values.iloc[
                latest_past_position
            ]

            # Missing performance values cannot produce a slope.
            if (
                pd.isna(historical_value)
                or pd.isna(latest_past_value)
            ):
                slopes.append(float("nan"))
                continue

            elapsed_hours = (
                timestamps[latest_past_position]
                - timestamps[historical_position]
            ).total_seconds() / 3600

            # Defensive protection against invalid time intervals.
            if elapsed_hours <= 0:
                slopes.append(float("nan"))
                continue

            slope = (
                latest_past_value - historical_value
            ) / elapsed_hours

            slopes.append(slope)

        time_indexed[
            f"normalized_pmpp_{window_name}_past_slope"
        ] = slopes

    return time_indexed.reset_index()

def engineer_tecnalia_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the complete TECNALIA feature-engineering pipeline.

    Pipeline order:
    1. Prepare and validate timestamps.
    2. Add environmental rolling statistics.
    3. Add first-order delta features.
    4. Add timestamp-aware short-term slopes.
    5. Add environmental-normalized performance indicators.
    6. Add historical performance trend indicators.

    The pipeline:
    - preserves the input row count,
    - does not resample the raw time series,
    - does not apply model-specific scaling,
    - does not use future observations,
    - does not create degradation labels,
    - preserves missing values as NaN.

    Returns
    -------
    pd.DataFrame
        Feature-engineered TECNALIA dataframe.
    """
    required_columns = [
        TIMESTAMP_COLUMN,
        "Front GPOA (W/m²)",
        "GHI (W/m²)",
        "Temp. Mod (°C)",
        "Amb. Temp. (°C)",
        "Wind Speed (m/s)",
        "Pmpp (W)",
        "normalized_pmpp",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    original_row_count = len(df)

    features = prepare_tecnalia_time_series(df)

    features = add_time_features(
        features
    )

    features = add_environmental_rolling_features(
        features
    )

    features = add_delta_features(
        features
    )

    features = add_slope_features(
        features
    )

    features = add_environmental_normalized_features(
        features
    )

    features = add_performance_trend_indicators(
        features
    )

    if len(features) != original_row_count:
        raise ValueError(
            "Feature engineering changed the number of rows: "
            f"{original_row_count} -> {len(features)}"
        )

    return features

def add_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add calendar and cyclical time features.

    Calendar features:
    - hour
    - day_of_week
    - day_of_year
    - month

    Cyclical features:
    - hour_sin
    - hour_cos
    - day_of_year_sin
    - day_of_year_cos

    The timestamp itself is preserved.

    Returns
    -------
    pd.DataFrame
        Dataframe with time-derived features added.
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
        raise ValueError(
            "Found invalid timestamps."
        )

    prepared["hour"] = (
        prepared[TIMESTAMP_COLUMN].dt.hour
    )

    prepared["day_of_week"] = (
        prepared[TIMESTAMP_COLUMN].dt.dayofweek
    )

    prepared["day_of_year"] = (
        prepared[TIMESTAMP_COLUMN].dt.dayofyear
    )

    prepared["month"] = (
        prepared[TIMESTAMP_COLUMN].dt.month
    )

    # ------------------------------------------------------------------
    # Cyclical hour encoding
    # ------------------------------------------------------------------
    prepared["hour_sin"] = np.sin(
        2 * np.pi * prepared["hour"] / 24
    )

    prepared["hour_cos"] = np.cos(
        2 * np.pi * prepared["hour"] / 24
    )

    # ------------------------------------------------------------------
    # Cyclical day-of-year encoding
    #
    # 365 is used as the annual cycle length for this feature.
    # ------------------------------------------------------------------
    prepared["day_of_year_sin"] = np.sin(
        2 * np.pi * prepared["day_of_year"] / 365
    )

    prepared["day_of_year_cos"] = np.cos(
        2 * np.pi * prepared["day_of_year"] / 365
    )

    return prepared