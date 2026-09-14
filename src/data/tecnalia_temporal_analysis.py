from pathlib import Path

import pandas as pd


def prepare_temporal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare TECNALIA data for temporal analysis."""

    result = df.copy()

    result["month"] = result["Fecha"].dt.to_period("M").astype(str)

    return result


def monthly_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize environmental and performance behavior by month."""

    data = prepare_temporal_data(df)

    return (
        data.groupby(["module", "month"])[
            [
                "Front GPOA (W/m²)",
                "Temp. Mod (°C)",
                "Pmpp (W)",
                "normalized_pmpp",
            ]
        ]
        .agg(["count", "mean", "median"])
        .round(4)
    )


def monthly_matched_gpoa(
    df: pd.DataFrame,
    lower: float = 800.0,
    upper: float = 1000.0,
) -> pd.DataFrame:
    """Compare monthly normalized performance at matched GPOA."""

    data = prepare_temporal_data(df)

    subset = data[
        (data["Front GPOA (W/m²)"] >= lower)
        & (data["Front GPOA (W/m²)"] < upper)
    ].copy()

    return (
        subset.groupby(["module", "month"])["normalized_pmpp"]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )


def monthly_environment_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize monthly environmental conditions."""

    data = prepare_temporal_data(df)

    return (
        data.groupby("month")[
            [
                "GHI (W/m²)",
                "Front GPOA (W/m²)",
                "Temp. Mod (°C)",
                "Amb. Temp. (°C)",
                "Wind Speed (m/s)",
            ]
        ]
        .mean()
        .round(4)
    )