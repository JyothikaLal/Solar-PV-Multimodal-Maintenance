from pathlib import Path

import pandas as pd


MODULE_RATINGS = {
    "Atersa": 330.0,
    "JaSolar3": 315.0,
    "NingboSolar": 175.0,
    "Photowatt": 155.0,
    "TrinaSolar": 185.0,
}

TECNALIA_MODULES = {
    "Atersa": "data_Atersa/data_Atersa.csv",
    "JaSolar3": "data_JaSolar3/data_JaSolar3.csv",
    "NingboSolar": "data_NingboSolar/data_NingboSolar.csv",
    "Photowatt": "data_Photowatt/data_Photowatt.csv",
    "TrinaSolar": "data_TrinaSolar/data_TrinaSolar.csv",
}


def load_module_data(
    raw_dir: str | Path = "data/raw/tecnalia/TECNALIA",
) -> pd.DataFrame:
    """Load TECNALIA telemetry and calculate normalized performance."""

    raw_dir = Path(raw_dir)
    frames = []

    for module, relative_path in TECNALIA_MODULES.items():
        path = raw_dir / relative_path

        df = pd.read_csv(path, sep=";")
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df["module"] = module

        # Analysis only: convert known sentinel to NaN.
        df = df.replace(-9999, float("nan"))

        df["rated_power_w"] = MODULE_RATINGS[module]
        df["normalized_pmpp"] = (
            df["Pmpp (W)"] / df["rated_power_w"]
        )

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def summarize_normalized_performance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize normalized Pmpp by module."""

    return (
        df.groupby("module")["normalized_pmpp"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(4)
    )


def compare_matched_gpoa(
    df: pd.DataFrame,
    lower: float = 800.0,
    upper: float = 1000.0,
) -> pd.DataFrame:
    """Compare normalized performance at matched GPOA."""

    subset = df[
        (df["Front GPOA (W/m²)"] >= lower)
        & (df["Front GPOA (W/m²)"] < upper)
    ].copy()

    return (
        subset.groupby("module")["normalized_pmpp"]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )


def temperature_adjusted_comparison(
    df: pd.DataFrame,
    gpoa_lower: float = 800.0,
    gpoa_upper: float = 1000.0,
) -> pd.DataFrame:
    """Compare normalized performance by module temperature."""

    subset = df[
        (df["Front GPOA (W/m²)"] >= gpoa_lower)
        & (df["Front GPOA (W/m²)"] < gpoa_upper)
    ].copy()

    subset["temperature_bin"] = pd.cut(
        subset["Temp. Mod (°C)"],
        bins=[-10, 10, 20, 30, 40, 50, 60, 80, 150],
        right=False,
    )

    return (
        subset.groupby(
            ["module", "temperature_bin"],
            observed=True,
        )["normalized_pmpp"]
        .agg(["count", "mean", "median"])
        .round(4)
    )


def gpoa_temperature_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize GPOA, temperature, and performance by module."""

    return (
        df.groupby("module")[
            [
                "Front GPOA (W/m²)",
                "Temp. Mod (°C)",
                "normalized_pmpp",
            ]
        ]
        .agg(["mean", "median", "std"])
        .round(4)
    )
