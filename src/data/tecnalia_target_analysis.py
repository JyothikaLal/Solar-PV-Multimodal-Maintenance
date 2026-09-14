import pandas as pd


def prepare_target_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare TECNALIA data for regression-target analysis."""

    data = df.copy()

    data["normalized_pmpp"] = (
        data["Pmpp (W)"] / data["rated_power_w"]
    )

    return data


def summarize_target(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize normalized performance by module."""

    data = prepare_target_data(data)

    return (
        data.groupby("module")["normalized_pmpp"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(4)
    )


def summarize_gpoa_ranges(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze normalized performance across GPOA operating ranges."""

    data = prepare_target_data(data)

    bins = [20, 200, 400, 600, 800, 1000, 1200, 1500, float("inf")]
    labels = [
        "20-200",
        "200-400",
        "400-600",
        "600-800",
        "800-1000",
        "1000-1200",
        "1200-1500",
        "1500+",
    ]

    data["gpoa_range"] = pd.cut(
        data["Front GPOA (W/m²)"],
        bins=bins,
        labels=labels,
        right=False,
    )

    return (
        data.dropna(subset=["gpoa_range"])
        .groupby(["module", "gpoa_range"], observed=True)[
            "normalized_pmpp"
        ]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(4)
    )


def summarize_temperature_ranges(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze normalized performance across module-temperature ranges."""

    data = prepare_target_data(data)

    bins = [-10, 0, 10, 20, 30, 40, 50, 60, 80, 150]
    labels = [
        "-10-0",
        "0-10",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "50-60",
        "60-80",
        "80+",
    ]

    data["temperature_range"] = pd.cut(
        data["Temp. Mod (°C)"],
        bins=bins,
        labels=labels,
        right=False,
    )

    return (
        data.dropna(subset=["temperature_range"])
        .groupby(
            ["module", "temperature_range"],
            observed=True,
        )["normalized_pmpp"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(4)
    )


def summarize_daytime_target(
    data: pd.DataFrame,
    ghi_threshold: float = 20.0,
) -> pd.DataFrame:
    """Summarize normalized performance for daytime observations."""

    data = prepare_target_data(data)

    daytime = data[
        data["GHI (W/m²)"] >= ghi_threshold
    ].copy()

    return (
        daytime.groupby("module")["normalized_pmpp"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(4)
    )
