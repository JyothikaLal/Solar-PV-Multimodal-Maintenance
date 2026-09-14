import pandas as pd


GPOA_THRESHOLDS = [20, 100, 200, 400, 600]


def analyze_gpoa_thresholds(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare candidate GPOA thresholds for regression filtering."""

    data = df.copy()

    data["normalized_pmpp"] = (
        data["Pmpp (W)"] / data["rated_power_w"]
    )

    rows = []

    for module, module_df in data.groupby("module"):
        total = len(module_df)

        for threshold in GPOA_THRESHOLDS:
            subset = module_df[
                module_df["Front GPOA (W/m²)"] >= threshold
            ].copy()

            target = subset["normalized_pmpp"].dropna()

            rows.append(
                {
                    "module": module,
                    "gpoa_threshold": threshold,
                    "total_rows": total,
                    "retained_rows": len(subset),
                    "retained_pct": len(subset) / total * 100,
                    "target_count": target.count(),
                    "target_mean": target.mean(),
                    "target_median": target.median(),
                    "target_std": target.std(),
                    "target_min": target.min(),
                    "target_max": target.max(),
                }
            )

    return pd.DataFrame(rows).round(4)


def analyze_common_threshold(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize retention across modules for each threshold."""

    result = analyze_gpoa_thresholds(df)

    return (
        result.groupby("gpoa_threshold")[
            [
                "total_rows",
                "retained_rows",
                "retained_pct",
                "target_count",
                "target_mean",
                "target_median",
                "target_std",
            ]
        ]
        .mean()
        .round(4)
    )
