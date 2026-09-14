from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ENVIRONMENTAL_COLUMNS = [
    "GHI (W/m²)",
    "Front GPOA (W/m²)",
    "Temp. Mod (°C)",
    "Amb. Temp. (°C)",
    "Wind Speed (m/s)",
]

TARGET_COLUMN = "Pmpp (W)"


def calculate_correlations(
    df: pd.DataFrame,
    ghi_threshold: float = 20.0,
) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlations with Pmpp."""

    daytime = df[df["GHI (W/m²)"] >= ghi_threshold].copy()

    records = []

    for module, module_df in daytime.groupby("module"):
        for feature in ENVIRONMENTAL_COLUMNS:
            pair = module_df[[feature, TARGET_COLUMN]].dropna()

            records.append(
                {
                    "module": module,
                    "feature": feature,
                    "pearson": pair[feature].corr(
                        pair[TARGET_COLUMN],
                        method="pearson",
                    ),
                    "spearman": pair[feature].corr(
                        pair[TARGET_COLUMN],
                        method="spearman",
                    ),
                    "n": len(pair),
                }
            )

    return pd.DataFrame(records)


def calculate_gpoa_bins(
    df: pd.DataFrame,
    ghi_threshold: float = 20.0,
) -> pd.DataFrame:
    """Summarize Pmpp across Front GPOA operating ranges."""

    daytime = df[df["GHI (W/m²)"] >= ghi_threshold].copy()

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

    daytime["gpoa_bin"] = pd.cut(
        daytime["Front GPOA (W/m²)"],
        bins=bins,
        labels=labels,
        right=False,
    )

    return (
        daytime.groupby(["module", "gpoa_bin"], observed=True)[
            ["Front GPOA (W/m²)", TARGET_COLUMN]
        ]
        .agg(["count", "mean", "median"])
        .round(3)
    )


def plot_gpoa_vs_pmpp(
    df: pd.DataFrame,
    output_dir: str | Path = "reports/figures/tecnalia",
    ghi_threshold: float = 20.0,
) -> None:
    """Plot Front GPOA against Pmpp for each module."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daytime = df[df["GHI (W/m²)"] >= ghi_threshold]

    for module, module_df in daytime.groupby("module"):
        plt.figure(figsize=(9, 5))

        plt.scatter(
            module_df["Front GPOA (W/m²)"],
            module_df[TARGET_COLUMN],
            s=4,
            alpha=0.25,
        )

        plt.title(f"{module}: Front GPOA vs Pmpp")
        plt.xlabel("Front GPOA (W/m²)")
        plt.ylabel("Pmpp (W)")
        plt.tight_layout()

        plt.savefig(
            output_dir / f"{module}_gpoa_vs_pmpp.png",
            dpi=150,
        )
        plt.close()