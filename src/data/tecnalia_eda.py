from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TECNALIA_MODULES = {
    "Atersa": "data_Atersa/data_Atersa.csv",
    "JaSolar3": "data_JaSolar3/data_JaSolar3.csv",
    "NingboSolar": "data_NingboSolar/data_NingboSolar.csv",
    "Photowatt": "data_Photowatt/data_Photowatt.csv",
    "TrinaSolar": "data_TrinaSolar/data_TrinaSolar.csv",
}

ANALYSIS_COLUMNS = [
    "GHI (W/m²)",
    "Front GPOA (W/m²)",
    "Pmpp (W)",
    "Vmpp(V)",
    "Impp(A)",
    "Temp. Mod (°C)",
    "Amb. Temp. (°C)",
    "Wind Speed (m/s)",
]


def load_tecnalia_modules(
    raw_dir: str | Path = "data/raw/tecnalia/TECNALIA",
) -> pd.DataFrame:
    """Load all TECNALIA module telemetry into one analysis dataframe."""

    raw_dir = Path(raw_dir)
    frames = []

    for module, relative_path in TECNALIA_MODULES.items():
        path = raw_dir / relative_path

        df = pd.read_csv(path, sep=";")
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df["module"] = module

        # Convert known sentinel values for analysis only.
        df = df.replace(-9999, float("nan"))

        # Remove physically impossible module-temperature readings.
        df.loc[df["Temp. Mod (°C)"] < -50, "Temp. Mod (°C)"] = float("nan")

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def summarize_operating_behavior(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return descriptive statistics for important operating variables."""

    summary = (
        df.groupby("module")[ANALYSIS_COLUMNS]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(3)
    )

    return summary


def summarize_day_night_behavior(
    df: pd.DataFrame,
    ghi_threshold: float = 20.0,
) -> pd.DataFrame:
    """Compare low-light and daytime operating behavior."""

    analysis = df.copy()

    analysis["operating_period"] = (
        analysis["GHI (W/m²)"] >= ghi_threshold
    ).map(
        {
            True: "daytime",
            False: "low_light",
        }
    )

    return (
        analysis.groupby(["module", "operating_period"])[
            ANALYSIS_COLUMNS
        ]
        .agg(["count", "mean", "median"])
        .round(3)
    )


def plot_distributions(
    df: pd.DataFrame,
    output_dir: str | Path = "reports/figures/tecnalia",
) -> None:
    """Save basic distribution plots for important operating variables."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for column in ANALYSIS_COLUMNS:
        plt.figure(figsize=(9, 5))

        for module, module_df in df.groupby("module"):
            module_df[column].dropna().plot(
                kind="kde",
                label=module,
            )

        plt.title(f"{column} Distribution")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        filename = (
            column.replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("²", "2")
            .replace("°C", "C")
            + "_distribution.png"
        )

        plt.savefig(output_dir / filename, dpi=150)
        plt.close()