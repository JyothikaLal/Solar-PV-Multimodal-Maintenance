from pathlib import Path

from src.data.tecnalia_eda import (
    load_tecnalia_modules,
    plot_distributions,
    summarize_day_night_behavior,
    summarize_operating_behavior,
)


OUTPUT_DIR = Path("reports/results/tecnalia")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_tecnalia_modules()

    print("Dataset shape:")
    print(df.shape)

    print("\nRows by module:")
    print(df["module"].value_counts().sort_index())

    print("\nTime coverage:")
    print(
        df.groupby("module")["Fecha"]
        .agg(["min", "max"])
    )

    print("\nOperating behavior:")
    print(summarize_operating_behavior(df))

    print("\nDay/night behavior:")
    print(summarize_day_night_behavior(df))

    summary = summarize_operating_behavior(df)
    summary.to_csv(
        OUTPUT_DIR / "operating_behavior_summary.csv"
    )

    day_night = summarize_day_night_behavior(df)
    day_night.to_csv(
        OUTPUT_DIR / "day_night_behavior_summary.csv"
    )

    plot_distributions(df)

    print("\nEDA outputs saved successfully.")


if __name__ == "__main__":
    main()