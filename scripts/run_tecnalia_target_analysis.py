from pathlib import Path

from src.data.tecnalia_module_analysis import load_module_data
from src.data.tecnalia_target_analysis import (
    summarize_daytime_target,
    summarize_gpoa_ranges,
    summarize_target,
    summarize_temperature_ranges,
)


OUTPUT_DIR = Path("reports/results/tecnalia")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_module_data()

    target = summarize_target(df)
    gpoa = summarize_gpoa_ranges(df)
    temperature = summarize_temperature_ranges(df)
    daytime = summarize_daytime_target(df)

    print("=== Normalized Target Summary ===")
    print(target.to_string())

    print("\n=== Normalized Target by GPOA Range ===")
    print(gpoa.to_string())

    print("\n=== Normalized Target by Module Temperature ===")
    print(temperature.to_string())

    print("\n=== Daytime Normalized Target ===")
    print(daytime.to_string())

    target.to_csv(
        OUTPUT_DIR / "target_normalized_summary.csv"
    )
    gpoa.to_csv(
        OUTPUT_DIR / "target_gpoa_ranges.csv"
    )
    temperature.to_csv(
        OUTPUT_DIR / "target_temperature_ranges.csv"
    )
    daytime.to_csv(
        OUTPUT_DIR / "target_daytime_summary.csv"
    )

    print("\nTarget analysis completed.")


if __name__ == "__main__":
    main()
