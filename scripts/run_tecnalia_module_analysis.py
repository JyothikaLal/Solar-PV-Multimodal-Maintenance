from pathlib import Path

from src.data.tecnalia_module_analysis import (
    compare_matched_gpoa,
    gpoa_temperature_summary,
    load_module_data,
    summarize_normalized_performance,
    temperature_adjusted_comparison,
)


OUTPUT_DIR = Path("reports/results/tecnalia")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_module_data()

    print("=== Module Ratings ===")
    print(
        df[["module", "rated_power_w"]]
        .drop_duplicates()
        .sort_values("module")
        .to_string(index=False)
    )

    print("\n=== Normalized Performance ===")
    normalized = summarize_normalized_performance(df)
    print(normalized)

    print("\n=== Matched GPOA: 800–1000 W/m² ===")
    matched_gpoa = compare_matched_gpoa(df)
    print(matched_gpoa)

    print("\n=== Temperature-Adjusted Comparison ===")
    temperature = temperature_adjusted_comparison(df)
    print(temperature)

    print("\n=== GPOA / Temperature / Performance Summary ===")
    summary = gpoa_temperature_summary(df)
    print(summary)

    normalized.to_csv(
        OUTPUT_DIR / "module_normalized_summary.csv"
    )

    matched_gpoa.to_csv(
        OUTPUT_DIR / "module_gpoa_comparison.csv"
    )

    temperature.to_csv(
        OUTPUT_DIR / "module_temperature_analysis.csv"
    )

    summary.to_csv(
        OUTPUT_DIR / "module_environment_summary.csv"
    )

    print("\nModule analysis completed.")


if __name__ == "__main__":
    main()