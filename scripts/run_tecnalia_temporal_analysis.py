from pathlib import Path

from src.data.tecnalia_module_analysis import load_module_data
from src.data.tecnalia_temporal_analysis import (
    monthly_environment_summary,
    monthly_matched_gpoa,
    monthly_summary,
)


OUTPUT_DIR = Path("reports/results/tecnalia")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_module_data()

    monthly = monthly_summary(df)
    matched = monthly_matched_gpoa(df)
    environment = monthly_environment_summary(df)

    print("=== Monthly Summary ===")
    print(monthly.to_string())

    print("\n=== Monthly Performance at Matched GPOA: 800–1000 W/m² ===")
    print(matched.to_string())

    print("\n=== Monthly Environmental Conditions ===")
    print(environment.to_string())

    monthly.to_csv(OUTPUT_DIR / "monthly_summary.csv")
    matched.to_csv(OUTPUT_DIR / "monthly_matched_gpoa.csv")
    environment.to_csv(
        OUTPUT_DIR / "monthly_environment_summary.csv"
    )

    print("\nTemporal analysis completed.")


if __name__ == "__main__":
    main()
