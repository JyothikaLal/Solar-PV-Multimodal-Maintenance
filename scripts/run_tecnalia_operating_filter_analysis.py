from pathlib import Path

from src.data.tecnalia_module_analysis import load_module_data
from src.data.tecnalia_operating_filter_analysis import (
    analyze_common_threshold,
    analyze_gpoa_thresholds,
)


OUTPUT_DIR = Path("reports/results/tecnalia")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_module_data()

    by_module = analyze_gpoa_thresholds(df)
    common = analyze_common_threshold(df)

    print("=== GPOA Threshold Analysis by Module ===")
    print(by_module.to_string(index=False))

    print("\n=== Average Threshold Summary ===")
    print(common.to_string())

    by_module.to_csv(
        OUTPUT_DIR / "operating_filter_by_module.csv",
        index=False,
    )

    common.to_csv(
        OUTPUT_DIR / "operating_filter_summary.csv",
    )

    print("\nOperating-condition analysis completed.")


if __name__ == "__main__":
    main()
