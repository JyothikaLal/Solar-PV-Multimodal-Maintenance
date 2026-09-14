from pathlib import Path

from src.data.tecnalia_eda import load_tecnalia_modules
from src.data.tecnalia_relationships import (
    calculate_correlations,
    calculate_gpoa_bins,
    plot_gpoa_vs_pmpp,
)


OUTPUT_DIR = Path("reports/results/tecnalia")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_tecnalia_modules()

    correlations = calculate_correlations(df)
    correlations.to_csv(
        OUTPUT_DIR / "environmental_pmpp_correlations.csv",
        index=False,
    )

    gpoa_bins = calculate_gpoa_bins(df)
    gpoa_bins.to_csv(
        OUTPUT_DIR / "gpoa_pmpp_bins.csv"
    )

    plot_gpoa_vs_pmpp(df)

    print("\n=== Environmental → Pmpp Correlations ===")
    print(correlations.to_string(index=False))

    print("\n=== Front GPOA → Pmpp Bins ===")
    print(gpoa_bins.to_string())

    print("\nRelationship analysis completed.")


if __name__ == "__main__":
    main()