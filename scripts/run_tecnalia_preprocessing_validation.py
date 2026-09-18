from pathlib import Path

from src.data.tecnalia_preprocessing import (
    build_tecnalia_validation_report,
    preprocess_tecnalia_module,
)


RAW_DIR = Path("data/raw/tecnalia/TECNALIA")
REPORT_DIR = Path("reports/results/tecnalia")

MODULE_FILES = {
    "Atersa": RAW_DIR / "data_Atersa/data_Atersa.csv",
    "JaSolar3": RAW_DIR / "data_JaSolar3/data_JaSolar3.csv",
    "NingboSolar": RAW_DIR / "data_NingboSolar/data_NingboSolar.csv",
    "Photowatt": RAW_DIR / "data_Photowatt/data_Photowatt.csv",
    "TrinaSolar": RAW_DIR / "data_TrinaSolar/data_TrinaSolar.csv",
}


def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_reports = []

    for module_name, file_path in MODULE_FILES.items():
        _, report = preprocess_tecnalia_module(
            file_path,
            module_name,
        )

        validation_reports.append(report)

    module_summary, missing_values = (
        build_tecnalia_validation_report(
            validation_reports
        )
    )

    module_summary.to_csv(
        REPORT_DIR / "preprocessing_module_summary.csv",
        index=False,
    )

    missing_values.to_csv(
        REPORT_DIR / "preprocessing_missing_values.csv",
        index=False,
    )

    print("Preprocessing validation completed.")
    print()
    print(module_summary.to_string(index=False))
    print()
    print("Reports written to:")
    print(REPORT_DIR)


if __name__ == "__main__":
    main()