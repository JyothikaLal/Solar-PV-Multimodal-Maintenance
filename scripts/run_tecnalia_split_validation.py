from pathlib import Path

import pandas as pd

from src.data.tecnalia_preprocessing import (
    preprocess_tecnalia_module,
)
from src.data.tecnalia_split import (
    build_tecnalia_split_manifest,
    validate_tecnalia_split_manifest,
)


RAW_ROOT = Path("data/raw/tecnalia/TECNALIA")

MODULE_FILES = {
    "Atersa": RAW_ROOT / "data_Atersa/data_Atersa.csv",
    "JaSolar3": RAW_ROOT / "data_JaSolar3/data_JaSolar3.csv",
    "NingboSolar": RAW_ROOT / "data_NingboSolar/data_NingboSolar.csv",
    "Photowatt": RAW_ROOT / "data_Photowatt/data_Photowatt.csv",
    "TrinaSolar": RAW_ROOT / "data_TrinaSolar/data_TrinaSolar.csv",
}

OUTPUT_PATH = Path(
    "reports/results/tecnalia/tecnalia_split_manifest.csv"
)


def main():
    module_dataframes = {}

    for module_name, file_path in MODULE_FILES.items():
        df, _ = preprocess_tecnalia_module(
            file_path,
            module_name,
        )
        module_dataframes[module_name] = df

    manifest = build_tecnalia_split_manifest(
        module_dataframes
    )

    validation = validate_tecnalia_split_manifest(
        manifest
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("TECNALIA split validation: PASS")
    print(f"Total rows: {validation['total_rows']}")
    print(f"Modules: {validation['module_count']}")
    print()

    for module_name, report in validation[
        "module_reports"
    ].items():
        print(module_name)
        print("-" * 50)
        print(f"Train:      {report['train_rows']}")
        print(f"Validation: {report['validation_rows']}")
        print(f"Test:       {report['test_rows']}")
        print(
            f"Train end:  {report['train_end']}"
        )
        print(
            f"Val start:  {report['validation_start']}"
        )
        print(
            f"Val end:    {report['validation_end']}"
        )
        print(
            f"Test start: {report['test_start']}"
        )
        print()

    print(f"Manifest: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()