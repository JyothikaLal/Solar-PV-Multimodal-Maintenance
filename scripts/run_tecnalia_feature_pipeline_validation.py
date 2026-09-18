from pathlib import Path

import pandas as pd

from src.data.tecnalia_features import engineer_tecnalia_features
from src.data.tecnalia_preprocessing import (
    preprocess_tecnalia_module,
)


TECNALIA_MODULES = {
    "Atersa": "data_Atersa/data_Atersa.csv",
    "JaSolar3": "data_JaSolar3/data_JaSolar3.csv",
    "NingboSolar": "data_NingboSolar/data_NingboSolar.csv",
    "Photowatt": "data_Photowatt/data_Photowatt.csv",
    "TrinaSolar": "data_TrinaSolar/data_TrinaSolar.csv",
}

RAW_ROOT = Path("data/raw/tecnalia/TECNALIA")
RESULTS_DIR = Path("reports/results/tecnalia")


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for module_name, relative_path in TECNALIA_MODULES.items():
        file_path = RAW_ROOT / relative_path

        print(f"\nProcessing {module_name}...")
        print(f"File: {file_path}")

        df, preprocessing_report = preprocess_tecnalia_module(
            file_path,
            module_name,
        )


        result = engineer_tecnalia_features(df)

        feature_columns = [
            column
            for column in result.columns
            if column not in df.columns
        ]

        missing_feature_count = int(
            result[feature_columns]
            .isna()
            .sum()
            .sum()
        )

        summary_rows.append(
            {
                "module_name": module_name,
                "input_rows": len(df),
                "output_rows": len(result),
                "input_columns": len(df.columns),
                "output_columns": len(result.columns),
                "new_feature_count": len(feature_columns),
                "missing_feature_values": missing_feature_count,
                "timestamp_start": result["Fecha"].min(),
                "timestamp_end": result["Fecha"].max(),
                "duplicate_timestamps": int(
                    result["Fecha"].duplicated().sum()
                ),
            }
        )

        print(
            f"Rows: {len(df)} -> {len(result)}"
        )
        print(
            f"Columns: {len(df.columns)} -> "
            f"{len(result.columns)}"
        )
        print(
            f"New features: {len(feature_columns)}"
        )
        print(
            f"Missing engineered values: "
            f"{missing_feature_count}"
        )

    summary = pd.DataFrame(summary_rows)

    output_path = (
        RESULTS_DIR
        / "feature_pipeline_validation_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print("\nValidation summary:")
    print(summary.to_string(index=False))

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()