from pathlib import Path

import pandas as pd
import numpy as np


TIMESTAMP_COLUMN = "Fecha"


def load_tecnalia_module(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load one TECNALIA module telemetry CSV.

    The raw CSV is read without modifying the source file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"TECNALIA file not found: {file_path}")

    df = pd.read_csv(file_path, sep=";")

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required timestamp column: {TIMESTAMP_COLUMN}"
        )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    invalid_timestamps = df[TIMESTAMP_COLUMN].isna().sum()

    if invalid_timestamps > 0:
        raise ValueError(
            f"Found {invalid_timestamps} invalid timestamps "
            f"in {file_path.name}"
        )

    if not df[TIMESTAMP_COLUMN].is_monotonic_increasing:
        df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    duplicate_timestamps = df[TIMESTAMP_COLUMN].duplicated().sum()

    if duplicate_timestamps > 0:
        raise ValueError(
            f"Found {duplicate_timestamps} duplicate timestamps "
            f"in {file_path.name}"
        )

    return df

def convert_tecnalia_sentinels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert known TECNALIA sentinel values to NaN.

    The input dataframe is copied so the caller's dataframe is not
    modified in place.
    """
    processed = df.copy()

    temperature_column = "Temp. Mod (°C)"

    if temperature_column in processed.columns:
        processed[temperature_column] = processed[temperature_column].replace(
            -9999,
            np.nan,
        )

    return processed

def analyze_tecnalia_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a missing-value summary for a TECNALIA dataframe.

    Returns one row per column with:
    - total rows
    - missing count
    - missing percentage
    - whether the column is completely missing
    """
    summary = pd.DataFrame(
        {
            "total_rows": len(df),
            "missing_count": df.isna().sum(),
        }
    )

    summary["missing_percentage"] = (
        summary["missing_count"] / len(df) * 100
    )

    summary["completely_missing"] = summary["missing_count"].eq(len(df))

    return summary.reset_index(names="column")

NONNEGATIVE_COLUMNS = [
    "Vmpp(V)",
    "Impp(A)",
    "Pmpp (W)",
    "Front GPOA (W/m²)",
    "Wind Speed (m/s)",
]


def validate_tecnalia_physical_ranges(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify physically invalid TECNALIA measurements.

    The function does not modify or remove rows.

    Returns one row per checked column containing:
    - negative_count
    - positive_infinity_count
    - negative_infinity_count
    - invalid_count
    """
    results = []

    for column in NONNEGATIVE_COLUMNS:
        if column not in df.columns:
            continue

        values = df[column]

        negative_count = (values < 0).sum()
        positive_infinity_count = (values == float("inf")).sum()
        negative_infinity_count = (values == float("-inf")).sum()

        results.append(
            {
                "column": column,
                "negative_count": int(negative_count),
                "positive_infinity_count": int(
                    positive_infinity_count
                ),
                "negative_infinity_count": int(
                    negative_infinity_count
                ),
                "invalid_count": int(
                    negative_count
                    + positive_infinity_count
                    + negative_infinity_count
                ),
            }
        )

    return pd.DataFrame(results)

def count_tecnalia_infinite_values(
    df: pd.DataFrame,
) -> int:
    """
    Count all positive and negative infinite values
    across the TECNALIA dataframe.
    """
    numeric_df = df.select_dtypes(include="number")

    return int(numeric_df.isin([float("inf"), float("-inf")]).sum().sum())

def validate_tecnalia_duplicates(
    df: pd.DataFrame,
) -> dict:
    """
    Validate timestamp uniqueness and chronological ordering.

    Duplicate timestamps are treated as a data-quality error rather
    than silently removed.
    """
    duplicate_count = int(
        df[TIMESTAMP_COLUMN].duplicated().sum()
    )

    is_sorted = bool(
        df[TIMESTAMP_COLUMN].is_monotonic_increasing
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate timestamps."
        )

    return {
        "duplicate_count": duplicate_count,
        "is_sorted": is_sorted,
    }

MODULE_RATED_POWER = {
    "Atersa": 330.0,
    "JaSolar3": 315.0,
    "NingboSolar": 175.0,
    "Photowatt": 155.0,
    "TrinaSolar": 185.0,
}

def add_module_metadata(
    df: pd.DataFrame,
    module_name: str,
) -> pd.DataFrame:
    """
    Add module identity, rated power, and normalized Pmpp.

    normalized_pmpp is calculated from the recorded Pmpp value
    divided by the module's STC maximum power.
    """
    if module_name not in MODULE_RATED_POWER:
        raise ValueError(
            f"Unknown TECNALIA module: {module_name}"
        )

    processed = df.copy()

    rated_power = MODULE_RATED_POWER[module_name]

    processed["module_name"] = module_name
    processed["rated_power"] = rated_power

    processed["normalized_pmpp"] = (
        processed["Pmpp (W)"] / rated_power
    )

    return processed

def preprocess_tecnalia_module(
    file_path: str | Path,
    module_name: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the complete TECNALIA preprocessing pipeline for one module.

    Returns:
        processed_df: cleaned dataframe with module metadata
                     and normalized Pmpp.
        validation_report: data-quality information collected
                           during preprocessing.
    """
    # 1. Load and validate timestamps.
    df = load_tecnalia_module(file_path)

    # 2. Convert known sentinel values to NaN.
    processed_df = convert_tecnalia_sentinels(df)

    # 3. Validate timestamp uniqueness and ordering.
    duplicate_results = validate_tecnalia_duplicates(
        processed_df
    )

    # 4. Validate physical ranges.
    physical_results = validate_tecnalia_physical_ranges(
        processed_df
    )

    # 5. Count infinite values.
    infinite_count = count_tecnalia_infinite_values(
        processed_df
    )

    # 6. Analyze missing values before adding derived columns.
    missing_summary = analyze_tecnalia_missing_values(
        processed_df
    )

    # 7. Add module metadata and normalized performance.
    processed_df = add_module_metadata(
        processed_df,
        module_name,
    )

    validation_report = {
        "file": Path(file_path).name,
        "module_name": module_name,
        "rows": len(processed_df),
        "columns": len(processed_df.columns),
        "timestamp_start": processed_df[TIMESTAMP_COLUMN].min(),
        "timestamp_end": processed_df[TIMESTAMP_COLUMN].max(),
        "duplicate_timestamps": duplicate_results[
            "duplicate_count"
        ],
        "timestamp_sorted": duplicate_results[
            "is_sorted"
        ],
        "infinite_values": infinite_count,
        "physical_ranges": physical_results,
        "missing_values": missing_summary,
    }

    return processed_df, validation_report

def build_tecnalia_validation_report(
    validation_reports: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert preprocessing validation reports into two tabular
    dataframes:

    1. module_summary:
       one row per module with core validation information.

    2. missing_values:
       one row per module/column containing missing-value statistics.
    """
    module_rows = []
    missing_rows = []

    for report in validation_reports:
        module_rows.append(
            {
                "module_name": report["module_name"],
                "file": report["file"],
                "rows": report["rows"],
                "columns": report["columns"],
                "timestamp_start": report["timestamp_start"],
                "timestamp_end": report["timestamp_end"],
                "duplicate_timestamps": report[
                    "duplicate_timestamps"
                ],
                "timestamp_sorted": report[
                    "timestamp_sorted"
                ],
                "infinite_values": report[
                    "infinite_values"
                ],
            }
        )

        missing_summary = report["missing_values"].copy()
        missing_summary.insert(
            0,
            "module_name",
            report["module_name"],
        )

        missing_rows.append(missing_summary)

    module_summary = pd.DataFrame(module_rows)
    missing_values = pd.concat(
        missing_rows,
        ignore_index=True,
    )

    return module_summary, missing_values