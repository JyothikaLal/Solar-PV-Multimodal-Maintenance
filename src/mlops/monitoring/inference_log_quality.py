from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATABASE_PATH = Path("inference_logs.db")

EXPECTED_CLASSES = [
    "No-Anomaly",
    "Cell",
    "Vegetation",
    "Diode",
    "Cell-Multi",
    "Shadowing",
    "Cracking",
    "Offline-Module",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Soiling",
    "Diode-Multi",
]

REQUIRED_TEXT_COLUMNS = [
    "inference_id",
    "telemetry_model_name",
    "telemetry_model_version",
    "thermal_model_name",
    "thermal_model_version",
    "thermal_anomaly_class",
    "thermal_evidence_state",
    "combined_evidence_state",
    "maintenance_priority",
]


REQUIRED_COLUMNS = [
    "inference_id",
    "prediction_timestamp",
    "telemetry_model_name",
    "telemetry_model_version",
    "thermal_model_name",
    "thermal_model_version",
    "telemetry_predicted_normalized_pmpp",
    "thermal_anomaly_class",
    "thermal_anomaly_confidence",
    "thermal_class_probabilities",
    "thermal_anomaly_detected",
    "thermal_evidence_state",
    "embedding_indicators",
    "health_telemetry_evidence_level",
    "health_thermal_evidence_level",
    "combined_evidence_state",
    "maintenance_priority",
    "created_at",
]


def _validate_probability_json(
    value: Any,
) -> tuple[bool, str | None]:
    """Validate one stored thermal probability JSON value."""

    if not isinstance(value, str) or not value.strip():
        return False, "missing"

    try:
        probabilities = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return False, "invalid_json"

    if not isinstance(probabilities, dict):
        return False, "not_object"

    if set(probabilities) != set(EXPECTED_CLASSES):
        return False, "invalid_class_set"

    values = np.asarray(
        [probabilities[class_name] for class_name in EXPECTED_CLASSES],
        dtype=float,
    )

    if not np.isfinite(values).all():
        return False, "non_finite"

    if (values < 0).any():
        return False, "negative"

    if not np.isclose(
        float(values.sum()),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        return False, "invalid_probability_sum"

    return True, None


def _validate_optional_json(
    value: Any,
) -> tuple[bool, str | None]:
    """Validate optional embedding-indicator JSON."""

    if value is None:
        return True, None

    if not isinstance(value, str) or not value.strip():
        return False, "invalid_json"

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return False, "invalid_json"

    if not isinstance(parsed, dict):
        return False, "not_object"

    return True, None


def load_inference_logs(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    """Load all persisted inference events."""

    database_path = Path(database_path)

    if not database_path.exists():
        raise FileNotFoundError(
            f"Inference-log database does not exist: {database_path}"
        )

    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM inference_logs",
            connection,
        )


def monitor_inference_log_quality(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """Validate persisted inference-event data quality."""

    frame = load_inference_logs(database_path)

    row_count = len(frame)

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(frame.columns)
    )

    if missing_columns:
        return {
            "database_path": str(database_path),
            "row_count": row_count,
            "required_columns_valid": False,
            "missing_columns": missing_columns,
            "quality_valid": False,
        }

    if row_count == 0:
        return {
            "database_path": str(database_path),
            "row_count": 0,
            "required_columns_valid": True,
            "quality_valid": True,
            "empty_log": True,
            "invalid_timestamp_count": 0,
            "invalid_created_at_count": 0,
            "non_finite_telemetry_prediction_count": 0,
            "invalid_thermal_confidence_count": 0,
            "invalid_probability_count": 0,
            "invalid_thermal_class_count": 0,
            "invalid_model_metadata_count": 0,
            "invalid_anomaly_flag_count": 0,
            "invalid_evidence_state_count": 0,
            "invalid_maintenance_priority_count": 0,
            "invalid_embedding_indicator_count": 0,
        }

    prediction_timestamps = pd.to_datetime(
        frame["prediction_timestamp"],
        errors="coerce",
        utc=True,
    )

    created_timestamps = pd.to_datetime(
        frame["created_at"],
        errors="coerce",
        utc=True,
    )

    invalid_timestamp_count = int(
        prediction_timestamps.isna().sum()
    )

    invalid_created_at_count = int(
        created_timestamps.isna().sum()
    )

    telemetry_values = pd.to_numeric(
        frame["telemetry_predicted_normalized_pmpp"],
        errors="coerce",
    )

    invalid_telemetry_prediction_count = int(
        (~np.isfinite(telemetry_values.to_numpy())).sum()
    )

    thermal_confidence = pd.to_numeric(
        frame["thermal_anomaly_confidence"],
        errors="coerce",
    )

    invalid_thermal_confidence_count = int(
        (
            ~np.isfinite(thermal_confidence.to_numpy())
            | (thermal_confidence.to_numpy() < 0)
            | (thermal_confidence.to_numpy() > 1)
        ).sum()
    )

    probability_errors: list[str | None] = []

    for value in frame["thermal_class_probabilities"]:
        _, error = _validate_probability_json(value)
        probability_errors.append(error)

    invalid_probability_count = sum(
        error is not None
        for error in probability_errors
    )

    invalid_thermal_class_count = int(
        (
            frame["thermal_anomaly_class"].isna()
            | ~frame["thermal_anomaly_class"].isin(
                EXPECTED_CLASSES
            )
        ).sum()
    )

    invalid_model_metadata_count = 0

    for column in [
        "telemetry_model_name",
        "telemetry_model_version",
        "thermal_model_name",
        "thermal_model_version",
    ]:
        invalid_model_metadata_count += int(
            frame[column]
            .isna()
            .sum()
        )
        invalid_model_metadata_count += int(
            frame[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

    anomaly_flag = pd.to_numeric(
        frame["thermal_anomaly_detected"],
        errors="coerce",
    )

    invalid_anomaly_flag_count = int(
        (
            ~anomaly_flag.isin([0, 1])
        ).sum()
    )

    invalid_evidence_state_count = int(
        (
            frame["thermal_evidence_state"].isna()
            | frame["combined_evidence_state"].isna()
            | frame["thermal_evidence_state"]
            .astype(str)
            .str.strip()
            .eq("")
            | frame["combined_evidence_state"]
            .astype(str)
            .str.strip()
            .eq("")
        ).sum()
    )

    invalid_maintenance_priority_count = int(
        (
            frame["maintenance_priority"].isna()
            | frame["maintenance_priority"]
            .astype(str)
            .str.strip()
            .eq("")
        ).sum()
    )

    invalid_embedding_indicator_count = sum(
        not _validate_optional_json(value)[0]
        for value in frame["embedding_indicators"]
    )

    quality_valid = all(
        count == 0
        for count in [
            invalid_timestamp_count,
            invalid_created_at_count,
            invalid_telemetry_prediction_count,
            invalid_thermal_confidence_count,
            invalid_probability_count,
            invalid_thermal_class_count,
            invalid_model_metadata_count,
            invalid_anomaly_flag_count,
            invalid_evidence_state_count,
            invalid_maintenance_priority_count,
            invalid_embedding_indicator_count,
        ]
    )

    return {
        "database_path": str(database_path),
        "row_count": row_count,
        "required_columns_valid": True,
        "quality_valid": quality_valid,
        "empty_log": False,
        "invalid_timestamp_count": invalid_timestamp_count,
        "invalid_created_at_count": invalid_created_at_count,
        "non_finite_telemetry_prediction_count": (
            invalid_telemetry_prediction_count
        ),
        "invalid_thermal_confidence_count": (
            invalid_thermal_confidence_count
        ),
        "invalid_probability_count": invalid_probability_count,
        "invalid_probability_errors": {
            error: probability_errors.count(error)
            for error in sorted(
                {
                    error
                    for error in probability_errors
                    if error is not None
                }
            )
        },
        "invalid_thermal_class_count": (
            invalid_thermal_class_count
        ),
        "invalid_model_metadata_count": (
            invalid_model_metadata_count
        ),
        "invalid_anomaly_flag_count": (
            invalid_anomaly_flag_count
        ),
        "invalid_evidence_state_count": (
            invalid_evidence_state_count
        ),
        "invalid_maintenance_priority_count": (
            invalid_maintenance_priority_count
        ),
        "invalid_embedding_indicator_count": (
            invalid_embedding_indicator_count
        ),
    }
