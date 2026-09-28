from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.mlops.inference_logging import InferenceLogStore
from src.mlops.monitoring.inference_log_quality import (
    monitor_inference_log_quality,
)


PROBABILITIES = {
    "No-Anomaly": 0.70,
    "Cell": 0.05,
    "Vegetation": 0.04,
    "Diode": 0.04,
    "Cell-Multi": 0.03,
    "Shadowing": 0.03,
    "Cracking": 0.02,
    "Offline-Module": 0.02,
    "Hot-Spot": 0.02,
    "Hot-Spot-Multi": 0.02,
    "Soiling": 0.02,
    "Diode-Multi": 0.01,
}


def make_record() -> dict[str, object]:
    return {
        "inference_id": "test-inference",
        "prediction_timestamp": (
            "2026-09-28T12:00:00+00:00"
        ),
        "telemetry_model_name": (
            "gradient_boosting_tuned"
        ),
        "telemetry_model_version": "candidate-5",
        "thermal_model_name": (
            "resnet18_finetuned"
        ),
        "thermal_model_version": "1",
        "telemetry_predicted_normalized_pmpp": 0.42,
        "thermal_anomaly_class": "No-Anomaly",
        "thermal_anomaly_confidence": 0.70,
        "thermal_class_probabilities": json.dumps(
            PROBABILITIES
        ),
        "thermal_anomaly_detected": 0,
        "thermal_evidence_state": "normal",
        "embedding_indicators": json.dumps(
            {"centroid_shift": 0.1}
        ),
        "health_telemetry_evidence_level": "low",
        "health_thermal_evidence_level": "low",
        "combined_evidence_state": "normal",
        "maintenance_priority": "routine",
        "created_at": (
            "2026-09-28T12:00:01+00:00"
        ),
    }


def insert_record(
    database_path,
    record: dict[str, object],
) -> None:
    columns = list(record)
    placeholders = ", ".join("?" for _ in columns)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            f"""
            INSERT INTO inference_logs (
                {", ".join(columns)}
            )
            VALUES ({placeholders})
            """,
            tuple(record[column] for column in columns),
        )
        connection.commit()


def test_empty_inference_log_is_valid(tmp_path):
    database_path = tmp_path / "inference_logs.db"

    InferenceLogStore(database_path)

    result = monitor_inference_log_quality(
        database_path
    )

    assert result["row_count"] == 0
    assert result["required_columns_valid"] is True
    assert result["quality_valid"] is True
    assert result["empty_log"] is True


def test_valid_inference_record_is_valid(tmp_path):
    database_path = tmp_path / "inference_logs.db"

    InferenceLogStore(database_path)

    insert_record(
        database_path,
        make_record(),
    )

    result = monitor_inference_log_quality(
        database_path
    )

    assert result["row_count"] == 1
    assert result["quality_valid"] is True
    assert result["invalid_probability_count"] == 0


@pytest.mark.parametrize(
    ("field", "value", "counter"),
    [
        (
            "prediction_timestamp",
            "not-a-timestamp",
            "invalid_timestamp_count",
        ),
        (
            "thermal_anomaly_confidence",
            1.5,
            "invalid_thermal_confidence_count",
        ),
        (
            "thermal_anomaly_class",
            "Unknown-Class",
            "invalid_thermal_class_count",
        ),
        (
            "thermal_model_version",
            "",
            "invalid_model_metadata_count",
        ),
        (
            "thermal_anomaly_detected",
            2,
            "invalid_anomaly_flag_count",
        ),
        (
            "embedding_indicators",
            "{not-valid-json}",
            "invalid_embedding_indicator_count",
        ),
    ],
)
def test_invalid_persisted_values_are_detected(
    tmp_path,
    field,
    value,
    counter,
):
    database_path = tmp_path / "inference_logs.db"

    InferenceLogStore(database_path)

    record = make_record()
    record[field] = value

    insert_record(
        database_path,
        record,
    )

    result = monitor_inference_log_quality(
        database_path
    )

    assert result[counter] == 1
    assert result["quality_valid"] is False


def test_invalid_probability_distribution_is_detected(
    tmp_path,
):
    database_path = tmp_path / "inference_logs.db"

    InferenceLogStore(database_path)

    record = make_record()

    invalid_probabilities = dict(PROBABILITIES)
    invalid_probabilities["Cell"] = 0.50

    record["thermal_class_probabilities"] = json.dumps(
        invalid_probabilities
    )

    insert_record(
        database_path,
        record,
    )

    result = monitor_inference_log_quality(
        database_path
    )

    assert result["invalid_probability_count"] == 1
    assert result["quality_valid"] is False


def test_non_finite_telemetry_prediction_is_detected(
    monkeypatch,
):
    record = make_record()
    record[
        "telemetry_predicted_normalized_pmpp"
    ] = np.nan

    frame = pd.DataFrame([record])

    monkeypatch.setattr(
        "src.mlops.monitoring.inference_log_quality.load_inference_logs",
        lambda _: frame.copy(),
    )

    result = monitor_inference_log_quality(
        "controlled_dataframe"
    )

    assert (
        result["non_finite_telemetry_prediction_count"]
        == 1
    )
    assert result["quality_valid"] is False


def test_missing_required_column_is_rejected(
    monkeypatch,
):
    record = make_record()
    frame = pd.DataFrame([record])

    frame = frame.drop(
        columns=["thermal_model_version"]
    )

    monkeypatch.setattr(
        "src.mlops.monitoring.inference_log_quality.load_inference_logs",
        lambda _: frame.copy(),
    )

    result = monitor_inference_log_quality(
        "controlled_dataframe"
    )

    assert result["required_columns_valid"] is False
    assert (
        "thermal_model_version"
        in result["missing_columns"]
    )
    assert result["quality_valid"] is False
