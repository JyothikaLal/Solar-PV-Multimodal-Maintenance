from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.fusion.contracts import (
    FusionOutput,
    TelemetryPrediction,
    ThermalPrediction,
)
from src.models.fusion.health_assessment import HealthAssessment


DEFAULT_INFERENCE_LOG_DB = "inference_logs.db"


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inference_logs (
    inference_id TEXT PRIMARY KEY,
    prediction_timestamp TEXT NOT NULL,

    telemetry_model_name TEXT NOT NULL,
    telemetry_model_version TEXT NOT NULL,

    thermal_model_name TEXT NOT NULL,
    thermal_model_version TEXT NOT NULL,

    telemetry_predicted_normalized_pmpp REAL NOT NULL,

    thermal_anomaly_class TEXT NOT NULL,
    thermal_anomaly_confidence REAL NOT NULL,
    thermal_class_probabilities TEXT NOT NULL,

    thermal_anomaly_detected INTEGER NOT NULL,
    thermal_evidence_state TEXT NOT NULL,

    embedding_indicators TEXT,

    health_telemetry_evidence_level TEXT,
    health_thermal_evidence_level TEXT,
    combined_evidence_state TEXT NOT NULL,
    maintenance_priority TEXT NOT NULL,

    created_at TEXT NOT NULL
);
"""


class InferenceLogStore:
    """Persistent SQLite store for individual inference events."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_INFERENCE_LOG_DB,
    ) -> None:
        self.database_path = Path(database_path)

        if self.database_path.parent != Path("."):
            self.database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(CREATE_TABLE_SQL)
            connection.commit()

    def log_inference(
        self,
        *,
        telemetry: TelemetryPrediction,
        thermal: ThermalPrediction,
        fusion: FusionOutput,
        health: HealthAssessment,
        prediction_timestamp: datetime | None = None,
        inference_id: str | None = None,
    ) -> str:
        """Persist one validated multimodal inference event."""

        timestamp = (
            prediction_timestamp
            or datetime.now(timezone.utc)
        )

        if timestamp.tzinfo is None:
            raise ValueError(
                "prediction_timestamp must be timezone-aware."
            )

        timestamp = timestamp.astimezone(timezone.utc)

        if fusion.telemetry != telemetry:
            raise ValueError(
                "Fusion telemetry output does not match "
                "the logged telemetry prediction."
            )

        if fusion.thermal != thermal:
            raise ValueError(
                "Fusion thermal output does not match "
                "the logged thermal prediction."
            )

        record_id = inference_id or str(uuid.uuid4())

        thermal_evidence = (
            1.0 - thermal.class_probabilities["No-Anomaly"]
        )

        health_telemetry_level = (
            health.telemetry.evidence_level
            if health.telemetry is not None
            else None
        )

        health_thermal_level = (
            health.thermal.evidence_level
            if health.thermal is not None
            else None
        )

        embedding_indicators = fusion.embedding_indicators

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inference_logs (
                    inference_id,
                    prediction_timestamp,
                    telemetry_model_name,
                    telemetry_model_version,
                    thermal_model_name,
                    thermal_model_version,
                    telemetry_predicted_normalized_pmpp,
                    thermal_anomaly_class,
                    thermal_anomaly_confidence,
                    thermal_class_probabilities,
                    thermal_anomaly_detected,
                    thermal_evidence_state,
                    embedding_indicators,
                    health_telemetry_evidence_level,
                    health_thermal_evidence_level,
                    combined_evidence_state,
                    maintenance_priority,
                    created_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    record_id,
                    timestamp.isoformat(),
                    telemetry.provenance.model_name,
                    telemetry.provenance.model_version,
                    thermal.provenance.model_name,
                    thermal.provenance.model_version,
                    telemetry.predicted_normalized_pmpp,
                    thermal.anomaly_class,
                    thermal.anomaly_confidence,
                    json.dumps(
                        thermal.class_probabilities,
                        sort_keys=True,
                    ),
                    int(fusion.thermal_anomaly_detected),
                    fusion.evidence_state,
                    (
                        json.dumps(
                            embedding_indicators,
                            sort_keys=True,
                        )
                        if embedding_indicators is not None
                        else None
                    ),
                    health_telemetry_level,
                    health_thermal_level,
                    health.combined_evidence_state,
                    health.maintenance_priority,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()

        return record_id

    def get_inference(
        self,
        inference_id: str,
    ) -> dict[str, Any] | None:
        """Return one inference record as a dictionary."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM inference_logs
                WHERE inference_id = ?
                """,
                (inference_id,),
            ).fetchone()

        if row is None:
            return None

        record = dict(row)

        record["thermal_class_probabilities"] = json.loads(
            record["thermal_class_probabilities"]
        )

        if record["embedding_indicators"] is not None:
            record["embedding_indicators"] = json.loads(
                record["embedding_indicators"]
            )

        record["thermal_anomaly_detected"] = bool(
            record["thermal_anomaly_detected"]
        )

        return record

    def count(self) -> int:
        """Return the number of stored inference events."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM inference_logs"
            ).fetchone()

        return int(row["count"])