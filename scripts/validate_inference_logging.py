from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops.inference_logging import InferenceLogStore
from src.models.fusion.contracts import (
    FusionInput,
    ModelProvenance,
    RAPTORMAPS_CLASS_NAMES,
    TelemetryPrediction,
    ThermalPrediction,
)
from src.models.fusion.decision_fusion import fuse_decision_outputs
from src.models.fusion.health_assessment import (
    ThermalAnomalyEvidence,
    TelemetryPerformanceEvidence,
    build_health_assessment,
    classify_telemetry_evidence,
    classify_thermal_evidence,
)


def build_telemetry_prediction() -> TelemetryPrediction:
    return TelemetryPrediction(
        predicted_normalized_pmpp=0.82,
        provenance=ModelProvenance(
            model_name="tecnalia_gradient_boosting",
            model_version="candidate-v1",
            checkpoint_path=(
                "models/tecnalia/final_model.joblib"
            ),
            validation_metric_name="mae",
            validation_metric_value=0.029539,
        ),
    )


def build_thermal_prediction() -> ThermalPrediction:
    probabilities = {
        class_name: 0.0
        for class_name in RAPTORMAPS_CLASS_NAMES
    }

    probabilities["Hot-Spot"] = 0.80
    probabilities["No-Anomaly"] = 0.20

    return ThermalPrediction(
        anomaly_class="Hot-Spot",
        anomaly_confidence=0.80,
        class_probabilities=probabilities,
        provenance=ModelProvenance(
            model_name="SolarPV_RaptorMaps_ResNet18",
            model_version="1",
            checkpoint_path=(
                "models/raptormaps/resnet18_finetune/"
                "best_model.pt"
            ),
            checkpoint_epoch=14,
            validation_metric_name="macro_f1",
            validation_metric_value=0.650738,
        ),
    )


def create_health_assessment():
    telemetry = TelemetryPerformanceEvidence(
        actual_normalized_pmpp=0.72,
        predicted_normalized_pmpp=0.82,
        performance_deviation=-0.10,
        evidence_level=classify_telemetry_evidence(-0.10),
    )

    thermal = ThermalAnomalyEvidence(
        anomaly_class="Hot-Spot",
        no_anomaly_probability=0.20,
        anomaly_evidence=0.80,
        evidence_level=classify_thermal_evidence(0.80),
    )

    return build_health_assessment(
        telemetry=telemetry,
        thermal=thermal,
    )


def main() -> None:
    telemetry = build_telemetry_prediction()
    thermal = build_thermal_prediction()

    fusion = fuse_decision_outputs(
        FusionInput(
            telemetry=telemetry,
            thermal=thermal,
            embedding_indicators={
                "embedding_distance": 0.42,
            },
        )
    )

    health = create_health_assessment()

    with TemporaryDirectory() as tmp_dir:
        database_path = Path(tmp_dir) / "inference_logs.db"

        store = InferenceLogStore(database_path)

        inference_id = store.log_inference(
            telemetry=telemetry,
            thermal=thermal,
            fusion=fusion,
            health=health,
            prediction_timestamp=datetime(
                2026,
                9,
                26,
                14,
                30,
                tzinfo=timezone.utc,
            ),
        )

        record = store.get_inference(inference_id)

        assert record is not None
        assert store.count() == 1

        print("Inference logging validation: PASS")
        print(f"inference_id: {inference_id}")
        print(
            "prediction_timestamp:",
            record["prediction_timestamp"],
        )
        print(
            "telemetry_model:",
            record["telemetry_model_name"],
            record["telemetry_model_version"],
        )
        print(
            "telemetry_predicted_normalized_pmpp:",
            record["telemetry_predicted_normalized_pmpp"],
        )
        print(
            "thermal_model:",
            record["thermal_model_name"],
            record["thermal_model_version"],
        )
        print(
            "thermal_anomaly_class:",
            record["thermal_anomaly_class"],
        )
        print(
            "thermal_anomaly_confidence:",
            record["thermal_anomaly_confidence"],
        )
        print(
            "thermal_anomaly_detected:",
            record["thermal_anomaly_detected"],
        )
        print(
            "thermal_evidence_state:",
            record["thermal_evidence_state"],
        )
        print(
            "combined_evidence_state:",
            record["combined_evidence_state"],
        )
        print(
            "maintenance_priority:",
            record["maintenance_priority"],
        )
        print(
            "embedding_indicators:",
            record["embedding_indicators"],
        )
        print(
            "stored_record_count:",
            store.count(),
        )


if __name__ == "__main__":
    main()