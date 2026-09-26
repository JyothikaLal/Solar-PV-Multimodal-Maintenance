from pathlib import Path

import mlflow
import pytest

from src.mlops.mlflow_tracking import (
    build_standard_tags,
    configure_tracking,
    get_or_create_experiment,
    log_artifact,
    log_dataset_metadata,
    log_metrics,
    log_params,
    start_run,
)


@pytest.fixture()
def isolated_tracking_db(tmp_path: Path):
    """Use a fresh MLflow SQLite database for each test."""
    database_path = tmp_path / "mlflow_test.db"
    uri = f"sqlite:///{database_path}"

    configure_tracking(uri)

    yield uri

    mlflow.end_run()


def test_configure_tracking_uses_explicit_uri(isolated_tracking_db):
    assert mlflow.get_tracking_uri() == isolated_tracking_db


def test_get_or_create_experiment_is_idempotent(isolated_tracking_db):
    name = "SolarPV_Test_Tracking_Experiment"

    first_id = get_or_create_experiment(name)
    second_id = get_or_create_experiment(name)

    assert first_id == second_id

    experiment = mlflow.get_experiment(first_id)

    assert experiment.name == name
    assert experiment.lifecycle_stage == "active"


def test_tracking_utility_logs_run_data(
    isolated_tracking_db,
    tmp_path: Path,
):
    experiment_name = "SolarPV_Test_Run_Experiment"

    artifact = tmp_path / "metrics.txt"
    artifact.write_text("test artifact\n", encoding="utf-8")

    with start_run(
        experiment_name=experiment_name,
        run_name="utility_test",
        tags={
            "project": "SolarPV",
            "stage": "test",
        },
    ) as run:
        log_params(
            {
                "learning_rate": 0.001,
                "features": ["a", "b"],
            }
        )

        log_metrics(
            {
                "accuracy": 0.95,
                "macro_f1": 0.91,
            }
        )

        log_dataset_metadata(
            dataset_name="test_dataset",
            dataset_version="v1",
            split_strategy="chronological",
            sample_counts={
                "train": 10,
                "validation": 5,
                "test": 5,
            },
        )

        log_artifact(artifact)

        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

    client = mlflow.MlflowClient()
    stored_run = client.get_run(run_id)

    assert stored_run.data.params["learning_rate"] == "0.001"
    assert stored_run.data.params["features"] == '["a","b"]'

    assert stored_run.data.metrics["accuracy"] == pytest.approx(0.95)
    assert stored_run.data.metrics["macro_f1"] == pytest.approx(0.91)

    assert stored_run.data.tags["dataset_name"] == "test_dataset"
    assert stored_run.data.tags["dataset_version"] == "v1"
    assert stored_run.data.tags["split_strategy"] == "chronological"
    assert stored_run.data.tags["sample_counts"] == (
        '{"test":5,"train":10,"validation":5}'
    )

    artifacts = client.list_artifacts(run_id)

    assert any(item.path == "metrics.txt" for item in artifacts)
    assert client.get_experiment(experiment_id).lifecycle_stage == "active"


def test_build_standard_tags():
    tags = build_standard_tags(
        dataset="TECNALIA",
        modality="telemetry",
        task="regression",
        model_family="gradient_boosting",
        stage="final",
        run_type="retrospective",
        split_strategy="frozen_manifest",
    )

    assert tags == {
        "project": "SolarPV_Multimodal_Maintenance",
        "dataset": "TECNALIA",
        "modality": "telemetry",
        "task": "regression",
        "model_family": "gradient_boosting",
        "stage": "final",
        "run_type": "retrospective",
        "split_strategy": "frozen_manifest",
    }
