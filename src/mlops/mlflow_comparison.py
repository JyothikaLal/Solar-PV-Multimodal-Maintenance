"""Utilities for comparing tracked Solar PV MLflow runs."""

from __future__ import annotations

from typing import Any

import mlflow

from src.mlops.mlflow_tracking import (
    EXPERIMENT_RAPTORMAPS_TRANSFER,
    EXPERIMENT_TECNALIA_REGRESSION,
    configure_tracking,
)


COMPARISON_EXPERIMENTS = (
    EXPERIMENT_TECNALIA_REGRESSION,
    EXPERIMENT_RAPTORMAPS_TRANSFER,
)


REGRESSION_METRICS = (
    "validation_mae",
    "validation_rmse",
    "validation_r2",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_mape_percent",
)


CLASSIFICATION_METRICS = (
    "best_validation_macro_f1",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "test_roc_auc_ovr_macro",
    "test_pr_auc_macro",
)


def get_comparison_runs(
    experiment_names: tuple[str, ...] = COMPARISON_EXPERIMENTS,
) -> list[dict[str, Any]]:
    """Return normalized metadata and metrics for tracked runs."""
    configure_tracking()

    client = mlflow.MlflowClient()
    rows: list[dict[str, Any]] = []

    for experiment_name in experiment_names:
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            continue

        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time ASC"],
        )

        for run in runs:
            tags = run.data.tags
            task = tags.get("task")

            row: dict[str, Any] = {
                "experiment": experiment_name,
                "run_id": run.info.run_id,
                "run_name": tags.get("mlflow.runName"),
                "status": run.info.status,
                "dataset": tags.get("dataset"),
                "modality": tags.get("modality"),
                "task": task,
                "model_family": tags.get("model_family"),
                "stage": tags.get("stage"),
                "run_type": tags.get("run_type"),
            }

            metric_names = (
                REGRESSION_METRICS
                if task == "regression"
                else CLASSIFICATION_METRICS
                if task == "classification"
                else ()
            )

            for metric_name in metric_names:
                row[metric_name] = run.data.metrics.get(metric_name)

            rows.append(row)

    return rows


def group_comparison_runs(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Group runs by dataset, modality, and task."""
    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = {}

    for row in rows:
        key = (
            str(row.get("dataset")),
            str(row.get("modality")),
            str(row.get("task")),
        )
        groups.setdefault(key, []).append(row)

    return groups
