from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.mlops.retraining.baseline as baseline


def test_tecnalia_baseline_uses_validation_metrics_only(tmp_path):
    artifact = tmp_path / "model_selection_summary.json"
    artifact.write_text(
        json.dumps(
            {
                "selected_model": "gradient_boosting_tuned",
                "validation_results": {
                    "mae": 0.029539,
                    "rmse": 0.061526,
                    "r2": 0.910609,
                },
                "final_test_results": {
                    "mae": 999.0,
                    "rmse": 999.0,
                    "r2": -999.0,
                },
            }
        )
    )

    result = baseline.resolve_tecnalia_production_baseline(
        artifact_path=artifact,
    )

    assert result.dataset == "TECNALIA"
    assert result.model_family == "gradient_boosting_tuned"
    assert result.validation_metrics == {
        "validation_mae": 0.029539,
        "validation_rmse": 0.061526,
        "validation_r2": 0.910609,
    }


def test_tecnalia_baseline_requires_selected_model(tmp_path):
    artifact = tmp_path / "model_selection_summary.json"
    artifact.write_text(
        json.dumps(
            {
                "selected_model": "xgboost_tuned",
                "validation_results": {
                    "mae": 0.03,
                    "rmse": 0.06,
                    "r2": 0.91,
                },
            }
        )
    )

    with pytest.raises(
        baseline.ProductionBaselineUnavailable,
        match="does not identify",
    ):
        baseline.resolve_tecnalia_production_baseline(
            artifact_path=artifact,
        )


def test_raptormaps_without_production_fails_closed(monkeypatch):
    monkeypatch.setattr(
        baseline,
        "_find_raptormaps_production_versions",
        lambda tracking_uri: [],
    )

    with pytest.raises(
        baseline.ProductionBaselineUnavailable,
        match="No production RaptorMaps model",
    ):
        baseline.resolve_raptormaps_production_baseline()


def test_raptormaps_candidate_does_not_count_as_production(monkeypatch):
    candidate = SimpleNamespace(
        version="1",
        tags={
            "lifecycle_stage": "candidate",
            "model_family": "resnet18_finetuned",
            "best_validation_macro_f1": "0.6507379837195436",
        },
    )

    monkeypatch.setattr(
        baseline,
        "_find_raptormaps_production_versions",
        lambda tracking_uri: [],
    )

    with pytest.raises(
        baseline.ProductionBaselineUnavailable,
    ):
        baseline.resolve_raptormaps_production_baseline()


def test_raptormaps_production_baseline_reads_validation_metric(
    monkeypatch,
):
    production = SimpleNamespace(
        version="3",
        tags={
            "lifecycle_stage": "production",
            "model_family": "resnet18_finetuned",
            "best_validation_macro_f1": "0.671234",
        },
    )

    monkeypatch.setattr(
        baseline,
        "_find_raptormaps_production_versions",
        lambda tracking_uri: [
            ("SolarPV_RaptorMaps_ResNet18", production)
        ],
    )

    result = baseline.resolve_raptormaps_production_baseline()

    assert result.dataset == "RaptorMaps"
    assert result.model_family == "resnet18_finetuned"
    assert result.validation_metrics == {
        "best_validation_macro_f1": 0.671234
    }
    assert result.source == "SolarPV_RaptorMaps_ResNet18:v3"


def test_multiple_raptormaps_production_versions_are_ambiguous(
    monkeypatch,
):
    production_1 = SimpleNamespace(
        version="2",
        tags={
            "lifecycle_stage": "production",
            "model_family": "resnet18_finetuned",
            "best_validation_macro_f1": "0.67",
        },
    )
    production_2 = SimpleNamespace(
        version="4",
        tags={
            "lifecycle_stage": "production",
            "model_family": "efficientnet_b0_finetuned",
            "best_validation_macro_f1": "0.68",
        },
    )

    monkeypatch.setattr(
        baseline,
        "_find_raptormaps_production_versions",
        lambda tracking_uri: [
            ("SolarPV_RaptorMaps_ResNet18", production_1),
            ("SolarPV_RaptorMaps_EfficientNetB0", production_2),
        ],
    )

    with pytest.raises(
        baseline.ProductionBaselineUnavailable,
        match="ambiguous",
    ):
        baseline.resolve_raptormaps_production_baseline()
