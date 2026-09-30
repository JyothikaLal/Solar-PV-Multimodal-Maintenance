from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from src.mlops.retraining.acceptance_contract import AcceptanceDecision
from src.mlops.retraining.evaluation import CandidateEvaluation
from src.mlops.retraining.registration import (
    _load_checkpoint_metadata,
    register_raptormaps_candidate,
)
from src.mlops.retraining.training import ClassificationTrainingResult


def _accepted_evaluation() -> CandidateEvaluation:
    acceptance = SimpleNamespace(
        decision=AcceptanceDecision.ACCEPT
    )

    return CandidateEvaluation(
        model_family="resnet18_finetuned",
        acceptance=acceptance,
        candidate_validation_metrics={
            "best_validation_macro_f1": 0.70,
            "validation_balanced_accuracy": 0.65,
        },
        production_validation_metrics={
            "best_validation_macro_f1": 0.65,
        },
    )


def _training_result(tmp_path: Path) -> ClassificationTrainingResult:
    checkpoint_path = tmp_path / "best_model.pt"

    torch.save(
        {
            "epoch": 7,
            "best_validation_macro_f1": 0.70,
            "model_state_dict": {},
        },
        checkpoint_path,
    )

    return ClassificationTrainingResult(
        model_family="resnet18_finetuned",
        validation_metrics={
            "macro_f1": 0.70,
            "balanced_accuracy": 0.65,
        },
        test_metrics={
            "accuracy": 0.78,
            "balanced_accuracy": 0.62,
            "macro_precision": 0.67,
            "macro_recall": 0.61,
            "macro_f1": 0.64,
            "weighted_f1": 0.77,
            "roc_auc_ovr_macro": 0.94,
            "pr_auc_macro": 0.66,
        },
        best_validation_macro_f1=0.70,
        checkpoint_path=checkpoint_path,
        history=[],
        class_names=[
            "Cell",
            "Cell-Multi",
            "Cracking",
            "Diode",
            "Diode-Multi",
            "Hot-Spot",
            "Hot-Spot-Multi",
            "No-Anomaly",
            "Offline-Module",
            "Shadowing",
            "Soiling",
            "Vegetation",
        ],
    )


def test_checkpoint_metadata_requires_required_fields(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "epoch": 3,
            "model_state_dict": {},
        },
        checkpoint_path,
    )

    try:
        _load_checkpoint_metadata(checkpoint_path)
    except ValueError as exc:
        assert "best_validation_macro_f1" in str(exc)
    else:
        raise AssertionError("Expected missing checkpoint field error.")


def test_rejected_candidate_is_not_registered(
    tmp_path,
    monkeypatch,
):
    result = _training_result(tmp_path)

    rejected = CandidateEvaluation(
        model_family="resnet18_finetuned",
        acceptance=SimpleNamespace(
            decision=AcceptanceDecision.REJECT
    ),
        candidate_validation_metrics={},
        production_validation_metrics={},
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("MLflow registration should not be called.")

    monkeypatch.setattr(
        "src.mlops.retraining.registration.mlflow",
        SimpleNamespace(
            set_tracking_uri=fail_if_called,
            start_run=fail_if_called,
        ),
    )

    try:
        register_raptormaps_candidate(
            result=result,
            evaluation=rejected,
        )
    except ValueError as exc:
        assert "ACCEPT" in str(exc)
    else:
        raise AssertionError("Expected rejected candidate to be blocked.")


def test_accepted_candidate_registers_as_candidate(
    tmp_path,
    monkeypatch,
):
    result = _training_result(tmp_path)
    evaluation = _accepted_evaluation()

    logged = {}

    class FakeRun:
        info = SimpleNamespace(run_id="test-run-123")

    class FakeRunContext:
        def __enter__(self):
            return FakeRun()

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: logged.update(
            tracking_uri=uri
        ),
        start_run=lambda **kwargs: FakeRunContext(),
        log_params=lambda params: logged.update(params=params),
        log_metric=lambda name, value: logged.setdefault(
            "metrics", {}
        ).update({name: value}),
        pyfunc=SimpleNamespace(
            log_model=lambda **kwargs: logged.update(
                model_logging=kwargs
            )
        ),
    )

    monkeypatch.setattr(
        "src.mlops.retraining.registration.mlflow",
        fake_mlflow,
    )

    monkeypatch.setattr(
        "src.mlops.retraining.registration.RaptorMapsMLflowModel",
        lambda **kwargs: kwargs,
    )

    monkeypatch.setattr(
        "src.mlops.retraining.registration.build_model_metadata",
        lambda **kwargs: logged.update(
            metadata=kwargs
        ) or {
            "lifecycle_stage": "candidate",
            "dataset": "RaptorMaps",
        },
    )

    monkeypatch.setattr(
        "src.mlops.retraining.registration.register_candidate_model",
        lambda **kwargs: logged.update(
            registration=kwargs
        ) or SimpleNamespace(version=5),
    )

    registered = register_raptormaps_candidate(
        result=result,
        evaluation=evaluation,
        tracking_uri="sqlite:///test.db",
    )

    assert registered.registered_model_name == "SolarPV_RaptorMaps_ResNet18"
    assert registered.model_family == "resnet18"
    assert registered.run_id == "test-run-123"
    assert registered.version == "5"

    assert (
        logged["registration"]["metadata"]["lifecycle_stage"]
        == "candidate"
    )

    assert (
        logged["metadata"]["test_metrics"]
        == result.test_metrics
    )
