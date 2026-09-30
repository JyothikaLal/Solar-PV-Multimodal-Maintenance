from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.mlops.retraining.pipeline as pipeline
from src.mlops.retraining.acceptance_contract import AcceptanceDecision


def _config(*, register_candidate: bool = True):
    return SimpleNamespace(
        seed=42,
        register_candidate=register_candidate,
        tracking_uri="sqlite:///mlflow.db",
        tecnalia=SimpleNamespace(
            raw_dir=Path("data/raw/tecnalia"),
            split_manifest=Path(
                "reports/results/tecnalia/tecnalia_split_manifest.csv"
            ),
        ),
        raptormaps=SimpleNamespace(
            raw_dir=Path("data/raw/raptormaps"),
            split_manifest=Path(
                "reports/results/raptormaps/raptormaps_split_manifest.csv"
            ),
        ),
    )


def _training_result(model_family: str):
    return SimpleNamespace(model_family=model_family)


def _evaluation(decision: AcceptanceDecision):
    return SimpleNamespace(
        acceptance=SimpleNamespace(decision=decision)
    )


def test_tecnalia_pipeline_runs_validation_training_and_evaluation(
    monkeypatch,
):
    calls = []

    validation_result = {"status": "valid"}
    training_result = _training_result("gradient_boosting_tuned")
    evaluation_result = _evaluation(AcceptanceDecision.REJECT)

    def fake_validate(*, raw_root, split_manifest_path):
        calls.append(
            (
                "validate",
                raw_root,
                split_manifest_path,
            )
        )
        return validation_result

    def fake_train():
        calls.append(("train",))
        return training_result

    def fake_evaluate(candidate, production_metrics):
        calls.append(
            (
                "evaluate",
                candidate,
                production_metrics,
            )
        )
        return evaluation_result

    monkeypatch.setattr(
        pipeline,
        "validate_tecnalia_retraining_data",
        fake_validate,
    )
    monkeypatch.setattr(
        pipeline,
        "train_tecnalia_candidate",
        fake_train,
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_tecnalia_candidate",
        fake_evaluate,
    )

    production_metrics = {
        "validation_mae": 0.029539,
        "validation_rmse": 0.061526,
        "validation_r2": 0.910609,
    }

    result = pipeline.run_tecnalia_retraining_pipeline(
        config=_config(),
        production_validation_metrics=production_metrics,
    )

    assert result.dataset == "TECNALIA"
    assert result.model_family == "gradient_boosting_tuned"
    assert result.validation is validation_result
    assert result.training_result is training_result
    assert result.evaluation is evaluation_result
    assert result.registration is None
    assert result.promotion_gate is None

    assert calls == [
        (
            "validate",
            Path("data/raw/tecnalia"),
            Path(
                "reports/results/tecnalia/tecnalia_split_manifest.csv"
            ),
        ),
        ("train",),
        (
            "evaluate",
            training_result,
            production_metrics,
        ),
    ]


def test_raptormaps_rejected_candidate_is_not_registered_or_promoted(
    monkeypatch,
):
    calls = []

    validation_result = {"status": "valid"}
    training_result = _training_result("resnet18_finetuned")
    evaluation_result = _evaluation(AcceptanceDecision.REJECT)

    def fake_validate(*, raw_root, split_manifest_path):
        calls.append(
            (
                "validate",
                raw_root,
                split_manifest_path,
            )
        )
        return validation_result

    def fake_train(*, model_family, checkpoint_path, seed):
        calls.append(
            (
                "train",
                model_family,
                checkpoint_path,
                seed,
            )
        )
        return training_result

    def fake_evaluate(candidate, production_metrics):
        calls.append(
            (
                "evaluate",
                candidate,
                production_metrics,
            )
        )
        return evaluation_result

    def fail_register(**kwargs):
        raise AssertionError(
            "Rejected candidates must not be registered."
        )

    def fail_promote(*args, **kwargs):
        raise AssertionError(
            "Rejected candidates must not reach the promotion gate."
        )

    monkeypatch.setattr(
        pipeline,
        "validate_raptormaps_retraining_data",
        fake_validate,
    )
    monkeypatch.setattr(
        pipeline,
        "train_raptormaps_candidate",
        fake_train,
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_raptormaps_candidate",
        fake_evaluate,
    )
    monkeypatch.setattr(
        pipeline,
        "register_raptormaps_candidate",
        fail_register,
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_promotion_gate",
        fail_promote,
    )

    production_metrics = {
        "validation_macro_f1": 0.60,
        "validation_balanced_accuracy": 0.65,
    }
    checkpoint_path = Path(
        "models/raptormaps/resnet18_finetune/best_model.pt"
    )

    result = pipeline.run_raptormaps_retraining_pipeline(
        config=_config(register_candidate=True),
        model_family="resnet18_finetuned",
        checkpoint_path=checkpoint_path,
        production_validation_metrics=production_metrics,
    )

    assert result.dataset == "RaptorMaps"
    assert result.model_family == "resnet18_finetuned"
    assert result.validation is validation_result
    assert result.training_result is training_result
    assert result.evaluation is evaluation_result
    assert result.registration is None
    assert result.promotion_gate is None

    assert calls == [
        (
            "validate",
            Path("data/raw/raptormaps"),
            Path(
                "reports/results/raptormaps/raptormaps_split_manifest.csv"
            ),
        ),
        (
            "train",
            "resnet18_finetuned",
            checkpoint_path,
            42,
        ),
        (
            "evaluate",
            training_result,
            production_metrics,
        ),
    ]


def test_raptormaps_accepted_candidate_registers_then_evaluates_promotion_gate(
    monkeypatch,
):
    calls = []

    validation_result = {"status": "valid"}
    training_result = _training_result("resnet18_finetuned")
    evaluation_result = _evaluation(AcceptanceDecision.ACCEPT)

    registration_result = SimpleNamespace(
        registered_model_name="SolarPV_RaptorMaps_ResNet18",
        model_family="resnet18_finetuned",
        version="2",
    )

    promotion_result = SimpleNamespace(
        decision="eligible",
        reasons=("Explicit registry promotion is still required.",),
    )

    def fake_validate(*, raw_root, split_manifest_path):
        calls.append(("validate", raw_root, split_manifest_path))
        return validation_result

    def fake_train(*, model_family, checkpoint_path, seed):
        calls.append(
            (
                "train",
                model_family,
                checkpoint_path,
                seed,
            )
        )
        return training_result

    def fake_evaluate(candidate, production_metrics):
        calls.append(
            (
                "evaluate",
                candidate,
                production_metrics,
            )
        )
        return evaluation_result

    def fake_register(*, result, evaluation, tracking_uri):
        calls.append(
            (
                "register",
                result,
                evaluation,
                tracking_uri,
            )
        )
        return registration_result

    def fake_promotion_gate(request):
        calls.append(("promotion_gate", request))
        return promotion_result

    monkeypatch.setattr(
        pipeline,
        "validate_raptormaps_retraining_data",
        fake_validate,
    )
    monkeypatch.setattr(
        pipeline,
        "train_raptormaps_candidate",
        fake_train,
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_raptormaps_candidate",
        fake_evaluate,
    )
    monkeypatch.setattr(
        pipeline,
        "register_raptormaps_candidate",
        fake_register,
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_promotion_gate",
        fake_promotion_gate,
    )

    production_metrics = {
        "validation_macro_f1": 0.60,
        "validation_balanced_accuracy": 0.65,
    }
    checkpoint_path = Path(
        "models/raptormaps/resnet18_finetune/best_model.pt"
    )

    result = pipeline.run_raptormaps_retraining_pipeline(
        config=_config(register_candidate=True),
        model_family="resnet18_finetuned",
        checkpoint_path=checkpoint_path,
        production_validation_metrics=production_metrics,
    )

    assert result.registration is registration_result
    assert result.promotion_gate is promotion_result

    assert [call[0] for call in calls] == [
        "validate",
        "train",
        "evaluate",
        "register",
        "promotion_gate",
    ]

    promotion_request = calls[-1][1]

    assert promotion_request.registered_model_name == (
        "SolarPV_RaptorMaps_ResNet18"
    )
    assert promotion_request.candidate_version == "2"
    assert promotion_request.acceptance_decision == (
        AcceptanceDecision.ACCEPT
    )
    assert promotion_request.current_lifecycle_stage == "candidate"

    assert promotion_request.candidate_metadata == {
        "dataset": "RaptorMaps",
        "dataset_version": "project_locked",
        "modality": "thermal",
        "task": "classification",
        "split_strategy": "frozen_raptormaps_manifest",
        "model_family": "resnet18_finetuned",
    }


def test_raptormaps_accepted_candidate_is_not_registered_when_disabled(
    monkeypatch,
):
    training_result = _training_result("efficientnet_b0_finetuned")
    evaluation_result = _evaluation(AcceptanceDecision.ACCEPT)

    monkeypatch.setattr(
        pipeline,
        "validate_raptormaps_retraining_data",
        lambda **kwargs: {"status": "valid"},
    )
    monkeypatch.setattr(
        pipeline,
        "train_raptormaps_candidate",
        lambda **kwargs: training_result,
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_raptormaps_candidate",
        lambda candidate, production_metrics: evaluation_result,
    )

    def fail_register(**kwargs):
        raise AssertionError(
            "Registration must remain disabled when "
            "config.register_candidate is false."
        )

    monkeypatch.setattr(
        pipeline,
        "register_raptormaps_candidate",
        fail_register,
    )

    result = pipeline.run_raptormaps_retraining_pipeline(
        config=_config(register_candidate=False),
        model_family="efficientnet_b0_finetuned",
        checkpoint_path=Path(
            "models/raptormaps/efficientnet_b0_finetune/best_model.pt"
        ),
        production_validation_metrics={
            "validation_macro_f1": 0.55,
            "validation_balanced_accuracy": 0.60,
        },
    )

    assert result.registration is None
    assert result.promotion_gate is None


def test_pipeline_requires_production_metrics():
    with pytest.raises(
        ValueError,
        match="Production validation metrics are required",
    ):
        pipeline._require_production_metrics(None)

def test_tecnalia_pipeline_resolves_baseline_before_training(
    monkeypatch,
):
    calls = []

    baseline_result = SimpleNamespace(
        validation_metrics={
            "validation_mae": 0.029539,
            "validation_rmse": 0.061526,
            "validation_r2": 0.910609,
        }
    )

    training_result = _training_result("gradient_boosting_tuned")
    evaluation_result = _evaluation(AcceptanceDecision.REJECT)

    def fake_resolve(*, dataset, tracking_uri):
        calls.append(("resolve", dataset, tracking_uri))
        return baseline_result

    def fake_validate(*, raw_root, split_manifest_path):
        calls.append(("validate",))
        return {"status": "valid"}

    def fake_train():
        calls.append(("train",))
        return training_result

    def fake_evaluate(candidate, production_metrics):
        calls.append(("evaluate", production_metrics))
        return evaluation_result

    monkeypatch.setattr(
        pipeline,
        "resolve_production_baseline",
        fake_resolve,
    )
    monkeypatch.setattr(
        pipeline,
        "validate_tecnalia_retraining_data",
        fake_validate,
    )
    monkeypatch.setattr(
        pipeline,
        "train_tecnalia_candidate",
        fake_train,
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_tecnalia_candidate",
        fake_evaluate,
    )

    result = pipeline.run_tecnalia_retraining_pipeline(
        config=_config(),
    )

    assert result.evaluation is evaluation_result

    assert calls == [
        ("resolve", "TECNALIA", "sqlite:///mlflow.db"),
        ("validate",),
        ("train",),
        (
            "evaluate",
            {
                "validation_mae": 0.029539,
                "validation_rmse": 0.061526,
                "validation_r2": 0.910609,
            },
        ),
    ]


def test_raptormaps_missing_baseline_stops_before_training(
    monkeypatch,
):
    def fail_train(**kwargs):
        raise AssertionError(
            "Training must not start without a production baseline."
        )

    monkeypatch.setattr(
        pipeline,
        "resolve_production_baseline",
        lambda **kwargs: (_ for _ in ()).throw(
            pipeline.ProductionBaselineUnavailable(
                "No production RaptorMaps model is registered."
            )
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "train_raptormaps_candidate",
        fail_train,
    )

    with pytest.raises(
        pipeline.ProductionBaselineUnavailable,
        match="No production RaptorMaps model",
    ):
        pipeline.run_raptormaps_retraining_pipeline(
            config=_config(),
            model_family="resnet18_finetuned",
            checkpoint_path=Path(
                "models/raptormaps/resnet18_finetune/best_model.pt"
            ),
        )