from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.mlops.retraining.acceptance_contract import AcceptanceDecision
from src.mlops.retraining.config import RetrainingConfig
from src.mlops.retraining.data_validation import (
    validate_raptormaps_retraining_data,
    validate_tecnalia_retraining_data,
)
from src.mlops.retraining.evaluation import (
    CandidateEvaluation,
    evaluate_raptormaps_candidate,
    evaluate_tecnalia_candidate,
)
from src.mlops.retraining.promotion_gate import (
    PromotionGateResult,
    PromotionRequest,
    evaluate_promotion_gate,
)
from src.mlops.retraining.registration import (
    CandidateRegistrationResult,
    register_raptormaps_candidate,
)
from src.mlops.retraining.training import (
    ClassificationTrainingResult,
    RegressionTrainingResult,
    train_raptormaps_candidate,
    train_tecnalia_candidate,
)
from src.mlops.retraining.baseline import (
    ProductionBaselineUnavailable,
    resolve_production_baseline,
)


@dataclass(frozen=True)
class RetrainingPipelineResult:
    dataset: str
    model_family: str
    validation: Mapping[str, Any]
    training_result: (
        RegressionTrainingResult | ClassificationTrainingResult
    )
    evaluation: CandidateEvaluation
    registration: CandidateRegistrationResult | None
    promotion_gate: PromotionGateResult | None


def _require_production_metrics(
    production_validation_metrics: Mapping[str, float] | None,
) -> Mapping[str, float]:
    if production_validation_metrics is None:
        raise ValueError(
            "Production validation metrics are required for "
            "candidate acceptance."
        )

    return production_validation_metrics


def run_tecnalia_retraining_pipeline(
    *,
    config: RetrainingConfig,
    production_validation_metrics: Mapping[str, float] | None = None,
) -> RetrainingPipelineResult:
    if production_validation_metrics is None:
        baseline = resolve_production_baseline(
            dataset="TECNALIA",
            tracking_uri=config.tracking_uri,
        )
        production_validation_metrics = baseline.validation_metrics
    validation = validate_tecnalia_retraining_data(
        raw_root=config.tecnalia.raw_dir,
        split_manifest_path=config.tecnalia.split_manifest,
    )

    training_result = train_tecnalia_candidate()

    evaluation = evaluate_tecnalia_candidate(
        training_result,
        _require_production_metrics(production_validation_metrics),
    )

    return RetrainingPipelineResult(
        dataset="TECNALIA",
        model_family=training_result.model_family,
        validation=validation,
        training_result=training_result,
        evaluation=evaluation,
        registration=None,
        promotion_gate=None,
    )


def run_raptormaps_retraining_pipeline(
    *,
    config: RetrainingConfig,
    model_family: str,
    checkpoint_path: Path,
    production_validation_metrics: Mapping[str, float] | None = None,
) -> RetrainingPipelineResult:
    if production_validation_metrics is None:
        baseline = resolve_production_baseline(
            dataset="RaptorMaps",
            tracking_uri=config.tracking_uri,
        )
        production_validation_metrics = baseline.validation_metrics

    validation = validate_raptormaps_retraining_data(
        raw_root=config.raptormaps.raw_dir,
        split_manifest_path=config.raptormaps.split_manifest,
    )

    training_result = train_raptormaps_candidate(
        model_family=model_family,
        checkpoint_path=checkpoint_path,
        seed=config.seed,
    )

    evaluation = evaluate_raptormaps_candidate(
        training_result,
        _require_production_metrics(production_validation_metrics),
    )

    registration = None
    promotion_gate = None

    if (
        config.register_candidate
        and evaluation.acceptance.decision
        == AcceptanceDecision.ACCEPT
    ):
        registration = register_raptormaps_candidate(
            result=training_result,
            evaluation=evaluation,
            tracking_uri=config.tracking_uri,
        )

        promotion_gate = evaluate_promotion_gate(
            PromotionRequest(
                registered_model_name=registration.registered_model_name,
                candidate_version=registration.version,
                candidate_metadata={
                    "dataset": "RaptorMaps",
                    "dataset_version": "project_locked",
                    "modality": "thermal",
                    "task": "classification",
                    "split_strategy": "frozen_raptormaps_manifest",
                    "model_family": registration.model_family,
                },
                acceptance_decision=evaluation.acceptance.decision,
                current_lifecycle_stage="candidate",
            )
        )

    return RetrainingPipelineResult(
        dataset="RaptorMaps",
        model_family=training_result.model_family,
        validation=validation,
        training_result=training_result,
        evaluation=evaluation,
        registration=registration,
        promotion_gate=promotion_gate,
    )