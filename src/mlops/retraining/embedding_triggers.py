from __future__ import annotations

from src.mlops.monitoring.embedding_drift import (
    monitor_embedding_drift,
)
from src.mlops.retraining.trigger_contract import (
    DEFAULT_DRIFT_THRESHOLDS,
    DriftThresholds,
)
from src.mlops.retraining.trigger_evaluator import (
    DriftSignal,
    evaluate_retraining_trigger,
    psi_signal,
)


SUPPORTED_EMBEDDING_MODELS = (
    "resnet18_finetuned",
    "efficientnet_b0_finetuned",
)


def build_raptormaps_embedding_drift_signal(
    model_name: str,
    current_embeddings,
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
) -> DriftSignal:
    """Convert RaptorMaps embedding distance PSI into a drift signal."""
    result = monitor_embedding_drift(
        model_name=model_name,
        current_embeddings=current_embeddings,
    )

    return psi_signal(
        name=f"raptormaps.embedding.psi:{model_name}",
        psi=float(result["distance_psi"]),
        thresholds=thresholds,
    )


def evaluate_raptormaps_embedding_drift(
    model_name: str,
    current_embeddings,
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
):
    """Evaluate one RaptorMaps embedding-drift signal."""
    signal = build_raptormaps_embedding_drift_signal(
        model_name=model_name,
        current_embeddings=current_embeddings,
        thresholds=thresholds,
    )

    return evaluate_retraining_trigger(
        signals=(signal,),
    )
