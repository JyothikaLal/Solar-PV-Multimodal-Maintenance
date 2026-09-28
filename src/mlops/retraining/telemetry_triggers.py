from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.mlops.monitoring.tecnalia_drift import (
    monitor_tecnalia_module_distribution,
    monitor_tecnalia_numeric_features,
)
from src.mlops.retraining.trigger_contract import (
    DEFAULT_DRIFT_THRESHOLDS,
    DriftSeverity,
    DriftThresholds,
)
from src.mlops.retraining.trigger_evaluator import (
    DriftSignal,
    evaluate_retraining_trigger,
    js_signal,
    psi_signal,
)


TELEMETRY_FEATURE_NAMES = (
    "Front GPOA (W/m²)",
    "GHI (W/m²)",
    "Temp. Mod (°C)",
    "Amb. Temp. (°C)",
    "Wind Speed (m/s)",
)


def build_tecnalia_feature_drift_signals(
    drift_results: pd.DataFrame,
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
) -> tuple[DriftSignal, ...]:
    """Convert TECNALIA feature PSI results into drift signals.

    The existing TECNALIA monitor remains responsible for calculating
    PSI and Wasserstein distance. This adapter only applies the
    retraining policy to PSI.
    """
    required_columns = {"feature_name", "psi"}

    missing = required_columns - set(drift_results.columns)

    if missing:
        raise ValueError(
            "TECNALIA feature drift results are missing columns: "
            f"{sorted(missing)}"
        )

    signals: list[DriftSignal] = []

    for feature in TELEMETRY_FEATURE_NAMES:
        matches = drift_results.loc[
            drift_results["feature_name"] == feature
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one drift result for feature "
                f"{feature!r}, found {len(matches)}."
            )

        psi_value = float(matches.iloc[0]["psi"])

        signals.append(
            psi_signal(
                name=f"tecnalia.feature.psi:{feature}",
                psi=psi_value,
                thresholds=thresholds,
            )
        )

    return tuple(signals)


def build_tecnalia_module_drift_signal(
    reference_modules: pd.Series,
    current_modules: pd.Series,
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
) -> DriftSignal:
    """Convert TECNALIA module-distribution JS divergence into a signal."""
    result = monitor_tecnalia_module_distribution(
        reference_modules=reference_modules,
        current_modules=current_modules,
    )

    return js_signal(
        name="tecnalia.module_distribution.js",
        divergence=float(result["js_divergence"]),
        thresholds=thresholds,
    )


def evaluate_tecnalia_drift(
    current: pd.DataFrame,
    reference_modules: pd.Series,
    current_modules: pd.Series,
    thresholds: DriftThresholds = DEFAULT_DRIFT_THRESHOLDS,
):
    """Evaluate the TECNALIA drift signals using the generic policy."""
    feature_results = monitor_tecnalia_numeric_features(
        current=current,
    )

    feature_signals = build_tecnalia_feature_drift_signals(
        drift_results=feature_results,
        thresholds=thresholds,
    )

    module_signal = build_tecnalia_module_drift_signal(
        reference_modules=reference_modules,
        current_modules=current_modules,
        thresholds=thresholds,
    )

    return evaluate_retraining_trigger(
        signals=(*feature_signals, module_signal),
    )
