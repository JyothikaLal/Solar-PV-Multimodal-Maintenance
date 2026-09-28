from __future__ import annotations

import pandas as pd
import pytest

from src.mlops.retraining.telemetry_triggers import (
    TELEMETRY_FEATURE_NAMES,
    build_tecnalia_feature_drift_signals,
    build_tecnalia_module_drift_signal,
)


def _feature_results(psi_values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_name": list(psi_values),
            "psi": list(psi_values.values()),
            "wasserstein_distance": [0.0] * len(psi_values),
        }
    )


def test_all_zero_psi_produces_no_drift():
    results = _feature_results(
        {feature: 0.0 for feature in TELEMETRY_FEATURE_NAMES}
    )

    signals = build_tecnalia_feature_drift_signals(results)

    assert len(signals) == 5
    assert all(signal.severity.value == "NONE" for signal in signals)


def test_monitor_level_psi_is_classified():
    results = _feature_results(
        {
            TELEMETRY_FEATURE_NAMES[0]: 0.15,
            TELEMETRY_FEATURE_NAMES[1]: 0.0,
            TELEMETRY_FEATURE_NAMES[2]: 0.0,
            TELEMETRY_FEATURE_NAMES[3]: 0.0,
            TELEMETRY_FEATURE_NAMES[4]: 0.0,
        }
    )

    signals = build_tecnalia_feature_drift_signals(results)

    assert signals[0].severity.value == "MONITOR"


def test_significant_level_psi_is_classified():
    results = _feature_results(
        {
            TELEMETRY_FEATURE_NAMES[0]: 0.20,
            TELEMETRY_FEATURE_NAMES[1]: 0.0,
            TELEMETRY_FEATURE_NAMES[2]: 0.0,
            TELEMETRY_FEATURE_NAMES[3]: 0.0,
            TELEMETRY_FEATURE_NAMES[4]: 0.0,
        }
    )

    signals = build_tecnalia_feature_drift_signals(results)

    assert signals[0].severity.value == "SIGNIFICANT"


def test_strong_level_psi_is_classified():
    results = _feature_results(
        {
            TELEMETRY_FEATURE_NAMES[0]: 0.25,
            TELEMETRY_FEATURE_NAMES[1]: 0.0,
            TELEMETRY_FEATURE_NAMES[2]: 0.0,
            TELEMETRY_FEATURE_NAMES[3]: 0.0,
            TELEMETRY_FEATURE_NAMES[4]: 0.0,
        }
    )

    signals = build_tecnalia_feature_drift_signals(results)

    assert signals[0].severity.value == "STRONG"


def test_missing_feature_result_is_rejected():
    results = _feature_results(
        {
            TELEMETRY_FEATURE_NAMES[0]: 0.0,
            TELEMETRY_FEATURE_NAMES[1]: 0.0,
            TELEMETRY_FEATURE_NAMES[2]: 0.0,
            TELEMETRY_FEATURE_NAMES[3]: 0.0,
        }
    )

    with pytest.raises(ValueError, match="Expected exactly one"):
        build_tecnalia_feature_drift_signals(results)


def test_duplicate_feature_result_is_rejected():
    results = pd.concat(
        [
            _feature_results(
                {feature: 0.0 for feature in TELEMETRY_FEATURE_NAMES}
            ),
            pd.DataFrame(
                {
                    "feature_name": [TELEMETRY_FEATURE_NAMES[0]],
                    "psi": [0.1],
                    "wasserstein_distance": [0.0],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Expected exactly one"):
        build_tecnalia_feature_drift_signals(results)


def test_missing_required_column_is_rejected():
    results = pd.DataFrame(
        {
            "feature_name": list(TELEMETRY_FEATURE_NAMES),
        }
    )

    with pytest.raises(ValueError, match="missing columns"):
        build_tecnalia_feature_drift_signals(results)


def test_module_distribution_zero_divergence():
    modules = pd.Series(
        ["Atersa", "JaSolar3", "NingboSolar", "Photowatt", "TrinaSolar"]
    )

    signal = build_tecnalia_module_drift_signal(
        reference_modules=modules,
        current_modules=modules,
    )

    assert signal.severity.value == "NONE"


def test_module_distribution_monitor_level():
    reference = pd.Series(["A"] * 90 + ["B"] * 10)
    current = pd.Series(["A"] * 70 + ["B"] * 30)

    signal = build_tecnalia_module_drift_signal(
        reference_modules=reference,
        current_modules=current,
    )

    assert signal.severity.value in {
        "NONE",
        "MONITOR",
        "SIGNIFICANT",
        "STRONG",
    }


def test_feature_signal_names_are_stable():
    results = _feature_results(
        {feature: 0.0 for feature in TELEMETRY_FEATURE_NAMES}
    )

    signals = build_tecnalia_feature_drift_signals(results)

    assert [signal.name for signal in signals] == [
        f"tecnalia.feature.psi:{feature}"
        for feature in TELEMETRY_FEATURE_NAMES
    ]
