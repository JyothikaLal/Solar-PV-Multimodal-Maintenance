from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.compare_custom_cnn_baselines import (
    METRICS,
    build_overall_comparison,
    build_per_class_comparison,
)


TASK18_DIR = Path(
    "reports/results/raptormaps/custom_cnn"
)

TASK25_DIR = Path(
    "reports/results/raptormaps/custom_cnn_task25"
)


def test_overall_comparison_has_expected_metrics():
    result = build_overall_comparison()

    assert list(result["metric"]) == METRICS
    assert len(result) == len(METRICS)

    assert result[
        "task18"
    ].notna().all()

    assert result[
        "task25"
    ].notna().all()

    assert result[
        "absolute_delta_task25_minus_task18"
    ].notna().all()


def test_macro_f1_delta_matches_saved_results():
    result = build_overall_comparison()

    row = result[
        result["metric"] == "macro_f1"
    ].iloc[0]

    assert row["task18"] == pytest.approx(
        0.4363234052965139
    )

    assert row["task25"] == pytest.approx(
        0.492170412075358
    )

    assert row[
        "absolute_delta_task25_minus_task18"
    ] == pytest.approx(
        0.05584700677884408
    )


def test_per_class_comparison_has_12_classes():
    result = build_per_class_comparison()

    assert len(result) == 12

    assert result[
        "task18_f1"
    ].notna().all()

    assert result[
        "task25_f1"
    ].notna().all()

    assert result[
        "f1_delta_task25_minus_task18"
    ].notna().all()


def test_support_is_unchanged_between_runs():
    result = build_per_class_comparison()

    assert (
        result["support_delta_task25_minus_task18"]
        == 0
    ).all()


def test_expected_result_files_exist():
    expected = [
        TASK18_DIR / "test_overall_metrics.csv",
        TASK18_DIR / "test_per_class_metrics.csv",
        TASK25_DIR / "test_overall_metrics.csv",
        TASK25_DIR / "test_per_class_metrics.csv",
    ]

    for path in expected:
        assert path.exists(), path
