import numpy as np
import pandas as pd
import pytest

from src.evaluation.tecnalia_regression_error_analysis import (
    calculate_group_metrics,
    calculate_largest_errors,
    calculate_residual_summary,
    create_gpoa_bins,
)


def make_predictions():
    return pd.DataFrame(
        {
            "model": [
                "model_a",
                "model_a",
                "model_b",
                "model_b",
            ],
            "module_name": [
                "Atersa",
                "Atersa",
                "JaSolar3",
                "JaSolar3",
            ],
            "y_true": [1.0, 2.0, 1.0, 3.0],
            "y_pred": [1.0, 1.0, 2.0, 2.0],
            "residual": [0.0, 1.0, -1.0, 1.0],
            "Front GPOA (W/m²)": [
                250.0,
                450.0,
                700.0,
                1100.0,
            ],
        }
    )


def test_group_metrics():
    df = make_predictions()

    result = calculate_group_metrics(
        df,
        "module_name",
    )

    assert len(result) == 2
    assert result.loc[
        result["module_name"] == "Atersa",
        "mae",
    ].iloc[0] == pytest.approx(0.5)


def test_gpoa_bins():
    result = create_gpoa_bins(make_predictions())

    assert result["gpoa_bin"].notna().all()
    assert result["gpoa_bin"].astype(str).tolist() == [
        "200-400",
        "400-600",
        "600-800",
        "1000-1200",
    ]


def test_residual_summary():
    result = calculate_residual_summary(
        make_predictions()
    )

    assert set(result["model"]) == {
        "model_a",
        "model_b",
    }


def test_largest_errors():
    result = calculate_largest_errors(
        make_predictions(),
        top_n=2,
    )

    assert len(result) == 2
    assert np.all(
        result["absolute_error"].to_numpy()
        == np.array([1.0, 1.0])
    )
