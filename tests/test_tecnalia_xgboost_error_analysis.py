from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.tecnalia_xgboost_error_analysis import (
    load_xgboost_predictions,
)


def test_load_xgboost_predictions(tmp_path: Path):
    path = tmp_path / "predictions.csv"

    pd.DataFrame(
        {
            "module_name": ["Atersa", "JaSolar3"],
            "Fecha": [
                "2025-06-19 20:00:00",
                "2025-06-19 20:10:00",
            ],
            "normalized_pmpp": [0.90, 0.85],
            "prediction": [0.88, 0.87],
            "residual": [-0.02, 0.02],
        }
    ).to_csv(path, index=False)

    result = load_xgboost_predictions(path)

    assert result["model"].unique().tolist() == ["xgboost"]

    assert result.columns.tolist() == [
        "model",
        "module_name",
        "Fecha",
        "y_true",
        "y_pred",
        "residual",
    ]

    assert result["y_true"].tolist() == pytest.approx(
        [0.90, 0.85]
    )

    assert result["y_pred"].tolist() == pytest.approx(
        [0.88, 0.87]
    )


def test_load_xgboost_predictions_rejects_missing_columns(
    tmp_path: Path,
):
    path = tmp_path / "invalid.csv"

    pd.DataFrame(
        {
            "module_name": ["Atersa"],
            "Fecha": ["2025-06-19 20:00:00"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required"):
        load_xgboost_predictions(path)
