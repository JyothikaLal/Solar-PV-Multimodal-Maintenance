import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from src.models.regression.tuning import (
    calculate_regression_metrics,
    create_gradient_boosting_candidates,
    create_xgboost_candidates,
)


def test_calculate_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert set(metrics) == {
        "mae",
        "rmse",
        "r2",
        "mape_percent",
    }

    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert np.isfinite(metrics["r2"])
    assert metrics["mape_percent"] > 0


def test_gradient_boosting_candidates():
    candidates = create_gradient_boosting_candidates()

    assert len(candidates) == 8
    assert all(
        isinstance(model, GradientBoostingRegressor)
        for _, model, _ in candidates
    )


def test_xgboost_candidates():
    candidates = create_xgboost_candidates()

    assert len(candidates) == 6
    assert all(
        isinstance(model, XGBRegressor)
        for _, model, _ in candidates
    )


def test_candidate_ids_are_unique():
    gb = create_gradient_boosting_candidates()
    xgb = create_xgboost_candidates()

    assert len({candidate_id for candidate_id, _, _ in gb}) == len(gb)
    assert len({candidate_id for candidate_id, _, _ in xgb}) == len(xgb)