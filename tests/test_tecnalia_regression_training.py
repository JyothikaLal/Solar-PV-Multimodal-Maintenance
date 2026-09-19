import numpy as np
import pytest

from src.training.train_tecnalia_regression import (
    calculate_mape,
    calculate_regression_metrics,
)


def test_calculate_mape():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 1.0])

    assert calculate_mape(y_true, y_pred) == pytest.approx(25.0)


def test_calculate_mape_ignores_zero_targets():
    y_true = np.array([0.0, 2.0])
    y_pred = np.array([1.0, 1.0])

    assert calculate_mape(y_true, y_pred) == pytest.approx(50.0)


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["r2"] == pytest.approx(0.5)
