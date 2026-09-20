from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


@dataclass(frozen=True)
class TuningResult:
    model_family: str
    candidate_id: int
    params: dict
    mae: float
    rmse: float
    r2: float
    mape_percent: float


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_pred - y_true

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    non_zero = y_true != 0

    if np.any(non_zero):
        mape = float(
            np.mean(
                np.abs(
                    (y_true[non_zero] - y_pred[non_zero])
                    / y_true[non_zero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "mape_percent": mape,
    }


def create_gradient_boosting_candidates():
    candidates = []

    candidate_id = 0

    for n_estimators in [200, 400]:
        for learning_rate in [0.03, 0.05]:
            for max_depth in [2, 3]:
                candidates.append(
                    (
                        candidate_id,
                        GradientBoostingRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            min_samples_leaf=5,
                            random_state=42,
                        ),
                        {
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "min_samples_leaf": 5,
                        },
                    )
                )

                candidate_id += 1

    return candidates


def create_xgboost_candidates():
    candidates = []

    candidate_id = 0

    for max_depth in [3, 5, 6]:
        for learning_rate in [0.03, 0.05]:
            candidates.append(
                (
                    candidate_id,
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=500,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        min_child_weight=3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.0,
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                    {
                        "n_estimators": 500,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "min_child_weight": 3,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.0,
                        "reg_lambda": 1.0,
                    },
                )
            )

            candidate_id += 1

    return candidates