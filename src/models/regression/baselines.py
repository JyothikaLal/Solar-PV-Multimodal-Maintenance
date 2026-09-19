from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression


RANDOM_STATE = 42


def create_linear_regression() -> LinearRegression:
    """Create the linear regression baseline."""

    return LinearRegression()


def create_random_forest_regressor() -> RandomForestRegressor:
    """Create the random forest regression baseline."""

    return RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_gradient_boosting_regressor() -> GradientBoostingRegressor:
    """Create the gradient boosting regression baseline."""

    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
        loss="squared_error",
    )


def create_regression_baseline_models() -> dict[str, object]:
    """Create all regression baseline models."""

    return {
        "linear_regression": create_linear_regression(),
        "random_forest": create_random_forest_regressor(),
        "gradient_boosting": create_gradient_boosting_regressor(),
    }
