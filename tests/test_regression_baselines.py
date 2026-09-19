from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression

from src.models.regression.baselines import (
    create_gradient_boosting_regressor,
    create_linear_regression,
    create_random_forest_regressor,
    create_regression_baseline_models,
)


def test_linear_regression_creation():
    model = create_linear_regression()

    assert isinstance(
        model,
        LinearRegression,
    )


def test_random_forest_regressor_creation():
    model = create_random_forest_regressor()

    assert isinstance(
        model,
        RandomForestRegressor,
    )

    assert model.n_estimators == 200


def test_gradient_boosting_regressor_creation():
    model = create_gradient_boosting_regressor()

    assert isinstance(
        model,
        GradientBoostingRegressor,
    )

    assert model.n_estimators == 200
    assert model.learning_rate == 0.05


def test_all_regression_baselines_are_created():
    models = create_regression_baseline_models()

    assert set(models) == {
        "linear_regression",
        "random_forest",
        "gradient_boosting",
    }
