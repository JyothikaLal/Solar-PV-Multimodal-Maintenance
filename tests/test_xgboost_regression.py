import numpy as np
import pandas as pd

from src.models.regression.xgboost_model import (
    create_xgboost_regressor,
)
from src.models.regression.xgboost_preprocessing import (
    create_xgboost_preprocessor,
)


def test_xgboost_model_configuration():
    model = create_xgboost_regressor()

    assert model.objective == "reg:squarederror"
    assert model.n_estimators == 500
    assert model.learning_rate == 0.05
    assert model.max_depth == 6
    assert model.random_state == 42


def test_xgboost_model_can_fit_small_dataset():
    model = create_xgboost_regressor()

    X = np.array(
        [
            [200.0, 25.0],
            [400.0, 30.0],
            [600.0, 35.0],
            [800.0, 40.0],
            [1000.0, 45.0],
            [1200.0, 50.0],
        ]
    )
    y = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()


def test_xgboost_preprocessor_handles_numeric_and_categorical_data():
    df = pd.DataFrame(
        {
            "Front GPOA (W/m²)": [200.0, np.nan, 500.0],
            "Temp. Mod (°C)": [25.0, 30.0, np.nan],
            "module_name": ["Atersa", "JaSolar3", "Atersa"],
        }
    )

    preprocessor = create_xgboost_preprocessor(
        numeric_features=[
            "Front GPOA (W/m²)",
            "Temp. Mod (°C)",
        ],
        categorical_features=["module_name"],
    )

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= 3
    assert np.isfinite(transformed).all()


def test_xgboost_preprocessor_handles_unknown_module():
    train = pd.DataFrame(
        {
            "Front GPOA (W/m²)": [200.0, 500.0],
            "module_name": ["Atersa", "JaSolar3"],
        }
    )

    test = pd.DataFrame(
        {
            "Front GPOA (W/m²)": [400.0],
            "module_name": ["NewModule"],
        }
    )

    preprocessor = create_xgboost_preprocessor(
        numeric_features=["Front GPOA (W/m²)"],
        categorical_features=["module_name"],
    )

    preprocessor.fit(train)
    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()
