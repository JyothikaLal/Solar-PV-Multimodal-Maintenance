import numpy as np
import pandas as pd

from src.models.regression.preprocessing import (
    create_regression_preprocessor,
)


NUMERIC = [
    "Front GPOA",
    "GHI",
    "Temp Mod",
    "Temp Amb",
    "Wind Speed",
]

CATEGORICAL = ["module_name"]


def make_data():
    return pd.DataFrame(
        {
            "Front GPOA": [250.0, 500.0, np.nan],
            "GHI": [230.0, 450.0, 300.0],
            "Temp Mod": [30.0, 35.0, 32.0],
            "Temp Amb": [25.0, 30.0, 27.0],
            "Wind Speed": [2.0, 3.0, 1.0],
            "module_name": ["Atersa", "JaSolar3", None],
        }
    )


def test_preprocessor_fits_and_transforms():
    data = make_data()

    preprocessor = create_regression_preprocessor(
        NUMERIC,
        CATEGORICAL,
    )

    transformed = preprocessor.fit_transform(data)

    assert transformed.shape[0] == len(data)
    assert transformed.shape[1] == 7
    assert np.isfinite(transformed).all()


def test_unknown_category_is_supported():
    train = make_data().iloc[:2].copy()
    test = make_data().iloc[:1].copy()
    test["module_name"] = "NewModule"

    preprocessor = create_regression_preprocessor(
        NUMERIC,
        CATEGORICAL,
    )

    preprocessor.fit(train)
    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_numeric_scaling_is_train_fitted():
    train = make_data().iloc[:2].copy()

    preprocessor = create_regression_preprocessor(
        NUMERIC,
        CATEGORICAL,
    )

    transformed = preprocessor.fit_transform(train)

    numeric_block = transformed[:, :5]

    assert np.allclose(numeric_block.mean(axis=0), 0.0)
    assert np.allclose(numeric_block.std(axis=0), 1.0)
