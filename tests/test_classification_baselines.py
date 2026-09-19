from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.models.classification.baselines import (
    create_baseline_models,
    create_decision_tree,
    create_dummy_classifier,
    create_logistic_regression,
    create_random_forest,
)


def test_create_dummy_classifier():
    model = create_dummy_classifier()

    assert isinstance(model, DummyClassifier)
    assert model.strategy == "most_frequent"


def test_create_logistic_regression():
    model = create_logistic_regression()

    assert isinstance(model, LogisticRegression)
    assert model.class_weight == "balanced"
    assert model.solver == "saga"
    assert model.max_iter == 5000
    assert model.tol == 1e-3
    assert model.random_state == 42


def test_create_decision_tree():
    model = create_decision_tree()

    assert isinstance(model, DecisionTreeClassifier)
    assert model.class_weight == "balanced"
    assert model.random_state == 42


def test_create_random_forest():
    model = create_random_forest()

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 200
    assert model.class_weight == "balanced"
    assert model.random_state == 42
    assert model.n_jobs == -1


def test_create_all_baseline_models():
    models = create_baseline_models()

    assert set(models) == {
        "dummy",
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }

    assert isinstance(
        models["dummy"],
        DummyClassifier,
    )

    assert isinstance(
        models["logistic_regression"],
        LogisticRegression,
    )

    assert isinstance(
        models["decision_tree"],
        DecisionTreeClassifier,
    )

    assert isinstance(
        models["random_forest"],
        RandomForestClassifier,
    )