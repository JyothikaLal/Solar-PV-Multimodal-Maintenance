from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def create_dummy_classifier() -> DummyClassifier:
    """
    Create the trivial majority-class baseline classifier.
    """
    return DummyClassifier(
        strategy="most_frequent"
    )


def create_logistic_regression() -> LogisticRegression:
    """
    Create the Logistic Regression baseline.

    SAGA is used for the multiclass optimization problem.
    The input pixels are already normalized to [0, 1], so
    additional feature standardization is not required.

    A larger iteration budget and practical convergence
    tolerance are used to avoid prematurely terminating
    the baseline optimization.
    """
    return LogisticRegression(
        class_weight="balanced",
        solver="saga",
        max_iter=5000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )


def create_decision_tree() -> DecisionTreeClassifier:
    """
    Create the Decision Tree baseline.

    Class balancing is used to reduce the influence of the
    majority class during training.
    """
    return DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def create_random_forest() -> RandomForestClassifier:
    """
    Create the Random Forest baseline.

    Class balancing is applied at the tree level.
    """
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_baseline_models() -> dict[str, object]:
    """
    Create all classical classification baseline models.

    Returns:
        Dictionary mapping model names to unfitted sklearn estimators.
    """
    return {
        "dummy": create_dummy_classifier(),
        "logistic_regression": create_logistic_regression(),
        "decision_tree": create_decision_tree(),
        "random_forest": create_random_forest(),
    }