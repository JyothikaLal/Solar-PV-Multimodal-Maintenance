import pandas as pd

from src.evaluation.classification_error_analysis import (
    extract_top_confusions,
)


def test_extract_top_confusions(tmp_path):
    matrix = pd.DataFrame(
        [
            [5, 2, 0],
            [1, 4, 1],
            [0, 2, 6],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    matrix.to_csv(
        tmp_path / "random_forest_test_confusion_matrix.csv"
    )

    result = extract_top_confusions(
        "random_forest",
        top_n=3,
        results_dir=tmp_path,
    )

    assert len(result) == 3
    assert result.iloc[0]["count"] == 2
    assert result.iloc[0]["true_class"] in {"A", "C"}
    assert result.iloc[0]["predicted_class"] in {"B", "A"}