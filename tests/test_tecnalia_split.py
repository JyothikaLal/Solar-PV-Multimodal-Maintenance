import pandas as pd
import pytest

from src.data.tecnalia_split import (
    create_tecnalia_temporal_split,
    prepare_tecnalia_split_input,
    validate_tecnalia_temporal_split,
)


def make_dataframe(n_rows=20):
    timestamps = pd.date_range(
        "2025-01-01",
        periods=n_rows,
        freq="10min",
    )

    return pd.DataFrame(
        {
            "Fecha": timestamps,
            "normalized_pmpp": range(n_rows),
            "module_name": ["Atersa"] * n_rows,
        }
    )


def test_split_creates_expected_labels():
    df = make_dataframe(20)

    result = create_tecnalia_temporal_split(df)

    assert set(result["split"]) == {
        "train",
        "validation",
        "test",
    }


def test_split_is_deterministic():
    df = make_dataframe(100)

    result_1 = create_tecnalia_temporal_split(df)
    result_2 = create_tecnalia_temporal_split(df)

    pd.testing.assert_frame_equal(result_1, result_2)


def test_split_preserves_row_count():
    df = make_dataframe(100)

    result = create_tecnalia_temporal_split(df)

    assert len(result) == len(df)


def test_split_uses_chronological_order():
    df = make_dataframe(100)

    result = create_tecnalia_temporal_split(df)

    train = result.loc[result["split"] == "train", "Fecha"]
    validation = result.loc[
        result["split"] == "validation",
        "Fecha",
    ]
    test = result.loc[result["split"] == "test", "Fecha"]

    assert train.max() < validation.min()
    assert validation.max() < test.min()


def test_split_counts_for_100_rows():
    df = make_dataframe(100)

    result = create_tecnalia_temporal_split(df)

    counts = result["split"].value_counts().to_dict()

    assert counts["train"] == 70
    assert counts["validation"] == 15
    assert counts["test"] == 15


def test_unsorted_input_is_sorted():
    df = make_dataframe(20).sample(
        frac=1,
        random_state=123,
    )

    result = create_tecnalia_temporal_split(df)

    assert result["Fecha"].is_monotonic_increasing


def test_duplicate_timestamps_are_rejected():
    df = make_dataframe(20)

    df.loc[1, "Fecha"] = df.loc[0, "Fecha"]

    with pytest.raises(ValueError, match="duplicate timestamps"):
        create_tecnalia_temporal_split(df)


def test_invalid_timestamp_is_rejected():
    df = make_dataframe(20)

    df["Fecha"] = df["Fecha"].astype(object)
    df.loc[0, "Fecha"] = "not-a-timestamp"

    with pytest.raises(ValueError, match="invalid timestamps"):
        create_tecnalia_temporal_split(df)


def test_missing_timestamp_column_is_rejected():
    df = pd.DataFrame({"value": range(10)})

    with pytest.raises(ValueError, match="Fecha"):
        create_tecnalia_temporal_split(df)


def test_input_dataframe_is_not_modified():
    df = make_dataframe(20)
    original = df.copy(deep=True)

    create_tecnalia_temporal_split(df)

    pd.testing.assert_frame_equal(df, original)


def test_module_identity_is_preserved():
    df = make_dataframe(100)
    df.loc[70:, "module_name"] = "JaSolar3"

    result = create_tecnalia_temporal_split(df)

    assert result["module_name"].tolist() == df.sort_values(
        "Fecha"
    )["module_name"].tolist()


def test_validation_function_passes_for_valid_split():
    df = make_dataframe(100)

    result = create_tecnalia_temporal_split(df)

    validation = validate_tecnalia_temporal_split(result)

    assert validation["total_rows"] == 100
    assert validation["train_rows"] == 70
    assert validation["validation_rows"] == 15
    assert validation["test_rows"] == 15
    assert validation["train_before_validation"] is True
    assert validation["validation_before_test"] is True
    assert validation["timestamp_overlap"] is False


def test_validation_rejects_missing_split_column():
    df = make_dataframe(20)

    with pytest.raises(ValueError, match="split"):
        validate_tecnalia_temporal_split(df)

from src.data.tecnalia_split import (
    build_tecnalia_split_manifest,
    validate_tecnalia_split_manifest,
)


def test_manifest_contains_all_modules():
    module_dataframes = {
        "Atersa": make_dataframe(20),
        "JaSolar3": make_dataframe(20),
        "NingboSolar": make_dataframe(20),
        "Photowatt": make_dataframe(20),
        "TrinaSolar": make_dataframe(20),
    }

    for module_name, df in module_dataframes.items():
        df["module_name"] = module_name

    manifest = build_tecnalia_split_manifest(
        module_dataframes
    )

    assert set(manifest["module_name"]) == {
        "Atersa",
        "JaSolar3",
        "NingboSolar",
        "Photowatt",
        "TrinaSolar",
    }


def test_manifest_preserves_module_row_counts():
    module_dataframes = {
        "Atersa": make_dataframe(20),
        "JaSolar3": make_dataframe(20),
        "NingboSolar": make_dataframe(20),
        "Photowatt": make_dataframe(20),
        "TrinaSolar": make_dataframe(20),
    }

    for module_name, df in module_dataframes.items():
        df["module_name"] = module_name

    manifest = build_tecnalia_split_manifest(
        module_dataframes
    )

    counts = manifest["module_name"].value_counts()

    assert counts["Atersa"] == 20
    assert counts["JaSolar3"] == 20
    assert counts["NingboSolar"] == 20
    assert counts["Photowatt"] == 20
    assert counts["TrinaSolar"] == 20


def test_manifest_has_no_module_timestamp_duplicates():
    module_dataframes = {
        "Atersa": make_dataframe(20),
        "JaSolar3": make_dataframe(20),
        "NingboSolar": make_dataframe(20),
        "Photowatt": make_dataframe(20),
        "TrinaSolar": make_dataframe(20),
    }

    for module_name, df in module_dataframes.items():
        df["module_name"] = module_name

    manifest = build_tecnalia_split_manifest(
        module_dataframes
    )

    assert not manifest[
        ["module_name", "Fecha"]
    ].duplicated().any()


def test_manifest_validation_passes():
    module_dataframes = {
        "Atersa": make_dataframe(20),
        "JaSolar3": make_dataframe(20),
        "NingboSolar": make_dataframe(20),
        "Photowatt": make_dataframe(20),
        "TrinaSolar": make_dataframe(20),
    }

    for module_name, df in module_dataframes.items():
        df["module_name"] = module_name

    manifest = build_tecnalia_split_manifest(
        module_dataframes
    )

    report = validate_tecnalia_split_manifest(
        manifest
    )

    assert report["total_rows"] == 100
    assert report["module_count"] == 5