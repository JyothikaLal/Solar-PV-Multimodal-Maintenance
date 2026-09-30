from __future__ import annotations

import pandas as pd
import pytest

from src.mlops.retraining.data_validation import (
    RetrainingDataValidationError,
    validate_raptormaps_retraining_data,
    validate_tecnalia_retraining_data,
)


def test_tecnalia_validation_rejects_missing_manifest(
    monkeypatch,
    tmp_path,
):
    import src.mlops.retraining.data_validation as module

    monkeypatch.setattr(
        module,
        "_require_tecnalia_raw_validation",
        lambda raw_root: {},
    )

    with pytest.raises(
        RetrainingDataValidationError,
        match="split manifest not found",
    ):
        validate_tecnalia_retraining_data(
            raw_root=tmp_path,
            split_manifest_path=(
                tmp_path / "missing.csv"
            ),
        )


def test_raptormaps_validation_rejects_missing_manifest(
    monkeypatch,
    tmp_path,
):
    import src.mlops.retraining.data_validation as module

    monkeypatch.setattr(
        module,
        "_require_raptormaps_raw_validation",
        lambda raw_root: {},
    )

    with pytest.raises(
        RetrainingDataValidationError,
        match="split manifest not found",
    ):
        validate_raptormaps_retraining_data(
            raw_root=tmp_path,
            split_manifest_path=(
                tmp_path / "missing.csv"
            ),
        )


def test_raptormaps_manifest_split_counts_are_positive(
    monkeypatch,
    tmp_path,
):
    import src.mlops.retraining.data_validation as module

    monkeypatch.setattr(
        module,
        "_require_raptormaps_raw_validation",
        lambda raw_root: {
            "integrity": {"status": "PASS"},
            "labels": {"status": "PASS"},
            "duplicates": {"status": "PASS"},
            "split": {"status": "PASS"},
        },
    )

    manifest = pd.DataFrame(
        {
            "metadata_id": ["1", "2", "3"],
            "image_filepath": [
                "1.jpg",
                "2.jpg",
                "3.jpg",
            ],
            "anomaly_class": [
                "Cell",
                "Diode",
                "Soiling",
            ],
            "split": [
                "train",
                "validation",
                "test",
            ],
        }
    )

    manifest_path = (
        tmp_path / "split_manifest.csv"
    )
    manifest.to_csv(
        manifest_path,
        index=False,
    )

    monkeypatch.setattr(
        module,
        "load_raptormaps_manifest",
        lambda manifest_path: manifest.copy(),
    )

    class DummyDataset:
        def __init__(self, count):
            self._count = count

        def __len__(self):
            return self._count

    class DummyLoader:
        def __init__(self, count):
            self.dataset = DummyDataset(count)

    monkeypatch.setattr(
        module,
        "create_raptormaps_dataloaders",
        lambda **kwargs: (
            DummyLoader(1),
            DummyLoader(1),
            DummyLoader(1),
        ),
    )

    result = (
        validate_raptormaps_retraining_data(
            raw_root=tmp_path,
            split_manifest_path=manifest_path,
        )
    )

    assert result["status"] == "PASS"
    assert result["split_counts"] == {
        "test": 1,
        "train": 1,
        "validation": 1,
    }
    assert result["loader_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_raptormaps_missing_supervised_split_is_rejected(
    monkeypatch,
    tmp_path,
):
    import src.mlops.retraining.data_validation as module

    monkeypatch.setattr(
        module,
        "_require_raptormaps_raw_validation",
        lambda raw_root: {
            "integrity": {"status": "PASS"},
            "labels": {"status": "PASS"},
            "duplicates": {"status": "PASS"},
            "split": {"status": "PASS"},
        },
    )

    manifest = pd.DataFrame(
        {
            "metadata_id": ["1", "2"],
            "image_filepath": [
                "1.jpg",
                "2.jpg",
            ],
            "anomaly_class": [
                "Cell",
                "Diode",
            ],
            "split": [
                "train",
                "validation",
            ],
        }
    )

    manifest_path = (
        tmp_path / "split_manifest.csv"
    )
    manifest.to_csv(
        manifest_path,
        index=False,
    )

    monkeypatch.setattr(
        module,
        "load_raptormaps_manifest",
        lambda manifest_path: manifest.copy(),
    )

    with pytest.raises(
        RetrainingDataValidationError,
        match="missing one or more supervised splits",
    ):
        validate_raptormaps_retraining_data(
            raw_root=tmp_path,
            split_manifest_path=manifest_path,
        )
