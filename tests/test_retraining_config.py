from pathlib import Path

import pytest
import yaml

from src.mlops.retraining.config import load_retraining_config


def test_load_project_retraining_config():
    config = load_retraining_config()

    assert config.seed == 42
    assert config.tracking_uri == "sqlite:///mlflow.db"
    assert config.dry_run is False
    assert config.register_candidate is True
    assert config.allow_promotion is False

    assert (
        config.tecnalia_model.model_family
        == "gradient_boosting_tuned"
    )
    assert (
        config.raptormaps_resnet18.model_family
        == "resnet18_finetuned"
    )
    assert (
        config.raptormaps_efficientnet_b0.model_family
        == "efficientnet_b0_finetuned"
    )


def test_paths_are_resolved_against_project_root(tmp_path: Path):
    config_path = tmp_path / "configs" / "config.yaml"
    config_path.parent.mkdir()

    config = {
        "project": {
            "name": "test",
        },
        "data": {
            "tecnalia": {
                "raw_dir": "data/raw/tecnalia",
                "split_manifest": "reports/tecnalia.csv",
            },
            "raptormaps": {
                "raw_dir": "data/raw/raptormaps",
                "split_manifest": "reports/raptormaps.csv",
            },
        },
        "reproducibility": {
            "seed": 42,
        },
        "runtime": {
            "device": "auto",
        },
        "mlflow": {
            "tracking_uri": "sqlite:///mlflow.db",
        },
        "retraining": {
            "dry_run": True,
            "register_candidate": False,
            "allow_promotion": False,
            "tecnalia": {
                "model_family": "gradient_boosting_tuned",
            },
            "raptormaps": {
                "resnet18_model_family": "resnet18_finetuned",
                "efficientnet_b0_model_family": "efficientnet_b0_finetuned",
            },
        },
    }

    config_path.write_text(yaml.safe_dump(config))

    loaded = load_retraining_config(config_path)

    assert loaded.tecnalia.split_manifest == (
        tmp_path / "reports/tecnalia.csv"
    )
    assert loaded.raptormaps.split_manifest == (
        tmp_path / "reports/raptormaps.csv"
    )


def test_missing_retraining_section_is_rejected(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {},
                "reproducibility": {"seed": 42},
                "runtime": {"device": "auto"},
                "mlflow": {"tracking_uri": "sqlite:///mlflow.db"},
            }
        )
    )

    with pytest.raises(ValueError, match="retraining"):
        load_retraining_config(config_path)


def test_allow_promotion_must_be_boolean(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "tecnalia": {"split_manifest": "tecnalia.csv"},
                    "raptormaps": {"split_manifest": "raptormaps.csv"},
                },
                "reproducibility": {"seed": 42},
                "runtime": {"device": "auto"},
                "mlflow": {"tracking_uri": "sqlite:///mlflow.db"},
                "retraining": {
                    "allow_promotion": "true",
                    "tecnalia": {
                        "model_family": "gradient_boosting_tuned"
                    },
                    "raptormaps": {
                        "resnet18_model_family": "resnet18_finetuned",
                        "efficientnet_b0_model_family": (
                            "efficientnet_b0_finetuned"
                        ),
                    },
                },
            }
        )
    )

    with pytest.raises(ValueError, match="allow_promotion"):
        load_retraining_config(config_path)
