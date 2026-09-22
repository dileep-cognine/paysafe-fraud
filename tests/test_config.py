import os
# pyrefly: ignore [missing-import]
import pytest
from pathlib import Path
from fraud_scoring.config import load_config, resolve_config_path, AppConfig


def test_load_dev_config():
    config = load_config("configs/dev.yaml", force_reload=True)
    assert isinstance(config, AppConfig)
    assert config.environment == "dev"
    assert config.model.name == "paysafe-fraud-detector"
    assert config.model.alias == "champion"
    assert config.evaluation.thresholds.min_roc_auc == 0.70
    assert "amount" in config.features.numerical_features


def test_load_ci_and_prod_configs():
    ci_cfg = load_config("configs/ci.yaml", force_reload=True)
    assert ci_cfg.environment == "ci"
    assert ci_cfg.model.alias == "challenger"
    assert ci_cfg.evaluation.thresholds.min_roc_auc == 0.70

    prod_cfg = load_config("configs/prod.yaml", force_reload=True)
    assert prod_cfg.environment == "prod"
    assert prod_cfg.model.alias == "champion"
    assert prod_cfg.evaluation.thresholds.min_roc_auc == 0.80
    assert prod_cfg.evaluation.thresholds.min_pr_auc == 0.45


def test_config_missing_file_fails_loudly():
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent.yaml", force_reload=True)


def test_config_invalid_schema_fails_loudly():
    temp_dir = Path(".pytest_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    bad_config = temp_dir / "bad_config.yaml"
    bad_config.write_text("evaluation:\n  thresholds:\n    min_roc_auc: 99.9\n", encoding="utf-8")

    try:
        with pytest.raises(ValueError, match="Configuration schema validation failed"):
            load_config(str(bad_config), force_reload=True)
    finally:
        if bad_config.exists():
            bad_config.unlink()


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///test_override.db")
    monkeypatch.setenv("MODEL_ALIAS", "test_candidate")

    config = load_config("configs/dev.yaml", force_reload=True)
    assert config.mlflow.tracking_uri == "sqlite:///test_override.db"
    assert config.model.alias == "test_candidate"
