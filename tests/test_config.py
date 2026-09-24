from pathlib import Path

import pytest

from fraud_scoring.config import (
    ENVIRONMENT_OVERRIDES,
    AppConfig,
    load_config,
    resolve_config_path,
)


@pytest.fixture(autouse=True)
def isolate_configuration_environment(monkeypatch):
    """Keep committed-profile tests independent of a developer's local .env."""
    for variable in ("APP_ENV", "CONFIG_PATH", *ENVIRONMENT_OVERRIDES):
        monkeypatch.delenv(variable, raising=False)


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


def test_config_missing_required_section_fails_loudly():
    temp_dir = Path(".pytest_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    incomplete_config = temp_dir / "incomplete_config.yaml"
    incomplete_config.write_text("environment: dev\n", encoding="utf-8")

    try:
        with pytest.raises(ValueError, match="Configuration schema validation failed"):
            load_config(str(incomplete_config), force_reload=True)
    finally:
        if incomplete_config.exists():
            incomplete_config.unlink()


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///test_override.db")
    monkeypatch.setenv("MODEL_ALIAS", "test_candidate")

    config = load_config("configs/dev.yaml", force_reload=True)
    assert config.mlflow.tracking_uri == "sqlite:///test_override.db"
    assert config.model.alias == "test_candidate"


def test_explicit_config_path_ignores_app_env(monkeypatch):
    """An explicit profile must not be relabelled by the runner's APP_ENV."""
    monkeypatch.setenv("APP_ENV", "ci")

    config = load_config("configs/dev.yaml", force_reload=True)

    assert config.environment == "dev"


def test_unknown_environment_fails_loudly(monkeypatch):
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    monkeypatch.setenv("APP_ENV", "missing-environment")

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        resolve_config_path()
