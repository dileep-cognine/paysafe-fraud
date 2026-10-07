"""Configuration management module for PaySafe Fraud Scoring.

Loads and validates environment-specific YAML configuration files and environment-variable
overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class StrictConfigModel(BaseModel):
    """Reject misspelled configuration keys rather than silently ignoring them."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictConfigModel):
    """Data paths, identifiers, and split settings for one environment."""

    raw_data_path: str = "data/raw/transactions.csv"
    processed_train_path: str = "data/processed/train.parquet"
    processed_test_path: str = "data/processed/test.parquet"
    features_path: str = "data/processed/features.json"
    target_column: str = "is_fraud"
    id_column: str = "transaction_id"
    test_size: float = Field(default=0.2, ge=0.01, le=0.9)
    random_state: int = 42


class FeaturesConfig(StrictConfigModel):
    """Feature groups and optional merchant-normalization settings."""

    numerical_features: List[str] = Field(
        default_factory=lambda: ["amount", "hour_of_day", "device_risk"]
    )
    categorical_features: List[str] = Field(default_factory=lambda: ["merchant_category"])
    derived_features: List[str] = Field(
        default_factory=lambda: ["is_night_transaction", "high_amount_risk"]
    )
    llm_category_mapping_enabled: bool = False
    llm_category_mapping_model: str = "qwen3-coder:30b"
    llm_category_mapping_base_url: str = "http://127.0.0.1:11434"
    llm_category_mapping_timeout_seconds: float = Field(default=5.0, gt=0, le=30)


class ModelConfig(StrictConfigModel):
    """Model identity, parameters, and local candidate artifact path."""

    name: str = "paysafe-fraud-detector"
    alias: str = "champion"
    algorithm: str = "HistGradientBoostingClassifier"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    artifact_path: str = "artifacts/model/candidate.joblib"


class EvaluationThresholds(StrictConfigModel):
    """Minimum metric values required by the evaluation quality gate."""

    min_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    min_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    min_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    min_roc_auc: float = Field(default=0.70, ge=0.0, le=1.0)
    min_pr_auc: float = Field(default=0.30, ge=0.0, le=1.0)
    min_precision_at_recall_80: float = Field(default=0.20, ge=0.0, le=1.0)


class EvaluationConfig(StrictConfigModel):
    """Evaluation thresholds and destination for measured metrics."""

    thresholds: EvaluationThresholds = Field(default_factory=EvaluationThresholds)
    metrics_path: str = "artifacts/evaluation/metrics.json"


class MLflowConfig(StrictConfigModel):
    """MLflow tracking endpoint and experiment identity."""

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "paysafe-fraud-scoring"


class ServerConfig(StrictConfigModel):
    """Runtime settings for the FastAPI server process."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "INFO"


class AppConfig(StrictConfigModel):
    """Complete configuration required to run one application environment."""

    environment: str
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
    server: ServerConfig


_CONFIG_CACHE: Optional[AppConfig] = None


ENVIRONMENT_OVERRIDES: Dict[str, tuple[str, str]] = {
    "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
    "MLFLOW_EXPERIMENT_NAME": ("mlflow", "experiment_name"),
    "MODEL_REGISTRY_NAME": ("model", "name"),
    "MODEL_ALIAS": ("model", "alias"),
    "API_HOST": ("server", "host"),
    "API_PORT": ("server", "port"),
    "API_WORKERS": ("server", "workers"),
    "LOG_LEVEL": ("server", "log_level"),
    "LLM_CATEGORY_MAPPING_ENABLED": ("features", "llm_category_mapping_enabled"),
    "LLM_CATEGORY_MAPPING_MODEL": ("features", "llm_category_mapping_model"),
    "LLM_CATEGORY_MAPPING_BASE_URL": ("features", "llm_category_mapping_base_url"),
    "LLM_CATEGORY_MAPPING_TIMEOUT_SECONDS": (
        "features",
        "llm_category_mapping_timeout_seconds",
    ),
}


def resolve_config_path(config_path: Optional[str | Path] = None) -> Path:
    """Resolve a profile path using the configured precedence order.

    Args:
        config_path: Explicit YAML path supplied by a caller.

    Returns:
        Existing YAML profile selected from the argument, `CONFIG_PATH`, or `APP_ENV`.

    Raises:
        FileNotFoundError: If the selected configuration file does not exist.
    """
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Specified configuration file not found: {path}")
        return path

    env_config = os.getenv("CONFIG_PATH")
    if env_config:
        path = Path(env_config)
        if not path.exists():
            raise FileNotFoundError(f"CONFIG_PATH points to non-existent file: {path}")
        return path

    app_env = os.getenv("APP_ENV", "dev").lower()
    default_path = Path(f"configs/{app_env}.yaml")
    if not default_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found for environment '{app_env}': {default_path}"
        )
    return default_path


def load_config(config_path: Optional[str | Path] = None, force_reload: bool = False) -> AppConfig:
    """Load and validate the selected application profile.

    Args:
        config_path: Explicit YAML path supplied by a caller.
        force_reload: Whether to bypass the cached implicit configuration.

    Returns:
        Validated application configuration with environment overrides applied.

    Raises:
        ValueError: If the YAML cannot be read or fails schema validation.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload and config_path is None:
        return _CONFIG_CACHE

    target_path = resolve_config_path(config_path)

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Failed to read configuration YAML at {target_path}: {e}") from e

    # Environment variables carry deployment-specific values and secrets.
    # YAML remains the committed, non-secret configuration baseline.
    for env_var, (section, key) in ENVIRONMENT_OVERRIDES.items():
        value = os.getenv(env_var)
        if value is not None and value.strip():
            raw_yaml.setdefault(section, {})[key] = value
    try:
        config = AppConfig(**raw_yaml)
    except ValidationError as e:
        raise ValueError(f"Configuration schema validation failed for {target_path}:\n{e}") from e

    if config_path is None:
        _CONFIG_CACHE = config

    return config
