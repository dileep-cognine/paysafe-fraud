"""Configuration management module for PaySafe Fraud Scoring.

Loads and validates environment-specific YAML configuration files and environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field, ValidationError


class DataConfig(BaseModel):
    raw_data_path: str = "data/raw/transactions.csv"
    processed_train_path: str = "data/processed/train.parquet"
    processed_test_path: str = "data/processed/test.parquet"
    features_path: str = "data/processed/features.json"
    target_column: str = "is_fraud"
    id_column: str = "transaction_id"
    test_size: float = Field(default=0.2, ge=0.01, le=0.9)
    random_state: int = 42


class FeaturesConfig(BaseModel):
    numerical_features: List[str] = Field(default_factory=lambda: ["amount", "hour_of_day", "device_risk"])
    categorical_features: List[str] = Field(default_factory=lambda: ["merchant_category"])
    derived_features: List[str] = Field(default_factory=lambda: ["is_night_transaction", "high_amount_risk"])


class ModelConfig(BaseModel):
    name: str = "paysafe-fraud-detector"
    alias: str = "champion"
    algorithm: str = "HistGradientBoostingClassifier"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EvaluationThresholds(BaseModel):
    min_roc_auc: float = Field(default=0.70, ge=0.0, le=1.0)
    min_pr_auc: float = Field(default=0.30, ge=0.0, le=1.0)
    min_precision_at_recall_80: float = Field(default=0.20, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    thresholds: EvaluationThresholds = Field(default_factory=EvaluationThresholds)


class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///mlruns.db"
    experiment_name: str = "paysafe-fraud-scoring"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1


class AppConfig(BaseModel):
    environment: str = "dev"
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


_CONFIG_CACHE: Optional[AppConfig] = None


def resolve_config_path(config_path: Optional[str | Path] = None) -> Path:
    """Resolve configuration file path based on argument, environment variable, or APP_ENV."""
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
        # Fallback to dev if specific env file is missing
        fallback = Path("configs/dev.yaml")
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Configuration file not found for environment '{app_env}': {default_path}")
    return default_path


def load_config(config_path: Optional[str | Path] = None, force_reload: bool = False) -> AppConfig:
    """Load, validate, and return the application configuration."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload and config_path is None:
        return _CONFIG_CACHE

    target_path = resolve_config_path(config_path)

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Failed to read configuration YAML at {target_path}: {e}") from e

    # Apply environment variable overrides for critical infrastructure settings
    if "MLFLOW_TRACKING_URI" in os.environ:
        raw_yaml.setdefault("mlflow", {})["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    if "MODEL_ALIAS" in os.environ:
        raw_yaml.setdefault("model", {})["alias"] = os.environ["MODEL_ALIAS"]
    if "APP_ENV" in os.environ:
        raw_yaml["environment"] = os.environ["APP_ENV"]

    try:
        config = AppConfig(**raw_yaml)
    except ValidationError as e:
        raise ValueError(f"Configuration schema validation failed for {target_path}:\n{e}") from e

    if config_path is None:
        _CONFIG_CACHE = config

    return config
