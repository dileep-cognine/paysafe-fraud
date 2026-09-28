"""Load an approved registry version and score validated serving transactions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from fraud_scoring.config import AppConfig
from fraud_scoring.features import build_serving_features


@dataclass(frozen=True)
class LoadedChampion:
    model: Any
    version: str


def load_champion(config: AppConfig) -> LoadedChampion:
    """Resolve the configured alias once and load that exact approved version."""
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    version = MlflowClient(tracking_uri=config.mlflow.tracking_uri).get_model_version_by_alias(
        config.model.name, config.model.alias
    )
    if version.tags.get("quality_gate_status") != "PASS":
        raise ValueError("Aliased model version has no passing quality-gate tag.")
    # Pin the resolved version so alias movement cannot change the model between
    # the registry lookup and loading, or change responses in this process.
    model_uri = f"models:/{config.model.name}/{version.version}"
    model = mlflow.sklearn.load_model(model_uri)
    classes = list(model.classes_)
    if classes.count(1) != 1:
        raise ValueError("Approved model must expose fraud class 1 probabilities.")
    return LoadedChampion(model=model, version=str(version.version))


def score_transaction(champion: LoadedChampion, transaction: dict[str, object]) -> float:
    """Use the shared serving builder and return the probability of class 1."""
    features = build_serving_features(pd.DataFrame([transaction]))
    classes = list(champion.model.classes_)
    fraud_index = classes.index(1)
    probabilities = np.asarray(champion.model.predict_proba(features), dtype=float)
    if probabilities.shape != (1, len(classes)):
        raise ValueError("Model returned an unexpected probability shape.")
    risk_score = float(probabilities[0, fraud_index])
    if not math.isfinite(risk_score) or not 0 <= risk_score <= 1:
        raise ValueError("Model returned an invalid fraud probability.")
    return risk_score
