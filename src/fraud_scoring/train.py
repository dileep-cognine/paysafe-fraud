"""Fit a reproducible candidate model from validated historical transactions."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_scoring.config import AppConfig, load_config
from fraud_scoring.data_validation import validate_and_enforce
from fraud_scoring.features import TRAINING_FEATURES, build_training_features


@dataclass
class CandidateArtifact:
    """Model, input contract, and source fingerprint needed for later evaluation."""

    pipeline: Pipeline
    dataset_sha256: str
    feature_names: tuple[str, ...]
    validation_indices: list[int]


def dataset_sha256(path: Path) -> str:
    """Fingerprint the exact raw dataset bytes used for training."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_indices(labels: pd.Series, config: AppConfig) -> tuple[list[int], list[int]]:
    """Return a repeatable stratified holdout using positional row indices."""
    counts = labels.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Training requires at least two rows from each target class.")
    train_idx, validation_idx = train_test_split(
        list(range(len(labels))),
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=labels,
    )
    return list(train_idx), list(validation_idx)


def train_candidate(data_path: Path, config: AppConfig) -> CandidateArtifact:
    """Validate, split, and fit preprocessing plus logistic regression together."""
    if config.model.algorithm != "LogisticRegression":
        raise ValueError(f"Unsupported model algorithm: {config.model.algorithm}")
    raw = validate_and_enforce(pd.read_csv(data_path), is_training=True)
    built = build_training_features(raw)
    train_idx, validation_idx = split_indices(built.target, config)

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), ["amount", "hour_of_day", "device_risk"]),
            ("merchant", OneHotEncoder(handle_unknown="ignore"), ["merchant_category"]),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(**config.model.parameters)),
        ]
    )
    pipeline.fit(built.features.iloc[train_idx], built.target.iloc[np.array(train_idx)])
    return CandidateArtifact(
        pipeline=pipeline,
        dataset_sha256=dataset_sha256(data_path),
        feature_names=TRAINING_FEATURES,
        validation_indices=validation_idx,
    )


def save_candidate(candidate: CandidateArtifact, path: Path) -> None:
    """Save the complete fitted preprocessing and classifier as one artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local fraud-scoring candidate")
    parser.add_argument("--config", help="YAML profile; defaults to APP_ENV / CONFIG_PATH")
    parser.add_argument("--data-path", type=Path, help="Override the configured raw CSV path")
    args = parser.parse_args()
    config = load_config(args.config)
    data_path = args.data_path or Path(config.data.raw_data_path)
    # Keep MLflow orchestration outside the estimator implementation.
    from fraud_scoring.mlflow_tracking import run_training_experiment

    outcome = run_training_experiment(data_path, config)
    print(f"Saved candidate to {config.model.artifact_path}")
    print(f"Saved evaluation to {config.evaluation.metrics_path}")
    print(f"MLflow run: {outcome.run_id} (model: {outcome.model_uri})")
    print(f"Quality gate: {'PASS' if outcome.evaluation.passed else 'FAIL'}")
    if not outcome.evaluation.passed:
        raise SystemExit("Quality gate FAILED: " + "; ".join(outcome.evaluation.failed_thresholds))


if __name__ == "__main__":
    main()
