"""Evaluate a local candidate model and apply configured quality thresholds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_scoring.config import AppConfig, EvaluationThresholds, load_config
from fraud_scoring.data_validation import validate_and_enforce
from fraud_scoring.features import TRAINING_FEATURES, build_training_features
from fraud_scoring.train import CandidateArtifact, dataset_sha256


@dataclass(frozen=True)
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    precision_at_recall_80: float
    validation_samples: int
    positive_samples: int
    negative_samples: int
    passed: bool
    failed_thresholds: tuple[str, ...]


def quality_gate(metrics: dict[str, float], thresholds: EvaluationThresholds) -> tuple[str, ...]:
    """Return every missed threshold; an empty tuple means the candidate passes."""
    required = {
        "precision": thresholds.min_precision,
        "recall": thresholds.min_recall,
        "f1": thresholds.min_f1,
        "roc_auc": thresholds.min_roc_auc,
        "pr_auc": thresholds.min_pr_auc,
        "precision_at_recall_80": thresholds.min_precision_at_recall_80,
    }
    return tuple(
        f"{name}: {metrics[name]:.4f} < {minimum:.4f}"
        for name, minimum in required.items()
        if metrics[name] < minimum
    )


def evaluate_candidate(
    candidate: CandidateArtifact, data_path: Path, config: AppConfig
) -> EvaluationResult:
    """Score only the candidate's untouched holdout from the same dataset bytes."""
    if dataset_sha256(data_path) != candidate.dataset_sha256:
        raise ValueError("Dataset SHA-256 differs from training; evaluation cannot reuse this holdout.")
    if candidate.feature_names != TRAINING_FEATURES:
        raise ValueError("Candidate feature contract differs from the current shared feature builder.")
    raw = validate_and_enforce(pd.read_csv(data_path), is_training=True)
    built = build_training_features(raw)
    x_valid = built.features.iloc[candidate.validation_indices]
    y_valid = built.target.iloc[np.array(candidate.validation_indices)]
    if len(y_valid) == 0 or y_valid.nunique() != 2:
        raise ValueError("Validation holdout must contain both target classes.")

    probabilities = candidate.pipeline.predict_proba(x_valid)[:, 1]
    predictions = candidate.pipeline.predict(x_valid)
    curve_precision, curve_recall, _ = precision_recall_curve(y_valid, probabilities)
    metrics = {
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "pr_auc": float(average_precision_score(y_valid, probabilities)),
        "precision_at_recall_80": float(max(curve_precision[curve_recall >= 0.80])),
    }
    failures = quality_gate(metrics, config.evaluation.thresholds)
    return EvaluationResult(
        **metrics,
        validation_samples=len(y_valid),
        positive_samples=int(y_valid.sum()),
        negative_samples=int((y_valid == 0).sum()),
        passed=not failures,
        failed_thresholds=failures,
    )


def save_evaluation(result: EvaluationResult, path: Path) -> None:
    """Write the exact measured result for local inspection and tracking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local fraud-scoring candidate")
    parser.add_argument("--config", help="YAML profile; defaults to APP_ENV / CONFIG_PATH")
    parser.add_argument("--data-path", type=Path, help="Override the configured raw CSV path")
    parser.add_argument("--model-path", type=Path, help="Override the configured candidate path")
    args = parser.parse_args()
    config = load_config(args.config)
    data_path = args.data_path or Path(config.data.raw_data_path)
    model_path = args.model_path or Path(config.model.artifact_path)
    candidate = joblib.load(model_path)
    if not isinstance(candidate, CandidateArtifact):
        raise TypeError(f"Artifact at {model_path} is not a fraud-scoring candidate.")
    result = evaluate_candidate(candidate, data_path, config)
    output = Path(config.evaluation.metrics_path)
    save_evaluation(result, output)
    print(json.dumps(asdict(result), indent=2))
    print(f"Saved evaluation to {output}")
    if not result.passed:
        raise SystemExit("Quality gate FAILED: " + "; ".join(result.failed_thresholds))


if __name__ == "__main__":
    main()
