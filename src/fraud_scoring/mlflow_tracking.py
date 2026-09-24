"""Record one validated training and evaluation lifecycle as an MLflow run."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from fraud_scoring.config import AppConfig
from fraud_scoring.data_validation import validate_and_enforce
from fraud_scoring.evaluate import EvaluationResult, evaluate_candidate, save_evaluation
from fraud_scoring.features import TARGET_COLUMN, build_serving_features
from fraud_scoring.train import CandidateArtifact, dataset_sha256, save_candidate, train_candidate


@dataclass(frozen=True)
class ExperimentOutcome:
    run_id: str
    model_uri: str
    candidate: CandidateArtifact
    evaluation: EvaluationResult


def configure_tracking(config: AppConfig) -> str:
    """Select the configured store and create or select its experiment."""
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    return mlflow.set_experiment(config.mlflow.experiment_name).experiment_id


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and len(commit) == 40 else None


def _log_parameters(config: AppConfig, candidate: CandidateArtifact) -> None:
    params = {
        "model_type": config.model.algorithm,
        "random_seed": config.data.random_state,
        "validation_split": config.data.test_size,
        "feature_names": ",".join(candidate.feature_names),
        "preprocessing.numeric": "StandardScaler",
        "preprocessing.merchant_category": "OneHotEncoder(handle_unknown=ignore)",
    }
    params.update({f"model.{key}": value for key, value in config.model.parameters.items()})
    mlflow.log_params(params)


def _log_results(result: EvaluationResult, raw: pd.DataFrame) -> None:
    measured = asdict(result)
    mlflow.log_metrics(
        {
            key: value
            for key, value in measured.items()
            if key not in {"passed", "failed_thresholds"} and isinstance(value, (int, float))
        }
    )
    mlflow.log_metrics(
        {
            "training_row_count": len(raw) - result.validation_samples,
            "validation_row_count": result.validation_samples,
            "fraud_count": int(raw[TARGET_COLUMN].sum()),
            "non_fraud_count": int((raw[TARGET_COLUMN] == 0).sum()),
        }
    )
    mlflow.set_tag("quality_gate_status", "PASS" if result.passed else "FAIL")
    if result.failed_thresholds:
        mlflow.set_tag("quality_gate_failures", "; ".join(result.failed_thresholds))


def _log_dataset(raw: pd.DataFrame, data_path: Path, candidate: CandidateArtifact) -> None:
    source = str(data_path.resolve())
    mlflow.log_input(
        mlflow.data.from_pandas(
            raw,
            source=source,
            targets=TARGET_COLUMN,
            name=data_path.name,
        ),
        context="training",
    )
    mlflow.set_tags(
        {
            "dataset_path": source,
            "dataset_sha256": candidate.dataset_sha256,
            "dataset_size_bytes": str(data_path.stat().st_size),
            "dataset_row_count": str(len(raw)),
        }
    )


def _log_model(candidate: CandidateArtifact, raw: pd.DataFrame) -> str:
    # Use the shared serving builder: identity and target never reach the model.
    example = build_serving_features(raw.drop(columns=[TARGET_COLUMN]).iloc[[0]])
    signature = infer_signature(example, candidate.pipeline.predict_proba(example))
    model_info = mlflow.sklearn.log_model(
        candidate.pipeline,
        name="candidate",
        signature=signature,
        input_example=example,
        pyfunc_predict_fn="predict_proba",
        serialization_format="cloudpickle",
    )
    return model_info.model_uri


def run_training_experiment(data_path: Path, config: AppConfig) -> ExperimentOutcome:
    """Train, evaluate, and track the real candidate, including a failed gate."""
    configure_tracking(config)
    with mlflow.start_run(run_name="baseline-logistic-regression") as run:
        mlflow.set_tags({
            "project": "paysafe-fraud-scoring",
            "environment": config.environment,
            "model_type": config.model.algorithm,
        })
        commit = _git_commit()
        if commit:
            mlflow.set_tag("git_commit", commit)

        candidate = train_candidate(data_path, config)
        save_candidate(candidate, Path(config.model.artifact_path))
        result = evaluate_candidate(candidate, data_path, config)
        metrics_path = Path(config.evaluation.metrics_path)
        save_evaluation(result, metrics_path)

        # Read only the already validated dataset, and verify it still matches
        # the candidate before recording its reference and input example.
        if dataset_sha256(data_path) != candidate.dataset_sha256:
            raise ValueError("Dataset changed during experiment logging.")
        raw = validate_and_enforce(pd.read_csv(data_path), is_training=True)
        _log_parameters(config, candidate)
        _log_results(result, raw)
        _log_dataset(raw, data_path, candidate)
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        model_uri = _log_model(candidate, raw)
        return ExperimentOutcome(run.info.run_id, model_uri, candidate, result)
