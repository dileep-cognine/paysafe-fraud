"""Record one validated training and evaluation lifecycle as an MLflow run."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature
from sklearn.metrics import confusion_matrix

from fraud_scoring.artifacts import CandidateArtifact
from fraud_scoring.config import AppConfig, load_config
from fraud_scoring.data_utils import dataset_sha256
from fraud_scoring.data_validation import validate_and_enforce
from fraud_scoring.evaluate import EvaluationResult, evaluate_candidate, save_evaluation
from fraud_scoring.features import (
    TARGET_COLUMN,
    build_serving_features,
    build_training_features,
)
from fraud_scoring.train import save_candidate, train_candidate


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


def _dvc_hash(data_path: Path) -> str | None:
    """Read the real content hash from the DVC pointer beside a tracked dataset."""
    pointer_path = Path(f"{data_path}.dvc")
    if not pointer_path.exists():
        return None

    try:
        metadata = yaml.safe_load(pointer_path.read_text(encoding="utf-8")) or {}
        outputs = metadata.get("outs", [])
        if not outputs or not isinstance(outputs[0], dict):
            return None

        digest = outputs[0].get("md5")
        return digest if isinstance(digest, str) and digest else None

    except (
        OSError,
        yaml.YAMLError,
        TypeError,
        AttributeError,
    ):
        return None


def _log_dataset(
    raw: pd.DataFrame,
    data_path: Path,
    candidate: CandidateArtifact,
) -> None:
    source = str(data_path.resolve())

    mlflow.log_input(
        from_pandas(
            raw,
            source=source,
            targets=TARGET_COLUMN,
            name=data_path.name,
        ),
        context="training",
    )

    tags = {
        "dataset_path": source,
        "dataset_sha256": candidate.dataset_sha256,
        "dataset_size_bytes": str(data_path.stat().st_size),
        "dataset_row_count": str(len(raw)),
    }

    dvc_hash = _dvc_hash(data_path)

    if dvc_hash is not None:
        tags["dataset_dvc_hash"] = dvc_hash

    mlflow.set_tags(tags)


def _log_evaluation_artifacts(candidate: CandidateArtifact, raw: pd.DataFrame) -> None:
    """Log the exact feature contract and validation confusion matrix."""
    built = build_training_features(raw)
    validation_indices = candidate.validation_indices
    y_valid = built.target.take(validation_indices)
    predictions = candidate.pipeline.predict(built.features.take(validation_indices))
    matrix = confusion_matrix(y_valid, predictions, labels=[0, 1]).tolist()

    with tempfile.TemporaryDirectory() as temporary_directory:
        artifact_root = Path(temporary_directory)
        feature_path = artifact_root / "feature_contract.json"
        matrix_path = artifact_root / "confusion_matrix.json"
        feature_path.write_text(
            json.dumps({"model_features": candidate.feature_names}, indent=2) + "\n",
            encoding="utf-8",
        )
        matrix_path.write_text(
            json.dumps(
                {
                    "labels": [0, 1],
                    "rows": "actual class",
                    "columns": "predicted class",
                    "matrix": matrix,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(feature_path), artifact_path="model")
        mlflow.log_artifact(str(matrix_path), artifact_path="evaluation")


def _log_model(candidate: CandidateArtifact, raw: pd.DataFrame) -> str:
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
    model_uri = model_info.model_uri
    if not isinstance(model_uri, str):
        raise TypeError("MLflow did not return a model URI for the logged candidate.")
    return model_uri


def run_training_experiment(data_path: Path, config: AppConfig) -> ExperimentOutcome:
    """Train, evaluate, and track the real candidate, including a failed gate."""
    configure_tracking(config)
    with mlflow.start_run(run_name="baseline-logistic-regression") as run:
        mlflow.set_tags(
            {
                "project": "paysafe-fraud-scoring",
                "environment": config.environment,
                "model_type": config.model.algorithm,
            }
        )
        commit = _git_commit()
        if commit:
            mlflow.set_tag("git_commit", commit)

        candidate = train_candidate(data_path, config)
        save_candidate(candidate, Path(config.model.artifact_path))
        result = evaluate_candidate(candidate, data_path, config)
        metrics_path = Path(config.evaluation.metrics_path)
        save_evaluation(result, metrics_path)

        if dataset_sha256(data_path) != candidate.dataset_sha256:
            raise ValueError("Dataset changed during experiment logging.")
        raw = validate_and_enforce(pd.read_csv(data_path), is_training=True)
        _log_parameters(config, candidate)
        _log_results(result, raw)
        _log_dataset(raw, data_path, candidate)
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        _log_evaluation_artifacts(candidate, raw)
        model_uri = _log_model(candidate, raw)
        return ExperimentOutcome(run.info.run_id, model_uri, candidate, result)


def main() -> None:
    """Run one explicitly requested, real MLflow training experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Train and track a fraud-scoring candidate")
    parser.add_argument("--config", help="YAML profile; defaults to APP_ENV / CONFIG_PATH")
    parser.add_argument("--data-path", type=Path, help="Override the configured CSV path")
    args = parser.parse_args()
    config = load_config(args.config)
    data_path = args.data_path or Path(config.data.raw_data_path)
    outcome = run_training_experiment(data_path, config)
    print(f"MLflow run: {outcome.run_id}")
    print(f"Logged model: {outcome.model_uri}")
    if not outcome.evaluation.passed:
        raise SystemExit("Quality gate FAILED: " + "; ".join(outcome.evaluation.failed_thresholds))


if __name__ == "__main__":
    main()
