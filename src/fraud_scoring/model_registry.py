"""Explicit, quality-gated MLflow model registration and alias promotion."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from fraud_scoring.config import AppConfig, load_config
from fraud_scoring.evaluate import quality_gate

GATE_METRICS = (
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "precision_at_recall_80",
)


class PromotionRejected(ValueError):
    """The requested MLflow model is not eligible for the configured alias."""


@dataclass(frozen=True)
class PromotionResult:
    model_name: str
    version: str
    alias: str
    run_id: str
    previous_version: str | None


def _require_eligible_run(client: MlflowClient, run_id: str, config: AppConfig):
    """Reapply Stage 4's quality gate to the run's recorded measurements."""
    try:
        run = client.get_run(run_id)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            raise PromotionRejected(f"MLflow run {run_id} was not found.") from exc
        raise
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None or run.info.experiment_id != experiment.experiment_id:
        raise PromotionRejected("Run does not belong to the configured MLflow experiment.")
    if run.info.status != "FINISHED":
        raise PromotionRejected("Run did not finish successfully.")
    if run.data.tags.get("environment") != config.environment:
        raise PromotionRejected("Run environment differs from the selected configuration.")
    if run.data.tags.get("quality_gate_status") != "PASS":
        raise PromotionRejected(
            "Model promotion rejected because evaluation thresholds were not met."
        )
    missing = [name for name in GATE_METRICS if name not in run.data.metrics]
    if missing:
        raise PromotionRejected(f"Run is missing quality-gate metrics: {missing}.")
    metrics = {name: run.data.metrics[name] for name in GATE_METRICS}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
        raise PromotionRejected("Run contains invalid quality-gate metrics.")
    failures = quality_gate(metrics, config.evaluation.thresholds)
    if failures:
        raise PromotionRejected(
            "Model promotion rejected because evaluation thresholds were not met: "
            + "; ".join(failures)
        )
    return run


def _require_logged_model(client: MlflowClient, model_uri: str, run_id: str) -> str:
    """Accept only the actual Stage 5 logged model belonging to this run."""
    prefix = "models:/"
    if not model_uri.startswith(prefix):
        raise PromotionRejected("Model URI must be a logged-model URI (models:/m-...).")
    model_id = model_uri[len(prefix):]
    if not model_id.startswith("m-") or "/" in model_id or "@" in model_id:
        raise PromotionRejected("Model URI must identify one logged model, not a registry alias.")
    try:
        logged_model = client.get_logged_model(model_id)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            raise PromotionRejected(f"Logged model {model_id} was not found.") from exc
        raise
    if logged_model.source_run_id != run_id or logged_model.status != "READY":
        raise PromotionRejected("Logged model is not ready or does not belong to the requested run.")
    return model_id


def _existing_version(client: MlflowClient, name: str, model_id: str, run_id: str):
    """Reuse an existing version so repeating a promotion does not duplicate it."""
    for version in client.search_model_versions(f"name = '{name}'"):
        if version.source == f"models:/{model_id}" and version.run_id == run_id:
            return version
    return None


def _current_alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise


def promote_model(config: AppConfig, run_id: str, model_uri: str) -> PromotionResult:
    """Register an eligible logged model and point the configured alias to it."""
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
    run = _require_eligible_run(client, run_id, config)
    model_id = _require_logged_model(client, model_uri, run_id)
    name, alias = config.model.name, config.model.alias
    previous_version = _current_alias_version(client, name, alias)

    tags = {
        "environment": config.environment,
        "quality_gate_status": "PASS",
        "training_run_id": run_id,
    }
    model_type = run.data.tags.get("model_type") or run.data.params.get("model_type")
    if model_type:
        tags["model_type"] = model_type
    if run.data.tags.get("git_commit"):
        tags["git_commit"] = run.data.tags["git_commit"]

    version = _existing_version(client, name, model_id, run_id)
    if version is None:
        # Register the Stage 5 artifact itself; no retraining or second model log.
        version = mlflow.register_model(model_uri=model_uri, name=name, tags=tags)
    else:
        for key, value in tags.items():
            client.set_model_version_tag(name, version.version, key, value)

    if version.run_id != run_id:
        raise PromotionRejected("Registered version does not trace back to the requested run.")
    client.set_registered_model_alias(name, alias, version.version)
    return PromotionResult(name, str(version.version), alias, run_id, previous_version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote an eligible MLflow candidate")
    commands = parser.add_subparsers(dest="command", required=True)
    promote = commands.add_parser("promote", help="register and assign the configured alias")
    promote.add_argument("--config", help="YAML profile; defaults to APP_ENV / CONFIG_PATH")
    promote.add_argument("--run-id", required=True, help="finished Stage 5 MLflow run ID")
    promote.add_argument("--model-uri", required=True, help="logged model URI printed by training")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        result = promote_model(config, args.run_id, args.model_uri)
    except PromotionRejected as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Promoted {result.model_name} version {result.version} from run {result.run_id} "
        f"to @{result.alias} (previous version: {result.previous_version or 'none'})."
    )


if __name__ == "__main__":
    main()
