"""Promotion uses the real MLflow run, gate and logged model."""

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from fraud_scoring.config import load_config
from fraud_scoring.generate_data import generate_synthetic_transactions
from fraud_scoring.mlflow_tracking import run_training_experiment
from fraud_scoring.model_registry import PromotionRejected, promote_model


def test_promotion_rejection_alias_movement_and_rollback(tmp_path):
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_transactions(n_records=600, fraud_ratio=0.15, random_seed=17).to_csv(
        data_path, index=False
    )
    base = load_config("configs/dev.yaml", force_reload=True)
    config = base.model_copy(
        update={
            "mlflow": base.mlflow.model_copy(
                update={
                    "tracking_uri": f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
                    "experiment_name": "stage-6-test",
                }
            ),
            "model": base.model.model_copy(
                update={
                    "name": "stage-6-fraud-model",
                    "alias": "champion",
                    "artifact_path": str(tmp_path / "candidate.joblib"),
                }
            ),
            "evaluation": base.evaluation.model_copy(
                update={"metrics_path": str(tmp_path / "metrics.json")}
            ),
        }
    )
    client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)

    first = run_training_experiment(data_path, config)
    assert first.evaluation.passed
    promoted_first = promote_model(config, first.run_id, first.model_uri)
    assert promoted_first.model_name == config.model.name
    assert promoted_first.alias == config.model.alias
    assert promoted_first.previous_version is None
    first_version = client.get_model_version(config.model.name, promoted_first.version)
    assert first_version.run_id == first.run_id
    assert first_version.tags["training_run_id"] == first.run_id
    assert first_version.tags["quality_gate_status"] == "PASS"
    assert str(client.get_model_version_by_alias(config.model.name, config.model.alias).version) == promoted_first.version
    assert mlflow.pyfunc.load_model(f"models:/{config.model.name}@{config.model.alias}")

    failed_thresholds = config.evaluation.thresholds.model_copy(update={"min_roc_auc": 1.0})
    failing_config = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"thresholds": failed_thresholds})}
    )
    failed = run_training_experiment(data_path, failing_config)
    assert not failed.evaluation.passed
    with pytest.raises(PromotionRejected, match="thresholds were not met"):
        promote_model(config, failed.run_id, failed.model_uri)
    assert str(client.get_model_version_by_alias(config.model.name, config.model.alias).version) == promoted_first.version
    assert len(client.search_model_versions(f"name = '{config.model.name}'")) == 1

    second = run_training_experiment(data_path, config)
    assert second.evaluation.passed
    with pytest.raises(PromotionRejected, match="does not belong"):
        promote_model(config, second.run_id, first.model_uri)
    promoted_second = promote_model(config, second.run_id, second.model_uri)
    assert promoted_second.previous_version == promoted_first.version
    assert str(client.get_model_version_by_alias(config.model.name, config.model.alias).version) == promoted_second.version
    assert client.get_model_version(config.model.name, promoted_first.version).run_id == first.run_id
    assert len(client.search_model_versions(f"name = '{config.model.name}'")) == 2

    # Re-promoting an earlier approved artifact moves the alias back without
    # deleting the newer version or creating a duplicate version.
    rollback = promote_model(config, first.run_id, first.model_uri)
    assert rollback.version == promoted_first.version
    assert rollback.previous_version == promoted_second.version
    assert str(client.get_model_version_by_alias(config.model.name, config.model.alias).version) == promoted_first.version
    assert len(client.search_model_versions(f"name = '{config.model.name}'")) == 2
