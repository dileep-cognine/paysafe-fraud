"""MLflow records real candidate experiments in an isolated local store."""

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fraud_scoring.config import load_config
from fraud_scoring.features import SERVING_FEATURES
from fraud_scoring.generate_data import generate_synthetic_transactions
from fraud_scoring.mlflow_tracking import run_training_experiment


@pytest.fixture
def experiment_setup(tmp_path):
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_transactions(n_records=600, fraud_ratio=0.15, random_seed=17).to_csv(
        data_path, index=False
    )
    config = load_config("configs/dev.yaml", force_reload=True)
    config = config.model_copy(
        update={
            "mlflow": config.mlflow.model_copy(
                update={
                    "tracking_uri": f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}",
                    "experiment_name": "stage-5-test",
                }
            ),
            "model": config.model.model_copy(
                update={"artifact_path": str(tmp_path / "candidate.joblib")}
            ),
            "evaluation": config.evaluation.model_copy(
                update={"metrics_path": str(tmp_path / "metrics.json")}
            ),
        }
    )
    return data_path, config


def test_run_records_model_data_metrics_and_gate(experiment_setup):
    data_path, config = experiment_setup
    outcome = run_training_experiment(data_path, config)
    client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name("stage-5-test")
    assert experiment is not None
    run = client.get_run(outcome.run_id)
    assert run.info.experiment_id == experiment.experiment_id
    assert run.data.params["model_type"] == "LogisticRegression"
    assert run.data.params["model.max_iter"] == "1000"
    assert run.data.metrics["roc_auc"] == outcome.evaluation.roc_auc
    assert run.data.metrics["training_row_count"] == 480
    assert run.data.metrics["validation_row_count"] == 120
    assert run.data.metrics["fraud_count"] + run.data.metrics["non_fraud_count"] == 600
    assert run.data.tags["quality_gate_status"] == ("PASS" if outcome.evaluation.passed else "FAIL")
    assert run.data.tags["dataset_sha256"] == outcome.candidate.dataset_sha256
    assert run.data.tags["dataset_path"] == str(data_path.resolve())
    assert len(run.inputs.dataset_inputs) == 1
    assert client.download_artifacts(outcome.run_id, "evaluation/metrics.json")

    model = mlflow.pyfunc.load_model(outcome.model_uri)
    input_schema = model.metadata.get_input_schema()
    assert [column.name for column in input_schema.inputs] == list(SERVING_FEATURES)
    assert model.metadata.get_output_schema() is not None
    assert model.metadata.saved_input_example_info is not None
    example = pd.read_csv(data_path).loc[[0], list(SERVING_FEATURES)]
    probabilities = model.predict(example)
    assert probabilities.shape == (1, 2)


def test_failed_gate_still_records_run(experiment_setup):
    data_path, config = experiment_setup
    thresholds = config.evaluation.thresholds.model_copy(update={"min_roc_auc": 1.0})
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(update={"thresholds": thresholds})
        }
    )
    outcome = run_training_experiment(data_path, config)
    run = MlflowClient(tracking_uri=config.mlflow.tracking_uri).get_run(outcome.run_id)
    assert not outcome.evaluation.passed
    assert run.data.tags["quality_gate_status"] == "FAIL"
    assert "roc_auc" in run.data.tags["quality_gate_failures"]
    assert run.data.metrics["roc_auc"] == outcome.evaluation.roc_auc
