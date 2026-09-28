"""Meaningful training and evaluation checks on deterministic synthetic rows."""

import joblib
import numpy as np
import pytest

from fraud_scoring.config import EvaluationThresholds, load_config
from fraud_scoring.evaluate import evaluate_candidate, quality_gate
from fraud_scoring.features import IDENTITY_COLUMN, TARGET_COLUMN, TRAINING_FEATURES
from fraud_scoring.generate_data import generate_synthetic_transactions
from fraud_scoring.train import save_candidate, train_candidate


@pytest.fixture
def training_csv(tmp_path):
    path = tmp_path / "transactions.csv"
    generate_synthetic_transactions(n_records=600, fraud_ratio=0.15, random_seed=17).to_csv(
        path, index=False
    )
    return path


def test_candidate_predicts_and_survives_artifact_reload(training_csv, tmp_path):
    config = load_config("configs/dev.yaml", force_reload=True)
    candidate = train_candidate(training_csv, config)
    artifact_path = tmp_path / "candidate.joblib"
    save_candidate(candidate, artifact_path)
    restored = joblib.load(artifact_path)

    assert restored.feature_names == TRAINING_FEATURES
    assert IDENTITY_COLUMN not in restored.feature_names
    assert TARGET_COLUMN not in restored.feature_names
    assert artifact_path.is_file()
    result = evaluate_candidate(restored, training_csv, config)
    assert result.validation_samples == 120
    assert result.positive_samples + result.negative_samples == 120
    assert all(
        0 <= getattr(result, metric) <= 1
        for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc", "precision_at_recall_80")
    )


def test_repeat_training_has_same_scores(training_csv):
    config = load_config("configs/dev.yaml", force_reload=True)
    first = train_candidate(training_csv, config)
    second = train_candidate(training_csv, config)

    assert first.validation_indices == second.validation_indices
    first_result = evaluate_candidate(first, training_csv, config)
    second_result = evaluate_candidate(second, training_csv, config)
    assert np.isclose(first_result.roc_auc, second_result.roc_auc)
    assert np.isclose(first_result.f1, second_result.f1)


def test_quality_gate_reports_pass_and_failure():
    thresholds = EvaluationThresholds(
        min_precision=0.3,
        min_recall=0.5,
        min_f1=0.4,
        min_roc_auc=0.7,
        min_pr_auc=0.3,
        min_precision_at_recall_80=0.2,
    )
    passing = dict(
        precision=0.4,
        recall=0.6,
        f1=0.48,
        roc_auc=0.8,
        pr_auc=0.5,
        precision_at_recall_80=0.3,
    )
    assert quality_gate(passing, thresholds) == ()
    failing = {**passing, "roc_auc": 0.6, "recall": 0.4}
    failures = quality_gate(failing, thresholds)
    assert len(failures) == 2
    assert any("roc_auc" in failure for failure in failures)
    assert any("recall" in failure for failure in failures)


def test_evaluation_rejects_changed_dataset(training_csv):
    config = load_config("configs/dev.yaml", force_reload=True)
    candidate = train_candidate(training_csv, config)
    with training_csv.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        evaluate_candidate(candidate, training_csv, config)
