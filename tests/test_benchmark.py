"""Tests for deterministic latency-summary calculations."""

import pytest

from fraud_scoring import benchmark
from fraud_scoring.benchmark import nearest_rank_percentile, run_benchmark, summarize_latencies


def test_nearest_rank_percentiles_use_end_to_end_samples() -> None:
    """Use the documented nearest-rank convention for benchmark reporting."""
    samples = [11.0, 7.0, 20.0, 13.0, 9.0]

    assert nearest_rank_percentile(samples, 0.50) == 11.0
    assert nearest_rank_percentile(samples, 0.95) == 20.0


def test_latency_summary_rejects_invalid_inputs() -> None:
    """Reject empty samples and invalid concurrency rather than reporting fake metrics."""
    with pytest.raises(ValueError, match="successful request"):
        nearest_rank_percentile([], 0.95)
    with pytest.raises(ValueError, match="Concurrency"):
        summarize_latencies([10.0], 0, 10.0)


def test_benchmark_warms_then_measures_concurrent_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise warm-up and concurrent measurement without requiring a live API."""
    calls: list[str] = []

    def fake_score_once(url: str, payload: dict[str, object], timeout_seconds: float) -> float:
        calls.append(url)
        return 12.5

    monkeypatch.setattr(benchmark, "_score_once", fake_score_once)

    summary, failures = run_benchmark(
        "http://example.test/score",
        {"transaction_id": "test"},
        warmup_requests=2,
        measured_requests=4,
        concurrency=2,
        timeout_seconds=1.0,
    )

    assert calls == ["http://example.test/score"] * 6
    assert failures == ()
    assert summary.request_count == 4
    assert summary.concurrency == 2
    assert summary.p50_ms == 12.5
