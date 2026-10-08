"""Measure end-to-end latency of a running fraud-scoring API instance."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_PAYLOAD: dict[str, object] = {
    "transaction_id": "benchmark-transaction",
    "amount": 125.50,
    "merchant_category": "electronics",
    "hour_of_day": 14,
    "device_risk": 0.23,
}


@dataclass(frozen=True)
class LatencySummary:
    """Measured end-to-end response-time distribution for successful requests."""

    request_count: int
    concurrency: int
    elapsed_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


def nearest_rank_percentile(samples_ms: list[float], percentile: float) -> float:
    """Return a nearest-rank latency percentile from non-empty samples.

    Args:
        samples_ms: Successful request durations in milliseconds.
        percentile: Requested percentile in the inclusive range 0 to 1.

    Returns:
        The selected duration in milliseconds.

    Raises:
        ValueError: If samples are empty, non-finite, or percentile is invalid.
    """
    if not samples_ms:
        raise ValueError("At least one successful request is required.")
    if not 0 <= percentile <= 1:
        raise ValueError("Percentile must be between 0 and 1.")
    if any(not math.isfinite(value) or value < 0 for value in samples_ms):
        raise ValueError("Latency samples must be finite non-negative values.")

    ordered = sorted(samples_ms)
    position = max(1, math.ceil(percentile * len(ordered)))
    return ordered[position - 1]


def summarize_latencies(
    samples_ms: list[float], concurrency: int, elapsed_ms: float
) -> LatencySummary:
    """Build a summary for successful end-to-end scoring requests.

    Args:
        samples_ms: Successful response durations in milliseconds.
        concurrency: Number of simultaneous request workers.
        elapsed_ms: Total wall-clock duration of the measured batch.

    Returns:
        Aggregate count and latency percentiles.

    Raises:
        ValueError: If concurrency or elapsed duration is invalid.
    """
    if concurrency < 1:
        raise ValueError("Concurrency must be at least 1.")
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise ValueError("Elapsed time must be a finite non-negative value.")

    return LatencySummary(
        request_count=len(samples_ms),
        concurrency=concurrency,
        elapsed_ms=elapsed_ms,
        min_ms=min(samples_ms),
        p50_ms=nearest_rank_percentile(samples_ms, 0.50),
        p95_ms=nearest_rank_percentile(samples_ms, 0.95),
        max_ms=max(samples_ms),
    )


def _score_once(url: str, payload: dict[str, object], timeout_seconds: float) -> float:
    """Call `/score` once and validate its response contract."""
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body: Any = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000

    if not isinstance(body, dict) or not isinstance(body.get("risk_score"), (int, float)):
        raise RuntimeError("Response did not contain a numeric risk_score.")
    if not isinstance(body.get("model_version"), str):
        raise RuntimeError("Response did not contain a model_version.")
    return elapsed_ms


def _load_payload(path: Path | None) -> dict[str, object]:
    """Load an optional JSON request body or return the safe sample payload."""
    if path is None:
        return DEFAULT_PAYLOAD.copy()
    loaded: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Payload file must contain a JSON object.")
    return loaded


def run_benchmark(
    url: str,
    payload: dict[str, object],
    warmup_requests: int,
    measured_requests: int,
    concurrency: int,
    timeout_seconds: float,
) -> tuple[LatencySummary, tuple[str, ...]]:
    """Warm the API, issue concurrent requests, and summarize client latency."""
    if warmup_requests < 0 or measured_requests < 1:
        raise ValueError("Warm-up requests must be non-negative and measured requests positive.")
    if concurrency < 1 or timeout_seconds <= 0:
        raise ValueError("Concurrency and timeout must be positive.")

    for _ in range(warmup_requests):
        _score_once(url, payload, timeout_seconds)

    started = time.perf_counter()
    samples_ms: list[float] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_score_once, url, payload, timeout_seconds)
            for _ in range(measured_requests)
        ]
        for future in as_completed(futures):
            try:
                samples_ms.append(future.result())
            except RuntimeError as exc:
                failures.append(str(exc))
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not samples_ms:
        raise RuntimeError("Benchmark failed: no measured requests succeeded.")
    return summarize_latencies(samples_ms, concurrency, elapsed_ms), tuple(failures)


def main() -> None:
    """Run the API latency benchmark and enforce an optional p95 budget."""
    parser = argparse.ArgumentParser(description="Benchmark the live fraud-scoring /score endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:8000/score")
    parser.add_argument("--payload-file", type=Path)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-p95-ms",
        type=float,
        default=200.0,
        help="Fail when p95 exceeds this budget; set 0 to report without a budget.",
    )
    args = parser.parse_args()

    summary, failures = run_benchmark(
        args.url,
        _load_payload(args.payload_file),
        args.warmup_requests,
        args.requests,
        args.concurrency,
        args.timeout_seconds,
    )
    print(json.dumps({"latency": asdict(summary), "failures": list(failures)}, indent=2))
    if failures:
        raise SystemExit(f"Benchmark failed: {len(failures)} request(s) did not succeed.")
    if args.max_p95_ms > 0 and summary.p95_ms > args.max_p95_ms:
        raise SystemExit(
            f"Benchmark failed: p95 {summary.p95_ms:.2f} ms exceeds {args.max_p95_ms:.2f} ms."
        )


if __name__ == "__main__":
    main()
