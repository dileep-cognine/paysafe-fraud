from __future__ import annotations

from dataclasses import dataclass

from sklearn.pipeline import Pipeline


@dataclass
class CandidateArtifact:
    """Fitted candidate and metadata required for repeatable evaluation."""

    pipeline: Pipeline
    dataset_sha256: str
    feature_names: tuple[str, ...]
    validation_indices: list[int]
