from __future__ import annotations

from dataclasses import dataclass

from sklearn.pipeline import Pipeline


@dataclass
class CandidateArtifact:
    pipeline: Pipeline
    dataset_sha256: str
    feature_names: list[str]
    validation_indices: list[int]