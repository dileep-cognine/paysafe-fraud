"""Shared data utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def dataset_sha256(path: Path) -> str:
    """Fingerprint the exact dataset bytes used by training/evaluation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()