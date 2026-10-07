"""Shared data utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def dataset_sha256(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a dataset file.

    Args:
        path: Dataset file to fingerprint.

    Returns:
        Hexadecimal SHA-256 digest of the file bytes.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()
