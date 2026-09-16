"""Shared timestamp-resolution protocol for temporal point-process models.

Raw event times are stored as UTC Unix seconds.  The released cross-platform
sources do not all expose the same timestamp precision, so the TPP analyses use
completed-minute gaps and conservative minute-censoring intervals.  Keeping the
conversion here prevents the manuscript and individual training scripts from
silently drifting apart.
"""

from __future__ import annotations

import numpy as np


MINUTE_INTERVAL_PROTOCOL = (
    "UTC Unix-second timestamps; exact duplicate timestamps collapsed; "
    "completed-minute gaps d=floor(delta_seconds/60); "
    "likelihood interval (max(d-1, 0), d+1] minutes"
)


def completed_gap_minutes(ts_seconds: np.ndarray) -> np.ndarray:
    """Return non-negative completed-minute gaps between sorted Unix seconds."""

    ts = np.asarray(ts_seconds, dtype=np.int64)
    if ts.ndim != 1:
        raise ValueError("timestamps must be a one-dimensional sequence")
    if ts.size < 2:
        return np.empty(0, dtype=np.int64)
    delta_seconds = np.diff(ts)
    if np.any(delta_seconds < 0):
        raise ValueError("timestamps must be sorted in non-decreasing order")
    return (delta_seconds // 60).astype(np.int64)


def conservative_minute_bounds(
    completed_minutes: np.ndarray, epsilon: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Return the common conservative censoring bounds used by the TPP fits."""

    d = np.asarray(completed_minutes, dtype=np.float64)
    if np.any(d < 0):
        raise ValueError("completed-minute gaps must be non-negative")
    lower = np.maximum(d - 1.0, 0.0)
    lower = np.where(lower == 0.0, epsilon, lower)
    upper = d + 1.0
    return lower, upper
