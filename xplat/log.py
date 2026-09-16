"""Elapsed-time logger shared by every experiment."""
from __future__ import annotations

import time

_T0 = time.time()


def reset_timer() -> None:
    global _T0
    _T0 = time.time()


def log(*parts) -> None:
    msg = " ".join(str(p) for p in parts)
    print(f"[{(time.time() - _T0) / 60:6.1f} min] {msg}", flush=True)
