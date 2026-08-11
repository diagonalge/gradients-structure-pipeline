"""In-process cancel flags + StructureJobCancelled for the pipeline."""

from __future__ import annotations

import threading
from uuid import UUID

_LOCK = threading.Lock()
_CANCELLED: set[str] = set()


def _key(job_id: UUID | str) -> str:
    return str(job_id)


def request_cancel(job_id: UUID | str) -> None:
    with _LOCK:
        _CANCELLED.add(_key(job_id))


def clear_cancel(job_id: UUID | str) -> None:
    with _LOCK:
        _CANCELLED.discard(_key(job_id))


def is_cancelled(job_id: UUID | str) -> bool:
    with _LOCK:
        return _key(job_id) in _CANCELLED


class StructureJobCancelled(RuntimeError):
    """Raised when a running structure job is cancelled."""
