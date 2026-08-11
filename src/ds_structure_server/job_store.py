"""Filesystem job tracking — no DB. status.json + unique errors + terminal cache."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any
from uuid import UUID

_LOCK = threading.Lock()
_TERMINAL_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TERMINAL_TTL_SEC = float(os.getenv("STRUCTURE_TERMINAL_CACHE_TTL", "600"))
_MAX_UNIQUE_ERRORS = 100


def data_root() -> Path:
    root = Path(os.getenv("STRUCTURE_DATA_DIR", "/tmp/ds-structure-data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_dir(job_id: UUID | str) -> Path:
    path = data_root() / str(job_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def status_path(job_id: UUID | str) -> Path:
    return job_dir(job_id) / "status.json"


def errors_path(job_id: UUID | str) -> Path:
    return job_dir(job_id) / "errors.jsonl"


def request_path(job_id: UUID | str) -> Path:
    return job_dir(job_id) / "request.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_request(job_id: UUID | str, payload: dict[str, Any]) -> None:
    request_path(job_id).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_status(job_id: UUID | str) -> dict[str, Any] | None:
    key = str(job_id)
    path = status_path(job_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with _LOCK:
        cached = _TERMINAL_CACHE.get(key)
        if cached is None:
            return None
        ts, payload = cached
        if time.time() - ts > _TERMINAL_TTL_SEC:
            _TERMINAL_CACHE.pop(key, None)
            return None
        return dict(payload)


def write_status(job_id: UUID | str, **fields: Any) -> dict[str, Any]:
    path = status_path(job_id)
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(fields)
    current["updated_at"] = _now()
    # Keep generate progress as a single overwritten line in status, not a log list.
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return current


def set_progress_line(job_id: UUID | str, message: str, **counts: Any) -> dict[str, Any]:
    """Overwrite the single generate-stage progress line + counters."""
    return write_status(job_id, progress=message, **counts)


def append_unique_error(job_id: UUID | str, message: str) -> None:
    """Persist unique error messages only (dedupe by exact message text)."""
    text = (message or "").strip()
    if not text:
        return
    path = errors_path(job_id)
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line).get("message", ""))
            except json.JSONDecodeError:
                continue
    if text in seen:
        return
    if len(seen) >= _MAX_UNIQUE_ERRORS:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": _now(), "level": "error", "message": text[:2000]}) + "\n")


def load_unique_errors(job_id: UUID | str) -> list[dict[str, Any]]:
    path = errors_path(job_id)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def build_public_status(job_id: UUID | str) -> dict[str, Any] | None:
    status = read_status(job_id)
    if status is None:
        return None
    errors = load_unique_errors(job_id) if status_path(job_id).exists() else status.get("errors") or []
    # Compact logs: milestones + unique errors + single progress line as info.
    logs: list[dict[str, Any]] = list(status.get("milestones") or [])
    progress = status.get("progress")
    if progress:
        logs.append(
            {
                "ts": status.get("updated_at") or _now(),
                "level": "info",
                "message": progress,
            }
        )
    logs.extend(errors)
    payload = dict(status)
    payload["job_id"] = str(job_id)
    payload["logs"] = logs
    payload.pop("milestones", None)
    return payload


def add_milestone(job_id: UUID | str, message: str) -> None:
    path = status_path(job_id)
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    milestones = list(current.get("milestones") or [])
    milestones.append({"ts": _now(), "level": "info", "message": message})
    current["milestones"] = milestones[-20:]
    current["updated_at"] = _now()
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def finalize_and_cleanup(job_id: UUID | str, final_status: dict[str, Any]) -> dict[str, Any]:
    """Write terminal status into cache, then delete on-disk job dir."""
    key = str(job_id)
    errors = load_unique_errors(job_id)
    payload = dict(final_status)
    payload["job_id"] = key
    payload["updated_at"] = _now()
    payload["errors"] = errors
    # Persist final snapshot to disk briefly then wipe.
    write_status(job_id, **{k: v for k, v in payload.items() if k != "job_id"})
    public = build_public_status(job_id) or payload
    with _LOCK:
        _TERMINAL_CACHE[key] = (time.time(), public)
    root = job_dir(job_id)
    shutil.rmtree(root, ignore_errors=True)
    return public
