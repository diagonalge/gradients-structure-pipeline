from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

from loguru import logger

from ds_structure_server import job_store
from ds_structure_server.job_runtime import StructureJobCancelled
from ds_structure_server.job_runtime import clear_cancel
from ds_structure_server.job_runtime import is_cancelled
from ds_structure_server.job_runtime import request_cancel
from ds_structure_server.models import CreateJobRequest
from ds_structure_server.pipeline.service import StructureAlreadyInstructError
from ds_structure_server.pipeline.service import StructureJobConfig
from ds_structure_server.pipeline.service import StructureSourceTooSmallError
from ds_structure_server.pipeline.service import assert_sources_ok_for_structure_job
from ds_structure_server.pipeline.service import run_structure_job
from ds_structure_server.pipeline.service import suggest_structure_for_sources
from ds_structure_server.s3 import upload_file_to_minio

COMBINED_DATASET_KEY = "dataset"
_MAX_JOBS = max(1, int(os.getenv("STRUCTURE_MAX_JOBS", "3")))
_JOB_SLOTS = threading.Semaphore(_MAX_JOBS)
_ACTIVE: set[str] = set()
_ACTIVE_LOCK = threading.Lock()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _on_progress(job_id: uuid.UUID, message: str, counts: dict[str, Any] | None = None) -> None:
    if is_cancelled(job_id):
        raise StructureJobCancelled("cancelled by user")
    level_error = "error" in message.lower() or message.startswith("fail")
    if level_error:
        job_store.append_unique_error(job_id, message)
        payload = dict(counts or {})
        payload.setdefault("stage", "generate")
        job_store.write_status(job_id, **payload)
        return

    stage = (counts or {}).get("stage")
    if stage == "generate" or "accepted " in message:
        fields = dict(counts or {})
        fields.setdefault("stage", "generate")
        job_store.set_progress_line(job_id, message, **fields)
        return

    # Milestone / stage transition
    job_store.add_milestone(job_id, message)
    if counts:
        job_store.write_status(job_id, **counts)


def _run_job(request: CreateJobRequest) -> None:
    job_id = request.job_id
    key = str(job_id)
    work_root = job_store.job_dir(job_id) / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    clear_cancel(job_id)
    bucket = os.getenv("S3_BUCKET_NAME", "gradients")

    acquired = False
    try:
        if not _JOB_SLOTS.acquire(timeout=1):
            raise RuntimeError(f"Structure server at capacity (max {_MAX_JOBS} concurrent jobs)")
        acquired = True
        with _ACTIVE_LOCK:
            _ACTIVE.add(key)

        job_store.write_status(
            job_id,
            status="running",
            started_at=_now(),
            source=request.sources[0],
            num_rows=request.num_rows,
            chunk_level=request.chunk_level,
            stage="validate",
            records=0,
            failures=0,
            goal=request.num_rows,
        )
        _on_progress(job_id, f"validating {len(request.sources)} source(s)", {"stage": "validate", "goal": request.num_rows})

        try:
            assert_sources_ok_for_structure_job(list(request.sources))
        except (StructureAlreadyInstructError, StructureSourceTooSmallError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc

        if is_cancelled(job_id):
            raise StructureJobCancelled("cancelled by user")

        _on_progress(
            job_id,
            f"running pipeline for {request.num_rows} rows",
            {"stage": "generate", "records": 0, "failures": 0, "goal": request.num_rows},
        )

        config = StructureJobConfig(
            source=request.sources[0],
            sources=list(request.sources),
            output_dir=work_root,
            num_rows=request.num_rows,
            personas=list(request.personas) if request.personas else None,
            chunk_level=request.chunk_level,
        )
        result = run_structure_job(
            config,
            on_progress=lambda message, counts=None: _on_progress(job_id, message, counts),
            is_cancelled=lambda: is_cancelled(job_id),
        )

        if is_cancelled(job_id):
            raise StructureJobCancelled("cancelled by user")

        _on_progress(job_id, "uploading training files", {**(result.counts or {}), "stage": "upload", "goal": request.num_rows})
        prefix = f"ds_structure_jobs/{job_id}/god_train"
        dataset_urls: dict[str, str] = {}
        for persona_name, path in result.persona_train_paths.items():
            if is_cancelled(job_id):
                raise StructureJobCancelled("cancelled by user")
            url = upload_file_to_minio(str(path), bucket, f"{prefix}/{path.name}")
            if not url:
                raise RuntimeError(f"Failed to upload training dataset for persona {persona_name!r}")
            dataset_urls[persona_name] = url

        if result.combined_train_path and result.combined_train_path.exists():
            combined_url = upload_file_to_minio(
                str(result.combined_train_path),
                bucket,
                f"{prefix}/{result.combined_train_path.name}",
            )
            if not combined_url:
                raise RuntimeError("Failed to upload combined training dataset")
            dataset_urls[COMBINED_DATASET_KEY] = combined_url
        elif dataset_urls:
            dataset_urls[COMBINED_DATASET_KEY] = next(iter(dataset_urls.values()))

        if COMBINED_DATASET_KEY not in dataset_urls:
            raise RuntimeError("ds-structure job produced no training files")

        personas = [persona.to_dict() for persona in result.personas]
        job_store.finalize_and_cleanup(
            job_id,
            {
                "status": "completed",
                "source": request.sources[0],
                "num_rows": request.num_rows,
                "chunk_level": request.chunk_level,
                "personas": personas,
                "counts": dict(result.counts or {}),
                "dataset_urls": dataset_urls,
                "error": None,
                "progress": f"completed: {result.counts}",
                "completed_at": _now(),
                "stage": "completed",
            },
        )
        logger.info(f"structure job {job_id} completed: {result.counts}")
    except StructureJobCancelled as exc:
        logger.info(f"structure job {job_id} cancelled")
        job_store.finalize_and_cleanup(
            job_id,
            {
                "status": "cancelled",
                "source": request.sources[0],
                "num_rows": request.num_rows,
                "chunk_level": request.chunk_level,
                "error": str(exc),
                "completed_at": _now(),
            },
        )
    except Exception as exc:
        logger.exception(f"structure job {job_id} failed: {exc}")
        job_store.append_unique_error(job_id, f"{type(exc).__name__}: {exc}")
        job_store.finalize_and_cleanup(
            job_id,
            {
                "status": "failed",
                "source": request.sources[0],
                "num_rows": request.num_rows,
                "chunk_level": request.chunk_level,
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at": _now(),
            },
        )
    finally:
        clear_cancel(job_id)
        with _ACTIVE_LOCK:
            _ACTIVE.discard(key)
        if acquired:
            _JOB_SLOTS.release()


def start_job_thread(request: CreateJobRequest) -> None:
    job_store.write_request(request.job_id, request.model_dump(mode="json"))
    job_store.write_status(
        request.job_id,
        status="pending",
        source=request.sources[0],
        num_rows=request.num_rows,
        chunk_level=request.chunk_level,
        goal=request.num_rows,
    )
    thread = threading.Thread(target=_run_job, args=(request,), name=f"structure-{request.job_id}", daemon=True)
    thread.start()


def cancel_job(job_id: uuid.UUID) -> dict[str, Any] | None:
    request_cancel(job_id)
    status = job_store.read_status(job_id)
    if status is None:
        return None
    if status.get("status") in {"completed", "failed", "cancelled"}:
        return job_store.build_public_status(job_id)
    # If not actively running in this process, mark cancelled immediately.
    with _ACTIVE_LOCK:
        active = str(job_id) in _ACTIVE
    if not active:
        return job_store.finalize_and_cleanup(
            job_id,
            {
                **status,
                "status": "cancelled",
                "error": "cancelled by user",
                "completed_at": _now(),
            },
        )
    job_store.write_status(job_id, status="cancelling", error="cancelled by user")
    return job_store.build_public_status(job_id)


def suggest(sources: list[str]) -> Any:
    return suggest_structure_for_sources(sources)
