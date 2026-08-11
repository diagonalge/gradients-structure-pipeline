from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from ds_structure_server.auth import ServiceAuth
from ds_structure_server import job_store
from ds_structure_server.models import (
    CreateJobRequest,
    CreateJobResponse,
    JobStatusResponse,
    SuggestRowsRequest,
    SuggestRowsResponse,
    SuggestedPersona,
)
from ds_structure_server.pipeline.service import StructureSourceTooSmallError
from ds_structure_server.worker import cancel_job
from ds_structure_server.worker import start_job_thread
from ds_structure_server.worker import suggest

app = FastAPI(
    title="ds-structure-server",
    version="0.1.0",
    description="Internal structure-generation worker for Gradients",
)


def _to_status_response(job_id: uuid.UUID, payload: dict[str, Any]) -> JobStatusResponse:
    counts = {
        k: payload.get(k)
        for k in ("stage", "records", "failures", "goal", "documents", "skipped", "updated_at")
        if payload.get(k) is not None
    } or None
    if payload.get("counts") and isinstance(payload["counts"], dict):
        counts = {**(counts or {}), **payload["counts"]}
    return JobStatusResponse(
        job_id=job_id,
        status=payload.get("status") or "pending",
        source=payload.get("source"),
        num_rows=payload.get("num_rows"),
        chunk_level=payload.get("chunk_level"),
        personas=payload.get("personas"),
        counts=counts,
        dataset_urls=payload.get("dataset_urls"),
        error=payload.get("error"),
        logs=payload.get("logs"),
        progress=payload.get("progress"),
        updated_at=payload.get("updated_at"),
        started_at=payload.get("started_at"),
        completed_at=payload.get("completed_at"),
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/jobs", response_model=CreateJobResponse)
def create_job(body: CreateJobRequest, _: ServiceAuth) -> CreateJobResponse:
    existing = job_store.read_status(body.job_id)
    if existing and existing.get("status") in {"pending", "running", "cancelling"}:
        return CreateJobResponse(job_id=body.job_id, status=existing.get("status") or "pending", message="already running")
    start_job_thread(body)
    return CreateJobResponse(job_id=body.job_id, status="pending")


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: uuid.UUID, _: ServiceAuth) -> JobStatusResponse:
    payload = job_store.build_public_status(job_id)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _to_status_response(job_id, payload)


@app.post(
    "/v1/jobs/{job_id}/cancel",
    response_model=JobStatusResponse,
    include_in_schema=False,
)
def cancel_job_endpoint(job_id: uuid.UUID, _: ServiceAuth) -> JobStatusResponse:
    payload = cancel_job(job_id)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _to_status_response(job_id, payload)


@app.post("/v1/suggest-rows", response_model=SuggestRowsResponse)
def suggest_rows(body: SuggestRowsRequest, _: ServiceAuth) -> SuggestRowsResponse:
    try:
        result = suggest(list(body.sources))
    except StructureSourceTooSmallError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return SuggestRowsResponse(
        suggested_rows=result.suggested_rows,
        personas=[SuggestedPersona(name=p.name, description=p.description) for p in result.personas],
        page_count=result.page_count,
        already_instruct=result.already_instruct,
        instruction_field=result.instruction_field,
        output_field=result.output_field,
        row_count=result.row_count,
    )


@app.exception_handler(Exception)
async def unhandled(_: Any, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
