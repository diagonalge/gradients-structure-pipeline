from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

ChunkLevel = Literal["line", "section", "paragraph", "page"]
JobStatus = Literal["pending", "running", "cancelling", "completed", "failed", "cancelled"]


class CreateJobRequest(BaseModel):
    job_id: UUID
    sources: list[str] = Field(..., min_length=1, max_length=50)
    num_rows: int = Field(..., ge=1)
    personas: list[str] | None = None
    chunk_level: ChunkLevel | None = None


class CreateJobResponse(BaseModel):
    job_id: UUID
    status: JobStatus = "pending"
    message: str = "accepted"


class SuggestRowsRequest(BaseModel):
    sources: list[str] = Field(..., min_length=1, max_length=50)


class SuggestedPersona(BaseModel):
    name: str
    description: str


class SuggestRowsResponse(BaseModel):
    suggested_rows: int
    personas: list[SuggestedPersona] = Field(default_factory=list)
    page_count: int | None = None
    already_instruct: bool = False
    instruction_field: str | None = None
    output_field: str | None = None
    row_count: int | None = None


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    source: str | None = None
    num_rows: int | None = None
    chunk_level: str | None = None
    personas: list[dict[str, Any]] | None = None
    counts: dict[str, Any] | None = None
    dataset_urls: dict[str, str] | None = None
    error: str | None = None
    logs: list[dict[str, Any]] | None = None
    progress: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
