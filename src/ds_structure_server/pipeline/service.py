from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlparse

import requests
from loguru import logger

from .inference import InferenceBackend, build_default_backend
from .instruct_detect import InstructPairDetection, detect_instruct_pair_from_path
from .loader import (
    _STRUCTURED_SUFFIXES,
    light_document_mode,
    load_hf_documents_isolated,
    load_local_documents,
)
from .models import ChunkLevel, DatasetProfile, Persona, SourceDocument
from .parsing import parse_document
from .pipeline import (
    _persona_train_path,
    apply_chunk_level,
    builtin_persona_by_name,
    estimate_generation_capacity,
    generic_builtin_personas,
    infer_dataset_profile,
    infer_personas,
    infer_personas_from_profile,
    infer_profile_and_extra_personas_fast,
    resolve_personas,
    run_pipeline,
    save_dataset_profile,
    save_personas,
)
from ds_structure_server.job_runtime import StructureJobCancelled

PresetName = Literal["line-analyst", "summarizer", "needle", "detail-researcher"]
BUILTIN_PRESET_NAMES = {"line-analyst", "summarizer", "needle", "detail-researcher"}
COMBINED_TRAIN_NAME = "dataset.jsonl"
DEFAULT_GENERIC_PERSONA_NAMES = ["line-analyst", "summarizer", "needle", "detail-researcher"]
MAX_INFERRED_PERSONAS = 2
MAX_STRUCTURE_SOURCES = 50
MAX_STRUCTURE_TOTAL_BYTES = int(os.getenv("STRUCTURE_MAX_TOTAL_BYTES", str(512 * 1024 * 1024)))
# Suggest / create-check defaults for ~16 vCPU / 16 GiB workers.
SUGGEST_DOC_LIMIT = int(os.getenv("STRUCTURE_SUGGEST_DOC_LIMIT", "64"))
# Calibrated from UKSI (~15 pages / ~27k chars → ~524 capacity) → ~50k chars ≈ 1000 rows.
MIN_STRUCTURE_CHARS = 50_000
MIN_STRUCTURE_ROWS = 1_000
SOURCE_TOO_SMALL_MESSAGE = (
    f"Source too small to generate at least {MIN_STRUCTURE_ROWS} training rows. "
    "Upload a longer document or combine multiple documents."
)
INSTRUCT_TOO_SMALL_MESSAGE = (
    f"Instruct dataset has fewer than {MIN_STRUCTURE_ROWS} training rows. "
    "Upload a larger dataset or combine multiple files."
)
ALREADY_INSTRUCT_CREATE_MESSAGE = (
    "Source is already an instruct/output training dataset. "
    "Use it directly for training instead of running ds-structure."
)


class StructureSourceTooSmallError(ValueError):
    """Raised when the source cannot honestly support MIN_STRUCTURE_ROWS rows."""


class StructureAlreadyInstructError(ValueError):
    """Raised on create when the source is already a supervised instruct dataset."""


def _internal_workers() -> int:
    """Pipeline concurrency; keep internal (env-tunable), not an API field."""
    from ds_structure_server.pipeline.inference import _env_concurrency

    return _env_concurrency(
        "DS_STRUCTURE_WORKERS",
        "STRUCTURE_LLM_MAX_CONCURRENCY",
        "OPENROUTER_MAX_CONCURRENCY",
    )


@dataclass
class StructureJobConfig:
    source: str
    """Primary Hugging Face dataset id (`org/name`) or HTTP(S) URL / local path."""

    output_dir: Path
    num_rows: int = 1000
    personas: list[str] | None = None
    sources: list[str] | None = None
    """Optional multi-source list; when set, all are merged into one training dataset."""

    # Internal knobs (not exposed on the public API).
    source_config: str | None = None
    split: str = "train"
    text_column: str | None = None
    id_column: str | None = None
    title_column: str | None = None
    metadata_columns: tuple[str, ...] = ()
    max_parse_rows: int = 1_000
    persona_sample_docs: int = 50
    chunk_level: ChunkLevel | None = None
    model: str = "Qwen/Qwen3-32B-TEE"
    temperature: float = 0.5
    top_p: float = 0.8
    top_k: int = 20
    seed: int = 42
    workers: int = field(default_factory=_internal_workers)
    max_input_tokens: int = 16_000
    shuffle_buffer: int = 1_000
    schema_sample_rows: int = 32

    def resolved_sources(self) -> list[str]:
        items: list[str] = []
        seen: set[str] = set()
        for raw in list(self.sources or []) + [self.source]:
            value = (raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            items.append(value)
        return items


@dataclass
class StructureJobResult:
    dataset_id: str
    personas: list[Persona]
    dataset_profile: DatasetProfile
    counts: dict[str, int]
    output_dir: Path
    persona_train_paths: dict[str, Path]
    combined_train_path: Path | None
    records_path: Path
    review_path: Path


@dataclass
class SuggestRowsResult:
    suggested_rows: int
    personas: list[Persona]
    page_count: int | None
    pool_sizes: dict[str, int]
    dataset_profile: DatasetProfile | None
    already_instruct: bool = False
    instruction_field: str | None = None
    output_field: str | None = None
    row_count: int | None = None


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _looks_like_hf_dataset(value: str) -> bool:
    if _looks_like_url(value) or Path(value).exists():
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value.strip()))


def download_source_url(url: str, destination_dir: Path, *, index: int = 0) -> Path:
    """Download a remote document into destination_dir and return the local path."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    name = Path(parsed.path).name or "source.bin"
    if not Path(name).suffix:
        name = f"{name}.bin"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    path = destination_dir / safe_name
    if path.exists() or index:
        stem, suffix = path.stem, path.suffix
        path = destination_dir / f"{index:02d}_{stem}{suffix}"
    written = 0
    logger.info("download: start url={}", url)
    try:
        with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "structured-ds/0.1"}) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > MAX_STRUCTURE_TOTAL_BYTES:
                raise ValueError(
                    f"Source exceeds {MAX_STRUCTURE_TOTAL_BYTES // (1024 * 1024)}MB limit: {url}"
                )
            with path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > MAX_STRUCTURE_TOTAL_BYTES:
                        raise ValueError(
                            f"Source exceeds {MAX_STRUCTURE_TOTAL_BYTES // (1024 * 1024)}MB limit: {url}"
                        )
                    handle.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if path.stat().st_size == 0:
        raise ValueError(f"Downloaded source was empty: {url}")
    logger.info("download: done path={} bytes={}", path.name, path.stat().st_size)
    return path


def _copy_local_into(destination_dir: Path, local_path: Path, *, index: int) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", local_path.name) or f"source_{index}"
    target = destination_dir / f"{index:02d}_{safe_name}"
    if local_path.is_dir():
        shutil.copytree(local_path, target, dirs_exist_ok=True)
    else:
        shutil.copy2(local_path, target)
    return target


def _safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    """Extract a zip into destination (no zip-slip). Nested .zip files are copied, not re-extracted."""
    from .loader import _safe_extract_zip as _extract

    return _extract(zip_path, destination)


def _expand_zip_sources(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.casefold() == ".zip":
            out = path.parent / f"{path.stem}_unzipped"
            if out.exists():
                shutil.rmtree(out, ignore_errors=True)
            _safe_extract_zip(path, out)
            expanded.append(out)
        else:
            expanded.append(path)
    return expanded


def materialize_sources(sources: list[str], download_dir: Path) -> Path:
    """Download/copy all sources into download_dir. Returns a file (single) or directory (multi)."""
    if not sources:
        raise ValueError("At least one source is required")
    if len(sources) > MAX_STRUCTURE_SOURCES:
        raise ValueError(f"At most {MAX_STRUCTURE_SOURCES} sources are allowed")
    if any(_looks_like_hf_dataset(source) for source in sources):
        raise ValueError("Hugging Face dataset ids can only be used as a single source via load_source_documents")

    download_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    total_bytes = 0
    for index, source in enumerate(sources):
        if _looks_like_url(source):
            path = download_source_url(source, download_dir, index=index)
        else:
            local = Path(source).expanduser()
            if not local.exists():
                raise ValueError(
                    "source must be a Hugging Face dataset id (org/name), an HTTP(S) URL, or an existing local path"
                )
            path = _copy_local_into(download_dir, local, index=index)
        total_bytes += path.stat().st_size if path.is_file() else sum(
            child.stat().st_size for child in path.rglob("*") if child.is_file()
        )
        if total_bytes > MAX_STRUCTURE_TOTAL_BYTES:
            raise ValueError(
                f"Combined sources exceed {MAX_STRUCTURE_TOTAL_BYTES // (1024 * 1024)}MB limit"
            )
        local_paths.append(path)

    local_paths = _expand_zip_sources(local_paths)
    if len(local_paths) == 1:
        return local_paths[0]
    # Multi-source (or multi-zip): stage everything under download_dir for a unified walk.
    if all(p.parent == download_dir or download_dir in p.parents for p in local_paths):
        return download_dir
    return download_dir


def load_source_documents(
    config: StructureJobConfig,
    *,
    download_dir: Path | None = None,
    limit: int | None = None,
) -> tuple[list[SourceDocument], Any, str]:
    pool_size = limit or max(config.num_rows, config.max_parse_rows, config.persona_sample_docs)
    sources = config.resolved_sources()
    if not sources:
        raise ValueError("At least one source is required")

    target_dir = download_dir or Path(tempfile.mkdtemp(prefix="structure-src-"))

    if len(sources) == 1 and _looks_like_hf_dataset(sources[0]):
        documents, schema = load_hf_documents_isolated(
            sources[0],
            split=config.split,
            config=config.source_config,
            limit=pool_size,
            seed=config.seed,
            shuffle_buffer=config.shuffle_buffer,
            schema_sample_rows=config.schema_sample_rows,
            text_column=config.text_column,
            id_column=config.id_column,
            title_column=config.title_column,
            metadata_columns=config.metadata_columns,
        )
        return documents, schema, sources[0]

    local_path = materialize_sources(sources, target_dir)
    documents, schema = load_local_documents(
        local_path if local_path.is_dir() or len(sources) == 1 else target_dir,
        limit=pool_size,
        seed=config.seed,
        schema_sample_rows=config.schema_sample_rows,
        text_column=config.text_column,
        id_column=config.id_column,
        title_column=config.title_column,
        metadata_columns=config.metadata_columns,
    )
    dataset_id = str(local_path if len(sources) == 1 else target_dir)
    return documents, schema, dataset_id


def _estimate_page_count(documents: list[SourceDocument]) -> int | None:
    if not documents:
        return None
    form_pages = 0
    for document in documents:
        text = document.text or ""
        if "\f" in text:
            form_pages += max(1, text.count("\f") + 1)
    if form_pages:
        return form_pages
    chars = sum(len(doc.text or "") for doc in documents)
    if chars <= 0:
        return None
    return max(1, round(chars / 1800))


def _total_chars(documents: list[SourceDocument]) -> int:
    return sum(len(doc.text or "") for doc in documents)


def _detect_ready_instruct(
    sources: list[str],
    *,
    download_dir: Path,
    local_path: Path | None = None,
) -> InstructPairDetection:
    """Fast path: CSV/JSON/JSONL (or single HF sample) already looks like instruct data."""
    if len(sources) == 1 and _looks_like_hf_dataset(sources[0]):
        # Lightweight sample via isolated loader rows is heavy; skip HF auto-ready for now
        # unless we can peek schema — treat as needs-structure unless columns are forced.
        return InstructPairDetection(already_instruct=False)

    try:
        resolved = local_path if local_path is not None else materialize_sources(sources, download_dir)
    except ValueError as exc:
        # HF-only / path errors mean "not a local structured upload" — treat as needs structuring.
        message = str(exc)
        if "Hugging Face" in message or "existing local path" in message or "At most" in message:
            return InstructPairDetection(already_instruct=False)
        raise

    candidates: list[Path] = []
    if resolved.is_file():
        candidates = [resolved]
    elif resolved.is_dir():
        candidates = sorted(
            child
            for child in resolved.rglob("*")
            if child.is_file() and child.suffix.casefold() in _STRUCTURED_SUFFIXES
        )

    if not candidates:
        return InstructPairDetection(already_instruct=False)

    # Single structured upload (common FE path): decide ready vs needs structuring.
    if len(candidates) == 1 and len(sources) == 1:
        return detect_instruct_pair_from_path(candidates[0])

    # Multiple structured files: ready only if every file is already instruct.
    detections = [detect_instruct_pair_from_path(path) for path in candidates]
    if detections and all(item.already_instruct for item in detections):
        total_rows = sum(int(item.row_count or 0) for item in detections)
        first = detections[0]
        return InstructPairDetection(
            already_instruct=True,
            instruction_field=first.instruction_field,
            output_field=first.output_field,
            row_count=total_rows,
            format=first.format,
        )
    return InstructPairDetection(already_instruct=False)


def _estimate_capacity_from_chars(chars: int, *, n_personas: int) -> int:
    """Calibrated floor: ~50k chars ≈ 1000 rows with the default ~5-persona mix."""
    if chars <= 0:
        return 0
    base = chars / 50.0
    scale = max(1.0, float(n_personas) / 5.0)
    return max(0, int(base * scale))


def _partition_persona_names(names: list[str]) -> tuple[list[str], list[str]]:
    presets: list[str] = []
    custom: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        if key in BUILTIN_PRESET_NAMES:
            presets.append(key)
        else:
            custom.append(name)
    return presets, custom


def build_default_personas(
    backend: InferenceBackend,
    documents: list[SourceDocument],
    *,
    dataset_profile: DatasetProfile | None = None,
    infer_extra: bool = True,
    inferred: list[Persona] | None = None,
) -> list[Persona]:
    """line-analyst + generics, plus up to 2 inferred DS-specific personas."""
    personas = generic_builtin_personas()
    if infer_extra and documents:
        extras = list(inferred or [])
        if not extras:
            try:
                extras = infer_personas(
                    backend,
                    documents,
                    maximum=MAX_INFERRED_PERSONAS,
                    include_builtins=False,
                    dataset_profile=dataset_profile,
                )
            except Exception:
                if dataset_profile is not None:
                    try:
                        extras = infer_personas_from_profile(
                            backend,
                            dataset_profile,
                            maximum=MAX_INFERRED_PERSONAS,
                        )
                    except Exception:
                        extras = []
        seen = {p.name.casefold() for p in personas}
        for persona in extras:
            key = persona.name.casefold()
            if key in seen or key in BUILTIN_PRESET_NAMES:
                continue
            personas.append(persona)
            seen.add(key)
            if sum(1 for p in personas if p.name.casefold() not in BUILTIN_PRESET_NAMES) >= MAX_INFERRED_PERSONAS:
                break
    return personas


def resolve_job_personas(
    backend: InferenceBackend,
    documents: list[SourceDocument],
    persona_names: list[str] | None,
    *,
    dataset_profile: DatasetProfile | None = None,
) -> list[Persona]:
    if not persona_names:
        return build_default_personas(backend, documents, dataset_profile=dataset_profile, infer_extra=True)

    presets, custom_names = _partition_persona_names(persona_names)
    if not presets and not custom_names:
        return build_default_personas(backend, documents, dataset_profile=dataset_profile, infer_extra=True)

    max_personas = len(presets) + len(custom_names)
    personas = resolve_personas(
        backend,
        documents,
        presets=presets,
        names=custom_names,
        max_personas=max_personas,
        infer=False,
        dataset_profile=dataset_profile,
    )
    by_name = {persona.name.casefold(): persona for persona in personas}
    ordered: list[Persona] = []
    for raw in persona_names:
        key = raw.strip().casefold()
        persona = by_name.get(key) or builtin_persona_by_name(key)
        if persona is not None and persona.name.casefold() not in {p.name.casefold() for p in ordered}:
            ordered.append(persona)
    for persona in personas:
        if persona.name.casefold() not in {p.name.casefold() for p in ordered}:
            ordered.append(persona)
    return ordered or personas


def suggest_structure_for_sources(
    sources: list[str],
    *,
    backend: InferenceBackend | None = None,
) -> SuggestRowsResult:
    """
    Analyse source(s): either already-instruct (200-ready flag) or structure capacity + personas.

    Raises StructureSourceTooSmallError when already-instruct row_count or unstructured
    capacity would be below MIN_STRUCTURE_ROWS.
    """
    cleaned = [str(item).strip() for item in sources if str(item).strip()]
    if not cleaned:
        raise ValueError("At least one source is required")
    if len(cleaned) > MAX_STRUCTURE_SOURCES:
        raise ValueError(f"At most {MAX_STRUCTURE_SOURCES} sources are allowed")

    logger.info("suggest: start sources={} count={}", cleaned[:3], len(cleaned))
    work = Path(tempfile.mkdtemp(prefix="structure-suggest-"))
    try:
        with light_document_mode(True):
            config = StructureJobConfig(
                source=cleaned[0],
                sources=cleaned,
                output_dir=work,
                num_rows=1,
            )
            src_dir = work / "src"
            doc_limit = max(SUGGEST_DOC_LIMIT, len(cleaned))

            if len(cleaned) == 1 and _looks_like_hf_dataset(cleaned[0]):
                logger.info("suggest: loading HF dataset limit={}", doc_limit)
                ready = InstructPairDetection(already_instruct=False)
                documents, _, _ = load_source_documents(
                    config,
                    download_dir=src_dir,
                    limit=doc_limit,
                )
            else:
                logger.info("suggest: materializing sources once")
                local_path = materialize_sources(cleaned, src_dir)
                logger.info("suggest: detecting already-instruct format")
                ready = _detect_ready_instruct(
                    cleaned,
                    download_dir=src_dir,
                    local_path=local_path,
                )
                if ready.already_instruct:
                    rows = int(ready.row_count or 0)
                    logger.info(
                        "suggest: already-instruct rows={} instruction={} output={}",
                        rows,
                        ready.instruction_field,
                        ready.output_field,
                    )
                    if rows < MIN_STRUCTURE_ROWS:
                        raise StructureSourceTooSmallError(INSTRUCT_TOO_SMALL_MESSAGE)
                    return SuggestRowsResult(
                        suggested_rows=rows,
                        personas=[],
                        page_count=None,
                        pool_sizes={},
                        dataset_profile=None,
                        already_instruct=True,
                        instruction_field=ready.instruction_field,
                        output_field=ready.output_field,
                        row_count=rows,
                    )
                logger.info("suggest: loading local documents limit={}", doc_limit)
                load_root = local_path if local_path.is_dir() or len(cleaned) == 1 else src_dir
                documents, _ = load_local_documents(
                    load_root,
                    limit=doc_limit,
                    seed=config.seed,
                    schema_sample_rows=config.schema_sample_rows,
                )

            if not documents:
                raise StructureSourceTooSmallError(SOURCE_TOO_SMALL_MESSAGE)

            total_chars = _total_chars(documents)
            logger.info("suggest: loaded docs={} chars={}", len(documents), total_chars)
            if total_chars < MIN_STRUCTURE_CHARS:
                raise StructureSourceTooSmallError(SOURCE_TOO_SMALL_MESSAGE)

            analysis = documents[: min(2, len(documents))]
            engine = backend or build_default_backend(
                config.model,
                max_input_tokens=config.max_input_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                seed=config.seed,
            )

            # One LLM skim call for profile + DS personas (quality-critical path).
            logger.info(
                "suggest: inferring dataset profile + personas from {} analysis doc(s)",
                len(analysis),
            )
            try:
                profile, extras = infer_profile_and_extra_personas_fast(
                    engine,
                    analysis,
                    max_inferred=MAX_INFERRED_PERSONAS,
                )
                logger.info("suggest: fast profile ok extras={}", len(extras))
            except Exception as exc:
                logger.warning("suggest: fast profile failed ({}), falling back", type(exc).__name__)
                profile = infer_dataset_profile(engine, analysis, sample_docs=1)
                extras = []
                try:
                    extras = infer_personas_from_profile(
                        engine, profile, maximum=MAX_INFERRED_PERSONAS
                    )
                except Exception as persona_exc:
                    logger.warning(
                        "suggest: profile-persona fallback failed ({})",
                        type(persona_exc).__name__,
                    )
                    extras = []

            personas = build_default_personas(
                engine,
                analysis,
                dataset_profile=profile,
                infer_extra=True,
                inferred=extras,
            )
            logger.info(
                "suggest: personas={} names={}",
                len(personas),
                [p.name for p in personas],
            )

            # Parse only the analysis skim docs, then extrapolate capacity by char ratio.
            # Avoids building full chunk pools for every loaded document.
            sample_capacity = 0
            pool_sizes: dict[str, int] = {}
            for document in analysis:
                try:
                    tree = parse_document(document)
                except Exception as exc:
                    logger.warning(
                        "suggest: parse failed doc={} ({})",
                        document.doc_id,
                        type(exc).__name__,
                    )
                    continue
                part_capacity, part_pools = estimate_generation_capacity(
                    tree, personas, dataset_profile=profile
                )
                sample_capacity += part_capacity
                for name, size in part_pools.items():
                    pool_sizes[name] = pool_sizes.get(name, 0) + size
                del tree

            analysis_chars = max(1, _total_chars(analysis))
            extrapolated = int(sample_capacity * (total_chars / analysis_chars))
            char_floor = _estimate_capacity_from_chars(total_chars, n_personas=len(personas))
            capacity = max(extrapolated, char_floor, sample_capacity)
            logger.info(
                "suggest: capacity sample={} extrapolated={} char_floor={} final={}",
                sample_capacity,
                extrapolated,
                char_floor,
                capacity,
            )

            if capacity < MIN_STRUCTURE_ROWS:
                raise StructureSourceTooSmallError(SOURCE_TOO_SMALL_MESSAGE)

            page_count = _estimate_page_count(documents)
            logger.info(
                "suggest: done suggested_rows={} page_count={} pool_sizes={}",
                capacity,
                page_count,
                pool_sizes,
            )
            return SuggestRowsResult(
                suggested_rows=int(capacity),
                personas=personas,
                page_count=page_count,
                pool_sizes=pool_sizes,
                dataset_profile=profile,
                already_instruct=False,
                row_count=None,
            )
    except Exception as exc:
        logger.exception("suggest: failed ({})", type(exc).__name__)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


def suggest_structure_for_source(source: str, *, backend: InferenceBackend | None = None) -> SuggestRowsResult:
    """Backward-compatible single-source suggest."""
    return suggest_structure_for_sources([source], backend=backend)


def suggest_rows_for_source(source: str) -> int:
    """Backward-compatible helper: return only the honest suggested row count."""
    return suggest_structure_for_source(source).suggested_rows


def assert_sources_ok_for_structure_job(sources: list[str]) -> None:
    """Fail-fast checks for create: refuse already-instruct and too-small sources."""
    cleaned = [str(item).strip() for item in sources if str(item).strip()]
    if not cleaned:
        raise ValueError("At least one source is required")
    work = Path(tempfile.mkdtemp(prefix="structure-create-check-"))
    try:
        ready = _detect_ready_instruct(cleaned, download_dir=work / "detect")
        if ready.already_instruct:
            raise StructureAlreadyInstructError(ALREADY_INSTRUCT_CREATE_MESSAGE)

        config = StructureJobConfig(source=cleaned[0], sources=cleaned, output_dir=work, num_rows=1)
        documents, _, _ = load_source_documents(
            config,
            download_dir=work / "src",
            limit=max(SUGGEST_DOC_LIMIT, len(cleaned)),
        )
        if not documents or _total_chars(documents) < MIN_STRUCTURE_CHARS:
            raise StructureSourceTooSmallError(SOURCE_TOO_SMALL_MESSAGE)

        # Cheap capacity floor using default personas (no LLM) — parse one at a time.
        personas = generic_builtin_personas()
        capacity = 0
        for document in documents:
            try:
                tree = parse_document(document)
            except Exception:
                continue
            part, _ = estimate_generation_capacity(tree, personas, dataset_profile=None)
            capacity += part
            del tree
        if capacity < MIN_STRUCTURE_ROWS:
            raise StructureSourceTooSmallError(SOURCE_TOO_SMALL_MESSAGE)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _combined_train_path(output_dir: Path) -> Path:
    return output_dir / "god_train" / COMBINED_TRAIN_NAME


def run_structure_job(
    config: StructureJobConfig,
    *,
    backend: InferenceBackend | None = None,
    on_progress: Callable[[str, dict[str, Any] | None], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> StructureJobResult:
    """Programmatic entry point used by the GOD API job worker."""
    if config.num_rows < 1:
        raise ValueError("num_rows must be at least 1")

    def progress(message: str, counts: dict[str, Any] | None = None) -> None:
        if is_cancelled is not None and is_cancelled():
            raise StructureJobCancelled("cancelled by user")
        stage = (counts or {}).get("stage")
        if on_progress is not None:
            on_progress(message, counts)
        else:
            logger.info("generate[{}]: {}", stage or "job", message)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = output_dir / "source_download"

    logger.info(
        "generate: start source={} num_rows={} workers={} chunk_level={} personas={}",
        config.source,
        config.num_rows,
        config.workers,
        config.chunk_level,
        config.personas,
    )
    progress("loading source documents", {"stage": "download", "goal": config.num_rows})
    documents, schema, dataset_id = load_source_documents(config, download_dir=download_dir)
    if not documents:
        raise ValueError("No non-empty documents were found for the given source")
    progress(
        f"loaded {len(documents)} document(s)",
        {"stage": "download", "documents": len(documents), "goal": config.num_rows},
    )

    (output_dir / "source_schema.json").write_text(json.dumps(schema.to_dict(), indent=2) + "\n")

    # Brief skim only — do not sample dozens of docs for profile/persona setup.
    analysis_documents = documents[: max(1, min(len(documents), 2))]
    engine = backend or build_default_backend(
        config.model,
        max_input_tokens=config.max_input_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        seed=config.seed,
    )

    progress("resolving personas / dataset profile", {"stage": "profile", "goal": config.num_rows})
    if not config.personas:
        try:
            logger.info("generate: inferring profile + personas (fast path)")
            dataset_profile, extras = infer_profile_and_extra_personas_fast(
                engine,
                analysis_documents,
                max_inferred=MAX_INFERRED_PERSONAS,
            )
            personas = build_default_personas(
                engine,
                analysis_documents,
                dataset_profile=dataset_profile,
                infer_extra=True,
                inferred=extras,
            )
        except Exception as exc:
            logger.warning("generate: fast profile failed ({}), falling back", type(exc).__name__)
            dataset_profile = infer_dataset_profile(engine, analysis_documents, sample_docs=1)
            personas = resolve_job_personas(
                engine,
                analysis_documents,
                None,
                dataset_profile=dataset_profile,
            )
    else:
        logger.info("generate: resolving requested personas={}", config.personas)
        dataset_profile = infer_dataset_profile(engine, analysis_documents, sample_docs=1)
        personas = resolve_job_personas(
            engine,
            analysis_documents,
            config.personas,
            dataset_profile=dataset_profile,
        )
    save_dataset_profile(output_dir / "dataset_profile.json", dataset_profile)
    if config.chunk_level:
        personas = apply_chunk_level(personas, config.chunk_level)
    save_personas(output_dir / "personas.json", personas)
    progress(
        f"using {len(personas)} persona(s): {', '.join(p.name for p in personas)}",
        {
            "stage": "profile",
            "goal": config.num_rows,
            "personas": [p.to_dict() for p in personas],
        },
    )

    progress(
        f"generating up to {config.num_rows} rows",
        {"stage": "generate", "records": 0, "failures": 0, "goal": config.num_rows},
    )
    counts = run_pipeline(
        engine,
        documents,
        personas,
        output_dir,
        dataset_id=dataset_id,
        examples_per_doc=1,
        seed=config.seed,
        resume=False,
        all_personas_per_doc=True,
        dataset_profile=dataset_profile,
        workers=config.workers,
        verify_grounding=False,
        resolve_references=True,
        write_review=False,
        target_rows=config.num_rows,
        on_progress=on_progress,
        is_cancelled=is_cancelled,
    )

    logger.info("generate: combining persona train files")
    persona_train_paths = {
        persona.name: path
        for persona in personas
        if (path := _persona_train_path(output_dir, persona.name)).exists()
    }
    logger.info("generate: persona train files={}", {name: str(path) for name, path in persona_train_paths.items()})

    combined_path = _combined_train_path(output_dir)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("wb") as handle:
        wrote = False
        for path in persona_train_paths.values():
            if not path.exists() or path.stat().st_size == 0:
                continue
            with path.open("rb") as src:
                shutil.copyfileobj(src, handle, length=64 * 1024)
            if handle.tell() > 0:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            wrote = True
    if not wrote:
        combined_path = None

    progress(f"generation finished: {counts}", {**counts, "stage": "generate", "goal": config.num_rows})
    logger.info("generate: done dataset_id={} counts={} combined={}", dataset_id, counts, combined_path)
    return StructureJobResult(
        dataset_id=dataset_id,
        personas=personas,
        dataset_profile=dataset_profile,
        counts=counts,
        output_dir=output_dir,
        persona_train_paths=persona_train_paths,
        combined_train_path=combined_path,
        records_path=output_dir / "records.jsonl",
        review_path=output_dir / "review.jsonl",
    )
