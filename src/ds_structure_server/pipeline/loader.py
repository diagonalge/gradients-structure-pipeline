from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from csv import DictReader
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger

from .models import SourceDocument


@dataclass(frozen=True)
class DatasetPreset:
    dataset_id: str
    text_column: str
    id_column: str | None = None
    title_column: str | None = None
    config: str | None = None


@dataclass(frozen=True)
class DiscoveredSchema:
    source: str
    source_kind: str
    text_column: str
    id_column: str | None
    title_column: str | None
    metadata_columns: tuple[str, ...]
    sampled_rows: int
    confidence: float
    text_candidates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_kind": self.source_kind,
            "text_column": self.text_column,
            "id_column": self.id_column,
            "title_column": self.title_column,
            "metadata_columns": list(self.metadata_columns),
            "sampled_rows": self.sampled_rows,
            "confidence": self.confidence,
            "text_candidates": list(self.text_candidates),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DiscoveredSchema:
        return cls(
            source=value["source"],
            source_kind=value["source_kind"],
            text_column=value["text_column"],
            id_column=value.get("id_column"),
            title_column=value.get("title_column"),
            metadata_columns=tuple(value.get("metadata_columns", [])),
            sampled_rows=value["sampled_rows"],
            confidence=value["confidence"],
            text_candidates=tuple(value.get("text_candidates", [])),
        )


PRESETS: dict[str, DatasetPreset] = {
    "open-patients": DatasetPreset(
        dataset_id="abuhoraira06/Open-Patients",
        text_column="description",
        id_column="_id",
    ),
    "technetium": DatasetPreset(
        dataset_id="temlm-foundation/Technetium-I",
        text_column="text",
        id_column="note_id",
        title_column="note_type",
    ),
    "canadian-caselaw": DatasetPreset(
        dataset_id="a2aj/canadian-case-law",
        text_column="unofficial_text_en",
        id_column="citation_en",
        title_column="name_en",
    ),
    "bailii": DatasetPreset(
        dataset_id="easymn/bailii_260505_raw",
        text_column="html_content",
        id_column="path",
        title_column="title",
    ),
    "cold-cases": DatasetPreset(
        dataset_id="harvard-lil/cold-cases",
        text_column="opinions[].opinion_text",
        id_column="id",
        title_column="case_name",
    ),
}


def nested_values(row: dict[str, Any], path: str | None) -> list[Any]:
    if not path:
        return []
    values: list[Any] = [row]
    for raw_part in path.split("."):
        part = raw_part.removesuffix("[]")
        next_values: list[Any] = []
        for value in values:
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and part in item:
                        next_values.append(item[part])
            elif isinstance(value, dict) and part in value:
                next_values.append(value[part])
        values = [item for value in next_values for item in value] if raw_part.endswith("[]") else next_values
    return values


def nested_value(row: dict[str, Any], path: str | None) -> Any:
    values = nested_values(row, path)
    return values[0] if values else None


def _joined_text(row: dict[str, Any], path: str) -> str:
    values = [value.strip() for value in nested_values(row, path) if isinstance(value, str) and value.strip()]
    return "\n\n".join(dict.fromkeys(values))


def _flatten_scalars(value: Any, path: str = "", depth: int = 0) -> Iterator[tuple[str, Any]]:
    if depth > 8:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _flatten_scalars(child, child_path, depth + 1)
    elif isinstance(value, list):
        list_path = f"{path}[]"
        for child in value[:8]:
            yield from _flatten_scalars(child, list_path, depth + 1)
    elif value is not None and path:
        yield path, value


def _path_tokens(path: str) -> set[str]:
    normalized = path.casefold().replace("[]", "").replace("-", "_")
    return {part for component in normalized.split(".") for part in component.split("_") if part}


def _text_name_score(path: str) -> float:
    leaf = path.casefold().replace("[]", "").split(".")[-1]
    exact = {
        "opinion_text": 14,
        "full_text": 14,
        "document_text": 14,
        "html_content": 13,
        "unofficial_text_en": 13,
        "text": 12,
        "content": 11,
        "body": 10,
        "document": 10,
        "description": 8,
        "opinion": 8,
        "transcript": 8,
        "article": 7,
        "summary": 4,
        "syllabus": 4,
        "headnotes": 3,
    }
    score = exact.get(leaf, 0)
    tokens = _path_tokens(path)
    if tokens & {"id", "uuid", "slug", "url", "path", "date", "name", "title", "citation"}:
        score -= 9
    return score


def _rank_text_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for path, value in _flatten_scalars(row):
            if isinstance(value, str) and value.strip():
                stats[path].append(value.strip())

    candidates = []
    for path, values in stats.items():
        average_chars = sum(len(value) for value in values) / len(values)
        coverage = len(values) / max(len(rows), 1)
        score = _text_name_score(path) + min(math.log10(average_chars + 1) * 2.5, 11) + coverage * 4
        if average_chars < 80:
            score -= 8
        candidates.append(
            {
                "path": path,
                "score": round(score, 3),
                "coverage": round(min(coverage, 1.0), 3),
                "average_chars": round(average_chars),
            }
        )
    return sorted(candidates, key=lambda candidate: (candidate["score"], candidate["average_chars"]), reverse=True)


def _rank_scalar_path(
    rows: list[dict[str, Any]],
    *,
    preferred: tuple[str, ...],
    maximum_average_chars: int,
) -> str | None:
    stats: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        for path, value in _flatten_scalars(row):
            if "[]" not in path and isinstance(value, (str, int)) and value not in ("", None):
                stats[path].append(value)

    ranked: list[tuple[float, str]] = []
    for path, values in stats.items():
        average_chars = sum(len(str(value)) for value in values) / len(values)
        if average_chars > maximum_average_chars:
            continue
        leaf = path.casefold().split(".")[-1]
        name_score = max((12 - index * 1.5 for index, name in enumerate(preferred) if leaf == name), default=0)
        uniqueness = len({str(value) for value in values}) / len(values)
        coverage = len(values) / max(len(rows), 1)
        ranked.append((name_score + uniqueness * 4 + coverage * 2, path))
    return max(ranked, default=(0, None))[1]


def discover_schema(
    rows: list[dict[str, Any]],
    *,
    source: str,
    source_kind: str,
    text_column: str | None = None,
    id_column: str | None = None,
    title_column: str | None = None,
    metadata_columns: tuple[str, ...] = (),
) -> DiscoveredSchema:
    if not rows:
        raise ValueError("Cannot discover a schema without sample rows")
    text_candidates = _rank_text_candidates(rows)
    selected_text = text_column or (text_candidates[0]["path"] if text_candidates else None)
    if not selected_text:
        raise ValueError("No non-empty text-like field was discovered")
    selected_id = id_column or _rank_scalar_path(
        rows,
        preferred=("id", "_id", "uuid", "slug", "document_id", "case_id", "citation"),
        maximum_average_chars=300,
    )
    selected_title = title_column or _rank_scalar_path(
        rows,
        preferred=("title", "case_name", "name", "case_name_short", "case_name_full", "headline", "subject", "slug"),
        maximum_average_chars=500,
    )
    if metadata_columns:
        selected_metadata = metadata_columns
    else:
        excluded = {selected_text, selected_id, selected_title}
        metadata_candidates = [
            path
            for path in (
                "date_filed",
                "court_full_name",
                "court_short_name",
                "court_jurisdiction",
                "precedential_status",
                "disposition",
                "nature_of_suit",
                "language",
                "category",
                "jurisdiction",
                "date",
            )
            if path not in excluded and any(nested_values(row, path) for row in rows)
        ]
        selected_metadata = tuple(metadata_candidates[:8])
    margin = (
        text_candidates[0]["score"] - text_candidates[1]["score"]
        if len(text_candidates) > 1
        else text_candidates[0]["score"]
        if text_candidates
        else 0
    )
    confidence = 1.0 if text_column else max(0.05, min(0.99, 0.55 + margin / 20))
    return DiscoveredSchema(
        source=source,
        source_kind=source_kind,
        text_column=selected_text,
        id_column=selected_id,
        title_column=selected_title,
        metadata_columns=selected_metadata,
        sampled_rows=len(rows),
        confidence=round(confidence, 3),
        text_candidates=tuple(text_candidates[:5]),
    )


def inspect_hf_schema(
    dataset_id: str,
    *,
    split: str = "train",
    config: str | None = None,
    sample_rows: int = 32,
    text_column: str | None = None,
    id_column: str | None = None,
    title_column: str | None = None,
    metadata_columns: tuple[str, ...] = (),
) -> DiscoveredSchema:
    dataset = load_dataset(dataset_id, config, split=split, streaming=True)
    iterator = iter(dataset)
    try:
        rows = list(islice(iterator, sample_rows))
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        del iterator
        del dataset
        gc.collect()
    return discover_schema(
        rows,
        source=dataset_id,
        source_kind="huggingface",
        text_column=text_column,
        id_column=id_column,
        title_column=title_column,
        metadata_columns=metadata_columns,
    )


def load_hf_documents_isolated(
    dataset_id: str,
    *,
    split: str = "train",
    config: str | None = None,
    limit: int,
    seed: int = 42,
    shuffle_buffer: int = 1_000,
    schema_sample_rows: int = 32,
    text_column: str | None = None,
    id_column: str | None = None,
    title_column: str | None = None,
    metadata_columns: tuple[str, ...] = (),
) -> tuple[list[SourceDocument], DiscoveredSchema]:
    with tempfile.TemporaryDirectory(prefix="unstructured-source-") as temporary_directory:
        payload_path = Path(temporary_directory) / "source.jsonl"
        command = [
            sys.executable,
            "-m",
            "ds_structure_server.pipeline.hf_source_worker",
            "--dataset",
            dataset_id,
            "--split",
            split,
            "--limit",
            str(limit),
            "--seed",
            str(seed),
            "--shuffle-buffer",
            str(shuffle_buffer),
            "--schema-sample-rows",
            str(schema_sample_rows),
            "--output",
            str(payload_path),
        ]
        for flag, value in (
            ("--config", config),
            ("--text-column", text_column),
            ("--id-column", id_column),
            ("--title-column", title_column),
        ):
            if value:
                command.extend((flag, value))
        for column in metadata_columns:
            command.extend(("--metadata-column", column))
        subprocess.run(command, check=True)

        lines_iter = payload_path.open("r", encoding="utf-8")
        try:
            first = next(lines_iter, None)
            if not first:
                raise ValueError("Hugging Face source worker returned no schema or documents")
            schema = DiscoveredSchema.from_dict(json.loads(first)["schema"])
            documents: list[SourceDocument] = []
            for line in lines_iter:
                text = line.strip()
                if not text:
                    continue
                value = json.loads(text)
                documents.append(
                    SourceDocument(
                        doc_id=value["doc_id"],
                        text=_truncate_chars(value["text"]),
                        title=value["title"],
                        metadata=value.get("metadata", {}),
                    )
                )
            return documents, schema
        finally:
            lines_iter.close()


_DOCUMENT_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}
_STRUCTURED_SUFFIXES = {".json", ".jsonl", ".csv"}
_ARCHIVE_SUFFIXES = {".zip"}
_MIN_PDF_CHARS_PER_PAGE = 40
_MIN_PDF_NONEMPTY_PAGE_RATIO = 0.2
# Keep OCR / PDF / document loads memory-bounded (small hosts ~4GiB).
_DEFAULT_OCR_DPI = int(os.getenv("STRUCTURE_OCR_DPI", "150"))
_DEFAULT_OCR_MAX_PAGES = int(os.getenv("STRUCTURE_OCR_MAX_PAGES", "40"))
_DEFAULT_NATIVE_PDF_MAX_PAGES = int(os.getenv("STRUCTURE_PDF_MAX_PAGES", "80"))
_MAX_DOCUMENT_CHARS = int(os.getenv("STRUCTURE_MAX_DOCUMENT_CHARS", "500000"))
_MAX_STRUCTURED_FILE_BYTES = int(os.getenv("STRUCTURE_MAX_STRUCTURED_FILE_BYTES", str(32 * 1024 * 1024)))
_MAX_ZIP_UNCOMPRESSED_BYTES = int(os.getenv("STRUCTURE_MAX_ZIP_UNCOMPRESSED_BYTES", str(200 * 1024 * 1024)))


def _safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    import shutil
    import zipfile

    destination.mkdir(parents=True, exist_ok=True)
    dest_root = destination.resolve()
    uncompressed = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                continue
            size = int(info.file_size or 0)
            uncompressed += size
            if uncompressed > _MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"Zip uncompressed size exceeds {_MAX_ZIP_UNCOMPRESSED_BYTES // (1024 * 1024)}MB: {zip_path}"
                )
            target = (destination / member).resolve()
            if not str(target).startswith(str(dest_root)):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 64)
    return destination


def _normalize_extracted_text(text: str) -> str:
    return "\n\n".join(part.strip() for part in text.splitlines() if part.strip())


def _truncate_chars(text: str, limit: int = _MAX_DOCUMENT_CHARS) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit]


def _pdf_native_text(
    path: Path,
    *,
    max_pages: int = _DEFAULT_NATIVE_PDF_MAX_PAGES,
    max_chars: int = _MAX_DOCUMENT_CHARS,
) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    total_pages = len(reader.pages)
    limit = total_pages if max_pages <= 0 else min(total_pages, max_pages)
    parts: list[str] = []
    used = 0
    for index in range(limit):
        raw = (reader.pages[index].extract_text() or "").strip()
        normalized = _normalize_extracted_text(raw)
        if not normalized:
            continue
        if max_chars > 0 and used + len(normalized) > max_chars:
            remaining = max_chars - used
            if remaining > 0:
                parts.append(normalized[:remaining])
            break
        parts.append(normalized)
        used += len(normalized)
    # Keep page boundaries for chunk_level=page sampling.
    return "\f".join(parts), total_pages


def _pdf_text_is_usable(text: str, page_count: int) -> bool:
    if page_count <= 0:
        return bool(text.strip())
    parts = [part for part in text.split("\f") if part.strip()] if "\f" in text else [part for part in text.split("\n\n") if part.strip()]
    nonempty_pages = len(parts)
    chars_per_page = len(text.replace("\f", "")) / page_count
    nonempty_ratio = nonempty_pages / page_count
    return chars_per_page >= _MIN_PDF_CHARS_PER_PAGE and nonempty_ratio >= _MIN_PDF_NONEMPTY_PAGE_RATIO


def _ocr_pdf(
    path: Path,
    *,
    dpi: int = _DEFAULT_OCR_DPI,
    max_pages: int = _DEFAULT_OCR_MAX_PAGES,
) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
    except ImportError as exc:
        raise RuntimeError(
            "OCR fallback requires optional deps: pip install 'ds-structure-server[ocr]' "
            f"(pytesseract + pdf2image). Missing while reading: {path}"
        ) from exc

    # Render + OCR one page at a time so peak RSS stays roughly constant.
    page_texts: list[str] = []
    last_page = max_pages if max_pages > 0 else None
    page_num = 1
    logger.info("ocr: start file={} dpi={} max_pages={}", path.name, dpi, max_pages)
    while last_page is None or page_num <= last_page:
        try:
            images = convert_from_path(
                str(path),
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
                grayscale=True,
                thread_count=1,
            )
        except (PDFInfoNotInstalledError, PDFPageCountError, OSError) as exc:
            if page_texts:
                break
            raise RuntimeError(
                f"OCR fallback requires poppler (`pdftoppm`) to render PDF pages: {path}"
            ) from exc
        except MemoryError:
            break
        if not images:
            break
        image = images[0]
        try:
            try:
                raw = pytesseract.image_to_string(image)
            except pytesseract.TesseractNotFoundError as exc:
                raise RuntimeError(
                    f"OCR fallback requires the `tesseract` binary on PATH: {path}"
                ) from exc
            except MemoryError:
                break
        finally:
            image.close()
            del images
        normalized = _normalize_extracted_text(raw)
        if normalized.strip():
            page_texts.append(normalized)
        if page_num == 1 or page_num % 5 == 0:
            logger.info("ocr: page {} done nonempty_pages={}", page_num, len(page_texts))
        page_num += 1
    logger.info("ocr: finished pages_attempted={} nonempty={}", page_num - 1, len(page_texts))
    return "\f".join(page_texts)


def _read_pdf(path: Path) -> str:
    logger.info("pdf: native extract {}", path.name)
    text, page_count = _pdf_native_text(path)
    if _pdf_text_is_usable(text, page_count):
        logger.info("pdf: native usable pages={} chars={}", page_count, len(text))
        return text
    logger.info("pdf: native sparse pages={} chars={} → OCR fallback", page_count, len(text))
    ocr_text = _ocr_pdf(path)
    if ocr_text.strip():
        logger.info("pdf: OCR ok chars={}", len(ocr_text))
        return ocr_text
    if text.strip():
        logger.warning("pdf: OCR empty; using sparse native text chars={}", len(text))
        return text
    raise ValueError(f"Could not extract text from PDF via native parsing or OCR: {path}")


def _read_document(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        from docx import Document

        parts: list[str] = []
        used = 0
        for paragraph in Document(path).paragraphs:
            text = (paragraph.text or "").strip()
            if not text:
                continue
            if used + len(text) + 2 > _MAX_DOCUMENT_CHARS:
                remaining = _MAX_DOCUMENT_CHARS - used
                if remaining > 0:
                    parts.append(text[:remaining])
                break
            parts.append(text)
            used += len(text) + 2
        return "\n\n".join(parts)
    # Bound plain-text / HTML reads without slurping multi-hundred-MB files.
    chunks: list[str] = []
    used = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while used < _MAX_DOCUMENT_CHARS:
            block = handle.read(min(64 * 1024, _MAX_DOCUMENT_CHARS - used))
            if not block:
                break
            chunks.append(block)
            used += len(block)
    return "".join(chunks)


def iter_structured_rows(path: Path, *, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield structured rows without requiring the whole file in memory (jsonl/csv)."""
    suffix = path.suffix.casefold()
    yielded = 0
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if max_rows is not None and yielded >= max_rows:
                    return
                text = line.strip()
                if not text:
                    continue
                value = json.loads(text)
                if isinstance(value, dict):
                    yield value
                    yielded += 1
        return
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in DictReader(handle):
                if max_rows is not None and yielded >= max_rows:
                    return
                if isinstance(row, dict):
                    yield row
                    yielded += 1
        return
    # JSON arrays/objects: size-capped full parse (true streaming needs ijson).
    if path.stat().st_size > _MAX_STRUCTURED_FILE_BYTES:
        raise ValueError(
            f"Structured JSON source exceeds {_MAX_STRUCTURED_FILE_BYTES // (1024 * 1024)}MB: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    rows: list[dict[str, Any]]
    if isinstance(value, list):
        rows = [row for row in value if isinstance(row, dict)]
    elif isinstance(value, dict):
        rows = []
        for key in ("rows", "records", "documents", "data", "items", "results"):
            nested = value.get(key)
            if isinstance(nested, list) and all(isinstance(row, dict) for row in nested):
                rows = nested
                break
        if not rows:
            rows = [value]
    else:
        raise ValueError(f"Structured source must contain objects: {path}")
    for row in rows:
        if max_rows is not None and yielded >= max_rows:
            return
        yield row
        yielded += 1


def sample_and_count_structured(
    path: Path,
    *,
    sample_limit: int = 200,
) -> tuple[list[dict[str, Any]], int]:
    """Stream structured sources: keep a small sample and an exact/cheap row count."""
    suffix = path.suffix.casefold()
    sample: list[dict[str, Any]] = []
    count = 0
    if suffix in {".jsonl", ".csv"}:
        for row in iter_structured_rows(path):
            count += 1
            if len(sample) < sample_limit:
                sample.append(row)
        return sample, count
    # JSON: one bounded parse; sample + full count from the in-memory list once.
    rows = list(iter_structured_rows(path))
    return rows[:sample_limit], len(rows)


def _structured_rows(path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    return list(iter_structured_rows(path, max_rows=max_rows))


def load_local_documents(
    input_path: Path,
    *,
    limit: int,
    seed: int = 42,
    schema_sample_rows: int = 32,
    text_column: str | None = None,
    id_column: str | None = None,
    title_column: str | None = None,
    metadata_columns: tuple[str, ...] = (),
) -> tuple[list[SourceDocument], DiscoveredSchema]:
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file() and path.suffix.casefold() in _ARCHIVE_SUFFIXES:
        unzipped = path.parent / f"{path.stem}_unzipped"
        if unzipped.exists():
            import shutil

            shutil.rmtree(unzipped, ignore_errors=True)
        path = _safe_extract_zip(path, unzipped)

    if path.is_dir():
        files = sorted(
            child
            for child in path.rglob("*")
            if child.is_file()
            and child.suffix.casefold() in (_DOCUMENT_SUFFIXES | _STRUCTURED_SUFFIXES)
        )
        random.Random(seed).shuffle(files)
        documents: list[SourceDocument] = []
        for file in files:
            if len(documents) >= limit:
                break
            suffix = file.suffix.casefold()
            if suffix in _DOCUMENT_SUFFIXES:
                text = _truncate_chars(_read_document(file))
                if text.strip():
                    documents.append(
                        SourceDocument(
                            doc_id=str(file.relative_to(path)),
                            text=text,
                            title=file.stem,
                            metadata={"path": str(file)},
                        )
                    )
                continue
            remaining = limit - len(documents)
            if remaining <= 0:
                break
            nested_docs, _ = load_local_documents(
                file,
                limit=remaining,
                seed=seed,
                schema_sample_rows=schema_sample_rows,
                text_column=text_column,
                id_column=id_column,
                title_column=title_column,
                metadata_columns=metadata_columns,
            )
            documents.extend(nested_docs)
        schema = DiscoveredSchema(
            source=str(path),
            source_kind="document-directory",
            text_column="$document",
            id_column="$relative_path",
            title_column="$filename",
            metadata_columns=("path",),
            sampled_rows=min(len(files), limit),
            confidence=1.0,
        )
        return [document for document in documents if document.text.strip()], schema

    suffix = path.suffix.casefold()
    if suffix in _DOCUMENT_SUFFIXES:
        text = _truncate_chars(_read_document(path))
        schema = DiscoveredSchema(
            source=str(path),
            source_kind="document",
            text_column="$document",
            id_column="$path",
            title_column="$filename",
            metadata_columns=("path",),
            sampled_rows=1,
            confidence=1.0,
        )
        return [SourceDocument(str(path), text, path.stem, {"path": str(path)})], schema
    if suffix not in _STRUCTURED_SUFFIXES:
        raise ValueError(f"Unsupported local source type: {suffix or '<none>'}")

    rows = _structured_rows(path, max_rows=max(limit, schema_sample_rows))
    sample = rows[:schema_sample_rows]
    schema = discover_schema(
        sample,
        source=str(path),
        source_kind=suffix.removeprefix("."),
        text_column=text_column,
        id_column=id_column,
        title_column=title_column,
        metadata_columns=metadata_columns,
    )
    random.Random(seed).shuffle(rows)
    documents = list(
        documents_from_rows(
            rows,
            source=str(path),
            schema=schema,
            limit=limit,
        )
    )
    return documents, schema


def _fallback_id(dataset_id: str, index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{dataset_id}:{index}:{digest}"


def documents_from_rows(
    rows: Iterator[dict[str, Any]] | list[dict[str, Any]],
    *,
    source: str,
    schema: DiscoveredSchema,
    limit: int,
) -> Iterator[SourceDocument]:
    emitted = 0
    for index, row in enumerate(rows):
        raw_text = _joined_text(row, schema.text_column)
        if not raw_text:
            continue
        raw_id = nested_value(row, schema.id_column)
        doc_id = str(raw_id).strip() if raw_id not in (None, "") else _fallback_id(source, index, raw_text)
        raw_title = nested_value(row, schema.title_column)
        title = str(raw_title).strip() if raw_title not in (None, "") else doc_id
        metadata = {column: nested_value(row, column) for column in schema.metadata_columns}
        yield SourceDocument(doc_id=doc_id, text=raw_text, title=title, metadata=metadata)
        emitted += 1
        if emitted >= limit:
            return


def stream_documents(
    dataset_id: str,
    text_column: str,
    *,
    split: str = "train",
    config: str | None = None,
    id_column: str | None = None,
    title_column: str | None = None,
    metadata_columns: tuple[str, ...] = (),
    limit: int = 3,
    seed: int = 42,
    shuffle_buffer: int = 1_000,
) -> Iterator[SourceDocument]:
    dataset = load_dataset(dataset_id, config, split=split, streaming=True)
    if shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    schema = DiscoveredSchema(
        source=dataset_id,
        source_kind="huggingface",
        text_column=text_column,
        id_column=id_column,
        title_column=title_column,
        metadata_columns=metadata_columns,
        sampled_rows=0,
        confidence=1.0,
    )
    iterator = iter(dataset)
    try:
        yield from documents_from_rows(iterator, source=dataset_id, schema=schema, limit=limit)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        del iterator
        del dataset
        gc.collect()
