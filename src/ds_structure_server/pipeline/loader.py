from __future__ import annotations

import gc
import hashlib
import json
import math
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

        lines = payload_path.read_text().splitlines()
        if not lines:
            raise ValueError("Hugging Face source worker returned no schema or documents")
        schema = DiscoveredSchema.from_dict(json.loads(lines[0])["schema"])
        documents = [
            SourceDocument(
                doc_id=value["doc_id"],
                text=value["text"],
                title=value["title"],
                metadata=value.get("metadata", {}),
            )
            for line in lines[1:]
            if (value := json.loads(line))
        ]
        return documents, schema


_DOCUMENT_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}
_STRUCTURED_SUFFIXES = {".json", ".jsonl", ".csv"}
_ARCHIVE_SUFFIXES = {".zip"}
_MIN_PDF_CHARS_PER_PAGE = 40
_MIN_PDF_NONEMPTY_PAGE_RATIO = 0.2
_DEFAULT_OCR_DPI = 200
_DEFAULT_OCR_MAX_PAGES = 200


def _safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    import shutil
    import zipfile

    destination.mkdir(parents=True, exist_ok=True)
    dest_root = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                continue
            target = (destination / member).resolve()
            if not str(target).startswith(str(dest_root)):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
    return destination


def _normalize_extracted_text(text: str) -> str:
    return "\n\n".join(part.strip() for part in text.splitlines() if part.strip())


def _pdf_native_text(path: Path) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    normalized = [_normalize_extracted_text(page) for page in pages if page.strip()]
    # Keep page boundaries for chunk_level=page sampling.
    return "\f".join(normalized), len(pages)


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
    import pytesseract
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError

    try:
        images = convert_from_path(
            str(path),
            dpi=dpi,
            first_page=1,
            last_page=max_pages if max_pages > 0 else None,
        )
    except (PDFInfoNotInstalledError, PDFPageCountError, OSError) as exc:
        raise RuntimeError(
            f"OCR fallback requires poppler (`pdftoppm`) to render PDF pages: {path}"
        ) from exc

    try:
        page_texts = [pytesseract.image_to_string(image) for image in images]
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            f"OCR fallback requires the `tesseract` binary on PATH: {path}"
        ) from exc
    return "\f".join(_normalize_extracted_text(page) for page in page_texts if page.strip())


def _read_pdf(path: Path) -> str:
    text, page_count = _pdf_native_text(path)
    if _pdf_text_is_usable(text, page_count):
        return text
    ocr_text = _ocr_pdf(path)
    if ocr_text.strip():
        return ocr_text
    if text.strip():
        return text
    raise ValueError(f"Could not extract text from PDF via native parsing or OCR: {path}")


def _read_document(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        from docx import Document

        return "\n\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    return path.read_text(errors="replace")


def _structured_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open(newline="") as handle:
            return list(DictReader(handle))
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("rows", "records", "documents", "data", "items", "results"):
            nested = value.get(key)
            if isinstance(nested, list) and all(isinstance(row, dict) for row in nested):
                return nested
        return [value]
    raise ValueError(f"Structured source must contain objects: {path}")


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
                text = _read_document(file)
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
        text = _read_document(path)
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

    rows = _structured_rows(path)
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
