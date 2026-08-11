from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import _STRUCTURED_SUFFIXES, sample_and_count_structured

# Prefer stronger / more specific names first.
INSTRUCTION_ALIASES = (
    "instruction",
    "instruct",
    "prompt",
    "question",
    "query",
    "user",
    "human",
    "input",
)
OUTPUT_ALIASES = (
    "output",
    "response",
    "completion",
    "answer",
    "target",
    "assistant",
    "label",
)

# Unstructured corpus columns — presence alone means "needs structuring" unless a true pair wins.
CORPUS_ALIASES = (
    "text",
    "content",
    "body",
    "document",
    "article",
    "passage",
    "context",
    "raw",
    "html",
    "markdown",
    "page_content",
)

MESSAGES_ALIASES = ("messages", "conversations", "conversation")


@dataclass(frozen=True)
class InstructPairDetection:
    already_instruct: bool
    instruction_field: str | None = None
    output_field: str | None = None
    row_count: int | None = None
    format: str | None = None  # "columns" | "messages" | None


def _field_map(keys: list[str] | set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key in keys:
        if isinstance(key, str) and key.strip():
            mapping.setdefault(key.casefold().strip(), key)
    return mapping


def _pick_field(mapping: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    return None


def _nonempty_str(value: Any, *, min_chars: int = 1) -> bool:
    return isinstance(value, str) and len(value.strip()) >= min_chars


def _union_keys(rows: list[dict[str, Any]], *, limit: int = 50) -> dict[str, str]:
    keys: list[str] = []
    for row in rows[:limit]:
        keys.extend(str(k) for k in row.keys() if isinstance(k, str))
    return _field_map(keys)


def _messages_pair(value: Any) -> bool:
    """True when value is a chat transcript with at least one user + assistant turn."""
    if not isinstance(value, list) or len(value) < 2:
        return False
    roles: set[str] = set()
    for turn in value:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from") or turn.get("speaker")
        content = turn.get("content") or turn.get("value") or turn.get("text")
        if isinstance(role, str) and _nonempty_str(content, min_chars=1):
            roles.add(role.casefold().strip())
    userish = roles & {"user", "human", "prompter", "instruction"}
    asstish = roles & {"assistant", "bot", "gpt", "model", "output"}
    return bool(userish and asstish)


def _column_pair_score(
    rows: list[dict[str, Any]],
    instruction_field: str,
    output_field: str,
) -> tuple[int, float]:
    usable = 0
    for row in rows:
        instr = row.get(instruction_field)
        out = row.get(output_field)
        if _nonempty_str(instr, min_chars=1) and _nonempty_str(out, min_chars=1):
            usable += 1
    ratio = usable / max(1, len(rows))
    return usable, ratio


def detect_instruct_pair_from_rows(
    rows: list[dict[str, Any]],
    *,
    min_rows: int = 1,
    majority_ratio: float = 0.7,
) -> InstructPairDetection:
    """
    Decide whether tabular/JSON rows are already a supervised instruct dataset.

    Ready examples: instruct/output, prompt/completion, question/answer, messages[].
    Not ready: document corpora (text/content/body), nested unstructured blobs, bare CSVs
    without a clear instruction↔answer pair.
    """
    dict_rows = [row for row in rows if isinstance(row, dict)]
    row_count = len(dict_rows)
    if row_count < min_rows:
        return InstructPairDetection(already_instruct=False, row_count=row_count)

    mapping = _union_keys(dict_rows)

    # Chat / messages format (ShareGPT, OpenAI-style).
    messages_field = _pick_field(mapping, MESSAGES_ALIASES)
    if messages_field:
        usable = sum(1 for row in dict_rows if _messages_pair(row.get(messages_field)))
        if usable >= max(1, int(row_count * majority_ratio)):
            return InstructPairDetection(
                already_instruct=True,
                instruction_field=messages_field,
                output_field=messages_field,
                row_count=row_count,
                format="messages",
            )

    instruction_field = _pick_field(mapping, INSTRUCTION_ALIASES)
    output_field = _pick_field(mapping, OUTPUT_ALIASES)
    if instruction_field and output_field and instruction_field != output_field:
        usable, ratio = _column_pair_score(dict_rows, instruction_field, output_field)
        if usable >= max(1, int(row_count * majority_ratio)) and ratio >= majority_ratio:
            return InstructPairDetection(
                already_instruct=True,
                instruction_field=instruction_field,
                output_field=output_field,
                row_count=row_count,
                format="columns",
            )

    # Explicitly not ready: looks like a document corpus / unstructured dump.
    return InstructPairDetection(already_instruct=False, row_count=row_count)


def looks_like_unstructured_corpus(rows: list[dict[str, Any]]) -> bool:
    """Heuristic: rows are prose documents, not instruct pairs."""
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not dict_rows:
        return False
    if detect_instruct_pair_from_rows(dict_rows).already_instruct:
        return False
    mapping = _union_keys(dict_rows)
    return _pick_field(mapping, CORPUS_ALIASES) is not None


def detect_instruct_pair_from_path(path: Path, *, sample_limit: int = 200) -> InstructPairDetection:
    suffix = path.suffix.casefold()
    if suffix not in _STRUCTURED_SUFFIXES:
        return InstructPairDetection(already_instruct=False)
    try:
        sample, row_count = sample_and_count_structured(path, sample_limit=sample_limit)
    except Exception:
        return InstructPairDetection(already_instruct=False)
    detected = detect_instruct_pair_from_rows(sample)
    if detected.already_instruct:
        return InstructPairDetection(
            already_instruct=True,
            instruction_field=detected.instruction_field,
            output_field=detected.output_field,
            row_count=row_count,
            format=detected.format,
        )
    return InstructPairDetection(already_instruct=False, row_count=row_count)
