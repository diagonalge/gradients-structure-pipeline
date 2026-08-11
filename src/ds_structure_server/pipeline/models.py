from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ChunkLevel = Literal["document", "section", "paragraph", "line", "needle", "page"]

CHUNK_LEVELS: tuple[ChunkLevel, ...] = ("document", "section", "paragraph", "line", "needle", "page")

# Public API selection (clear FE options). Maps onto internal ChunkLevel.
ApiChunkLevel = Literal["line", "section", "paragraph", "page"]
API_CHUNK_LEVELS: tuple[ApiChunkLevel, ...] = ("line", "section", "paragraph", "page")


@dataclass(frozen=True)
class DatasetProfile:
    domain: str
    source_kind: str
    content_signals: tuple[str, ...]
    extractable_assets: tuple[str, ...]
    preferred_task_types: tuple[str, ...]
    instruction_guidance: str
    answer_guidance: str
    avoid: tuple[str, ...]
    recommended_chunk_level: ChunkLevel | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetProfile:
        def _strings(key: str) -> tuple[str, ...]:
            raw = value.get(key, [])
            if not isinstance(raw, list):
                raise ValueError(f"Dataset profile field {key!r} must be a list")
            items = [item.strip() for item in raw if isinstance(item, str) and item.strip()]
            return tuple(items)

        required = ("domain", "source_kind", "instruction_guidance", "answer_guidance")
        if any(not isinstance(value.get(key), str) or not value[key].strip() for key in required):
            raise ValueError(f"Dataset profile is missing required string fields: {value!r}")
        chunk_level = value.get("recommended_chunk_level")
        if chunk_level is not None:
            if not isinstance(chunk_level, str) or chunk_level not in CHUNK_LEVELS:
                raise ValueError(f"Unsupported recommended_chunk_level: {chunk_level!r}")
        return cls(
            domain=value["domain"].strip(),
            source_kind=value["source_kind"].strip(),
            content_signals=_strings("content_signals"),
            extractable_assets=_strings("extractable_assets"),
            preferred_task_types=_strings("preferred_task_types"),
            instruction_guidance=value["instruction_guidance"].strip(),
            answer_guidance=value["answer_guidance"].strip(),
            avoid=_strings("avoid"),
            recommended_chunk_level=chunk_level,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    def prompt_block(self) -> str:
        return "\n".join(
            [
                f"Domain: {self.domain}",
                f"Source kind: {self.source_kind}",
                f"Content signals: {', '.join(self.content_signals) or 'n/a'}",
                f"Extractable assets: {', '.join(self.extractable_assets) or 'n/a'}",
                f"Preferred task types (use across the job; do not collapse to one): "
                f"{', '.join(self.preferred_task_types) or 'n/a'}",
                f"Instruction guidance: {self.instruction_guidance}",
                f"Answer guidance: {self.answer_guidance}",
                f"Avoid: {', '.join(self.avoid) or 'n/a'}",
                f"Recommended chunk level: {self.recommended_chunk_level or 'persona-default'}",
                "Diversity: each row should pursue a different supported task when CONTEXT allows; "
                "rotate among the preferred task types rather than one familiar stem.",
                "Shape: write reusable named-subject Q&A from CONTEXT as ground-truth data points; "
                "do not mention studies, articles, papers, or authors in the pair.",
            ]
        )


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    text: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Section:
    title: str
    path: str
    text: str
    start: int
    end: int
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class DocumentTree:
    document: SourceDocument
    cleaned_text: str
    sections: tuple[Section, ...]


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    chunk_level: ChunkLevel
    question_style: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Persona:
        required = ("name", "description", "chunk_level", "question_style")
        if any(not isinstance(value.get(key), str) or not value[key].strip() for key in required):
            raise ValueError(f"Persona is missing required string fields: {value!r}")
        if value["chunk_level"] not in CHUNK_LEVELS:
            raise ValueError(f"Unsupported chunk level: {value['chunk_level']}")
        return cls(**{key: value[key].strip() for key in required})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SampledChunk:
    text: str
    target_text: str
    section_path: str
    level: ChunkLevel
    method: str
    start: int
    end: int


@dataclass(frozen=True)
class GeneratedPair:
    instruction: str
    output: str
    standalone: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GeneratedPair:
        instruction = value.get("instruction")
        output = value.get("output")
        standalone = value.get("standalone")
        if not isinstance(instruction, str) or len(instruction.strip()) < 8:
            raise ValueError("Generated instruction is missing or too short")
        if not isinstance(output, str) or len(output.strip()) < 2:
            raise ValueError("Generated output is missing or too short")
        if not isinstance(standalone, bool):
            raise ValueError("Generated standalone flag must be boolean")
        return cls(instruction.strip(), output.strip(), standalone)
