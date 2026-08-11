from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from .inference import InferenceBackend
from ds_structure_server.job_runtime import StructureJobCancelled
from .models import (
    CHUNK_LEVELS,
    ChunkLevel,
    DatasetProfile,
    DocumentTree,
    GeneratedPair,
    Persona,
    SampledChunk,
    SourceDocument,
)
from .parsing import parse_document, split_sentences, is_non_content_heading, is_non_content_text
from .references import (
    extract_reference_mentions,
    format_output_with_reference_context,
    resolve_reference_excerpts,
)

BUILTIN_PERSONAS = (
    Persona(
        name="line-analyst",
        description="A careful reader who extracts and clarifies atomic facts from short source spans.",
        chunk_level="line",
        question_style="Ask a precise standalone question answerable from the local span.",
    ),
    Persona(
        name="summarizer",
        description="A reader who needs an accurate high-level understanding of a document or major section.",
        chunk_level="section",
        question_style="Ask for a concise synthesis of the central facts, reasoning, or outcome.",
    ),
    Persona(
        name="needle",
        description="A reader looking for one precise fact hidden among surrounding information.",
        chunk_level="needle",
        question_style="Ask a specific, independently understandable question about the target fact.",
    ),
    Persona(
        name="detail-researcher",
        description="A domain reader checking concrete details and relationships in the source.",
        chunk_level="paragraph",
        question_style="Ask a standalone factual or explanatory question answered directly by the passage.",
    ),
)

GENERIC_PERSONA_NAMES = ("line-analyst", "summarizer", "needle", "detail-researcher")

_DEDUP_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "tell",
        "me",
        "give",
        "help",
        "please",
        "with",
        "from",
        "into",
        "about",
        "that",
        "this",
        "these",
        "those",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "a",
    }
)

_LOW_VALUE_CHUNK_RE = re.compile(
    r"(?is)\b(?:"
    r"(?:from|in|see|cf\.?)\s+(?:the\s+)?(?:next|previous|following|preceding|earlier|later)\s+"
    r"(?:chapter|section|page|paragraph|exercise)|"
    r"as\s+we\s+(?:shall|will)\s+see|"
    r"later\s+(?:on\s+)?(?:in\s+)?(?:this|the)\s+(?:chapter|section|book)|"
    r"in\s+the\s+(?:next|previous|following)\s+(?:chapter|section)|"
    r"discussed\s+(?:in|below|above|later)|"
    r"see\s+(?:chapter|section|page)\s+\d|"
    r"to\s+be\s+(?:defined|proved|shown|discussed)\s+(?:later|below|in\s+chapter)"
    r")\b"
)
_FRAGMENT_START_RE = re.compile(r"^[a-z(,;:\-]")
_TASK_PROMPT_LINE_RE = re.compile(
    r"(?im)^(?:\(?\d+(?:\.\d+)*\)?[.)]?\s*)?(?:"
    r"prove|show that|show|compute|find|solve|determine|verify|evaluate|calculate|"
    r"derive|demonstrate|construct|give an example|explain how|explain why|"
    r"what is|why does|true or false|fill in|complete the|let\s+.+\s+show"
    r")\b"
)
_ANSWER_BEARING_RE = re.compile(
    r"(?i)\b(?:"
    r"therefore|thus|hence|because|since|implies|equivalently|in other words|"
    r"is defined as|means that|we have|it follows|follows that|proof|solution|"
    r"answer|result(?:s)? in|equals|denote[sd]?|consists of|is called"
    r")\b"
)


_PERSONA_SYSTEM = """You design realistic user personas for document-grounded instruction datasets.
Return only valid JSON. Do not include generic assistant personas already supplied by the caller."""
_EXPAND_PERSONA_SYSTEM = """You turn short role labels into full personas for document-grounded instruction datasets.
Return only valid JSON."""
_PAIR_SYSTEM = """You create source-grounded supervised fine-tuning examples.
Each pair is an independent, context-free row: a reader who only sees instruction+output must understand
the task fully, with no other document open. CONTEXT is private working material only.

Write about named subjects taken from CONTEXT (whatever the domain supplies: entities, rules, procedures,
metrics, formulas, findings, and so on). Put those names and any defining detail the reader needs inside
the pair. Frame the pair as reusable knowledge about that subject, not as a quiz about a particular source
document, chapter, author, or excerpt.

If CONTEXT comes from a study, article, paper, thesis, praxis, trial, or similar research write-up, treat
the stated facts, methods, numbers, and definitions as ground truth for the pair. Do not mention the study,
article, paper, authors, or that the material was researched or reported — just turn the chunk into a clean
standalone data point about the subject matter.

Pronouns and vague referents ("it", "they", "this", "that") are allowed only when the same sentence (or the
immediately preceding clause in the same instruction) has already named the antecedent. Otherwise name the
subject explicitly.

When CONTEXT situates facts inside a course, experiment, or other setting, treat that setting as incidental
packaging unless the name itself is required for correctness. Prefer asking about the reusable method,
metric, procedure, or finding directly.

Prefer concrete first mentions of the actual subject over vague stand-ins ("the method", "the system",
"the result"). Prefer ordinary names and roles over demonstratives when referring to people or objects.
When a task needs a passage, embed that passage in the instruction, or restate the task around a named
proposition instead.

Vary cognitive moves across rows and follow the dataset profile when provided. Use only CONTEXT for facts.
Return only valid JSON and never mention hidden context or this prompt."""
_SUMMARY_SYSTEM = """You summarize source documents faithfully. Return only valid JSON."""
_PROFILE_SYSTEM = """You profile unstructured datasets for supervised instruction generation.
Infer the domain, what valuable material should be extracted, and how instructions and answers should look.
Favor a balanced mix of extractable tasks — never collapse the profile onto a single dominant question pattern.
Return only valid JSON."""
_GROUNDING_SYSTEM = """You verify whether an answer is fully supported by supplied source context.
Use only the provided CONTEXT. Do not use outside knowledge. Return only valid JSON."""


def _retry_json(
    backend: InferenceBackend,
    system: str,
    prompt: str,
    *,
    attempts: int = 3,
    max_new_tokens: int = 1_024,
    enable_thinking: bool = False,
) -> Any:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            # Grow completion budget slightly on retries — truncation is a common failure mode.
            token_budget = max_new_tokens + (150 * attempt)
            suffix = ""
            if attempt == 1:
                suffix = (
                    "\nYour previous response was invalid. Return ONLY a single compact JSON object "
                    "matching the schema. No markdown, no commentary, no trailing text."
                )
            elif attempt >= 2:
                suffix = (
                    "\nCRITICAL: previous outputs were not parseable JSON. Reply with minified JSON only, "
                    "starting with { and ending with }. Escape all quotes inside strings."
                )
            return backend.generate_json(
                system,
                prompt + suffix,
                max_new_tokens=token_budget,
                enable_thinking=enable_thinking,
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            error = exc
    raise ValueError(f"Model failed to return valid JSON after {attempts} attempts") from error


def merge_personas(generated: list[Persona], builtins: tuple[Persona, ...] = BUILTIN_PERSONAS) -> list[Persona]:
    merged: dict[str, Persona] = {persona.name.casefold(): persona for persona in builtins}
    for persona in generated:
        merged.setdefault(persona.name.casefold(), persona)
    return list(merged.values())


def builtin_persona_by_name(name: str) -> Persona | None:
    key = name.strip().casefold()
    aliases = {
        "needle-in-haystack": "needle",
        "needle_in_haystack": "needle",
        "needle-in-the-haystack": "needle",
        "line_analyst": "line-analyst",
        "lineanalyst": "line-analyst",
    }
    key = aliases.get(key, key)
    for persona in BUILTIN_PERSONAS:
        if persona.name.casefold() == key:
            return persona
    return None


def generic_builtin_personas() -> list[Persona]:
    return [persona for persona in BUILTIN_PERSONAS if persona.name in GENERIC_PERSONA_NAMES]


def normalize_instruction_for_dedup(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip().casefold())
    value = value.strip(" ?.!,;:\"'")
    return value


def instruction_exact_key(text: str) -> str:
    return hashlib.sha1(normalize_instruction_for_dedup(text).encode("utf-8")).hexdigest()


def instruction_near_fingerprint(text: str, *, max_tokens: int = 12) -> str:
    normalized = normalize_instruction_for_dedup(text)
    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'-]*", normalized)
        if token not in _DEDUP_STOPWORDS and len(token) > 1
    ]
    unique = sorted(set(tokens))[:max_tokens]
    payload = " ".join(unique)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


_CONTEXT_BOUND_INSTRUCTION_RE = re.compile(
    r"(?:"
    r"\baccording to\b|"
    r"\bas (?:described|outlined|stated|noted|mentioned|discussed|presented|reported) in\b|"
    r"\bdescribed in (?:the\s+)?(?:\w+\s+){0,4}(?:study|paper|article|thesis|praxis|pipeline|overview|document|source|text)\b|"
    r"\bbased on (?:the )?(?:textbook|section|passage|chapter|document|source|text|excerpt|material|study|paper|praxis)\b|"
    r"\b(?:the |this |that )?(?:textbook|passage|excerpt|chapter|document|source material)\b|"
    r"\b(?:in|from|per) the (?:textbook|section|passage|chapter|document|source|study|paper|article|thesis|praxis)\b|"
    r"\b(?:textbook|section|chapter) (?:title|heading)\b|"
    r"\bprovided (?:textbook|section|passage|chapter|document|text|excerpt)\b|"
    r"\bthe (?:above|following|given|supplied|attached) (?:text|passage|section|excerpt|document)\b|"
    r"\brefer(?:ring)? to the (?:text|passage|section|textbook|document|source)\b|"
    r"\b(?:the|this|that)\s+(?:study|paper|article|thesis|praxis|dissertation|research)\b|"
    r"\bthe authors?\b|"
    r"\bin this research\b"
    r")",
    re.IGNORECASE,
)

# Vague subject pointers that leave the reader needing CONTEXT to know which thing is meant.
_UNDERSPECIFIED_SUBJECT_RE = re.compile(
    r"(?i)\b(?:the|this|that)\s+"
    r"(?:procedural\s+mechanism|experimental\s+design|proposed\s+(?:system|method|approach|framework|model|pipeline)|"
    r"methodology|pipeline|framework|approach|mechanism|procedure|process|technique|"
    r"method|system|model|algorithm|evaluation|metric|feedback\s+system)"
    r"\b(?!\s+(?:of|for|called|known\s+as|named)\s+[A-Za-z0-9])"
)


def instruction_cites_source_context(text: str) -> bool:
    """True when the instruction depends on unseen source provenance or scaffolding."""
    value = (text or "").strip()
    if not value:
        return True
    if _CONTEXT_BOUND_INSTRUCTION_RE.search(value):
        return True
    # Title/heading leaks often appear wrapped in asterisks.
    if re.search(r"\*[^*]{3,80}\*", value):
        return True
    return False


def pair_self_containment_failure(instruction: str, output: str) -> str | None:
    """Return a short reason when the pair is not independently understandable, else None."""
    instr = (instruction or "").strip()
    out = (output or "").strip()
    if not instr or not out:
        return "Empty instruction or output"
    combined = f"{instr}\n{out}"
    if instruction_cites_source_context(instr) or _CONTEXT_BOUND_INSTRUCTION_RE.search(out):
        return "Pair depends on unseen source provenance"
    if _UNDERSPECIFIED_SUBJECT_RE.search(combined):
        return "Pair uses an underspecified subject instead of a named one"
    return None


def instruction_stem_key(text: str, *, max_tokens: int = 5) -> str:
    """Coarse opening-stem key used to cap repeated instruction templates."""
    normalized = normalize_instruction_for_dedup(text)
    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'-]*", normalized)
        if token not in _DEDUP_STOPWORDS and len(token) > 1
    ]
    if not tokens:
        return normalized[:48]
    return " ".join(tokens[:max_tokens])


class InstructionDeduper:
    """Job-wide exact + near-dup + stem-diversity + context-bound gate.

    Exact/near hashes live in a job-local dbm file so RAM does not grow with
    every accepted row. Stem counts stay a small in-memory dict.
    """

    def __init__(
        self,
        *,
        max_stem_share: float = 0.14,
        min_before_stem_cap: int = 3,
        store_path: Path | None = None,
    ) -> None:
        import dbm

        self._lock = threading.Lock()
        self._stems: dict[str, int] = defaultdict(int)
        self._accepted = 0
        self._max_stem_share = max(0.05, min(1.0, max_stem_share))
        self._min_before_stem_cap = max(1, min_before_stem_cap)
        self._store_path = Path(store_path) if store_path else None
        self._db = None
        if self._store_path is not None:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = dbm.open(str(self._store_path), "c")
        else:
            self._exact: set[str] = set()
            self._near: set[str] = set()

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None

    def _has(self, kind: bytes, digest: str) -> bool:
        key = kind + digest.encode("ascii")
        if self._db is not None:
            return key in self._db
        bucket = self._exact if kind == b"e:" else self._near
        return digest in bucket

    def _put(self, kind: bytes, digest: str) -> None:
        key = kind + digest.encode("ascii")
        if self._db is not None:
            self._db[key] = b"1"
            return
        bucket = self._exact if kind == b"e:" else self._near
        bucket.add(digest)

    def try_claim(self, instruction: str) -> bool:
        if instruction_cites_source_context(instruction):
            return False
        exact = instruction_exact_key(instruction)
        near = instruction_near_fingerprint(instruction)
        stem = instruction_stem_key(instruction)
        with self._lock:
            if self._has(b"e:", exact) or self._has(b"n:", near):
                return False
            if self._accepted >= self._min_before_stem_cap:
                projected = self._stems[stem] + 1
                share = projected / (self._accepted + 1)
                if self._stems[stem] >= 2 and share > self._max_stem_share:
                    return False
            self._put(b"e:", exact)
            self._put(b"n:", near)
            self._stems[stem] += 1
            self._accepted += 1
            return True


# Canonical cognitive task focuses (fallbacks when profile is thin).
DEFAULT_TASK_FOCUSES = (
    "define_or_clarify_term",  # knowledge recall
    "explain_mechanism_or_reasoning",  # explanation
    "transform_summarize_or_structure",  # transformation
    "compare_or_differentiate",  # comparison
    "apply_rule_or_procedure",  # application
    "infer_what_follows",  # reasoning
    "critique_or_verify_claim",  # critique
    "decide_when_to_use",  # decision
)

def expand_persona_stubs(
    backend: InferenceBackend,
    names: list[str],
    documents: list[SourceDocument],
    *,
    dataset_profile: DatasetProfile | None = None,
) -> list[Persona]:
    """Turn short role labels (e.g. 'doctor') into full Persona objects using the source profile."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = re.sub(r"[^a-z0-9]+", "-", raw.strip().casefold()).strip("-")
        if not name or name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    if not cleaned:
        return []
    if not documents:
        raise ValueError("At least one document is required to expand persona stubs")

    profile_block = (
        f"Dataset profile to respect:\n{dataset_profile.prompt_block()}\n"
        if dataset_profile is not None
        else "No dataset profile was supplied; infer persona details from the samples alone.\n"
    )
    prompt = f"""Expand each role label into one concrete persona for this dataset.
{profile_block}
Rules:
- Keep the persona name as a short kebab-case slug derived from the label.
- description: who they are and what they need from a single source row.
- chunk_level: one of document|section|paragraph|line|needle|page.
- question_style: abstract instruction behavior covering MULTIPLE task shapes (not one fixed request stem).
  It must be adaptable across rows.
- Do not specialize the persona so every row becomes the same request template.
- Tasks must be answerable from one row without aggregating across rows.
- Prefer the dataset profile's extractable assets and preferred task types.

Return only:
{{"personas": [
  {{"name": "short-kebab-name", "description": "broad role and goal",
    "chunk_level": "document|section|paragraph|line|needle|page", "question_style": "adaptable instruction behavior"}}
]}}

Role labels to expand:
{json.dumps(cleaned, ensure_ascii=False)}

Sample rows:
{json.dumps(_persona_excerpts(documents[:8]), ensure_ascii=False)}
"""
    value = _retry_json(backend, _EXPAND_PERSONA_SYSTEM, prompt, max_new_tokens=900)
    expanded = _parse_personas(value, limit=len(cleaned))
    by_name = {persona.name.casefold(): persona for persona in expanded}
    ordered: list[Persona] = []
    for label in cleaned:
        persona = by_name.get(label)
        if persona is None:
            # Fall back to first unused expanded persona, or synthesize a minimal one.
            for candidate in expanded:
                if candidate.name.casefold() not in {item.name.casefold() for item in ordered}:
                    persona = candidate
                    break
        if persona is None:
            persona = Persona(
                name=label,
                description=f"A {label.replace('-', ' ')} using information from one source document.",
                chunk_level="paragraph",
                question_style=f"Ask a standalone question a {label.replace('-', ' ')} would need answered.",
            )
        ordered.append(persona)
    return ordered


def resolve_personas(
    backend: InferenceBackend,
    documents: list[SourceDocument],
    *,
    presets: list[str] | None = None,
    names: list[str] | None = None,
    max_personas: int = 1,
    infer: bool | None = None,
    dataset_profile: DatasetProfile | None = None,
) -> list[Persona]:
    """Combine preset builtins, custom name stubs, and optional inferred personas up to max_personas.

    Priority: presets, then expanded custom names, then inferred fill for remaining slots.
    `infer` defaults to True when presets+names leave unused slots under max_personas.
    """
    if max_personas < 1:
        raise ValueError("max_personas must be at least 1")
    preset_names = presets if presets is not None else ["summarizer"]
    custom_names = names or []

    seeded: list[Persona] = []
    seen: set[str] = set()
    for raw in preset_names:
        persona = builtin_persona_by_name(raw)
        if persona is None:
            raise ValueError(f"Unknown persona preset: {raw!r}")
        key = persona.name.casefold()
        if key in seen:
            continue
        seen.add(key)
        seeded.append(persona)
        if len(seeded) >= max_personas:
            return seeded

    remaining_names = [name for name in custom_names if re.sub(r"[^a-z0-9]+", "-", name.strip().casefold()).strip("-") not in seen]
    if remaining_names and len(seeded) < max_personas:
        expanded = expand_persona_stubs(backend, remaining_names, documents, dataset_profile=dataset_profile)
        for persona in expanded:
            key = persona.name.casefold()
            if key in seen:
                continue
            seen.add(key)
            seeded.append(persona)
            if len(seeded) >= max_personas:
                return seeded

    slots_left = max_personas - len(seeded)
    should_infer = infer if infer is not None else slots_left > 0
    if should_infer and slots_left > 0:
        inferred = infer_personas(
            backend,
            documents,
            maximum=slots_left,
            include_builtins=False,
            dataset_profile=dataset_profile,
        )
        for persona in inferred:
            key = persona.name.casefold()
            if key in seen:
                continue
            seen.add(key)
            seeded.append(persona)
            if len(seeded) >= max_personas:
                break
    if not seeded:
        summarizer = builtin_persona_by_name("summarizer")
        assert summarizer is not None
        seeded = [summarizer]
    return seeded[:max_personas]


def _parse_personas(value: Any, limit: int) -> list[Persona]:
    raw_personas = value.get("personas", []) if isinstance(value, dict) else []
    if not isinstance(raw_personas, list):
        raise ValueError("Persona response must contain a personas list")
    generated_by_name: dict[str, Persona] = {}
    for item in raw_personas:
        if isinstance(item, dict):
            persona = Persona.from_dict(item)
            generated_by_name.setdefault(persona.name.casefold(), persona)
        if len(generated_by_name) >= limit:
            break
    return list(generated_by_name.values())


def _persona_excerpts(documents: list[SourceDocument], max_chars: int = 700) -> list[dict[str, str]]:
    return [
        {
            "title": document.title,
            "excerpt": re.sub(r"\s+", " ", document.text).strip()[:max_chars],
        }
        for document in documents
    ]


def _brief_skim_excerpts(
    documents: list[SourceDocument],
    *,
    max_docs: int = 4,
    max_chars: int = 3_600,
    pages_per_doc: int = 6,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Random interior page skims for profile/persona inference (avoid front-matter bias)."""
    if not documents:
        return []
    rng = random.Random(seed)
    # Prefer longer docs, then randomly sample among the stronger half.
    ranked = sorted(documents, key=lambda doc: len(doc.text or ""), reverse=True)
    pool = ranked[: max(max_docs, min(len(ranked), max_docs * 2))]
    if len(pool) > max_docs:
        chosen_docs = rng.sample(pool, max_docs)
    else:
        chosen_docs = list(pool)

    out: list[dict[str, str]] = []
    per_doc_budget = max(400, max_chars // max(1, len(chosen_docs)))

    for document in chosen_docs:
        raw = (document.text or "").strip()
        if not raw:
            continue
        pages = [part.strip() for part in raw.split("\f") if part.strip()]
        if len(pages) <= 1:
            window = max(800, min(2_000, per_doc_budget // 2))
            chunks = [
                raw[i : i + window] for i in range(0, len(raw), window) if raw[i : i + window].strip()
            ]
            pages = chunks or [raw]
        body = [p for p in pages if not is_non_content_text(p)]
        if len(body) < max(2, pages_per_doc // 2):
            body = pages
        skip = min(max(1, len(body) // 10), max(0, len(body) - 1))
        interior = body[skip:] or body
        k = min(pages_per_doc, len(interior))
        sampled = rng.sample(interior, k) if k < len(interior) else list(interior)
        order = {id(p): i for i, p in enumerate(pages)}
        sampled.sort(key=lambda p: order.get(id(p), 0))
        pieces: list[str] = []
        used = 0
        for page in sampled:
            cleaned = re.sub(r"\s+", " ", page).strip()
            if not cleaned or is_non_content_text(cleaned):
                continue
            room = per_doc_budget - used
            if room <= 0:
                break
            take = cleaned[: min(len(cleaned), max(180, room))]
            pieces.append(take)
            used += len(take) + 1
        if not pieces:
            continue
        out.append({"title": document.title, "excerpt": " … ".join(pieces)})
    return out


def infer_profile_and_extra_personas_fast(
    backend: InferenceBackend,
    documents: list[SourceDocument],
    *,
    max_inferred: int = 2,
) -> tuple[DatasetProfile, list[Persona]]:
    """One brief-skim LLM call: dataset profile + up to ``max_inferred`` DS personas."""
    if not documents:
        raise ValueError("At least one document is required")
    skim = _brief_skim_excerpts(documents, max_docs=4, max_chars=3_600, pages_per_doc=6)
    if not skim:
        raise ValueError("Document skim is empty")
    prompt = f"""Briefly skim the source excerpts and return:
1) a compact dataset profile for the SUBSTANTIVE body content
2) exactly {max_inferred} domain-specific personas

Critical sampling notes:
- Excerpts are random interior pages. IGNORE publisher/front-matter boilerplate even if remnants appear
  (ProQuest banners, copyright notices, approval sheets, candidate information, tables of contents,
  Creative Commons legalese, "Information to All Users").
- Profile the actual research/domain content (methods, findings, entities, procedures in the body).
- preferred_task_types must reflect that body content — never invent thesis-submission / approval-workflow
  / formatting-guideline tasks unless the corpus is genuinely about institutional submission policy.

Do NOT include line-analyst, summarizer, needle, or detail-researcher — those are already added.
Keep preferred_task_types to 4 short abstract cognitive-move labels (2-5 words or snake_case), e.g.
"explain_named_mechanism", "compare_named_methods", "extract_criteria" — never full sample questions
or ready-made instruction templates.
Persona description and question_style: one short sentence each.
Prefer chunk_level paragraph or section unless line is clearly better.

Return only JSON:
{{
  "domain": "short domain label",
  "source_kind": "textbook|case_law|clinical_notes|code|research_paper|news|other",
  "content_signals": ["signal1", "signal2"],
  "extractable_assets": ["asset1", "asset2"],
  "preferred_task_types": ["task1", "task2", "task3", "task4"],
  "instruction_guidance": "standalone named-subject rules; include necessary defining details in the pair; never point at the source document",
  "answer_guidance": "concise factual answers that name the subject; no 'the study found' phrasing",
  "avoid": ["repeating the same task type or instruction stem across rows",
            "front-matter, copyright, or submission-admin tasks",
            "referring to the study/paper/authors or using bare definite phrases like 'the method' without naming it"],
  "recommended_chunk_level": "document|section|paragraph|line|needle|page",
  "personas": [
    {{"name": "short-kebab-name", "description": "who they are and their goal",
      "chunk_level": "document|section|paragraph|line|needle|page",
      "question_style": "adaptable multi-shape instruction behavior"}}
  ]
}}

Source skim:
{json.dumps(skim, ensure_ascii=False)}
"""
    value = _retry_json(backend, _PROFILE_SYSTEM, prompt, max_new_tokens=900, attempts=3)
    if not isinstance(value, dict):
        raise ValueError("Fast suggest response must be a JSON object")
    profile = _normalize_dataset_profile(DatasetProfile.from_dict(value))
    personas = _parse_personas(value, max_inferred)
    return profile, personas[:max_inferred]


def infer_dataset_profile(
    backend: InferenceBackend,
    documents: list[SourceDocument],
    *,
    sample_docs: int = 4,
) -> DatasetProfile:
    if not documents:
        raise ValueError("At least one document is required to infer a dataset profile")
    prompt = f"""Identify this dataset and decide what supervised instruction data should extract from it.
Use only a brief skim of random interior pages (do not assume you read the whole corpus).
IGNORE publisher front matter (ProQuest, copyright, approval sheets, TOC, Creative Commons legalese).
Do not invent a domain that is unsupported by the body-content samples.
Do NOT profile the corpus as thesis-submission / approval-workflow admin unless that is truly the topic.
Focus on modalities that make the eventual instruction/output pairs faithful to the source.

Diversity requirements (critical):
- List 4 to 6 distinct preferred_task_types that are actually supported by the samples. Cover different cognitive
  moves (e.g. define, explain mechanism, diagnose/classify, apply a rule/procedure, compare, extract criteria,
  contraindications/edge cases) — not minor wording variants of one task.
- Phrase preferred_task_types as short abstract cognitive-move labels (snake_case or 2-5 words), never as
  full instruction templates or sample questions.
- Do NOT overweight one modality. For clinical/medical sources, treatments are only ONE of several equal options
  alongside presentation/findings, diagnostic criteria, differentials, pathophysiology, complications, and
  contraindications/monitoring.
- instruction_guidance must describe ABSTRACT phrasing rules only (standalone, named subject, no demonstratives,
  include necessary defining clauses in the pair, never refer to the source document).
  It must NOT contain a concrete sample question, template, or single canonical stem such as
  "What is the first-line treatment for ...?".
- answer_guidance should allow concise factual answers, short lists, and brief explanations that name the subject —
  not "the study found…" phrasing and not only "recommend treatment" style responses.
- Put "repeating the same task type or instruction stem across rows" in avoid.
- Also avoid front-matter / copyright / submission-admin tasks and study/paper/author pointers.

Examples of modality adaptation (balanced — do not reduce a domain to one bullet):
- Math textbooks: formulas, identities, theorem statements, proof steps, worked calculations.
- Case law: holdings, procedural posture, standards of review, rule applications, remedies.
- Clinical / medical handbooks: findings, diagnostic criteria, differentials, mechanisms, treatments,
  contraindications, complications — treat these as co-equal.
- Code or API docs: signatures, failure modes, edge cases, repair steps, invariants.
- Research papers / dissertations: methods, experimental setups, results, claims, definitions, comparisons.

Return only:
{{
  "domain": "short domain label",
  "source_kind": "textbook|case_law|clinical_notes|code|research_paper|news|other",
  "content_signals": ["signal1", "signal2", "signal3"],
  "extractable_assets": ["asset1", "asset2", "asset3"],
  "preferred_task_types": ["task1", "task2", "task3", "task4"],
  "instruction_guidance": "abstract phrasing rules only; named subjects; include necessary defining clauses; no study/paper pointers",
  "answer_guidance": "how answers should be phrased across the diverse task types; name the subject; no 'the study found'",
  "avoid": ["repeating the same task type or instruction stem across rows",
            "front-matter, copyright, or submission-admin tasks",
            "referring to the study/paper/authors or bare definite phrases like 'the method' without naming it"],
  "recommended_chunk_level": "document|section|paragraph|line|needle|page"
}}

Sample skim:
{json.dumps(_brief_skim_excerpts(documents, max_docs=sample_docs, max_chars=3_600, pages_per_doc=6), ensure_ascii=False)}
"""
    value = _retry_json(backend, _PROFILE_SYSTEM, prompt, max_new_tokens=900, attempts=3)
    if not isinstance(value, dict):
        raise ValueError("Dataset profile response must be a JSON object")
    profile = DatasetProfile.from_dict(value)
    return _normalize_dataset_profile(profile)


def _abstract_task_label(task: str) -> str:
    """Turn sentence-like preferred_task_types into short cognitive-move labels."""
    raw = (task or "").strip()
    if not raw:
        return raw
    lowered = raw.casefold()
    words = re.findall(r"[a-z0-9]+", lowered)
    already_short = len(words) <= 5 and not re.match(
        r"^(explain|outline|evaluate|compare|identify|define|describe|list|summarize|help)\b",
        lowered,
    )
    if already_short and ("_" in raw or "-" in raw or raw.islower()):
        return raw
    verb_moves = (
        (r"^(?:explain|describe)\b", "explain_mechanism_or_application"),
        (r"^(?:outline|walk)\b", "apply_or_sequence_supported_procedure"),
        (r"^(?:evaluate|assess)\b", "evaluate_supported_criteria"),
        (r"^compare\b", "compare_or_differentiate"),
        (r"^(?:identify|list)\b.*(?:metric|measure|indicator)", "identify_supported_metrics"),
        (r"^(?:identify|list)\b", "extract_supported_facts_or_assets"),
        (r"^define\b", "define_or_clarify_term"),
        (r"^summarize\b", "transform_summarize_or_structure"),
        (r"^help\b", "apply_rule_or_procedure"),
    )
    for pattern, label in verb_moves:
        if re.match(pattern, lowered):
            return label
    if len(words) > 5:
        return "_".join(words[:4])
    return raw


def _normalize_dataset_profile(profile: DatasetProfile) -> DatasetProfile:
    """Clamp profile fields so one task pattern cannot dominate generation."""
    tasks = list(profile.preferred_task_types)
    # Drop near-duplicate task labels (same when casefolded / underscored).
    deduped: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        key = re.sub(r"[\s\-]+", "_", task.strip().casefold())
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(task.strip())
    if len(deduped) < 3:
        # Fallback mix; generation still has to stay CONTEXT-supported.
        for extra in DEFAULT_TASK_FOCUSES:
            if extra not in seen:
                deduped.append(extra)
                seen.add(extra)
            if len(deduped) >= 4:
                break

    guidance = profile.instruction_guidance.strip()
    # Strip accidental concrete stems/examples that overfit all rows to one pattern.
    if "?" in guidance or re.search(
        r"\be\.g\.\b|\bfor example\b|\bfirst-line treatment\b|\blist (?:the )?diagnostic criteria\b",
        guidance,
        re.I,
    ):
        guidance = (
            "Write standalone instructions that name the subject explicitly. "
            "Vary the cognitive task and the surface wording across rows; "
            "do not reuse one stem or template opener."
        )

    abstract_tasks: list[str] = []
    for task in deduped[:6]:
        lowered = task.casefold()
        if re.match(r"^list\b", lowered) and "criteria" in lowered:
            abstract_tasks.append("state_supported_criteria_or_essentials")
        elif "first-line treatment" in lowered or lowered.startswith("recommend treatment"):
            abstract_tasks.append("identify_supported_management_or_treatment_options")
        else:
            abstract_tasks.append(_abstract_task_label(task))
    tasks = tuple(dict.fromkeys(abstract_tasks))  # preserve order, drop dups

    avoid = list(profile.avoid)
    for item in (
        "repeating the same task type or instruction stem across rows",
        "referencing the source document, section, or passage by name or provenance",
    ):
        if not any(item.casefold() in existing.casefold() for existing in avoid):
            avoid.insert(0, item)

    return DatasetProfile(
        domain=profile.domain,
        source_kind=profile.source_kind,
        content_signals=profile.content_signals,
        extractable_assets=profile.extractable_assets,
        preferred_task_types=tasks,
        instruction_guidance=guidance,
        answer_guidance=profile.answer_guidance,
        avoid=tuple(avoid),
        recommended_chunk_level=profile.recommended_chunk_level,
    )


def save_dataset_profile(path: Path, profile: DatasetProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2) + "\n")


def load_dataset_profile(path: Path) -> DatasetProfile:
    return DatasetProfile.from_dict(json.loads(path.read_text()))


def assign_task_focuses(task_types: tuple[str, ...] | list[str], n_rows: int) -> list[str]:
    """Round-robin task focuses so rows do not collapse onto one preferred_task_type."""
    types = [item.strip() for item in task_types if isinstance(item, str) and item.strip()]
    if not types:
        types = list(DEFAULT_TASK_FOCUSES)
    if n_rows <= 0:
        return []
    return [types[index % len(types)] for index in range(n_rows)]


def infer_personas(
    backend: InferenceBackend,
    documents: list[SourceDocument],
    maximum: int = 5,
    *,
    include_builtins: bool = True,
    analysis_batch_size: int = 50,
    dataset_profile: DatasetProfile | None = None,
) -> list[Persona]:
    if not documents:
        raise ValueError("At least one document is required to infer personas")

    # Fast path: brief skim + one call (no batch/consolidate round-trip).
    if maximum <= 2 or len(documents) <= 2:
        profile_block = (
            f"Dataset profile to respect:\n{dataset_profile.prompt_block()}\n"
            if dataset_profile is not None
            else "No dataset profile was supplied; infer personas from the skim alone.\n"
        )
        prompt = f"""From a brief head/middle/tail skim, propose exactly {maximum} broad, distinct personas.
{profile_block}
Do not include summarizer, needle-in-haystack, detail-researcher, or line-analyst.
Each persona must work on one document/row and produce a self-contained instruction+answer pair.
Keep description and question_style to one short sentence; prefer chunk_level paragraph or section.

Return only:
{{"personas": [
  {{"name": "short-kebab-name", "description": "who they are and their goal",
    "chunk_level": "document|section|paragraph|line|needle|page", "question_style": "adaptable instruction behavior"}}
]}}

Source skim:
{json.dumps(_brief_skim_excerpts(documents, max_docs=min(2, len(documents)), max_chars=900), ensure_ascii=False)}
"""
        value = _retry_json(backend, _PERSONA_SYSTEM, prompt, max_new_tokens=700, attempts=3)
        generated = _parse_personas(value, maximum)
        if not generated:
            raise ValueError("Persona inference returned no personas")
        generated = generated[:maximum]
        return merge_personas(generated) if include_builtins else generated

    profile_block = (
        f"Dataset profile to respect:\n{dataset_profile.prompt_block()}\n"
        if dataset_profile is not None
        else "No dataset profile was supplied; infer personas from the samples alone.\n"
    )
    candidate_personas: list[Persona] = []
    candidate_limit = min(maximum + 3, 10)
    for batch_number, start in enumerate(range(0, len(documents), analysis_batch_size), start=1):
        batch = documents[start : start + analysis_batch_size]
        prompt = f"""Analyze dataset sample batch {batch_number} and propose up to {candidate_limit} broad user roles.
Infer who would realistically use the recurring information in this dataset and what they would ask an assistant to do.
{profile_block}
Each role must be useful for typical rows, not tied to one named entity, event, topic instance, or sample excerpt.
The pipeline gives the role one row or document at a time, so its tasks must be answerable from one row without
requiring statistics, comparisons, or evidence from other rows.
The final training projection contains only the instruction and answer, not the source document. Exclude roles whose
core task is summarizing a whole document, extracting source-specific metadata, translating an unseen passage, or
otherwise acting on content that cannot be compactly established inside the instruction.
Prefer roles that turn source facts into self-contained reasoning, decisions, explanations, or recommendations.
If the dataset profile lists extractable assets such as formulas, holdings, or code signatures, personas should
exercise those assets rather than only asking generic reading-comprehension questions.
The legacy question_style field describes instruction behavior, not a sample question. Keep it abstract and
adaptable across rows, use action-oriented wording, and do not refer to "this" or "that" source entity.
It may describe questions, direct commands, requests, or workflow tasks.
Describe behavior such as "Simplify source language for a general audience", not literal templates such as
"Translate this paragraph".
Return only:
{{"personas": [
  {{"name": "short-kebab-name", "description": "broad role and goal",
    "chunk_level": "document|section|paragraph|line|needle|page", "question_style": "adaptable instruction behavior"}}
]}}

Sample skim:
{json.dumps(_brief_skim_excerpts(batch, max_docs=min(3, len(batch)), max_chars=700), ensure_ascii=False)}
"""
        value = _retry_json(backend, _PERSONA_SYSTEM, prompt, max_new_tokens=900, attempts=3)
        candidate_personas.extend(_parse_personas(value, candidate_limit))

    prompt = f"""Consolidate these batch-level candidates into exactly {maximum} broad, distinct personas.
They represent analysis of {len(documents)} rows from the dataset.
{profile_block}
Each persona must represent a realistic user and source-grounded tasks that work for a typical individual row.
Do not specialize around one named entity, event, narrow topic instance, or example excerpt.
Because generation receives one row at a time, exclude roles whose core task requires aggregating multiple rows.
Because the final training pair does not include the source, also exclude roles centered on whole-document summaries,
source-specific metadata extraction, or transformations of an unseen passage. The required facts or target must fit
naturally inside a standalone instruction.
Make the personas operationally distinct: they should request different kinds of reasoning or transformation,
not merely use different job titles for the same task.
Align personas with the dataset profile's preferred task types and extractable assets when provided.
The legacy question_style field describes instruction behavior. Keep it abstract, action-oriented, and adaptable;
it may describe questions, commands, requests, or multi-step tasks, but it must not contain example facts or
refer to "this" or "that" source entity.
Describe behavior such as "Extract procedural history from one document", not a literal instruction containing
"this case", "this document", or "this paragraph".
Do not include summarizer, needle-in-haystack, or generic fact-recall personas because those are always added.

Return:
{{"personas": [
  {{"name": "short-kebab-name", "description": "who they are and their goal",
    "chunk_level": "document|section|paragraph|line|needle|page", "question_style": "instructions they would give"}}
]}}

Batch-level candidates:
{json.dumps([persona.to_dict() for persona in candidate_personas], ensure_ascii=False)}
"""
    value = _retry_json(backend, _PERSONA_SYSTEM, prompt, max_new_tokens=900, attempts=3)
    generated = _parse_personas(value, maximum)
    if len(generated) != maximum:
        raise ValueError(f"Expected {maximum} unique generated personas, received {len(generated)}")
    return merge_personas(generated) if include_builtins else generated


def infer_personas_from_profile(
    backend: InferenceBackend,
    dataset_profile: DatasetProfile,
    maximum: int = 2,
) -> list[Persona]:
    """Compact profile-only persona inference (fallback when sample-based JSON fails)."""
    if maximum < 1:
        raise ValueError("maximum must be at least 1")
    prompt = f"""Propose exactly {maximum} broad, distinct personas for this dataset profile.
Do not include summarizer, needle-in-haystack, detail-researcher, or line-analyst — those are already added.
Each persona must work on one document/row at a time and produce a self-contained instruction+answer pair.
Prefer chunk_level paragraph or section unless line-level is clearly better.
Keep description and question_style short (one sentence each).

Dataset profile:
{dataset_profile.prompt_block()}

Return only:
{{"personas": [
  {{"name": "short-kebab-name", "description": "who they are and their goal",
    "chunk_level": "document|section|paragraph|line|needle|page", "question_style": "instruction behavior"}}
]}}
"""
    value = _retry_json(backend, _PERSONA_SYSTEM, prompt, max_new_tokens=900, attempts=3)
    generated = _parse_personas(value, maximum)
    if not generated:
        raise ValueError("Profile-only persona inference returned no personas")
    return generated[:maximum]


def save_personas(path: Path, personas: list[Persona]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"personas": [persona.to_dict() for persona in personas]}, indent=2) + "\n")


def load_personas(path: Path) -> list[Persona]:
    value = json.loads(path.read_text())
    items = value.get("personas", [])
    if not isinstance(items, list):
        raise ValueError("Persona file must contain a personas list")
    return [Persona.from_dict(item) for item in items]


def summarize_document(
    backend: InferenceBackend,
    tree: DocumentTree,
    max_piece_chars: int = 12_000,
    *,
    workers: int = 1,
) -> str:
    """Build one document overview with a single LLM call (truncated to model context).

    `max_piece_chars` and `workers` are retained for call-site compatibility but ignored:
    multi-piece map-reduce was slower on remote APIs and unnecessary for a header summary.
    """
    del max_piece_chars, workers  # unused; kept for signature compatibility

    full_text = "\n\n".join(section.text for section in tree.sections if section.text.strip())
    if not full_text.strip():
        raise ValueError("Document has no useful prose to summarize")

    max_input = int(getattr(backend, "max_input_tokens", 16_000) or 16_000)
    # Leave headroom for system prompt, instructions, and completion.
    budget_tokens = max(2_000, max_input - 1_500)
    count_tokens = getattr(backend, "count_tokens", None)
    if callable(count_tokens):
        if count_tokens(full_text) <= budget_tokens:
            source = full_text
        else:
            # Approximate char cut, then trim until within budget.
            approx_chars = max(4_000, budget_tokens * 4)
            source = full_text[:approx_chars]
            while len(source) > 4_000 and count_tokens(source) > budget_tokens:
                source = source[: int(len(source) * 0.9)]
    else:
        source = full_text[: budget_tokens * 4]

    truncated = len(source) < len(full_text)
    prompt = (
        'Return JSON {"summary": "..."} with a faithful document overview in 3-6 sentences '
        "(about 120-220 words). Cover document type, scope, and the main topics or provisions. "
        "Do not invent details that are not present.\n"
    )
    if truncated:
        prompt += (
            "Note: only the beginning of the document is provided (context-window limit); "
            "summarize what is present.\n"
        )
    prompt += "\nDOCUMENT:\n" + source

    value = _retry_json(
        backend,
        _SUMMARY_SYSTEM,
        prompt,
        # Thinking burns the token budget on Qwen3/OpenRouter and truncates JSON.
        max_new_tokens=1_000,
        attempts=3,
        enable_thinking=False,
    )
    summary = value.get("summary") if isinstance(value, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Summary response is missing summary")
    return summary.strip()


def apply_chunk_level(personas: list[Persona], chunk_level: ChunkLevel) -> list[Persona]:
    if chunk_level not in CHUNK_LEVELS:
        raise ValueError(f"Unsupported chunk level: {chunk_level}")
    return [
        Persona(
            name=persona.name,
            description=persona.description,
            chunk_level=chunk_level,
            question_style=persona.question_style,
        )
        for persona in personas
    ]


def is_prompt_only_chunk(text: str) -> bool:
    """True when text mostly poses tasks/questions without answer material."""
    value = text.strip()
    if not value:
        return True
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return True

    task_lines = 0
    numbered_prompts = 0
    for line in lines:
        if line.endswith("?") or _TASK_PROMPT_LINE_RE.search(line):
            task_lines += 1
        if re.match(r"^\(?\d+(?:\.\d+)*\)?[.)]?\s+\S", line) and (
            line.endswith("?") or _TASK_PROMPT_LINE_RE.search(line) or len(line) < 180
        ):
            numbered_prompts += 1

    answer_hits = len(_ANSWER_BEARING_RE.findall(value))
    question_marks = value.count("?")
    task_ratio = task_lines / len(lines)
    numbered_ratio = numbered_prompts / len(lines)

    if task_ratio >= 0.55 and answer_hits <= 1:
        return True
    if question_marks >= 3 and answer_hits == 0 and task_ratio >= 0.35:
        return True
    if numbered_ratio >= 0.5 and numbered_prompts >= 3 and answer_hits <= 1:
        return True
    return False


def is_high_value_chunk(text: str, *, minimum_chars: int = 80) -> bool:
    value = text.strip()
    if len(value) < minimum_chars:
        return False
    if is_non_content_text(value):
        return False
    if is_prompt_only_chunk(value):
        return False
    if _LOW_VALUE_CHUNK_RE.search(value):
        # Allow long passages that only mention a cross-reference in passing.
        if len(value) < 500 or _LOW_VALUE_CHUNK_RE.sub("", value).strip() == "":
            return False
        cross_ref_ratio = sum(len(match.group(0)) for match in _LOW_VALUE_CHUNK_RE.finditer(value)) / len(value)
        if cross_ref_ratio > 0.12:
            return False
    if _FRAGMENT_START_RE.match(value):
        return False
    words = re.findall(r"\b[\w'-]+\b", value)
    if len(words) < 12:
        return False
    alpha = sum(char.isalpha() for char in value)
    if alpha / max(len(value), 1) < 0.45:
        return False
    return True


def _candidate_sections(tree: DocumentTree) -> list:
    sections = [
        section
        for section in tree.sections
        if not is_non_content_heading(section.title)
        and is_high_value_chunk(section.text, minimum_chars=160)
    ]
    if sections:
        return sections
    return [
        section
        for section in tree.sections
        if not is_non_content_heading(section.title)
        and len(section.text) >= 160
        and not is_prompt_only_chunk(section.text)
        and not is_non_content_text(section.text, heading=section.title)
    ]


def _candidate_pages(tree: DocumentTree) -> list[tuple[str, str, int]]:
    """Return (page_text, path_label, start_offset) candidates for page-level sampling."""
    text = tree.cleaned_text
    if "\f" in text:
        raw_pages = [part.strip() for part in text.split("\f") if part.strip()]
    else:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        raw_pages = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            if current and current_len + len(para) > 2_500:
                raw_pages.append("\n\n".join(current))
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += len(para) + 2
        if current:
            raw_pages.append("\n\n".join(current))

    results: list[tuple[str, str, int]] = []
    search_from = 0
    for index, page in enumerate(raw_pages, start=1):
        if is_non_content_text(page):
            continue
        if not is_high_value_chunk(page, minimum_chars=80) and not (
            len(page) >= 80 and not is_prompt_only_chunk(page)
        ):
            continue
        offset = text.find(page, search_from)
        if offset < 0:
            offset = search_from
        results.append((page, f"Page {index}", offset))
        search_from = offset + max(len(page), 1)
    return results


def _candidate_paragraphs(section) -> list[str]:
    paragraphs = [
        paragraph for paragraph in section.paragraphs if is_high_value_chunk(paragraph, minimum_chars=80)
    ]
    if paragraphs:
        return paragraphs
    return [
        paragraph
        for paragraph in section.paragraphs
        if len(paragraph) >= 80 and not is_prompt_only_chunk(paragraph)
    ]


def _candidate_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if is_high_value_chunk(line, minimum_chars=40):
            lines.append(line)
    if lines:
        return lines
    return [
        line.strip()
        for line in text.splitlines()
        if len(line.strip()) >= 40 and not is_prompt_only_chunk(line.strip())
    ]



_MAX_CHUNK_POOL = int(os.environ.get("STRUCTURE_MAX_CHUNK_POOL", "5000"))


def list_candidate_chunks(
    tree: DocumentTree,
    persona: Persona,
    max_context_chars: int = 16_000,
) -> list[SampledChunk]:
    """Split the document into the persona's chunk units, in document order."""
    if not tree.sections:
        raise ValueError("Document has no useful sections")
    level = persona.chunk_level
    chunks: list[SampledChunk] = []

    def _append(chunk: SampledChunk) -> bool:
        chunks.append(chunk)
        return len(chunks) < _MAX_CHUNK_POOL

    if level == "document":
        text = tree.cleaned_text[:max_context_chars]
        if not is_high_value_chunk(text, minimum_chars=160):
            raise ValueError("Document text is not usable for generation")
        return [SampledChunk(text, text, tree.document.title, "document", "document-prefix", 0, len(text))]

    if level == "page":
        for page_text, page_path, page_start in _candidate_pages(tree):
            text = page_text[:max_context_chars]
            if is_prompt_only_chunk(text) or not is_high_value_chunk(text, minimum_chars=120):
                continue
            if not _append(
                SampledChunk(
                    text,
                    text,
                    page_path,
                    "page",
                    "page-unit",
                    page_start,
                    page_start + len(text),
                )
            ):
                break
        if not chunks:
            raise ValueError("Document has no usable pages for generation")
        return chunks

    sections = _candidate_sections(tree)
    if not sections:
        raise ValueError("Document has no usable sections for generation")

    if level == "section":
        for section in sections:
            text = section.text[:max_context_chars]
            if is_prompt_only_chunk(text) or not is_high_value_chunk(text, minimum_chars=160):
                continue
            if not _append(
                SampledChunk(
                    text,
                    text,
                    section.path,
                    "section",
                    "native-section",
                    section.start,
                    section.start + len(text),
                )
            ):
                break
        if not chunks:
            raise ValueError("No high-value section remained after filtering")
        return chunks

    for section in sections:
        if len(chunks) >= _MAX_CHUNK_POOL:
            break
        if level == "line":
            lines = _candidate_lines(section.text)
            for index, target in enumerate(lines):
                left = max(0, index - 1)
                right = min(len(lines), index + 2)
                window = "\n".join(lines[left:right])[: max(500, min(1_500, max_context_chars))]
                line_offset = section.text.find(target)
                line_start = section.start + max(line_offset, 0)
                if not _append(
                    SampledChunk(
                        window,
                        target,
                        section.path,
                        "line",
                        "line-window",
                        line_start,
                        line_start + len(target),
                    )
                ):
                    break
            continue

        useful_paragraphs = _candidate_paragraphs(section)
        if not useful_paragraphs:
            continue

        if level == "needle":
            envelope = section.text[:max_context_chars]
            for paragraph in useful_paragraphs:
                if len(chunks) >= _MAX_CHUNK_POOL:
                    break
                paragraph_offset = section.text.find(paragraph)
                start = section.start + max(paragraph_offset, 0)
                sentences = [
                    sentence
                    for sentence in split_sentences(paragraph)
                    if is_high_value_chunk(sentence, minimum_chars=40)
                ] or [paragraph]
                for target in sentences:
                    if not _append(
                        SampledChunk(
                            envelope,
                            target,
                            section.path,
                            "needle",
                            "sentence-target-in-section-envelope",
                            start,
                            start + len(paragraph),
                        )
                    ):
                        break
            continue

        # paragraph (default for detail-researcher)
        window_paragraphs = [
            paragraph for paragraph in section.paragraphs if is_high_value_chunk(paragraph, minimum_chars=60)
        ] or [paragraph for paragraph in section.paragraphs if not is_prompt_only_chunk(paragraph)]
        if not window_paragraphs:
            continue
        for paragraph in useful_paragraphs:
            if len(chunks) >= _MAX_CHUNK_POOL:
                break
            if paragraph not in window_paragraphs:
                continue
            paragraph_offset = section.text.find(paragraph)
            start = section.start + max(paragraph_offset, 0)
            window = _paragraph_window(
                window_paragraphs,
                window_paragraphs.index(paragraph),
                max_context_chars=min(4_000, max_context_chars),
            )
            if is_prompt_only_chunk(window) or not is_high_value_chunk(window, minimum_chars=120):
                continue
            if not _append(
                SampledChunk(
                    window,
                    paragraph[:max_context_chars],
                    section.path,
                    "paragraph",
                    "paragraph-window",
                    start,
                    start + len(paragraph),
                )
            ):
                break

    if not chunks:
        raise ValueError(f"No usable {level} chunks remained after filtering")
    return chunks


def assign_chunk_indexes(n_chunks: int, n_rows: int) -> list[int]:
    """Evenly map row slots onto a chunk list (document order, no random choice)."""
    if n_chunks <= 0:
        raise ValueError("No candidate chunks available")
    if n_rows <= 0:
        return []
    if n_rows == 1:
        return [0]
    if n_rows <= n_chunks:
        # Unique, evenly spaced spans across the document.
        return [int(i * n_chunks / n_rows) for i in range(n_rows)]
    # More rows than chunks: cycle so each chunk is used about equally often.
    return [i % n_chunks for i in range(n_rows)]


def assign_chunks_evenly(chunks: list[SampledChunk], n_rows: int) -> list[SampledChunk]:
    """Map rows onto the chunk list with even document coverage (no random choice)."""
    indexes = assign_chunk_indexes(len(chunks), n_rows)
    return [chunks[i] for i in indexes]


def _paragraph_anchor_chunks(tree: DocumentTree, max_context_chars: int = 16_000) -> list[SampledChunk]:
    """Paragraph-level anchors used for shared multi-persona sampling."""
    anchor_persona = Persona(
        name="_paragraph-anchor",
        description="Internal paragraph anchor",
        chunk_level="paragraph",
        question_style="n/a",
    )
    try:
        return list_candidate_chunks(tree, anchor_persona, max_context_chars=max_context_chars)
    except ValueError:
        return []


def _chunks_overlapping(pool: list[SampledChunk], anchor: SampledChunk) -> list[SampledChunk]:
    return [chunk for chunk in pool if chunk.start < anchor.end and chunk.end > anchor.start]


def _synthesize_line_from_anchor(anchor: SampledChunk, rng: random.Random) -> SampledChunk | None:
    source = (anchor.target_text or anchor.text or "").strip()
    if not source:
        return None
    lines = _candidate_lines(source)
    if not lines:
        return None
    target = rng.choice(lines)
    index = lines.index(target)
    left = max(0, index - 1)
    right = min(len(lines), index + 2)
    window = "\n".join(lines[left:right])[:1_500]
    offset = source.find(target)
    start = anchor.start + max(offset, 0)
    return SampledChunk(
        window,
        target,
        anchor.section_path,
        "line",
        "line-from-anchor",
        start,
        start + len(target),
    )


def _project_chunk_for_persona(
    persona: Persona,
    anchor: SampledChunk,
    pool: list[SampledChunk],
    rng: random.Random,
) -> SampledChunk | None:
    """Map a shared paragraph anchor onto a native chunk for ``persona``."""
    overlapping = _chunks_overlapping(pool, anchor)
    if overlapping:
        return rng.choice(overlapping)
    if persona.chunk_level == "paragraph":
        return anchor
    if persona.chunk_level == "line":
        return _synthesize_line_from_anchor(anchor, rng)
    if pool:
        return min(pool, key=lambda chunk: abs(chunk.start - anchor.start))
    return None


def _sample_without_replacement(
    items: list[SampledChunk],
    k: int,
    rng: random.Random,
) -> list[SampledChunk]:
    if k <= 0 or not items:
        return []
    if k >= len(items):
        shuffled = list(items)
        rng.shuffle(shuffled)
        return shuffled
    return rng.sample(items, k)


def build_generation_work(
    tree: DocumentTree,
    personas: list[Persona],
    pools: dict[str, list[SampledChunk]],
    *,
    goal: int,
    profile_tasks: list[str],
    seed: int,
    sample_mode: str = "auto",
) -> list[tuple[int, Persona, SampledChunk, int, str]]:
    """Build a shuffled work queue: random native subsample or shared anchors.

    When ``goal`` is below first-pass capacity, subsample without replacement.
    ``sample_mode``:
      - ``shared_anchor``: K random paragraph anchors, projected to each persona
      - ``independent``: per-persona random subsample from native pools
      - ``auto``: shared_anchor when 2+ personas, else independent
    """
    if goal < 1:
        return []
    active = [persona for persona in personas if pools.get(persona.name)]
    if not active:
        return []

    mode = sample_mode
    if mode == "auto":
        mode = "shared_anchor" if len(active) >= 2 else "independent"
    if mode not in {"shared_anchor", "independent"}:
        raise ValueError(f"Unsupported sample_mode: {sample_mode}")

    rng = random.Random(seed)
    tasks = list(profile_tasks) or list(DEFAULT_TASK_FOCUSES)
    angle_passes = max(1, min(4, len(tasks)))
    # Equal soft share per persona; K = ceil(goal / n_personas) shared anchors.
    slots_per_persona = max(1, (goal + len(active) - 1) // len(active))
    total_pool = sum(len(pools.get(persona.name) or []) for persona in active)
    full_coverage = goal >= total_pool
    # First pass covers K (or full pool); one extra angle pass for failure/dedup headroom.
    max_passes = angle_passes if full_coverage else min(angle_passes, 2)

    work: list[tuple[Persona, SampledChunk, int, str]] = []
    task_cursor = 0

    def _next_task_focus() -> str:
        nonlocal task_cursor
        focus = tasks[task_cursor % len(tasks)]
        task_cursor += 1
        return focus

    def _append_projected(anchor: SampledChunk) -> None:
        for persona in active:
            pool = pools.get(persona.name) or []
            chunk = _project_chunk_for_persona(persona, anchor, pool, rng)
            if chunk is None:
                continue
            try:
                pool_index = pool.index(chunk)
            except ValueError:
                pool_index = abs(hash((chunk.start, chunk.end, chunk.level))) % max(len(pool), 1)
            work.append((persona, chunk, pool_index, _next_task_focus()))

    if mode == "shared_anchor":
        anchors = _paragraph_anchor_chunks(tree)
        if not anchors:
            mode = "independent"
        else:
            for pass_i in range(max_passes):
                if full_coverage:
                    n_anchors = len(anchors)
                elif pass_i == 0:
                    n_anchors = min(len(anchors), slots_per_persona)
                else:
                    # Spare anchors (~50%) for retries after dedup/failures.
                    n_anchors = min(len(anchors), max(1, (slots_per_persona + 1) // 2))
                chosen = _sample_without_replacement(anchors, n_anchors, rng)
                for anchor in chosen:
                    _append_projected(anchor)

    if mode == "independent":
        work = []
        task_cursor = 0
        for pass_i in range(max_passes):
            for persona in active:
                pool = pools.get(persona.name) or []
                if not pool:
                    continue
                if full_coverage:
                    k = len(pool)
                elif pass_i == 0:
                    k = min(len(pool), slots_per_persona)
                else:
                    k = min(len(pool), max(1, (slots_per_persona + 1) // 2))
                chosen = _sample_without_replacement(pool, k, rng)
                for chunk in chosen:
                    try:
                        pool_index = pool.index(chunk)
                    except ValueError:
                        pool_index = 0
                    work.append((persona, chunk, pool_index, _next_task_focus()))

    if not work:
        return []

    rng.shuffle(work)
    return [
        (index, persona, chunk, pool_index, task_focus)
        for index, (persona, chunk, pool_index, task_focus) in enumerate(work)
    ]


def sample_chunk(
    tree: DocumentTree,
    persona: Persona,
    rng: random.Random,
    max_context_chars: int = 16_000,
) -> SampledChunk:
    """Compatibility helper: return one chunk from the pre-split candidate list."""
    del rng  # assignment is even/deterministic; kept for call-site compatibility
    chunks = list_candidate_chunks(tree, persona, max_context_chars=max_context_chars)
    return chunks[0]


def _paragraph_window(paragraphs: tuple[str, ...] | list[str], target_index: int, *, max_context_chars: int) -> str:
    """Expand around a target paragraph until the local window is usefully large."""
    if not paragraphs:
        return ""
    target_index = max(0, min(target_index, len(paragraphs) - 1))
    left = right = target_index
    window = paragraphs[target_index]
    min_chars = min(1_200, max_context_chars)
    while len(window) < min_chars and (left > 0 or right + 1 < len(paragraphs)):
        take_left = left > 0 and (right + 1 >= len(paragraphs) or (target_index - left) <= (right - target_index))
        if take_left:
            left -= 1
            neighbor = paragraphs[left]
            if not is_high_value_chunk(neighbor, minimum_chars=40):
                continue
            candidate = f"{neighbor}\n\n{window}"
        else:
            right += 1
            neighbor = paragraphs[right]
            if not is_high_value_chunk(neighbor, minimum_chars=40):
                continue
            candidate = f"{window}\n\n{neighbor}"
        if len(candidate) > max_context_chars and len(window) >= min(400, min_chars):
            break
        window = candidate
    return window[:max_context_chars]


def sample_pair_augmentation(
    rng: random.Random,
    *,
    task_focus: str | None = None,
    form_index: int | None = None,
) -> dict[str, str]:
    specificity = rng.choices(
        [
            "Preserve exact details that materially affect the task; remove only irrelevant source-specific identity.",
            "Generalize nonessential entity attributes to a truthful role, type, or category.",
            "When precision is not material, replace a nonessential exact number or date "
            "with a truthful bounded range or category.",
            "Use the minimum source details needed for a useful, answerable instruction.",
        ],
        weights=[0.35, 0.30, 0.20, 0.15],
        k=1,
    )[0]
    forms = [
        "Write a short direct question that names the subject.",
        "Write a crisp imperative that names the subject; pick a natural verb for the ask.",
        "Write a conversational request that names the subject.",
        "Ask why named subject X happens or how named mechanism M works.",
        "Ask for a contrast between two explicitly named things from CONTEXT.",
        "Give a short named scenario and ask how to apply a named rule or concept.",
        "Ask what follows from explicitly stated facts about a named subject.",
        "Ask whether a named claim is correct or supported, and why.",
        "Ask when a named approach should be used or avoided, with named criteria.",
        "Ask for a procedure or workflow, naming the artifact being worked on.",
    ]
    if form_index is None:
        instruction_form = rng.choice(forms)
    else:
        instruction_form = forms[form_index % len(forms)]
    return {
        "specificity": specificity,
        "entity_style": rng.choice(
            [
                "Introduce generalized entities by an indefinite role or type before referring to them.",
                "Prefer domain-relevant roles and categories over incidental names or identifiers.",
                "Retain a proper name or identifier only when it is central to the task or answer.",
            ]
        ),
        "instruction_form": instruction_form,
        "task_focus": (task_focus or "use a CONTEXT-supported task distinct from neighboring rows").strip(),
    }


def generate_pair(
    backend: InferenceBackend,
    persona: Persona,
    chunk: SampledChunk,
    header: str,
    *,
    require_standalone: bool,
    augmentation: dict[str, str],
    dataset_profile: DatasetProfile | None = None,
    repair_hint: str | None = None,
) -> GeneratedPair:
    profile_block = (
        f"Dataset profile:\n{dataset_profile.prompt_block()}\n"
        if dataset_profile is not None
        else "Dataset profile: none supplied; stay faithful to CONTEXT alone.\n"
    )
    task_focus = augmentation.get("task_focus") or "CONTEXT-supported task"
    repair_block = ""
    if repair_hint:
        repair_block = f"\nRewrite note: {repair_hint.strip()}\n"
    prompt = f"""Create exactly one source-grounded instruction/answer pair.
CONTEXT is private. Write the pair as reusable domain knowledge a reader can use with no other document open:
name the subjects, and put any needed definition, setting, formula, or short clause inside instruction+output.
{repair_block}
Persona: {persona.name}
Goal: {persona.description}
Question style (tendency only — rotate stems): {persona.question_style}
Standalone required: {str(require_standalone).lower()}

{profile_block}
How to use CONTEXT:
- Use only CONTEXT (and TARGET when provided). Do not invent document titles, study names, or author framing.
- If CONTEXT is from a study, article, paper, thesis, praxis, or trial: accept its statements as ground truth
  and emit a normal domain data point. Do not mention the study/article/paper/authors in instruction or output.
- If CONTEXT places a fact inside a course, experiment, or other setting, lift the reusable subject (named method,
  metric, procedure, finding) into the pair. Keep a setting name only when it is necessary to identify that
  subject; otherwise drop the packaging and ask about the subject directly.
- Assigned task focus is a cognitive goal label only — do NOT paste it as the instruction text.
- Assigned task focus for this row: {task_focus}
- Realize that goal with a fresh surface form. Vary opener, syntax, and length so this row does not look like
  a template with different nouns slotted in.
- Choose a task CONTEXT can fully answer. Prefer the stated fragment over inventing missing steps, numbers, or claims.
- If CONTEXT only poses a question without the answer material, pick a different supported task.

Variation profile for this example:
- Task focus (goal, not wording): {task_focus}
- Specificity: {augmentation["specificity"]}
- Entities: {augmentation["entity_style"]}
- Instruction form: {augmentation["instruction_form"]}

Target shape:
- Instruction form and shape examples are inspiration only — invent a natural wording for this row; do not
  copy an example stem or fill the same template across rows.
- Make the instruction sound like a real standalone ask: vary opener, length, and syntax from row to row.
- Ground every substantive claim in CONTEXT.
- Name the actual subject from CONTEXT on first use; include any defining detail the reader needs.
- Pronouns like "it"/"they"/"this" need a local named antecedent in the same instruction; otherwise name the subject.
- Ask about the named method, metric, procedure, or finding itself; do not hang the task on incidental
  study/course/trial/experiment setting language, and do not narrate that the fact came from research.
- Phrase the answer as a direct factual statement about the subject, not as a report of what a study found.
- For translate / summarize / classify / extract tasks, embed the target passage in the instruction, or reframe
  around a named proposition so the pair stays self-contained.
- Prefer indefinite roles when no proper name exists ("a tenant", "an API client"); keep proper names when
  they are central to correctness.
- Phrase both fields as ordinary statements about the named subject.
- Preserve names, numbers, formulas, and terminology required for a correct answer; apply the variation profile
  only when it preserves truth.
- If a fully supported answer is not possible, narrow the task rather than guessing.

Ground-truth framing (BAD → GOOD):
BAD: "What did the study find about Model X?" / "It found that Model X reduced latency by 31%."
GOOD: "How did Model X affect inference latency under the tested configuration?" /
      "Model X reduced inference latency by 31% under the tested configuration."

BAD: "According to the 2026 research, which factor predicted Y?" / "Factor Z."
GOOD: "Which factor predicted Y in the evaluated population?" /
      "Factor Z predicted Y in the evaluated population."

BAD: "What approach did the authors propose?" / "They proposed Adaptive Routing."
GOOD: "What approach uses dynamically selected routes to perform [named function]?" /
      "Adaptive Routing uses dynamically selected routes to perform [named function]."

BAD: "What happened in this experiment when temperature reached 80°C?" / "The reaction rate doubled."
GOOD: "What happens to the measured reaction rate at 80°C under the described reaction conditions?" /
      "The measured reaction rate doubles at 80°C under those conditions."

Local antecedents (reject bare pronouns; allow when the subject is named in-instruction):
BAD: "Why does it improve accuracy?"
GOOD: "Why does grouped-query attention reduce KV-cache memory, and how does it affect inference efficiency?"

Shape examples (illustrations of variety only — invent your own wording from CONTEXT):
- Medical: "What findings suggest Y?", "Explain the mechanism of Z", "When is treatment A contraindicated for X?"
- Legal: "Explain the rule that applies when a landlord retains a deposit." → answer: "The applicable rule is..."
- Software: "Tell me why a parser fails when input ends with a delimiter." → answer: "The parser fails because..."
- Math: "State the relationship between the differential of f at a and f'(a)." (include the formula when present)
- Research: "How does method A differ from method B on metric M under condition C?"
- Procedures: "Walk me through how named procedure P produces outcome O with the stated settings."

Before returning, silently check: grounded in CONTEXT, subjects named (or pronouns with local antecedents),
no study/article/author framing, wording not a copied template.

Return:
{{"instruction": "...", "output": "...", "standalone": true}}

TARGET:
{chunk.target_text}

CONTEXT:
{chunk.text}
"""
    value = _retry_json(
        backend,
        _PAIR_SYSTEM,
        prompt,
        max_new_tokens=900,
        attempts=3,
        enable_thinking=False,
    )
    if not isinstance(value, dict):
        raise ValueError("Q&A response must be a JSON object")
    pair = GeneratedPair.from_dict(value)
    if require_standalone and not pair.standalone:
        raise ValueError("Model did not generate a standalone instruction")
    return pair


def verify_pair_grounding(
    backend: InferenceBackend,
    *,
    chunk_text: str,
    instruction: str,
    output: str,
) -> dict[str, Any]:
    prompt = f"""Decide whether OUTPUT is fully grounded in CONTEXT for the given INSTRUCTION.
Use only CONTEXT. Do not reward answers that are merely plausible from general knowledge.
Mark grounded=false if OUTPUT adds any substantive fact, number, formula, step, name, rule, diagnosis,
holding, code behavior, or conclusion that CONTEXT does not support.
Paraphrase is allowed. Invention is not.
If the instruction asks for more than CONTEXT can support, mark grounded=false.
Critical case: if CONTEXT mainly contains unanswered questions or prompts without the supporting facts,
procedure, ruling, result, or explanation, and OUTPUT tries to answer them anyway, mark grounded=false.
The model must not fill in answers from outside knowledge when CONTEXT only asked the question.
When grounded=true, quote the exact supporting passages from CONTEXT in supporting_excerpts. Prefer short
verbatim excerpts over paraphrase. Include every passage needed to justify OUTPUT.

Return only:
{{
  "grounded": true,
  "supporting_excerpts": ["verbatim excerpt from CONTEXT that supports the output"],
  "unsupported_claims": ["optional short claim not supported by CONTEXT"],
  "reason": "one short sentence"
}}

INSTRUCTION:
{instruction}

OUTPUT:
{output}

CONTEXT:
{chunk_text}
"""
    value = _retry_json(backend, _GROUNDING_SYSTEM, prompt, max_new_tokens=500)
    if not isinstance(value, dict) or not isinstance(value.get("grounded"), bool):
        raise ValueError("Grounding response must include boolean grounded")
    unsupported = value.get("unsupported_claims", [])
    if unsupported is None:
        unsupported = []
    if not isinstance(unsupported, list):
        raise ValueError("unsupported_claims must be a list")
    supporting = value.get("supporting_excerpts", [])
    if supporting is None:
        supporting = []
    if not isinstance(supporting, list):
        raise ValueError("supporting_excerpts must be a list")
    reason = value.get("reason", "")
    if not isinstance(reason, str):
        reason = ""
    return {
        "grounded": value["grounded"],
        "supporting_excerpts": [str(item).strip() for item in supporting if str(item).strip()],
        "unsupported_claims": [str(item) for item in unsupported if str(item).strip()],
        "reason": reason.strip(),
    }


def generate_grounded_pair(
    backend: InferenceBackend,
    persona: Persona,
    tree: DocumentTree,
    *,
    header_prefix: str,
    summary: str,
    rng: random.Random,
    dataset_profile: DatasetProfile | None = None,
    max_attempts: int = 5,
    verify_grounding: bool = True,
    assigned_chunk: SampledChunk | None = None,
    chunk_pool: list[SampledChunk] | None = None,
    pool_index: int = 0,
    task_focus: str | None = None,
    form_index: int | None = None,
    deduper: InstructionDeduper | None = None,
) -> tuple[SampledChunk, GeneratedPair, dict[str, Any], dict[str, str], str]:
    """Generate a pair from a pre-assigned chunk (with ordered fallbacks).

    When verify_grounding is True, an extra LLM call checks CONTEXT support and
    rejects ungrounded answers. Grounding details are not written into the training dataset.
    """
    last_error: Exception | None = None
    # Do not copy the full pool list — iterate the shared reference.
    pool = chunk_pool if chunk_pool is not None else list_candidate_chunks(tree, persona)
    if not pool and assigned_chunk is None:
        raise ValueError("No candidate chunks available")

    ordered: list[SampledChunk] = []
    if assigned_chunk is not None:
        ordered.append(assigned_chunk)
    if pool:
        start = pool_index % len(pool)
        for offset in range(len(pool)):
            candidate = pool[(start + offset) % len(pool)]
            if assigned_chunk is not None and candidate is assigned_chunk:
                continue
            if assigned_chunk is not None and (
                candidate.start == assigned_chunk.start
                and candidate.end == assigned_chunk.end
                and candidate.section_path == assigned_chunk.section_path
            ):
                continue
            ordered.append(candidate)
    if not ordered:
        ordered = list(pool)

    for attempt_i, chunk in enumerate(ordered[:max_attempts]):
        if is_prompt_only_chunk(chunk.text) or is_non_content_text(chunk.text, heading=chunk.section_path):
            last_error = ValueError("Sampled chunk is prompt-only without answerable source material")
            continue
        header = f"{header_prefix}{chunk.section_path}\nDocument summary: {summary}"
        augmentation = sample_pair_augmentation(
            rng,
            task_focus=task_focus,
            form_index=(form_index + attempt_i) if form_index is not None else None,
        )
        repair_hint: str | None = None
        # A few local rewrites on the same chunk before falling through to another chunk.
        for _local_try in range(3):
            try:
                pair = generate_pair(
                    backend,
                    persona,
                    chunk,
                    header,
                    require_standalone=True,
                    augmentation=augmentation,
                    dataset_profile=dataset_profile,
                    repair_hint=repair_hint,
                )
            except Exception as exc:  # noqa: BLE001 - retry across chunk/pair attempts
                last_error = exc
                repair_hint = (
                    "Previous draft failed to parse or was incomplete. Produce one self-contained "
                    "instruction/output pair that names its subjects from CONTEXT."
                )
                continue
            failure = pair_self_containment_failure(pair.instruction, pair.output)
            if failure:
                last_error = ValueError(failure)
                repair_hint = (
                    "Previous draft was not independently understandable. Rewrite so the instruction "
                    "names the concrete subject from CONTEXT and states reusable domain knowledge; "
                    "put any needed defining detail inside the pair."
                )
                continue
            if deduper is not None and not deduper.try_claim(pair.instruction):
                last_error = ValueError("Duplicate, near-duplicate, or overused instruction stem rejected")
                repair_hint = (
                    "Previous draft reused a common instruction template or stem. Keep the same cognitive "
                    "goal, but rewrite with a different opener and syntactic shape; name the subject."
                )
                # Give one rewrite a chance; otherwise move to another chunk.
                if _local_try >= 1:
                    break
                continue
            if not verify_grounding:
                return (
                    chunk,
                    pair,
                    {
                        "grounded": None,
                        "supporting_excerpts": [],
                        "unsupported_claims": [],
                        "reason": "grounding check skipped",
                    },
                    augmentation,
                    header,
                )
            try:
                grounding = verify_pair_grounding(
                    backend,
                    chunk_text=chunk.text,
                    instruction=pair.instruction,
                    output=pair.output,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                repair_hint = (
                    "Previous draft could not be verified. Produce a narrower answer fully supported by CONTEXT."
                )
                continue
            if grounding["grounded"]:
                return chunk, pair, grounding, augmentation, header
            last_error = ValueError(
                "Generated pair failed grounding check: "
                + (grounding["reason"] or ", ".join(grounding["unsupported_claims"]) or "unsupported claims")
            )
            repair_hint = (
                "Previous draft added unsupported claims. Narrow the task so every answer claim is present "
                "in CONTEXT, and keep subjects named."
            )
    raise last_error or ValueError("Failed to generate a grounded pair")


def _existing_doc_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value.get("doc_id"), str):
                ids.add(value["doc_id"])
    return ids


def _existing_doc_record_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    with path.open() as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = value.get("doc_id")
            if isinstance(doc_id, str):
                counts[doc_id] = counts.get(doc_id, 0) + 1
    return counts


def _existing_doc_personas(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    pairs = set()
    with path.open() as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            persona = value.get("persona")
            if (
                isinstance(value.get("doc_id"), str)
                and isinstance(persona, dict)
                and isinstance(persona.get("name"), str)
            ):
                pairs.add((value["doc_id"], persona["name"].casefold()))
    return pairs


def _append_jsonl(path: Path, value: dict[str, Any], *, lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, default=str) + "\n"
    if lock is None:
        with path.open("a") as handle:
            handle.write(line)
        return
    with lock:
        with path.open("a") as handle:
            handle.write(line)


def _build_example_record(
    backend: InferenceBackend,
    *,
    document: SourceDocument,
    persona: Persona,
    tree: DocumentTree,
    summary: str,
    rng: random.Random,
    dataset_id: str,
    dataset_profile: DatasetProfile | None,
    seed: int,
    verify_grounding: bool = True,
    resolve_references: bool = False,
    assigned_chunk: SampledChunk | None = None,
    chunk_pool: list[SampledChunk] | None = None,
    pool_index: int = 0,
    task_focus: str | None = None,
    form_index: int | None = None,
    deduper: InstructionDeduper | None = None,
) -> dict[str, Any]:
    chunk, pair, grounding, augmentation, header = generate_grounded_pair(
        backend,
        persona,
        tree,
        header_prefix=f"{document.title} > ",
        summary=summary,
        rng=rng,
        dataset_profile=dataset_profile,
        verify_grounding=verify_grounding,
        assigned_chunk=assigned_chunk,
        chunk_pool=chunk_pool,
        pool_index=pool_index,
        task_focus=task_focus,
        form_index=form_index,
        deduper=deduper,
    )
    context = f"{header}\n\n{chunk.text}"
    token_counter = getattr(backend, "count_tokens", None)
    tokens = token_counter(context) if callable(token_counter) else max(1, len(context) // 4)

    reference_excerpts: list[dict[str, Any]] = []
    output_with_references = pair.output
    if resolve_references:
        # Cheap regex gate: only pay for an LLM resolve call when the answer
        # (or instruction) actually cites sections/clauses/etc.
        citations = extract_reference_mentions(f"{pair.instruction}\n{pair.output}")
        if citations:
            reference_excerpts = resolve_reference_excerpts(
                backend,
                tree,
                pair.instruction,
                pair.output,
                preferred_text=chunk.text,
            )
            output_with_references = format_output_with_reference_context(
                pair.output,
                reference_excerpts,
            )

    # Training projection only needs instruct/output; keep a slim operational record.
    record: dict[str, Any] = {
        "persona": persona.to_dict(),
        "instruction": pair.instruction,
        "output": pair.output,
        "dataset_id": dataset_id,
        "doc_id": document.doc_id,
        "doc_title": document.title,
        "section_path": chunk.section_path,
        "level": chunk.level,
        "method": chunk.method,
        "tokens": tokens,
        "model": backend.model_name,
        "standalone": pair.standalone,
        "decontextualized": True,
        "augmentation": augmentation,
        "seed": seed,
        "grounded": grounding.get("grounded"),
    }
    if resolve_references:
        record["reference_excerpts"] = reference_excerpts
        record["output_with_references"] = output_with_references
        record["context"] = context
        record["source_chunk"] = chunk.text
        record["target_text"] = chunk.target_text
        record["document_summary"] = summary
        record["dataset_profile"] = dataset_profile.to_dict() if dataset_profile else None
        record["source_metadata"] = document.metadata
        record["header"] = header
        record["offsets"] = {"start": chunk.start, "end": chunk.end, "basis": "cleaned_text"}
        record["grounding"] = grounding
    return record


def _persona_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().casefold()).strip("-")
    return slug or "persona"


def _persona_train_path(output_dir: Path, persona_name: str) -> Path:
    return output_dir / "god_train" / f"{_persona_slug(persona_name)}.jsonl"


def _write_example_outputs(
    record: dict[str, Any],
    *,
    records_path: Path,
    review_path: Path | None,
    train_dir: Path,
    lock: threading.Lock,
) -> None:
    persona_name = ""
    persona = record.get("persona")
    if isinstance(persona, dict) and isinstance(persona.get("name"), str):
        persona_name = persona["name"]
    train_path = train_dir / f"{_persona_slug(persona_name)}.jsonl"
    train_output = record.get("output_with_references") or record["output"]
    _append_jsonl(records_path, record, lock=lock)
    _append_jsonl(train_path, {"instruct": record["instruction"], "output": train_output}, lock=lock)
    if review_path is None:
        return
    _append_jsonl(
        review_path,
        {
            "persona": record["persona"],
            "generated": {
                "instruction": record["instruction"],
                "output": record["output"],
                "output_with_references": record.get("output_with_references") or record["output"],
            },
            "reference_excerpts": record.get("reference_excerpts") or [],
            "augmentation": record.get("augmentation"),
            "grounding": record.get("grounding"),
            "dataset_profile": record.get("dataset_profile"),
            "source": {
                "dataset_id": record.get("dataset_id"),
                "doc_id": record.get("doc_id"),
                "doc_title": record.get("doc_title"),
                "document_summary": record.get("document_summary"),
                "section_path": record.get("section_path"),
                "chunk": record.get("source_chunk"),
                "target_text": record.get("target_text"),
                "offsets": record.get("offsets"),
            },
        },
        lock=lock,
    )


def estimate_generation_capacity(
    tree: DocumentTree,
    personas: list[Persona],
    *,
    dataset_profile: DatasetProfile | None = None,
    angle_passes: int | None = None,
) -> tuple[int, dict[str, int]]:
    """Honest upper bound: every persona × full chunk pool × angle passes."""
    profile_tasks = dataset_profile.preferred_task_types if dataset_profile is not None else ()
    passes = angle_passes if angle_passes is not None else max(1, min(4, len(profile_tasks) or 3))
    pool_sizes: dict[str, int] = {}
    total = 0
    for persona in personas:
        try:
            size = len(list_candidate_chunks(tree, persona))
        except ValueError:
            size = 0
        pool_sizes[persona.name] = size
        total += size * passes
    return max(0, total), pool_sizes


def run_pipeline(
    backend: InferenceBackend,
    documents: list[SourceDocument],
    personas: list[Persona],
    output_dir: Path,
    *,
    dataset_id: str,
    examples_per_doc: int = 3,
    seed: int = 42,
    resume: bool = True,
    all_personas_per_doc: bool = False,
    dataset_profile: DatasetProfile | None = None,
    workers: int = 1,
    verify_grounding: bool = True,
    resolve_references: bool = False,
    write_review: bool = False,
    persona_quotas: dict[str, int] | None = None,
    target_rows: int | None = None,
    sample_mode: str = "auto",
    on_progress: Callable[[str, dict[str, Any] | None], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, int]:
    """Generate rows from native chunk pools with random subsample when partial.

    ``target_rows`` is the combined accept target. When below capacity, chunks are
    randomly subsampled (shared paragraph anchors for multi-persona jobs by
    default). Chunk units stay at each persona's native level — never enlarged.
    ``persona_quotas`` is ignored (kept for call-site compatibility).
    """
    del persona_quotas  # random subsample / full-pool coverage replaces quota slicing
    records_path = output_dir / "records.jsonl"
    review_path = (output_dir / "review.jsonl") if write_review else None
    train_dir = output_dir / "god_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failures.jsonl"
    completed_pairs = _existing_doc_personas(records_path) if resume and all_personas_per_doc else set()
    existing_counts = _existing_doc_record_counts(records_path) if resume else {}
    counts = {"documents": 0, "records": 0, "failures": 0, "skipped": 0}
    write_lock = threading.Lock()
    accept_lock = threading.Lock()
    worker_count = max(1, workers)
    deduper = InstructionDeduper(store_path=output_dir / "dedupe.db")
    goal = max(1, int(target_rows or examples_per_doc or 1))
    progress_every = max(1, min(50, goal // 20 or 1))
    last_reported = 0
    logger.info(
        "pipeline: start docs={} personas={} goal={} workers={}",
        len(documents),
        [p.name for p in personas],
        goal,
        worker_count,
    )

    def _ensure_not_cancelled() -> None:
        if is_cancelled is not None and is_cancelled():
            raise StructureJobCancelled("cancelled by user")

    def _emit(message: str, *, error: bool = False) -> None:
        _ensure_not_cancelled()
        payload = {
            "stage": "generate",
            "goal": goal,
            "documents": counts["documents"],
            "records": counts["records"],
            "failures": counts["failures"],
            "skipped": counts["skipped"],
        }
        # Prefer job-store callback logging (worker → nohup); fall back when no callback.
        if error:
            if on_progress is not None:
                on_progress(f"error: {message}", payload)
            else:
                logger.error("pipeline: {}", message)
            return
        if on_progress is not None:
            on_progress(message, payload)
        else:
            logger.info("pipeline: {}", message)

    for document in documents:
        if counts["records"] >= goal:
            break
        if all_personas_per_doc and all(
            (document.doc_id, persona.name.casefold()) in completed_pairs for persona in personas
        ):
            counts["skipped"] += 1
            continue
        if not all_personas_per_doc and existing_counts.get(document.doc_id, 0) >= examples_per_doc:
            counts["skipped"] += 1
            continue
        try:
            _emit(f"parsing document {document.doc_id}")
            tree = parse_document(document)
            summary = summarize_document(backend, tree, workers=worker_count)
            if not tree.sections:
                raise ValueError("No useful prose remained after filtering")
            counts["documents"] += 1

            active_personas = [
                persona
                for persona in personas
                if (document.doc_id, persona.name.casefold()) not in completed_pairs
            ] or list(personas)

            pools: dict[str, list[SampledChunk]] = {}
            for persona in active_personas:
                try:
                    pools[persona.name] = list_candidate_chunks(tree, persona)
                except ValueError:
                    pools[persona.name] = []

            profile_tasks = (
                list(dataset_profile.preferred_task_types) if dataset_profile is not None else []
            )
            if not profile_tasks:
                profile_tasks = list(DEFAULT_TASK_FOCUSES)

            work = build_generation_work(
                tree,
                active_personas,
                pools,
                goal=goal - counts["records"],
                profile_tasks=profile_tasks,
                seed=seed ^ (hash(document.doc_id) & 0xFFFFFFFF),
                sample_mode=sample_mode,
            )
            if not work:
                raise ValueError("No usable chunks for any persona")
            work_index = output_dir / "work_index.jsonl"
            with work_index.open("a", encoding="utf-8") as handle:
                for index, persona, chunk, pool_index, task_focus in work:
                    handle.write(
                        json.dumps(
                            {
                                "doc_id": document.doc_id,
                                "index": index,
                                "persona": persona.name,
                                "pool_index": pool_index,
                                "task_focus": task_focus,
                                "start": chunk.start,
                                "end": chunk.end,
                            }
                        )
                        + "\n"
                    )
            _emit(f"generating from {document.doc_id}: {len(work)} chunk tasks queued")

            stop_flag = threading.Event()

            def _report_accept() -> None:
                nonlocal last_reported
                if (
                    counts["records"] == goal
                    or counts["records"] - last_reported >= progress_every
                    or counts["records"] <= 5
                ):
                    last_reported = counts["records"]
                    _emit(f"accepted {counts['records']}/{goal} rows ({counts['failures']} failures)")

            def _run_one(
                index: int,
                persona: Persona,
                chunk: SampledChunk,
                pool_index: int,
                task_focus: str,
            ) -> bool:
                _ensure_not_cancelled()
                if stop_flag.is_set():
                    return False
                with accept_lock:
                    if counts["records"] >= goal:
                        stop_flag.set()
                        return False
                local_rng = random.Random((seed + 1_000_003 * index) ^ (hash(document.doc_id) & 0xFFFFFFFF))
                record = _build_example_record(
                    backend,
                    document=document,
                    persona=persona,
                    tree=tree,
                    summary=summary,
                    rng=local_rng,
                    dataset_id=dataset_id,
                    dataset_profile=dataset_profile,
                    seed=seed,
                    verify_grounding=verify_grounding,
                    resolve_references=resolve_references,
                    assigned_chunk=chunk,
                    chunk_pool=pools.get(persona.name) or [chunk],
                    pool_index=pool_index,
                    task_focus=task_focus,
                    form_index=index,
                    deduper=deduper,
                )
                with accept_lock:
                    if counts["records"] >= goal:
                        stop_flag.set()
                        return False
                    _write_example_outputs(
                        record,
                        records_path=records_path,
                        review_path=review_path,
                        train_dir=train_dir,
                        lock=write_lock,
                    )
                    counts["records"] += 1
                    _report_accept()
                    if counts["records"] >= goal:
                        stop_flag.set()
                    return True

            if worker_count <= 1 or len(work) <= 1:
                for index, persona, chunk, pool_index, task_focus in work:
                    if stop_flag.is_set() or counts["records"] >= goal:
                        break
                    try:
                        _run_one(index, persona, chunk, pool_index, task_focus)
                    except StructureJobCancelled:
                        raise
                    except Exception as exc:
                        _append_jsonl(
                            failures_path,
                            {
                                "dataset_id": dataset_id,
                                "doc_id": document.doc_id,
                                "persona": persona.name,
                                "error": type(exc).__name__,
                                "message": str(exc),
                            },
                            lock=write_lock,
                        )
                        with accept_lock:
                            counts["failures"] += 1
                        _emit(f"{type(exc).__name__} on {persona.name}/{document.doc_id}: {exc}", error=True)
            else:
                # Keep only worker_count futures in flight — submitting the full
                # 20k-item queue at once OOMs small pods.
                max_inflight = min(worker_count, len(work))
                with ThreadPoolExecutor(max_workers=max_inflight) as pool:
                    work_iter = iter(work)
                    futures: dict[Any, Persona] = {}
                    pending: set[Any] = set()

                    def _submit_one() -> bool:
                        try:
                            index, persona, chunk, pool_index, task_focus = next(work_iter)
                        except StopIteration:
                            return False
                        fut = pool.submit(_run_one, index, persona, chunk, pool_index, task_focus)
                        futures[fut] = persona
                        pending.add(fut)
                        return True

                    while len(pending) < max_inflight and _submit_one():
                        pass

                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            persona = futures.pop(future)
                            try:
                                future.result()
                            except StructureJobCancelled:
                                stop_flag.set()
                                for leftover in pending:
                                    leftover.cancel()
                                raise
                            except Exception as exc:
                                _append_jsonl(
                                    failures_path,
                                    {
                                        "dataset_id": dataset_id,
                                        "doc_id": document.doc_id,
                                        "persona": persona.name,
                                        "error": type(exc).__name__,
                                        "message": str(exc),
                                    },
                                    lock=write_lock,
                                )
                                with accept_lock:
                                    counts["failures"] += 1
                                _emit(
                                    f"{type(exc).__name__} on {persona.name}/{document.doc_id}: {exc}",
                                    error=True,
                                )
                            if stop_flag.is_set() or counts["records"] >= goal:
                                stop_flag.set()
                                for leftover in pending:
                                    leftover.cancel()
                                pending.clear()
                                break
                            _submit_one()

        except StructureJobCancelled:
            raise
        except Exception as exc:
            _append_jsonl(
                failures_path,
                {"dataset_id": dataset_id, "doc_id": document.doc_id, "error": type(exc).__name__, "message": str(exc)},
                lock=write_lock,
            )
            counts["failures"] += 1
            _emit(f"{type(exc).__name__} on document {document.doc_id}: {exc}", error=True)

    deduper.close()
    logger.info("pipeline: finished counts={}", counts)
    return counts
