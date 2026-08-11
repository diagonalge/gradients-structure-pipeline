from __future__ import annotations

import json
import re
from typing import Any

from .inference import InferenceBackend
from .models import DocumentTree

# Named units: section 2.1, Theorem 3, Schedule 3, § 12, etc.
_NAMED_REFERENCE_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"\b(?P<kind>"
    r"articles?|sections?|chapters?|clauses?|regulations?|schedules?|"
    r"paragraphs?|appendices|appendix|parts?|figures?|tables?|"
    r"theorems?|lemmas?|propositions?|corollaries?|equations?|"
    r"definitions?|examples?|exercises?|rules?|provisions?"
    r")\s+(?P<label>[A-Z]?\d+(?:[.:\-]\d+)*(?:\([a-z0-9]+\))?[A-Za-z]?)\b"
    r"|\b(?P<abbr>art\.|sec\.|ch\.|reg\.|para\.|fig\.|eq\.)\s*"
    r"(?P<abbr_label>[A-Z]?\d+(?:[.:\-]\d+)*(?:\([a-z0-9]+\))?[A-Za-z]?)\b"
    r"|§\s*(?P<section_mark>\d+(?:[.:\-]\d+)*)\b"
    r")"
)

# Opaque layout labels that only make sense with the source outline.
_CLAUSE_LABEL_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"(?:sub-)?paragraphs?\s+\((?P<p_letter>[a-z0-9]+)\)(?:\((?P<p_sub>[ivx]{1,4}|\d+)\))?"
    r"|\((?P<letter>[a-z])\)\((?P<sub>[ivx]{1,4}|\d+)\)"
    r")"
)

_KIND_ALIASES = {
    "art": "article",
    "article": "article",
    "articles": "article",
    "sec": "section",
    "section": "section",
    "sections": "section",
    "ch": "chapter",
    "chapter": "chapter",
    "chapters": "chapter",
    "clause": "clause",
    "clauses": "clause",
    "reg": "regulation",
    "regulation": "regulation",
    "regulations": "regulation",
    "schedule": "schedule",
    "schedules": "schedule",
    "para": "paragraph",
    "paragraph": "paragraph",
    "paragraphs": "paragraph",
    "appendix": "appendix",
    "appendices": "appendix",
    "part": "part",
    "parts": "part",
    "fig": "figure",
    "figure": "figure",
    "figures": "figure",
    "table": "table",
    "tables": "table",
    "theorem": "theorem",
    "theorems": "theorem",
    "lemma": "lemma",
    "lemmas": "lemma",
    "proposition": "proposition",
    "propositions": "proposition",
    "corollary": "corollary",
    "corollaries": "corollary",
    "eq": "equation",
    "equation": "equation",
    "equations": "equation",
    "definition": "definition",
    "definitions": "definition",
    "example": "example",
    "examples": "example",
    "exercise": "exercise",
    "exercises": "exercise",
    "rule": "rule",
    "rules": "rule",
    "provision": "provision",
    "provisions": "provision",
}

_RESOLVE_SYSTEM = """You resolve citations using only the supplied CONTEXT.
Expand every cited unit into the concrete content it states (the duty, amount, item, or rule).
Do not invent missing text. Never write topic blurbs like "refers to X" or "is a key component".
If a numeric amount is not in CONTEXT, say so explicitly while stating what the unit requires.
Return only valid JSON."""


def extract_reference_mentions(text: str, *, limit: int = 12) -> list[dict[str, str]]:
    """Pull named citations and opaque clause labels from free text."""
    mentions: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(kind: str, label: str, raw: str, reference: str | None = None) -> None:
        nonlocal mentions
        reference = reference or f"{kind} {label}"
        key = reference.casefold()
        if key in seen:
            return
        seen.add(key)
        mentions.append({"kind": kind, "label": label, "reference": reference, "raw": raw})

    for match in _NAMED_REFERENCE_RE.finditer(text or ""):
        if match.group("kind"):
            kind_raw = match.group("kind")
            label = match.group("label")
        elif match.group("abbr"):
            kind_raw = match.group("abbr").rstrip(".")
            label = match.group("abbr_label")
        else:
            kind_raw = "section"
            label = match.group("section_mark")
        kind = _KIND_ALIASES.get(kind_raw.casefold().rstrip("."), kind_raw.casefold())
        label = (label or "").strip()
        if label:
            _add(kind, label, match.group(0).strip())
        if len(mentions) >= limit:
            return mentions

    for match in _CLAUSE_LABEL_RE.finditer(text or ""):
        if match.group("letter"):
            letter = match.group("letter")
            sub = match.group("sub")
            label = f"({letter})({sub})"
        else:
            letter = match.group("p_letter")
            sub = match.group("p_sub")
            label = f"({letter})({sub})" if sub else f"({letter})"
        raw = match.group(0).strip()
        _add("clause", label, raw, reference=f"clause {label}")
        if len(mentions) >= limit:
            break
    return mentions


def _build_lookup_context(
    tree: DocumentTree,
    *,
    preferred_text: str | None,
    max_chars: int,
) -> str:
    parts: list[str] = []
    if preferred_text and preferred_text.strip():
        parts.append("SAMPLED PASSAGE:\n" + preferred_text.strip())
    for section in tree.sections:
        body = section.text.strip()
        if not body:
            continue
        parts.append(f"SECTION {section.path or section.title}:\n{body}")
    if not parts and tree.cleaned_text.strip():
        parts.append(tree.cleaned_text.strip())
    joined = "\n\n".join(parts)
    if len(joined) <= max_chars:
        return joined
    if preferred_text and preferred_text.strip():
        head = ("SAMPLED PASSAGE:\n" + preferred_text.strip())[: max_chars // 3]
        remaining = max_chars - len(head) - 20
        rest = tree.cleaned_text.strip()
        return head + "\n\nDOCUMENT:\n" + rest[:remaining]
    return joined[:max_chars]


def resolve_reference_excerpts(
    backend: InferenceBackend,
    tree: DocumentTree,
    instruction: str,
    output: str,
    *,
    preferred_text: str | None = None,
    max_context_chars: int = 28_000,
) -> list[dict[str, Any]]:
    """
    Use the LLM to expand cited units into the concrete content they state.

    Detects named citations and opaque clause labels such as (a)(ii); resolution is
    model-based and domain-agnostic.
    """
    probe = f"{instruction}\n{output}"
    mentions = extract_reference_mentions(probe)
    if not mentions:
        return []

    context = _build_lookup_context(tree, preferred_text=preferred_text, max_chars=max_context_chars)
    citation_list = [item["reference"] for item in mentions]
    prompt = f"""The ANSWER cites labeled source units and/or opaque clause labels. For each citation,
search CONTEXT and state the concrete content of that unit.

Citations include names like "section 30" / "theorem 3" and clause labels like "clause (a)(ii)"
or "sub-paragraph (b)(i)".

Hard rules:
- Use only CONTEXT. If missing or too ambiguous, set found=false and explain briefly in reason.
- Prefer the most specific match. For clause labels such as (a)(ii), resolve them inside the
  surrounding provision in SAMPLED PASSAGE / CONTEXT (e.g. expand (a)(ii) to the year-before
  gross expenditure limb), not some unrelated "(a)" elsewhere.
- statement must state the CONTENT, not the topic:
  BAD: "Section 30 refers to amounts for different categories of dwellings."
  BAD: "Section 30 is a key component in the calculation of council tax."
  GOOD: "Under section 30 of the Act the billing authority sets the council-tax amounts for
  different categories of dwellings. CONTEXT does not give the numeric figures themselves;
  it requires the notice to state those amounts / the percentage change in them."
  BAD: "Clause (a)(ii) is part of the information list."
  GOOD: "Clause (a)(ii): the levying body's gross expenditure for the year before the levy year."
  GOOD: "Clause (b)(ii): the amount of the body's levy for the year before the levy year
  (if a levy was issued)."
- If ANSWER says "the amounts mentioned in (a)(i) and (b)(i)", expand EACH label—(a)(i),
  (b)(i), (a)(ii), (b)(ii)—to the actual item those labels denote in CONTEXT. Do not omit
  any of them.
- excerpt: short verbatim quote from CONTEXT that contains the resolved content.
- Keep excerpt under ~500 characters.

Citations to resolve:
{json.dumps(citation_list, ensure_ascii=False)}

Also resolve any additional clause labels like (a)(ii) that appear in ANSWER even if omitted above.

Return only:
{{
  "references": [
    {{
      "reference": "clause (a)(ii)",
      "found": true,
      "statement": "concrete content this unit states",
      "excerpt": "verbatim supporting quote from CONTEXT",
      "reason": ""
    }}
  ]
}}

INSTRUCTION:
{instruction}

ANSWER:
{output}

CONTEXT:
{context}
"""
    value = backend.generate_json(_RESOLVE_SYSTEM, prompt, max_new_tokens=1_200)
    if not isinstance(value, dict):
        raise ValueError("Reference resolver must return a JSON object")
    raw_items = value.get("references", [])
    if not isinstance(raw_items, list):
        raise ValueError("Reference resolver references must be a list")

    by_name = {item["reference"].casefold(): item for item in mentions}
    findings: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        reference = str(item.get("reference") or "").strip()
        if not reference:
            continue
        found = bool(item.get("found"))
        statement = str(item.get("statement") or "").strip()
        # Avoid leaking prompt vocabulary into user-facing expansions.
        statement = (
            statement.replace("CONTEXT does not", "This document does not")
            .replace("CONTEXT ", "the source ")
            .replace("CONTEXT", "the source")
        )
        excerpt = str(item.get("excerpt") or "").strip()
        mention = by_name.get(reference.casefold())
        findings.append(
            {
                "reference": reference,
                "raw": mention["raw"] if mention else reference,
                "matched": found and bool(statement or excerpt),
                "statement": statement or None,
                "excerpt": excerpt or None,
                "reason": None if found else str(item.get("reason") or "not found in context"),
                "resolver": "llm",
            }
        )

    present = {str(item["reference"]).casefold() for item in findings}
    for mention in mentions:
        if mention["reference"].casefold() in present:
            continue
        findings.append(
            {
                "reference": mention["reference"],
                "raw": mention["raw"],
                "matched": False,
                "statement": None,
                "excerpt": None,
                "reason": "omitted by resolver",
                "resolver": "llm",
            }
        )
    return findings


_REFERENCE_MARKER = "What the referenced material states:"


def format_output_with_reference_context(output: str, excerpts: list[dict[str, Any]]) -> str:
    """Append model-resolved citation explanations on a new line after the answer."""
    matched = [item for item in excerpts if item.get("matched") and (item.get("statement") or item.get("excerpt"))]
    if not matched:
        return output
    if _REFERENCE_MARKER in output:
        return output
    blocks = [output.strip(), "", _REFERENCE_MARKER]
    for item in matched:
        blocks.append("")
        blocks.append(f"[{item['reference']}]")
        if item.get("statement"):
            blocks.append(str(item["statement"]).strip())
        if item.get("excerpt"):
            blocks.append(f'Source: "{str(item["excerpt"]).strip()}"')
    return "\n".join(blocks)
