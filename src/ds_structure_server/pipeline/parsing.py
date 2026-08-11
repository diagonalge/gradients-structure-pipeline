from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

from .models import DocumentTree, Section, SourceDocument

_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+.+")
_NAMED_HEADING_RE = re.compile(
    r"^(?:section|chapter|part|article)\s+[A-Z0-9IVXLC.-]+(?:\s*[:—-]\s*.+)?$",
    re.IGNORECASE,
)
_SECTION_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)+\s+\S+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# Navigational / bibliographic / admin headings — skip body under these.
_NON_CONTENT_HEADING_RE = re.compile(
    r"(?i)^(?:#+\s*)?(?:"
    r"table of contents|contents|list of (?:figures|tables|illustrations|algorithms|abbreviations)|"
    r"index|glossary|nomenclature|colophon|"
    r"references|bibliography|works cited|further reading|"
    r"acknowledg(?:e)?ments?|dedication|preface|foreword|"
    r"copyright(?: page)?|license|permissions?|"
    r"about the authors?|about this (?:book|document|guide)|"
    r"revision history|document control|change log|version history|"
    r"approval(?: sheet)?|candidate information|submission (?:guidelines?|requirements?|procedures?)|"
    r"formatting (?:guidelines?|requirements?|instructions?)|"
    r"information to all users|publisher(?:'s)? note"
    r")\s*$"
)
_REFERENCE_HEADING_RE = _NON_CONTENT_HEADING_RE  # backward-compatible alias
_BOILERPLATE_RE = re.compile(
    r"^(?:page\s+\d+(?:\s+of\s+\d+)?|copyright\b.*|all rights reserved\.?|https?://\S+)$",
    re.IGNORECASE,
)
# Lexical cues common to packaging / legal / TOC pages across publishers and formats.
_NON_CONTENT_CUE_RE = re.compile(
    r"(?i)\b(?:"
    r"table of contents|list of (?:figures|tables|illustrations)|"
    r"all rights reserved|copyright (?:©|\(c\)|notice)|creative commons|"
    r"reproduced (?:with permission|by permission)|information to all users|"
    r"approval sheet|candidate information|submission (?:guideline|requirement|procedure)|"
    r"formatting (?:guideline|requirement|instruction)|document control|"
    r"revision history|version history|change log|"
    r"this (?:work|document|publication) is protected|"
    r"no part of this (?:publication|work|document) may be|"
    r"printed in the united states|\bissn\b|\bisbn\b"
    r")\b"
)
_TOC_LEADER_RE = re.compile(r"\.{3,}|…{1,}|·{3,}|\s+\d{1,4}\s*$")
_MIN_PARAGRAPH_CHARS = 200


def is_non_content_heading(heading: str) -> bool:
    """True for navigational/admin/bib headings that should not drive instruct pairs."""
    value = (heading or "").strip().lstrip("#").strip()
    if not value or len(value) > 160:
        return False
    return bool(_NON_CONTENT_HEADING_RE.match(value))


def is_non_content_text(text: str, *, heading: str | None = None) -> bool:
    """Generic detector for TOC, legal, packaging, and other non-substantive passages.

    Uses structure + cue density rather than a single corpus-specific brand list, so it
    applies across PDFs, handbooks, specs, and scraped docs.
    """
    value = (text or "").strip()
    if not value:
        return True
    if heading and is_non_content_heading(heading):
        return True

    # Short banner / footer only.
    if len(value) < 80 and (_BOILERPLATE_RE.match(value) or _NON_CONTENT_CUE_RE.search(value)):
        return True

    head = value[:1_200]
    cue_hits = len(_NON_CONTENT_CUE_RE.findall(head))
    if cue_hits >= 2 and len(value) < 4_000:
        return True
    if cue_hits >= 1 and len(value) < 600:
        return True

    lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
    short_ratio = 0.0
    leader_ratio = 0.0
    page_ratio = 0.0
    if len(lines) >= 6:
        short = sum(1 for ln in lines if len(ln) <= 60)
        leaders = sum(1 for ln in lines if _TOC_LEADER_RE.search(ln))
        pageish = sum(1 for ln in lines if re.search(r"\b\d{1,4}\s*$", ln) and len(ln) <= 80)
        short_ratio = short / len(lines)
        leader_ratio = leaders / len(lines)
        page_ratio = pageish / len(lines)
        # Classic TOC / index layout: many short lines with leaders or trailing page nums.
        if short_ratio >= 0.7 and (leader_ratio >= 0.35 or page_ratio >= 0.45):
            return True
        if leader_ratio >= 0.5 and len(lines) >= 8:
            return True

    # High digit / low prose density (page lists, form fields, revision tables).
    alpha = sum(ch.isalpha() for ch in value)
    digit = sum(ch.isdigit() for ch in value)
    alpha_ratio = alpha / max(len(value), 1)
    digit_ratio = digit / max(len(value), 1)
    if (
        len(value) >= 200
        and digit_ratio >= 0.28
        and alpha_ratio < 0.40
        and (cue_hits >= 1 or (len(lines) >= 8 and short_ratio >= 0.65))
    ):
        return True

    words = re.findall(r"[A-Za-z]{3,}", value)
    if len(words) >= 30:
        unique = len({w.casefold() for w in words})
        if unique / len(words) < 0.28 and cue_hits >= 1:
            return True
    return False


def clean_document_text(raw_text: str) -> str:
    # Preserve PDF page breaks so chunk_level=page can sample real pages.
    if "\f" in raw_text:
        pages = []
        for page in raw_text.split("\f"):
            cleaned = clean_document_text(page)
            if cleaned.strip() and not is_non_content_text(cleaned):
                pages.append(cleaned)
        return "\f".join(pages)

    text = raw_text
    if re.search(r"<(?:html|body|p|div|h[1-6]|br|table)\b", text, re.IGNORECASE):
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()
        text = soup.get_text("\n")
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return reflow_hard_wrapped_text(text.strip())


def reflow_hard_wrapped_text(text: str) -> str:
    """Rebuild paragraphs when OCR/PDF text has one physical line per fragment."""
    lines = [line.strip() for line in text.splitlines()]
    if not lines:
        return ""

    nonempty = [line for line in lines if line]
    if not nonempty:
        return ""

    avg_len = sum(len(line) for line in nonempty) / len(nonempty)
    blank_ratio = (len(lines) - len(nonempty)) / max(len(lines), 1)
    # Born-digital prose already has real paragraphs; leave it alone.
    # OCR/PDF wraps look like short lines separated by frequent blank lines.
    hard_wrapped = avg_len < 65 and blank_ratio >= 0.3 and len(nonempty) >= 4
    if not hard_wrapped:
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(" ".join(current))
            current = []

    for line in nonempty:
        if _is_heading(line) or _BOILERPLATE_RE.match(line):
            flush()
            blocks.append(line)
            continue
        if _SECTION_NUMBER_RE.match(line):
            flush()
            current = [line]
            continue
        if current and _should_start_new_paragraph(current, line):
            flush()
        current.append(line)
    flush()
    return "\n\n".join(blocks)


def _should_start_new_paragraph(current: list[str], nxt: str) -> bool:
    joined = " ".join(current)
    if len(joined) < _MIN_PARAGRAPH_CHARS:
        return False
    prev = current[-1]
    if not prev.endswith((".", "!", "?", ":", ";")):
        return False
    if _SECTION_NUMBER_RE.match(nxt) or _is_heading(nxt):
        return True
    first = nxt[:1]
    return first.isupper() or first.isdigit()


def split_sentences(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _SENTENCE_RE.split(text) if part.strip())


def _is_heading(line: str) -> bool:
    candidate = line.strip()
    if not candidate or len(candidate) > 120 or candidate.endswith((".", "?", "!")):
        return False
    # Textbook running headers/footers often include a page number.
    if re.match(r"^\d{1,3}\s+\S+", candidate) or re.search(r"\b\d{1,3}$", candidate):
        return False
    if _MARKDOWN_HEADING_RE.match(candidate) or _NAMED_HEADING_RE.match(candidate):
        return True
    letters = [char for char in candidate if char.isalpha()]
    return len(letters) >= 4 and sum(char.isupper() for char in letters) / len(letters) >= 0.8


def _useful_paragraph(paragraph: str) -> bool:
    value = paragraph.strip()
    if len(value) < 40 or _BOILERPLATE_RE.match(value):
        return False
    words = re.findall(r"\b[\w'-]+\b", value)
    if len(words) < 8:
        return False
    alpha = sum(char.isalpha() for char in value)
    return alpha / max(len(value), 1) >= 0.45


def _split_paragraphs(body: str) -> tuple[str, ...]:
    return tuple(
        paragraph.strip() for paragraph in re.split(r"\n\s*\n|\n(?=[•*-]\s)", body) if _useful_paragraph(paragraph)
    )


def parse_document(document: SourceDocument) -> DocumentTree:
    cleaned = clean_document_text(document.text)
    lines = cleaned.splitlines()
    sections: list[Section] = []
    heading = document.title
    body_lines: list[str] = []
    body_start = 0
    cursor = 0
    skip_references = False

    def flush(end: int) -> None:
        nonlocal body_lines
        body = "\n".join(body_lines).strip()
        paragraphs = _split_paragraphs(body)
        if paragraphs:
            section_text = "\n\n".join(paragraphs)
            sections.append(
                Section(
                    title=heading,
                    path=heading,
                    text=section_text,
                    start=body_start,
                    end=end,
                    paragraphs=paragraphs,
                )
            )
        body_lines = []

    for line in lines:
        stripped = line.strip()
        line_end = cursor + len(line)
        if _is_heading(stripped):
            flush(cursor)
            heading = stripped.lstrip("#").strip()
            body_start = line_end + 1
            skip_references = is_non_content_heading(heading)
        elif not skip_references and not _BOILERPLATE_RE.match(stripped):
            body_lines.append(line)
        cursor = line_end + 1
    flush(len(cleaned))

    if not sections:
        paragraphs = _split_paragraphs(cleaned)
        if not paragraphs and _useful_paragraph(cleaned):
            paragraphs = (cleaned,)
        sections = (
            [Section(document.title, document.title, "\n\n".join(paragraphs), 0, len(cleaned), paragraphs)]
            if paragraphs
            else []
        )

    # Drop sections whose heading or body is packaging / TOC / legal-only.
    filtered = tuple(
        section
        for section in sections
        if not is_non_content_heading(section.title) and not is_non_content_text(section.text, heading=section.title)
    )
    if filtered:
        sections = filtered

    return DocumentTree(document=document, cleaned_text=cleaned, sections=tuple(sections))
