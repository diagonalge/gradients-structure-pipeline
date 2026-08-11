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
_REFERENCE_HEADING_RE = re.compile(r"^(references|bibliography|works cited|table of contents|contents)$", re.IGNORECASE)
_BOILERPLATE_RE = re.compile(
    r"^(?:page\s+\d+(?:\s+of\s+\d+)?|copyright\b.*|all rights reserved\.?|https?://\S+)$",
    re.IGNORECASE,
)
_MIN_PARAGRAPH_CHARS = 200


def clean_document_text(raw_text: str) -> str:
    # Preserve PDF page breaks so chunk_level=page can sample real pages.
    if "\f" in raw_text:
        pages = [clean_document_text(page) for page in raw_text.split("\f")]
        return "\f".join(page for page in pages if page.strip())

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
            skip_references = bool(_REFERENCE_HEADING_RE.match(heading))
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

    return DocumentTree(document=document, cleaned_text=cleaned, sections=tuple(sections))
