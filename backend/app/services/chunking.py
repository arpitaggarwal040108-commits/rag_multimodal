import re
from typing import Any

PAGE_MARKER = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)
DEFAULT_TARGET_WORDS = 420
DEFAULT_OVERLAP_WORDS = 60


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _split_oversize_sentence(sentence: str, target_words: int, overlap_words: int) -> list[str]:
    words = sentence.split()
    if len(words) <= target_words:
        return [sentence.strip()]
    step = max(1, target_words - overlap_words)
    return [" ".join(words[start:start + target_words]) for start in range(0, len(words), step)]


def _page_sections(text: str, pages: list[dict] | None) -> list[tuple[int | None, str]]:
    matches = list(PAGE_MARKER.finditer(text or ""))
    sections: list[tuple[int | None, str]] = []
    if matches:
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((int(match.group(1)), body))
    elif pages:
        for item in pages:
            body = str(item.get("text", "")).strip()
            if body:
                sections.append((int(item.get("page", 1)), body))
    elif text and text.strip():
        sections.append((None, text.strip()))
    return sections


def semantic_chunk(
    text: str,
    source: str,
    target_words: int = DEFAULT_TARGET_WORDS,
    pages: list[dict] | None = None,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict]:
    """Split text by page and sentence boundaries with controlled word overlap.

    Chunk sizes are word-based (not tokenizer-specific). Pages are never merged,
    preserving page citations; overlap is applied only between chunks on a page.
    """
    if target_words < 50:
        raise ValueError("target_words must be at least 50.")
    if overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("overlap_words must be >= 0 and smaller than target_words.")
    if not text or not text.strip():
        return []

    sections = _page_sections(text, pages)
    chunks: list[dict] = []
    for page_number, page_text in sections:
        # Preserve Markdown line breaks and headings as meaningful split points.
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", page_text) if part.strip()]
        units: list[str] = []
        for sentence in sentences:
            units.extend(_split_oversize_sentence(sentence, target_words, overlap_words))

        current: list[str] = []
        current_words = 0
        last_heading: str | None = None

        def emit() -> None:
            nonlocal current, current_words
            if not current:
                return
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append({
                    "source": source,
                    "chunk_id": len(chunks),
                    "page": page_number,
                    "text": chunk_text,
                    "word_count": _word_count(chunk_text),
                    "chunk_type": _infer_chunk_type(chunk_text),
                    "section_title": last_heading,
                    "overlap_words": overlap_words if len(chunks) else 0,
                })
            current, current_words = [], 0

        for unit in units:
            if re.match(r"^#{1,6}\s+", unit):
                last_heading = re.sub(r"^#{1,6}\s+", "", unit).strip()
            unit_words = _word_count(unit)
            if current and current_words + unit_words > target_words:
                previous = list(current)
                emit()
                # Seed the next chunk with whole trailing sentences until the
                # overlap target is reached; then trim only as much as needed
                # so the next unit never pushes the chunk beyond its target.
                overlap: list[str] = []
                overlap_count = 0
                if overlap_words:
                    for prev in reversed(previous):
                        overlap.insert(0, prev)
                        overlap_count += _word_count(prev)
                        if overlap_count >= overlap_words:
                            break
                while overlap and overlap_count + unit_words > target_words:
                    overlap_count -= _word_count(overlap.pop(0))
                current = overlap
                current_words = max(0, overlap_count)
            current.append(unit)
            current_words += unit_words
            if unit_words >= target_words:
                emit()
        emit()
    return chunks


def chunk_structured_blocks(
    blocks: list[dict] | None,
    source: str,
    start_id: int = 0,
    target_words: int = DEFAULT_TARGET_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict]:
    """Create typed chunks for extracted tables and equations.

    Large Markdown tables are split by row while repeating their header in each
    chunk. Equations stay intact unless an individual equation exceeds the limit.
    """
    output: list[dict] = []
    for block in blocks or []:
        kind = str(block.get("type", "text")).lower()
        body = str(block.get("text", "")).strip()
        if not body:
            continue
        page = block.get("page")
        if page is not None:
            try: page = int(page)
            except (TypeError, ValueError): page = None
        metadata = {k: v for k, v in block.items() if k not in {"type", "text", "page"}}

        pieces: list[str] = []
        if kind == "table":
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            table_rows = [line for line in lines if line.startswith("|") and line.endswith("|")]
            if len(table_rows) >= 3:
                header = table_rows[0]
                separator = table_rows[1] if re.search(r"\|\s*:?-{2,}", table_rows[1]) else None
                data_rows = table_rows[2:] if separator else table_rows[1:]
                prefix = [line for line in lines if line not in table_rows]
                current_rows: list[str] = []
                count = _word_count(header) + (_word_count(separator) if separator else 0)
                for row in data_rows:
                    rw = _word_count(row)
                    if current_rows and count + rw > target_words:
                        pieces.append("\n".join(prefix + [header] + ([separator] if separator else []) + current_rows))
                        current_rows, count = [], _word_count(header) + (_word_count(separator) if separator else 0)
                    current_rows.append(row)
                    count += rw
                if current_rows or not pieces:
                    pieces.append("\n".join(prefix + [header] + ([separator] if separator else []) + current_rows))
            else:
                pieces = [body]
        elif kind in {"equation", "formula"}:
            pieces = _split_oversize_sentence(body, target_words, overlap_words)
        elif kind in {"text", "figure_text", "caption"}:
            page_info = [{"page": page, "text": body}] if page is not None else None
            split = semantic_chunk(
                body,
                source,
                target_words=target_words,
                pages=page_info,
                overlap_words=overlap_words,
            )
            pieces = [item["text"] for item in split] or [body]
        else:
            pieces = [body]

        for piece in pieces:
            output.append({
                "source": source,
                "chunk_id": start_id + len(output),
                "page": page,
                "text": piece,
                "word_count": _word_count(piece),
                "chunk_type": "equation" if kind == "formula" else kind,
                "section_title": block.get("caption") or block.get("section_title"),
                "overlap_words": overlap_words if kind == "text" else 0,
                **metadata,
            })
    return output


def _infer_chunk_type(text: str) -> str:
    body = text.strip()
    if re.search(r"<table\b|\|\s*:?-{2,}\s*\|", body, flags=re.IGNORECASE):
        return "table"
    if re.search(r"\\begin\{(?:equation|align|gather|math)|\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]", body):
        return "equation"
    return "text"
