
import re


PAGE_MARKER = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)


def semantic_chunk(
    text: str,
    source: str,
    target_words: int = 500,
    pages: list[dict] | None = None,
) -> list[dict]:
    """
    Split extracted text into chunks while preserving page metadata.

    Supports text containing [Page N] markers and the optional structured
    pages list returned by extraction.py.
    """
    if target_words < 50:
        raise ValueError("target_words must be at least 50.")

    if not text or not text.strip():
        return []

    # Prefer explicit page markers embedded in extracted text.
    sections = []
    matches = list(PAGE_MARKER.finditer(text))

    if matches:
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            page_text = text[start:end].strip()
            if page_text:
                sections.append((int(match.group(1)), page_text))
    elif pages:
        for page in pages:
            page_text = str(page.get("text", "")).strip()
            if page_text:
                sections.append((int(page.get("page", 1)), page_text))
    else:
        sections.append((None, text.strip()))

    chunks = []
    chunk_id = 0

    for page_number, page_text in sections:
        # Split into sentence-like units, preserving useful context.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", page_text)
            if sentence.strip()
        ]

        current = []
        current_words = 0

        def emit():
            nonlocal chunk_id, current, current_words

            if not current:
                return

            chunk_text = " ".join(current).strip()

            chunks.append({
                "source": source,
                "chunk_id": chunk_id,
                "page": page_number,
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
            })

            chunk_id += 1
            current = []
            current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # Avoid splitting a sentence unless it exceeds the target itself.
            if current and current_words + sentence_words > target_words:
                emit()

            if sentence_words > target_words:
                words = sentence.split()
                for start in range(0, len(words), target_words):
                    piece = " ".join(words[start:start + target_words])

                    if current:
                        emit()

                    current = [piece]
                    current_words = len(piece.split())
                    emit()
                continue

            current.append(sentence)
            current_words += sentence_words

        emit()

    return chunks
