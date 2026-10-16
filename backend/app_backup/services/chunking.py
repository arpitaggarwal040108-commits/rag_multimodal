import re

def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if paragraphs:
        return paragraphs
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def semantic_chunk(text: str, source: str, target_words: int = 500, pages: list | None = None) -> list[dict]:
    """
    Local, zero-API chunking.
    Groups adjacent paragraphs/sentences toward target_words without calling a hosted LLM.
    This is structure-aware chunking, not LLM-level semantic segmentation.
    """
    text = text.strip()
    if not text:
        return []
    paragraphs = _split_paragraphs(text)
    chunks_text = []
    current = []
    count = 0
    for paragraph in paragraphs:
        words = paragraph.split()
        # Split unusually long paragraphs into sentence groups first.
        segments = [paragraph]
        if len(words) > target_words * 1.5:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
            segments = sentences if sentences else [paragraph]
        for segment in segments:
            n = len(segment.split())
            if current and count + n > target_words:
                chunks_text.append("\n\n".join(current))
                current, count = [], 0
            current.append(segment)
            count += n
    if current:
        chunks_text.append("\n\n".join(current))
    result = []
    for idx, part in enumerate(chunks_text):
        result.append({
            "text": part, "source": source, "chunk_id": idx,
            "block": 0, "page": None, "section": None
        })
    return result
