import re
from collections import Counter

def _terms(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text)}

def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    No API key and no hosted LLM.
    Produces a grounded extractive response from retrieved passages.
    It does not synthesize new claims; citations refer to source/chunk metadata.
    """
    if not chunks:
        return {"answer": "The indexed papers did not provide relevant passages.", "sources": []}

    q_terms = _terms(question)
    selected = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
        scored = []
        for sentence in sentences:
            terms = _terms(sentence)
            overlap = len(q_terms & terms)
            # Rank score provides retrieval context; overlap helps select query-related evidence.
            scored.append((overlap, sentence))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = [s for score, s in scored[:2] if score > 0]
        if not best and sentences:
            best = sentences[:1]
        for sentence in best:
            selected.append((sentence, chunk))
    # Deduplicate exact sentences while preserving order.
    seen = set()
    lines = []
    for sentence, chunk in selected:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f'{sentence} [Source: {chunk.get("source")}, Chunk {chunk.get("chunk_id")}]')
        if len(lines) >= 8:
            break

    if lines:
        answer = (
            "The following evidence was retrieved from your indexed papers. "
            "This local mode extracts relevant statements rather than generating a free-form synthesis:\n\n"
            + "\n\n".join(lines)
        )
    else:
        answer = "The retrieved passages did not contain enough clearly relevant text to answer this question."

    sources = [{
        "chunk_index": c.get("global_id", c.get("chunk_id")),
        "source": c.get("source"),
        "chunk_id": c.get("chunk_id"),
        "text": c.get("text", ""),
        "score": c.get("score")
    } for c in chunks]
    return {"answer": answer, "sources": sources}
