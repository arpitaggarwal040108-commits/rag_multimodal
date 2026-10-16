
import re
from collections import Counter

from app.services.vision import explain_research_images


_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this",
    "from", "have", "has", "had", "you", "your", "their", "they", "them",
    "into", "about", "than", "then", "when", "where", "what", "which",
    "who", "how", "why", "can", "could", "would", "should", "will",
    "not", "but", "also", "more", "most", "some", "such", "each",
    "other", "using", "used", "use", "based", "between", "within",
    "paper", "study", "studies", "research", "results", "shown",
    "figure", "table", "section", "page", "chunk", "explain", "describe",
    "discuss", "tell", "give", "find", "according", "following",
    "compare", "comparison", "summarize", "summary", "provide",
}

MIN_SENTENCE_LENGTH = 30


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
    return {
        word for word in words
        if len(word) > 2 and word not in _STOPWORDS
    }


def _clean_text(text: str) -> str:
    text = str(text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        _clean_text(text),
    )
    return [
        part.strip()
        for part in parts
        if len(part.strip()) >= MIN_SENTENCE_LENGTH
    ]


def _score_sentence(
    sentence: str,
    query_terms: set[str],
    document_frequency: Counter,
    total_chunks: int,
) -> float:
    sentence_terms = _terms(sentence)

    if not query_terms or not sentence_terms:
        return 0.0

    matched = query_terms & sentence_terms

    if not matched:
        return 0.0

    coverage = len(matched) / len(query_terms)

    idf = {
        term: 1.0 + (
            (total_chunks + 1) /
            (document_frequency.get(term, 0) + 1)
        ) ** 0.5
        for term in query_terms
    }

    total_weight = sum(idf.values())
    matched_weight = sum(idf[term] for term in matched)

    weighted_coverage = (
        matched_weight / total_weight if total_weight else coverage
    )

    return 0.60 * coverage + 0.40 * weighted_coverage


def _source_name(chunk: dict) -> str:
    return str(
        chunk.get("original_filename")
        or chunk.get("source")
        or "unknown"
    )


def _abstention(message: str) -> dict:
    return {
        "answer": message,
        "sources": [],
    }


def _generate_text_answer(
    question: str,
    chunks: list[dict],
) -> dict:
    """Create a source-cited extractive answer without a language model."""
    query_terms = _terms(question)

    if not query_terms:
        return _abstention(
            "Please enter a more specific research question."
        )

    document_frequency = Counter()

    for chunk in chunks:
        document_frequency.update(_terms(chunk.get("text", "")))

    candidates = []
    seen = set()

    for chunk in chunks:
        for sentence in _split_sentences(chunk.get("text", "")):
            normalized = re.sub(
                r"\W+", " ", sentence.lower()
            ).strip()

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)

            score = _score_sentence(
                sentence,
                query_terms,
                document_frequency,
                max(len(chunks), 1),
            )

            if score >= 0.16:
                candidates.append({
                    "score": score,
                    "sentence": sentence,
                    "chunk": chunk,
                })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    if not candidates:
        return _abstention(
            "The retrieved text does not contain enough matching evidence "
            "to answer this question reliably."
        )

    selected = candidates[:6]
    lines = []
    sources = []
    seen_sources = set()

    for item in selected:
        chunk = item["chunk"]
        source = chunk.get("source", "unknown")
        chunk_id = chunk.get(
            "chunk_id",
            chunk.get("global_id", "unknown"),
        )
        page = chunk.get("page")

        citation = f"[Source: {source}, Chunk {chunk_id}"

        if page is not None:
            citation += f", Page {page}"

        citation += "]"

        lines.append(f"- {item['sentence']} {citation}")

        key = (source, chunk_id, page)

        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "source": source,
                "original_filename": _source_name(chunk),
                "chunk_id": chunk_id,
                "page": page,
                "chunk_index": chunk.get("global_id", chunk_id),
                "score": chunk.get("score"),
                "text": chunk.get("text", ""),
            })

    return {
        "answer": (
            "## Research evidence\n\n"
            "The following are extracted passages, not a synthesized "
            "or independently verified conclusion.\n\n"
            "## Relevant passages\n\n"
            + "\n\n".join(lines)
            + "\n\n## Limitations\n\n"
            "Check the cited pages for context. Similar wording alone "
            "does not prove that a passage supports a claim."
        ),
        "sources": sources,
    }


def generate_answer(
    question: str,
    chunks: list[dict],
    visual_results: list[dict] | None = None,
) -> dict:
    """
    Answer using retrieved images when available, otherwise use text.

    The original two-argument call remains supported.
    """
    question = _clean_text(question)

    if not question:
        return _abstention("Please enter a research question.")

    chunks = [
        chunk for chunk in chunks
        if isinstance(chunk, dict)
        and _clean_text(chunk.get("text", ""))
    ]

    visual_results = visual_results or []

    if not chunks and not visual_results:
        return _abstention(
            "No relevant text or visual evidence was retrieved."
        )

    text_context = "\n\n".join(
        f"[Source: {_source_name(chunk)}, "
        f"Page: {chunk.get('page', 'unknown')}]\n"
        f"{_clean_text(chunk.get('text', ''))}"
        for chunk in chunks[:6]
    )

    if visual_results:
        try:
            visual_answer = explain_research_images(
                question=question,
                visual_results=visual_results,
                text_context=text_context,
            )

            if visual_answer.get("available"):
                sources = []

                for chunk in chunks:
                    sources.append({
                        "source": chunk.get("source", "unknown"),
                        "original_filename": _source_name(chunk),
                        "chunk_id": chunk.get(
                            "chunk_id",
                            chunk.get("global_id", "unknown"),
                        ),
                        "page": chunk.get("page"),
                        "chunk_index": chunk.get(
                            "global_id",
                            chunk.get("chunk_id", "unknown"),
                        ),
                        "score": chunk.get("score"),
                        "text": chunk.get("text", ""),
                    })

                return {
                    "answer": visual_answer["answer"],
                    "sources": sources,
                    "visual_sources": visual_answer["visual_sources"],
                    "answer_mode": "local_vision",
                    "vision_model": visual_answer["model"],
                }

        except RuntimeError as exc:
            # Do not silently claim that image understanding succeeded.
            # Fall back to text and disclose why.
            fallback = _generate_text_answer(question, chunks)
            fallback["answer"] = (
                "## Visual analysis unavailable\n\n"
                f"{str(exc)}\n\n"
                "## Text evidence\n\n"
                + fallback["answer"]
            )
            fallback["answer_mode"] = "text_fallback"
            return fallback

    result = _generate_text_answer(question, chunks)
    result["answer_mode"] = "extractive_text"
    return result
