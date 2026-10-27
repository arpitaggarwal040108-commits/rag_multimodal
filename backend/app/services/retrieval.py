import re
from collections import defaultdict

from app.services.embeddings import embed_texts
from app.services.vector_store import search, search_visuals

_CANDIDATE_MULTIPLIER = 6
_MIN_CANDIDATES = 40
_RRF_K = 60
_MIN_MINILM_SIMILARITY = 0.16
_MIN_CLIP_SIMILARITY = 0.08
_MIN_COMBINED_SIMILARITY = 0.19

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
}

def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
    return {word for word in words if len(word) > 2 and word not in _STOPWORDS}

def _lexical_overlap(query: str, text: str) -> float:
    query_terms = _tokens(query)
    if not query_terms:
        return 0.0
    return len(query_terms & _tokens(text)) / len(query_terms)

def _clip_score(hits: list[tuple[int, float]]) -> dict[int, float]:
    return {idx: score for idx, score in hits}

def _rank_candidates(query: str, clip_hits: list[tuple[int, float]],
                     minilm_hits: list[tuple[int, float]], chunks: list[dict],
                     top_k: int) -> list[dict]:
    clip_scores = _clip_score(clip_hits)
    minilm_scores = _clip_score(minilm_hits)
    rrf_scores: dict[int, float] = defaultdict(float)
    for hits in (clip_hits, minilm_hits):
        for rank, (idx, _) in enumerate(hits, start=1):
            rrf_scores[idx] += 1.0 / (_RRF_K + rank)

    ranked = []
    for idx in set(clip_scores) | set(minilm_scores):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        clip = clip_scores.get(idx, -1.0)
        minilm = minilm_scores.get(idx, -1.0)
        lexical = _lexical_overlap(query, text)
        # Embedding cutoffs can be overly strict for OCR-heavy or table-heavy
        # passages. Keep borderline candidates when the query has a clear
        # lexical match, while retaining a floor against unrelated chunks.
        if minilm < _MIN_MINILM_SIMILARITY and lexical < 0.34:
            continue
        if clip < _MIN_CLIP_SIMILARITY and minilm < 0.27 and lexical < 0.34:
            continue
        clip_for_score = max(clip, 0.0)
        minilm_for_score = max(minilm, 0.0)
        combined_similarity = 0.75 * minilm_for_score + 0.25 * clip_for_score
        if combined_similarity < _MIN_COMBINED_SIMILARITY and lexical < 0.34:
            continue
        final_score = combined_similarity + min(lexical, 0.5) * 0.08
        item = dict(chunk)
        item.update({
            "score": round(float(final_score), 6),
            "clip_score": round(float(clip), 6),
            "minilm_score": round(float(minilm), 6),
            "lexical_overlap": round(float(lexical), 6),
            "rrf_score": round(float(rrf_scores.get(idx, 0.0)), 6),
        })
        ranked.append(item)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    source_counts: dict[str, int] = defaultdict(int)
    per_source_limit = max(4, top_k)
    for item in ranked:
        source = str(item.get("source", item.get("original_filename", "unknown")))
        if source_counts[source] >= per_source_limit:
            continue
        selected.append(item)
        source_counts[source] += 1
        if len(selected) >= top_k:
            break
    return selected

def rerank(query: str, top_k: int = 8, workspace_id: str = "default") -> list[dict]:
    query = str(query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    query_clip = embed_texts([query], "clip")
    query_minilm = embed_texts([query], "minilm")
    candidate_k = max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES)
    clip_hits, minilm_hits, chunks = search(
        query_clip, query_minilm, top_k=candidate_k, workspace_id=workspace_id
    )
    return _rank_candidates(query, clip_hits, minilm_hits, chunks, top_k)

def retrieve_visuals(query: str, top_k: int = 8, workspace_id: str = "default") -> list[dict]:
    query = str(query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    query_vector = embed_texts([query], "clip")
    results = search_visuals(query_vector, top_k=top_k, workspace_id=workspace_id)
    return [item for item in results if isinstance(item, dict) and item.get("path")]
