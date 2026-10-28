import os
import re
from collections import defaultdict
from functools import lru_cache

from app.services.embeddings import embed_texts
from app.services.vector_store import all_chunks, search, search_visuals

_CANDIDATE_MULTIPLIER = 6
_MIN_CANDIDATES = 40
_RRF_K = 60
_ENABLE_CROSS_ENCODER = os.getenv("ENABLE_CROSS_ENCODER", "false").strip().lower() in {"1", "true", "yes", "on"}
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from", "have", "has", "had", "you", "your", "their", "they", "them", "into", "about", "than", "then", "when", "where", "what", "which", "who", "how", "why", "can", "could", "would", "should", "will", "not", "but", "also", "more", "most", "some", "such", "each", "other", "using", "used", "use", "based", "between", "within", "paper", "study", "studies", "research", "results", "shown", "figure", "table", "section", "page", "chunk", "explain", "describe", "discuss", "tell", "give", "find", "according", "following",
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", str(text).lower())
    return [word for word in words if len(word) > 1 and word not in _STOPWORDS]


def _lexical_overlap(query: str, text: str) -> float:
    query_terms = set(_tokens(query))
    if not query_terms:
        return 0.0
    return len(query_terms & set(_tokens(text))) / len(query_terms)


def _bm25_hits(query: str, chunks: list[dict], top_k: int) -> list[tuple[int, float]]:
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [_tokens(str(chunk.get("text", ""))) for chunk in chunks]
        if not tokenized or not any(tokenized):
            return []
        model = BM25Okapi(tokenized)
        scores = model.get_scores(_tokens(query))
        # Only non-negative, non-zero candidates are meaningful for this query.
        ids = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
        return [(idx, float(scores[idx])) for idx in ids[:top_k] if float(scores[idx]) > 0]
    except Exception:
        # Keep retrieval operational even if optional BM25 dependencies fail.
        scored = [(idx, _lexical_overlap(query, str(chunk.get("text", "")))) for idx, chunk in enumerate(chunks)]
        return [(idx, score) for idx, score in sorted(scored, key=lambda item: item[1], reverse=True)[:top_k] if score > 0]


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL, max_length=1024)


def _rank_candidates(query: str, clip_hits: list[tuple[int, float]], text_hits: list[tuple[int, float]], chunks: list[dict], top_k: int) -> list[dict]:
    clip_scores = {idx: score for idx, score in clip_hits}
    text_scores = {idx: score for idx, score in text_hits}
    bm25_hits = _bm25_hits(query, chunks, max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES))
    bm25_scores = {idx: score for idx, score in bm25_hits}
    rrf_scores: dict[int, float] = defaultdict(float)
    for hits in (clip_hits, text_hits, bm25_hits):
        for rank, (idx, _) in enumerate(hits, start=1):
            rrf_scores[idx] += 1.0 / (_RRF_K + rank)

    candidate_ids = set(clip_scores) | set(text_scores) | set(bm25_scores)
    ranked = []
    for idx in candidate_ids:
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        content = str(chunk.get("text", "")).strip()
        if not content:
            continue
        clip = clip_scores.get(idx, 0.0)
        text_score = text_scores.get(idx, 0.0)
        lexical = _lexical_overlap(query, content)
        dense_blend = 0.80 * max(text_score, 0.0) + 0.20 * max(clip, 0.0)
        rrf = rrf_scores.get(idx, 0.0)
        rrf_normalized = min(1.0, rrf / (3.0 / (_RRF_K + 1)))
        # Dense retrieval handles paraphrases; normalized rank fusion and BM25
        # restore exact technical terms such as equation names and model IDs.
        final_score = 0.58 * dense_blend + 0.32 * rrf_normalized + 0.10 * lexical
        if final_score <= 0 and lexical <= 0:
            continue
        item = dict(chunk)
        item.update({
            "score": round(float(final_score), 6),
            "clip_score": round(float(clip), 6),
            "text_score": round(float(text_score), 6),
            "minilm_score": round(float(text_score), 6),  # response compatibility
            "bm25_score": round(float(bm25_scores.get(idx, 0.0)), 6),
            "lexical_overlap": round(float(lexical), 6),
            "rrf_score": round(float(rrf), 6),
        })
        ranked.append(item)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    if _ENABLE_CROSS_ENCODER and ranked:
        candidates = ranked[:max(top_k * 4, min(40, len(ranked)))]
        try:
            pairs = [(query, str(item.get("text", ""))[:8000]) for item in candidates]
            scores = _cross_encoder().predict(pairs, show_progress_bar=False)
            for item, score in zip(candidates, scores):
                item["reranker_score"] = float(score)
            ranked = sorted(candidates, key=lambda item: item.get("reranker_score", -1e9), reverse=True)
        except Exception as exc:
            print(f"[Research Prism] Cross-encoder unavailable; using hybrid ranking: {exc}")

    selected, source_counts = [], defaultdict(int)
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
    if not all_chunks(workspace_id):
        return []
    query_clip = embed_texts([query], "clip")
    query_text = embed_texts([query], "minilm")
    candidate_k = max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES)
    clip_hits, text_hits, chunks = search(query_clip, query_text, top_k=candidate_k, workspace_id=workspace_id)
    if not chunks:
        return []
    return _rank_candidates(query, clip_hits, text_hits, chunks, top_k)


def retrieve_visuals(query: str, top_k: int = 8, workspace_id: str = "default") -> list[dict]:
    query = str(query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    query_vector = embed_texts([query], "clip")
    results = search_visuals(query_vector, top_k=top_k, workspace_id=workspace_id)
    return [item for item in results if isinstance(item, dict) and item.get("path")]
