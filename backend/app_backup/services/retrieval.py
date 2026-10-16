from app.services.embeddings import embed_texts
from app.services.vector_store import search

def rerank(query: str, top_k: int = 5) -> list[dict]:
    q_clip = embed_texts([query], "clip")
    q_minilm = embed_texts([query], "minilm")
    clip_hits, minilm_hits, chunks = search(q_clip, q_minilm, top_k=5)
    scores = {}
    for hits in (clip_hits, minilm_hits):
        for rank, (idx, cosine) in enumerate(hits, start=1):
            entry = scores.setdefault(idx, {"rrf": 0.0, "cosines": []})
            entry["rrf"] += 1.0 / (60 + rank)
            entry["cosines"].append(cosine)
    ranked = []
    for idx, data in scores.items():
        avg_cosine = sum(data["cosines"]) / len(data["cosines"])
        final_score = 0.6 * data["rrf"] + 0.4 * avg_cosine
        item = dict(chunks[idx])
        item.update({
            "score": round(float(final_score), 6),
            "rrf_score": round(float(data["rrf"]), 6),
            "cosine_score": round(float(avg_cosine), 6)
        })
        ranked.append(item)
    return sorted(ranked, key=lambda x: x["score"], reverse=True)[:top_k]
