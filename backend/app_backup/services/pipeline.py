from pathlib import Path
from app.services.extraction import extract_text
from app.services.chunking import semantic_chunk
from app.services.embeddings import embed_texts
from app.services.vector_store import add_chunks, all_chunks, clear
from app.services.retrieval import rerank
from app.services.generation import generate_answer

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "data" / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

def ingest_file(path: Path) -> dict:
    extracted = extract_text(path)
    if not extracted["text"].strip():
        raise ValueError("No text could be extracted from this file.")
    chunks = semantic_chunk(extracted["text"], path.name, pages=extracted.get("pages"))
    if not chunks:
        raise ValueError("Semantic chunking produced no passages.")
    texts = [c["text"] for c in chunks]
    clip_vectors = embed_texts(texts, "clip")
    minilm_vectors = embed_texts(texts, "minilm")
    add_chunks(chunks, clip_vectors, minilm_vectors)
    return {"chunks_created": len(chunks), "total_chunks": len(all_chunks())}

def query_rag(question: str, top_k: int = 5) -> dict:
    if not all_chunks():
        raise ValueError("No papers are indexed. Upload a paper first.")
    ranked = rerank(question, top_k)
    result = generate_answer(question, ranked)
    result["retrieved_chunks"] = len(ranked)
    return result

def list_papers() -> dict:
    grouped = {}
    for chunk in all_chunks():
        grouped[chunk.get("source", "unknown")] = grouped.get(chunk.get("source", "unknown"), 0) + 1
    return {"papers": [{"filename": name, "chunks": count} for name, count in sorted(grouped.items())],
            "paper_count": len(grouped), "total_chunks": sum(grouped.values())}

def clear_all() -> dict:
    clear()
    for path in PAPERS_DIR.iterdir():
        if path.is_file():
            path.unlink(missing_ok=True)
    return {"message": "Uploaded papers, chunks, and FAISS indices cleared."}
