"""Vector-store adapter. Qdrant is the default; FAISS remains a compatibility option."""
import os

_backend = os.getenv("VECTOR_BACKEND", "qdrant").strip().lower()
if _backend == "qdrant":
    from app.services.vector_store_qdrant import add_chunks, add_visuals, all_chunks, all_visuals, clear, delete_document, search, search_visuals
elif _backend == "faiss":
    from app.services.vector_store_faiss import add_chunks, add_visuals, all_chunks, all_visuals, clear, delete_document, search, search_visuals
else:
    raise RuntimeError("VECTOR_BACKEND must be 'qdrant' or 'faiss'.")

__all__ = ["add_chunks", "add_visuals", "all_chunks", "all_visuals", "clear", "delete_document", "search", "search_visuals"]
