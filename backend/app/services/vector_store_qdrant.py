from __future__ import annotations

from pathlib import Path
import os
import re
import threading
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
QDRANT_DIR = DATA_DIR / "qdrant"
QDRANT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_COLLECTION = os.getenv("QDRANT_CHUNKS_COLLECTION", "research_prism_chunks_v1")
VISUALS_COLLECTION = os.getenv("QDRANT_VISUALS_COLLECTION", "research_prism_visuals_v1")
_lock = threading.RLock()
_client = QdrantClient(path=str(QDRANT_DIR))


def _safe_id(workspace_id: str) -> str:
    value = str(workspace_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("Invalid workspace ID.")
    return value


def _workspace_filter(workspace_id: str | None):
    if workspace_id is None:
        return None
    return models.Filter(must=[models.FieldCondition(key="workspace_id", match=models.MatchValue(value=_safe_id(workspace_id)))])


def _vecs(vectors: np.ndarray, count: int, name: str) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != count or vectors.shape[1] <= 0:
        raise ValueError(f"{name} must be a 2D array with {count} rows.")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero vector.")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _collection_exists(name: str) -> bool:
    return _client.collection_exists(name)


def _ensure_chunks_collection(clip_dim: int, text_dim: int) -> None:
    if not _collection_exists(CHUNKS_COLLECTION):
        _client.create_collection(
            collection_name=CHUNKS_COLLECTION,
            vectors_config={
                "clip": models.VectorParams(size=clip_dim, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
            },
        )
        _client.create_payload_index(collection_name=CHUNKS_COLLECTION, field_name="workspace_id", field_schema=models.PayloadSchemaType.KEYWORD)
        _client.create_payload_index(collection_name=CHUNKS_COLLECTION, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD)
        _client.create_payload_index(collection_name=CHUNKS_COLLECTION, field_name="chunk_type", field_schema=models.PayloadSchemaType.KEYWORD)
        return
    info = _client.get_collection(CHUNKS_COLLECTION)
    config = info.config.params.vectors
    if not isinstance(config, dict) or "clip" not in config or "text" not in config:
        raise RuntimeError("Existing Qdrant collection has an incompatible vector schema. Back it up and use a new collection name or rebuild it.")
    if config["clip"].size != clip_dim or config["text"].size != text_dim:
        raise RuntimeError(
            f"Embedding dimensions changed (existing clip/text={config['clip'].size}/{config['text'].size}, "
            f"new={clip_dim}/{text_dim}). Do not mix models in one index; use a new collection or re-index into a clean Qdrant store."
        )


def _ensure_visual_collection(dim: int) -> None:
    if not _collection_exists(VISUALS_COLLECTION):
        _client.create_collection(collection_name=VISUALS_COLLECTION, vectors_config={"clip": models.VectorParams(size=dim, distance=models.Distance.COSINE)})
        _client.create_payload_index(collection_name=VISUALS_COLLECTION, field_name="workspace_id", field_schema=models.PayloadSchemaType.KEYWORD)
        _client.create_payload_index(collection_name=VISUALS_COLLECTION, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD)
        return
    config = _client.get_collection(VISUALS_COLLECTION).config.params.vectors
    if not isinstance(config, dict) or "clip" not in config or config["clip"].size != dim:
        raise RuntimeError("Existing Qdrant visual collection has an incompatible CLIP dimension. Use a new collection or rebuild it.")


def _scroll(collection: str, workspace_id: str | None = None, with_vectors: bool = False):
    if not _collection_exists(collection):
        return []
    result = []
    offset = None
    flt = _workspace_filter(workspace_id)
    while True:
        points, offset = _client.scroll(
            collection_name=collection, scroll_filter=flt, limit=256, offset=offset,
            with_payload=True, with_vectors=with_vectors,
        )
        result.extend(points)
        if offset is None:
            break
    return result


def add_chunks(chunks: list[dict], clip_vectors: np.ndarray, minilm_vectors: np.ndarray, workspace_id: str = "default") -> None:
    if not chunks:
        return
    workspace_id = _safe_id(workspace_id)
    clip_vectors = _vecs(clip_vectors, len(chunks), "CLIP vectors")
    text_vectors = _vecs(minilm_vectors, len(chunks), "text vectors")
    with _lock:
        _ensure_chunks_collection(clip_vectors.shape[1], text_vectors.shape[1])
        existing = all_chunks(workspace_id)
        next_global = max((int(item.get("global_id", -1)) for item in existing), default=-1) + 1
        points = []
        for offset, chunk in enumerate(chunks):
            item = dict(chunk)
            item["workspace_id"] = workspace_id
            item["global_id"] = next_global + offset
            uid = str(uuid.uuid4())
            item["chunk_uid"] = uid
            payload = dict(item)
            points.append(models.PointStruct(
                id=uid,
                vector={"clip": clip_vectors[offset].tolist(), "text": text_vectors[offset].tolist()},
                payload=payload,
            ))
        for start in range(0, len(points), 128):
            _client.upsert(collection_name=CHUNKS_COLLECTION, points=points[start:start + 128], wait=True)


def add_visuals(visuals: list[dict], vectors: np.ndarray, workspace_id: str = "default") -> None:
    if not visuals:
        return
    workspace_id = _safe_id(workspace_id)
    vectors = _vecs(vectors, len(visuals), "visual vectors")
    with _lock:
        _ensure_visual_collection(vectors.shape[1])
        points = []
        for index, visual in enumerate(visuals):
            item = dict(visual)
            item["workspace_id"] = workspace_id
            uid = str(uuid.uuid4())
            item["visual_uid"] = uid
            points.append(models.PointStruct(id=uid, vector={"clip": vectors[index].tolist()}, payload=item))
        for start in range(0, len(points), 128):
            _client.upsert(collection_name=VISUALS_COLLECTION, points=points[start:start + 128], wait=True)


def all_chunks(workspace_id: str | None = None) -> list[dict]:
    with _lock:
        points = _scroll(CHUNKS_COLLECTION, workspace_id, with_vectors=False)
        records = [dict(point.payload or {}) for point in points]
        records.sort(key=lambda item: (str(item.get("workspace_id", "")), int(item.get("global_id", 0))))
        return records


def all_visuals(workspace_id: str | None = None) -> list[dict]:
    with _lock:
        points = _scroll(VISUALS_COLLECTION, workspace_id, with_vectors=False)
        records = [dict(point.payload or {}) for point in points]
        records.sort(key=lambda item: (str(item.get("workspace_id", "")), str(item.get("source", "")), int(item.get("page", 0) or 0), str(item.get("path", ""))))
        return records


def search(query_clip: np.ndarray, query_minilm: np.ndarray, top_k: int = 5, workspace_id: str = "default"):
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    workspace_id = _safe_id(workspace_id)
    clip_query = _vecs(query_clip, 1, "CLIP query")[0]
    text_query = _vecs(query_minilm, 1, "text query")[0]
    with _lock:
        chunks = all_chunks(workspace_id)
        if not chunks or not _collection_exists(CHUNKS_COLLECTION):
            return [], [], chunks

        info = _client.get_collection(CHUNKS_COLLECTION)
        config = info.config.params.vectors
        if isinstance(config, dict):
            expected_clip = config.get("clip").size if config.get("clip") else None
            expected_text = config.get("text").size if config.get("text") else None
            if expected_clip != clip_query.shape[0] or expected_text != text_query.shape[0]:
                raise RuntimeError(
                    f"Embedding dimension mismatch: Qdrant expects clip/text={expected_clip}/{expected_text}, "
                    f"but query models produced {clip_query.shape[0]}/{text_query.shape[0]}. "
                    "Use the same model used during indexing or rebuild the workspace index."
                )
        id_lookup = {str(item.get("chunk_uid")): i for i, item in enumerate(chunks)}
        flt = _workspace_filter(workspace_id)
        clip_resp = _client.query_points(
            collection_name=CHUNKS_COLLECTION, query=clip_query.tolist(), using="clip",
            query_filter=flt, limit=min(top_k, len(chunks)), with_payload=True,
        )
        text_resp = _client.query_points(
            collection_name=CHUNKS_COLLECTION, query=text_query.tolist(), using="text",
            query_filter=flt, limit=min(top_k, len(chunks)), with_payload=True,
        )
        clip_hits = [
            (id_lookup[str(point.payload.get("chunk_uid"))], float(point.score))
            for point in clip_resp.points
            if point.payload and str(point.payload.get("chunk_uid")) in id_lookup
        ]
        text_hits = [
            (id_lookup[str(point.payload.get("chunk_uid"))], float(point.score))
            for point in text_resp.points
            if point.payload and str(point.payload.get("chunk_uid")) in id_lookup
        ]
        return clip_hits, text_hits, chunks


def search_visuals(query_vectors: np.ndarray, top_k: int = 5, workspace_id: str = "default") -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    workspace_id = _safe_id(workspace_id)
    query_vector = _vecs(query_vectors, 1, "visual query")[0]
    with _lock:
        if not _collection_exists(VISUALS_COLLECTION):
            return []
        flt = _workspace_filter(workspace_id)
        response = _client.query_points(
            collection_name=VISUALS_COLLECTION, query=query_vector.tolist(), using="clip",
            query_filter=flt, limit=top_k, with_payload=True,
        )
        results = []
        for point in response.points:
            item = dict(point.payload or {})
            item["score"] = float(point.score)
            results.append(item)
        return results


def delete_document(source: str, workspace_id: str = "default") -> dict:
    workspace_id = _safe_id(workspace_id)
    flt = models.Filter(must=[
        models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id)),
        models.FieldCondition(key="source", match=models.MatchValue(value=source)),
    ])
    counts = {}
    with _lock:
        for label, collection in (("chunks", CHUNKS_COLLECTION), ("visuals", VISUALS_COLLECTION)):
            if _collection_exists(collection):
                counts[label] = _client.count(collection_name=collection, count_filter=flt, exact=True).count
                _client.delete(collection_name=collection, points_selector=models.FilterSelector(filter=flt), wait=True)
            else:
                counts[label] = 0
    return {"removed_chunks": counts["chunks"], "removed_visuals": counts["visuals"]}


def clear(workspace_id: str | None = None) -> dict:
    with _lock:
        if workspace_id is None:
            for collection in (CHUNKS_COLLECTION, VISUALS_COLLECTION):
                if _collection_exists(collection):
                    _client.delete_collection(collection_name=collection)
            return {"message": "All Research Prism Qdrant collections cleared."}
        workspace_id = _safe_id(workspace_id)
        flt = _workspace_filter(workspace_id)
        for collection in (CHUNKS_COLLECTION, VISUALS_COLLECTION):
            if _collection_exists(collection):
                _client.delete(collection_name=collection, points_selector=models.FilterSelector(filter=flt), wait=True)
        return {"message": f"Workspace '{workspace_id}' index cleared."}
