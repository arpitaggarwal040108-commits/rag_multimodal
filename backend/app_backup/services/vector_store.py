from pathlib import Path
import json
import threading
import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "indices"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_PATH = INDEX_DIR / "chunks.json"
CLIP_PATH = INDEX_DIR / "clip.index"
MINILM_PATH = INDEX_DIR / "minilm.index"
_lock = threading.RLock()
_chunks: list[dict] = []
_clip_index = None
_minilm_index = None

def _save():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_clip_index, str(CLIP_PATH))
    faiss.write_index(_minilm_index, str(MINILM_PATH))
    temp = CHUNKS_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(CHUNKS_PATH)

def _load():
    global _chunks, _clip_index, _minilm_index
    if CHUNKS_PATH.exists() and CLIP_PATH.exists() and MINILM_PATH.exists():
        try:
            chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
            clip = faiss.read_index(str(CLIP_PATH))
            minilm = faiss.read_index(str(MINILM_PATH))
            if clip.ntotal == minilm.ntotal == len(chunks):
                _chunks, _clip_index, _minilm_index = chunks, clip, minilm
        except Exception:
            # Leave store empty rather than loading mismatched/corrupt files.
            _chunks, _clip_index, _minilm_index = [], None, None

def add_chunks(chunks: list[dict], clip_vectors: np.ndarray, minilm_vectors: np.ndarray):
    global _clip_index, _minilm_index
    if not chunks:
        return
    if len(chunks) != len(clip_vectors) or len(chunks) != len(minilm_vectors):
        raise ValueError("Chunk and embedding counts do not match.")
    with _lock:
        if _clip_index is None:
            _clip_index = faiss.IndexFlatIP(clip_vectors.shape[1])
        if _minilm_index is None:
            _minilm_index = faiss.IndexFlatIP(minilm_vectors.shape[1])
        if _clip_index.d != clip_vectors.shape[1] or _minilm_index.d != minilm_vectors.shape[1]:
            raise RuntimeError("Embedding dimensions do not match existing FAISS indices.")
        start = len(_chunks)
        for i, chunk in enumerate(chunks):
            item = dict(chunk)
            item["global_id"] = start + i
            _chunks.append(item)
        _clip_index.add(np.ascontiguousarray(clip_vectors, dtype=np.float32))
        _minilm_index.add(np.ascontiguousarray(minilm_vectors, dtype=np.float32))
        _save()

def search(query_clip: np.ndarray, query_minilm: np.ndarray, top_k: int = 5):
    with _lock:
        if not _chunks or _clip_index is None or _minilm_index is None:
            raise ValueError("No indexed chunks yet. Upload a paper first.")
        s1, ids1 = _clip_index.search(np.asarray(query_clip, dtype=np.float32), min(top_k, _clip_index.ntotal))
        s2, ids2 = _minilm_index.search(np.asarray(query_minilm, dtype=np.float32), min(top_k, _minilm_index.ntotal))
        clip_hits = [(int(i), float(s)) for i, s in zip(ids1[0], s1[0]) if i >= 0]
        minilm_hits = [(int(i), float(s)) for i, s in zip(ids2[0], s2[0]) if i >= 0]
        return clip_hits, minilm_hits, list(_chunks)

def all_chunks():
    with _lock:
        return list(_chunks)

def clear():
    global _chunks, _clip_index, _minilm_index
    with _lock:
        _chunks, _clip_index, _minilm_index = [], None, None
        for p in (CHUNKS_PATH, CLIP_PATH, MINILM_PATH):
            p.unlink(missing_ok=True)
    return {"message": "Indexed chunks and vector indices cleared."}

_load()
