
from pathlib import Path
import json
import re
import threading

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "indices"
WORKSPACES_DIR = INDEX_DIR / "workspaces"

LEGACY_CHUNKS = INDEX_DIR / "chunks.json"
LEGACY_CLIP = INDEX_DIR / "clip.index"
LEGACY_MINILM = INDEX_DIR / "minilm.index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.RLock()
_stores: dict[str, dict] = {}


def _safe_id(workspace_id: str) -> str:
    value = str(workspace_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("Invalid workspace ID.")
    return value


def _paths(workspace_id: str) -> dict:
    workspace_id = _safe_id(workspace_id)
    folder = WORKSPACES_DIR / workspace_id
    folder.mkdir(parents=True, exist_ok=True)

    return {
        "folder": folder,
        "chunks": folder / "chunks.json",
        "clip": folder / "clip.index",
        "minilm": folder / "minilm.index",
        "visual": folder / "visual.index",
        "visuals": folder / "visuals.json",
    }


def _empty_store() -> dict:
    return {
        "chunks": [],
        "clip": None,
        "minilm": None,
        "visual": None,
        "visuals": [],
    }


def _validate_vectors(
    vectors: np.ndarray,
    expected_count: int | None = None,
    name: str = "vectors",
) -> np.ndarray:
    """Validate embedding shape and reject non-finite values."""
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")

    if expected_count is not None and len(vectors) != expected_count:
        raise ValueError(f"{name} count does not match metadata.")

    if vectors.shape[1] <= 0:
        raise ValueError(f"{name} must have a nonzero embedding dimension.")

    if not np.isfinite(vectors).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return np.ascontiguousarray(vectors, dtype=np.float32)


def _write_json_atomic(path: Path, data) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def _save(workspace_id: str) -> None:
    """Persist the selected workspace's indices and metadata."""
    store = _stores[workspace_id]
    paths = _paths(workspace_id)

    if store["clip"] is None or store["minilm"] is None:
        paths["chunks"].unlink(missing_ok=True)
        paths["clip"].unlink(missing_ok=True)
        paths["minilm"].unlink(missing_ok=True)
    else:
        if (
            store["clip"].ntotal != len(store["chunks"])
            or store["minilm"].ntotal != len(store["chunks"])
        ):
            raise RuntimeError("Text-index counts do not match chunk metadata.")

        faiss.write_index(store["clip"], str(paths["clip"]))
        faiss.write_index(store["minilm"], str(paths["minilm"]))
        _write_json_atomic(paths["chunks"], store["chunks"])

    if store["visual"] is None or not store["visuals"]:
        paths["visual"].unlink(missing_ok=True)
        paths["visuals"].unlink(missing_ok=True)
    else:
        if store["visual"].ntotal != len(store["visuals"]):
            raise RuntimeError("Visual-index counts do not match metadata.")

        faiss.write_index(store["visual"], str(paths["visual"]))
        _write_json_atomic(paths["visuals"], store["visuals"])


def _load_workspace(workspace_id: str) -> dict:
    paths = _paths(workspace_id)
    store = _empty_store()

    text_keys = ("chunks", "clip", "minilm")
    present_text = [paths[key].exists() for key in text_keys]

    if any(present_text) and not all(present_text):
        raise RuntimeError(
            f"Workspace '{workspace_id}' has incomplete text-index files. "
            "Files were preserved; restore or rebuild the index."
        )

    if all(present_text):
        try:
            chunks = json.loads(
                paths["chunks"].read_text(encoding="utf-8")
            )
            clip = faiss.read_index(str(paths["clip"]))
            minilm = faiss.read_index(str(paths["minilm"]))

            if not isinstance(chunks, list):
                raise RuntimeError("Chunk metadata must be a list.")

            if (
                clip.ntotal != minilm.ntotal
                or clip.ntotal != len(chunks)
            ):
                raise RuntimeError("Text-index counts do not match.")

            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise RuntimeError("Invalid chunk metadata entry.")

                stored_workspace = chunk.get("workspace_id")
                if (
                    stored_workspace is not None
                    and stored_workspace != workspace_id
                ):
                    raise RuntimeError(
                        "Chunk metadata contains a different workspace ID."
                    )

            store.update({
                "chunks": chunks,
                "clip": clip,
                "minilm": minilm,
            })
        except Exception as exc:
            raise RuntimeError(
                f"Could not load text index for workspace '{workspace_id}'. "
                "Original files were preserved."
            ) from exc

    visual_keys = ("visual", "visuals")
    present_visual = [paths[key].exists() for key in visual_keys]

    if any(present_visual) and not all(present_visual):
        raise RuntimeError(
            f"Workspace '{workspace_id}' has incomplete visual-index files. "
            "Original files were preserved."
        )

    if all(present_visual):
        try:
            visuals = json.loads(
                paths["visuals"].read_text(encoding="utf-8")
            )
            visual_index = faiss.read_index(str(paths["visual"]))

            if not isinstance(visuals, list):
                raise RuntimeError("Visual metadata must be a list.")

            if visual_index.ntotal != len(visuals):
                raise RuntimeError("Visual-index counts do not match.")

            for visual in visuals:
                if not isinstance(visual, dict):
                    raise RuntimeError("Invalid visual metadata entry.")

                stored_workspace = visual.get("workspace_id")
                if (
                    stored_workspace is not None
                    and stored_workspace != workspace_id
                ):
                    raise RuntimeError(
                        "Visual metadata contains a different workspace ID."
                    )

            store["visual"] = visual_index
            store["visuals"] = visuals
        except Exception as exc:
            raise RuntimeError(
                f"Could not load visual index for workspace '{workspace_id}'. "
                "Original files were preserved."
            ) from exc

    return store


def _get_store(workspace_id: str = "default") -> dict:
    workspace_id = _safe_id(workspace_id)

    if workspace_id not in _stores:
        _stores[workspace_id] = _load_workspace(workspace_id)

    return _stores[workspace_id]


def migrate_legacy_index() -> None:
    """Import legacy text indexes into default workspace without deleting them."""
    with _lock:
        paths = _paths("default")

        if all(
            paths[key].exists()
            for key in ("chunks", "clip", "minilm")
        ):
            _stores["default"] = _load_workspace("default")
            return

        legacy_files = (LEGACY_CHUNKS, LEGACY_CLIP, LEGACY_MINILM)

        if not all(path.exists() for path in legacy_files):
            _stores["default"] = _load_workspace("default")
            return

        chunks = json.loads(LEGACY_CHUNKS.read_text(encoding="utf-8"))
        clip = faiss.read_index(str(LEGACY_CLIP))
        minilm = faiss.read_index(str(LEGACY_MINILM))

        if (
            not isinstance(chunks, list)
            or clip.ntotal != minilm.ntotal
            or clip.ntotal != len(chunks)
        ):
            raise RuntimeError(
                "Legacy index counts do not match. "
                "Original files were preserved."
            )

        migrated_chunks = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise RuntimeError("Legacy chunk metadata is invalid.")

            item = dict(chunk)
            item["workspace_id"] = "default"
            migrated_chunks.append(item)

        previous = _stores.get("default")

        _stores["default"] = {
            "chunks": migrated_chunks,
            "clip": clip,
            "minilm": minilm,
            "visual": None,
            "visuals": [],
        }

        try:
            _save("default")
        except Exception:
            if previous is None:
                _stores.pop("default", None)
            else:
                _stores["default"] = previous
            raise


def add_chunks(
    chunks: list[dict],
    clip_vectors: np.ndarray,
    minilm_vectors: np.ndarray,
    workspace_id: str = "default",
) -> None:
    if not chunks:
        return

    workspace_id = _safe_id(workspace_id)

    clip_vectors = _validate_vectors(
        clip_vectors, len(chunks), "CLIP vectors"
    )
    minilm_vectors = _validate_vectors(
        minilm_vectors, len(chunks), "MiniLM vectors"
    )

    with _lock:
        store = _get_store(workspace_id)

        if clip_vectors.shape[1] <= 0 or minilm_vectors.shape[1] <= 0:
            raise ValueError("Embedding dimensions must be nonzero.")

        if store["clip"] is not None and (
            store["clip"].d != clip_vectors.shape[1]
        ):
            raise ValueError("CLIP embedding dimension mismatch.")

        if store["minilm"] is not None and (
            store["minilm"].d != minilm_vectors.shape[1]
        ):
            raise ValueError("MiniLM embedding dimension mismatch.")

        previous = {
            "chunks": list(store["chunks"]),
            "clip": faiss.clone_index(store["clip"])
            if store["clip"] is not None else None,
            "minilm": faiss.clone_index(store["minilm"])
            if store["minilm"] is not None else None,
            "visual": faiss.clone_index(store["visual"])
            if store["visual"] is not None else None,
            "visuals": list(store["visuals"]),
        }

        try:
            if store["clip"] is None:
                store["clip"] = faiss.IndexFlatIP(clip_vectors.shape[1])

            if store["minilm"] is None:
                store["minilm"] = faiss.IndexFlatIP(minilm_vectors.shape[1])

            start = len(store["chunks"])

            for i, chunk in enumerate(chunks):
                item = dict(chunk)
                item["workspace_id"] = workspace_id
                item["global_id"] = start + i
                store["chunks"].append(item)

            store["clip"].add(clip_vectors)
            store["minilm"].add(minilm_vectors)
            _save(workspace_id)

        except Exception:
            _stores[workspace_id] = previous
            raise


def add_visuals(
    visuals: list[dict],
    vectors: np.ndarray,
    workspace_id: str = "default",
) -> None:
    """Store visual embeddings and their workspace-specific metadata."""
    if not visuals:
        return

    workspace_id = _safe_id(workspace_id)
    vectors = _validate_vectors(vectors, len(visuals), "Visual vectors")

    with _lock:
        store = _get_store(workspace_id)

        if (
            store["visual"] is not None
            and store["visual"].d != vectors.shape[1]
        ):
            raise ValueError("Visual embedding dimension mismatch.")

        previous_index = (
            faiss.clone_index(store["visual"])
            if store["visual"] is not None else None
        )
        previous_visuals = list(store["visuals"])

        try:
            if store["visual"] is None:
                store["visual"] = faiss.IndexFlatIP(vectors.shape[1])

            start = len(store["visuals"])

            for i, visual in enumerate(visuals):
                item = dict(visual)
                item["workspace_id"] = workspace_id
                item["visual_id"] = start + i
                store["visuals"].append(item)

            store["visual"].add(vectors)
            _save(workspace_id)

        except Exception:
            store["visual"] = previous_index
            store["visuals"] = previous_visuals
            raise


def search(
    query_clip: np.ndarray,
    query_minilm: np.ndarray,
    top_k: int = 5,
    workspace_id: str = "default",
):
    """Search only the requested workspace's text indexes."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    workspace_id = _safe_id(workspace_id)

    with _lock:
        store = _get_store(workspace_id)
        chunks = store["chunks"]

        if not chunks or store["clip"] is None or store["minilm"] is None:
            raise ValueError(
                f"No indexed text chunks in workspace '{workspace_id}'. "
                "Upload a paper first."
            )

        query_clip = _validate_vectors(query_clip, 1, "CLIP query")
        query_minilm = _validate_vectors(query_minilm, 1, "MiniLM query")

        if query_clip.shape[1] != store["clip"].d:
            raise ValueError("CLIP query embedding dimension mismatch.")

        if query_minilm.shape[1] != store["minilm"].d:
            raise ValueError("MiniLM query embedding dimension mismatch.")

        k1 = min(top_k, store["clip"].ntotal)
        k2 = min(top_k, store["minilm"].ntotal)

        scores1, ids1 = store["clip"].search(query_clip, k1)
        scores2, ids2 = store["minilm"].search(query_minilm, k2)

        clip_hits = [
            (int(idx), float(score))
            for idx, score in zip(ids1[0], scores1[0])
            if idx >= 0
        ]
        minilm_hits = [
            (int(idx), float(score))
            for idx, score in zip(ids2[0], scores2[0])
            if idx >= 0
        ]

        return clip_hits, minilm_hits, list(chunks)


def search_visuals(
    query_vectors: np.ndarray,
    top_k: int = 5,
    workspace_id: str = "default",
) -> list[dict]:
    """Retrieve visual matches exclusively from the requested workspace."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    workspace_id = _safe_id(workspace_id)
    query_vectors = _validate_vectors(
        query_vectors, 1, "Visual query vectors"
    )

    with _lock:
        store = _get_store(workspace_id)
        index = store["visual"]

        if index is None or not store["visuals"]:
            return []

        if query_vectors.shape[1] != index.d:
            raise ValueError("Visual query embedding dimension mismatch.")

        scores, ids = index.search(
            query_vectors, min(top_k, index.ntotal)
        )

        results = []

        for idx, score in zip(ids[0], scores[0]):
            if idx < 0:
                continue

            item = dict(store["visuals"][int(idx)])
            item["score"] = float(score)
            results.append(item)

        return results


def all_chunks(workspace_id: str | None = None) -> list[dict]:
    with _lock:
        if workspace_id is not None:
            return list(_get_store(workspace_id)["chunks"])

        results = []

        for folder in WORKSPACES_DIR.iterdir():
            if (
                folder.is_dir()
                and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", folder.name)
            ):
                results.extend(_get_store(folder.name)["chunks"])

        return results


def delete_document(
    source: str,
    workspace_id: str = "default",
) -> dict:
    """Remove one document's text and visual entries from one workspace."""
    workspace_id = _safe_id(workspace_id)

    with _lock:
        store = _get_store(workspace_id)

        keep = [
            i for i, chunk in enumerate(store["chunks"])
            if chunk.get("source") != source
        ]
        removed = len(store["chunks"]) - len(keep)

        visual_keep = [
            i for i, visual in enumerate(store["visuals"])
            if visual.get("source") != source
        ]

        removed_visuals = len(store["visuals"]) - len(visual_keep)

        if removed == 0 and removed_visuals == 0:
            return {"removed_chunks": 0, "removed_visuals": 0}

        old = {
            "chunks": store["chunks"],
            "clip": store["clip"],
            "minilm": store["minilm"],
            "visual": store["visual"],
            "visuals": store["visuals"],
        }

        try:
            if store["clip"] is not None and store["minilm"] is not None:
                clip_vectors = np.asarray(
                    [store["clip"].reconstruct(i) for i in keep],
                    dtype=np.float32,
                ).reshape(-1, store["clip"].d)

                minilm_vectors = np.asarray(
                    [store["minilm"].reconstruct(i) for i in keep],
                    dtype=np.float32,
                ).reshape(-1, store["minilm"].d)
            else:
                clip_vectors = minilm_vectors = None

            new_chunks = []

            for new_id, old_id in enumerate(keep):
                item = dict(store["chunks"][old_id])
                item["global_id"] = new_id
                new_chunks.append(item)

            if keep:
                new_clip = faiss.IndexFlatIP(clip_vectors.shape[1])
                new_minilm = faiss.IndexFlatIP(minilm_vectors.shape[1])
                new_clip.add(np.ascontiguousarray(clip_vectors))
                new_minilm.add(np.ascontiguousarray(minilm_vectors))
            else:
                new_clip = new_minilm = None

            old_visual_index = store["visual"]

            if old_visual_index is not None and visual_keep:
                visual_vectors = np.asarray(
                    [
                        old_visual_index.reconstruct(i)
                        for i in visual_keep
                    ],
                    dtype=np.float32,
                ).reshape(-1, old_visual_index.d)

                new_visual_index = faiss.IndexFlatIP(
                    visual_vectors.shape[1]
                )
                new_visual_index.add(np.ascontiguousarray(visual_vectors))
            else:
                new_visual_index = None

            store.update({
                "chunks": new_chunks,
                "clip": new_clip,
                "minilm": new_minilm,
                "visual": new_visual_index,
                "visuals": [
                    dict(store["visuals"][i]) for i in visual_keep
                ],
            })

            _save(workspace_id)

        except Exception:
            _stores[workspace_id] = old
            raise

        return {
            "removed_chunks": removed,
            "removed_visuals": removed_visuals,
        }


def clear(workspace_id: str | None = None) -> dict:
    """Clear selected workspace(s), preserving legacy index files."""
    with _lock:
        if workspace_id is None:
            targets = set(_stores.keys())
            targets.update(
                folder.name
                for folder in WORKSPACES_DIR.iterdir()
                if (
                    folder.is_dir()
                    and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", folder.name)
                )
            )
        else:
            targets = {_safe_id(workspace_id)}

        previous = {}

        for target in targets:
            store = _get_store(target)
            previous[target] = {
                "chunks": list(store["chunks"]),
                "clip": faiss.clone_index(store["clip"])
                if store["clip"] is not None else None,
                "minilm": faiss.clone_index(store["minilm"])
                if store["minilm"] is not None else None,
                "visual": faiss.clone_index(store["visual"])
                if store["visual"] is not None else None,
                "visuals": list(store["visuals"]),
            }

        try:
            for target in targets:
                _stores[target] = _empty_store()
                _save(target)

        except Exception:
            _stores.update(previous)
            raise

        return {"message": "Selected workspace index cleared."}


migrate_legacy_index()
