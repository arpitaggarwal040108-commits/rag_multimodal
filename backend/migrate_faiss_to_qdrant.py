"""Copy existing FAISS vectors into Qdrant without deleting the old index files.

Important: this preserves the old MiniLM/CLIP vectors. Keep TEXT_EMBEDDING_MODEL
set to sentence-transformers/all-MiniLM-L6-v2 to query the migrated index. For
BGE-M3, prefer re-index_all_papers.py --yes from source PDFs instead.
"""
from pathlib import Path
import json
import sys
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

import numpy as np
import faiss
from app.services.vector_store_qdrant import add_chunks, add_visuals, all_chunks

INDEX_ROOT = ROOT / "data" / "indices"


def read_vectors(index_path: Path):
    index = faiss.read_index(str(index_path))
    if not index.ntotal:
        return np.empty((0, index.d), dtype=np.float32)
    return np.asarray([index.reconstruct(i) for i in range(index.ntotal)], dtype=np.float32)


def migrate_workspace(folder: Path, workspace_id: str):
    chunks_path, clip_path, text_path = folder / "chunks.json", folder / "clip.index", folder / "minilm.index"
    if not all(p.is_file() for p in (chunks_path, clip_path, text_path)):
        return False, f"Missing one or more text index files in {folder}"
    if all_chunks(workspace_id):
        return False, f"Qdrant already has chunks in workspace '{workspace_id}'; skipped to avoid duplicates"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    clip_vectors = read_vectors(clip_path)
    text_vectors = read_vectors(text_path)
    if len(chunks) != len(clip_vectors) or len(chunks) != len(text_vectors):
        return False, f"Index counts do not match for workspace '{workspace_id}'"
    add_chunks(chunks, clip_vectors, text_vectors, workspace_id=workspace_id)
    visual_meta_path, visual_index_path = folder / "visuals.json", folder / "visual.index"
    if visual_meta_path.is_file() and visual_index_path.is_file():
        visuals = json.loads(visual_meta_path.read_text(encoding="utf-8"))
        vectors = read_vectors(visual_index_path)
        if len(visuals) == len(vectors) and visuals:
            add_visuals(visuals, vectors, workspace_id=workspace_id)
    return True, f"Migrated workspace '{workspace_id}'"


def main():
    results = []
    workspaces_dir = INDEX_ROOT / "workspaces"
    if workspaces_dir.is_dir():
        for folder in sorted(workspaces_dir.iterdir()):
            if folder.is_dir():
                results.append(migrate_workspace(folder, folder.name))
    # Handle the pre-workspace legacy index layout as the default workspace.
    if all((INDEX_ROOT / name).is_file() for name in ("chunks.json", "clip.index", "minilm.index")):
        results.append(migrate_workspace(INDEX_ROOT, "default"))
    for ok, message in results:
        print(("OK: " if ok else "SKIP: ") + message)
    if not results:
        print("No legacy FAISS indexes found. Upload/re-index source files instead.")

if __name__ == "__main__":
    main()
