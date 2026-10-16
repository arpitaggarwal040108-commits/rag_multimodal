
from pathlib import Path
import re
import shutil

from app.services.extraction import extract_text
from app.services.chunking import semantic_chunk
from app.services.embeddings import embed_texts, embed_images
from app.services.vector_store import (
    add_chunks,
    add_visuals,
    all_chunks,
    clear,
    delete_document,
    search_visuals,
)
from app.services.retrieval import rerank
from app.services.generation import generate_answer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
VISUALS_DIR = DATA_DIR / "visuals"

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
VISUALS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_component(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value[:120] or "default"


def _validate_workspace(workspace_id: str) -> str:
    workspace_id = str(workspace_id or "").strip()

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", workspace_id):
        raise ValueError("Invalid workspace ID.")

    return workspace_id


def _validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    return top_k


def _paper_directory(workspace_id: str = "default") -> Path:
    if workspace_id == "default":
        return PAPERS_DIR

    directory = PAPERS_DIR / _safe_component(workspace_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _visual_directory(workspace_id: str, filename: str) -> Path:
    directory = (
        VISUALS_DIR
        / _safe_component(workspace_id)
        / _safe_component(Path(filename).stem)
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _visual_url(image_path: Path) -> str | None:
    try:
        relative_path = image_path.resolve().relative_to(
            VISUALS_DIR.resolve()
        )
    except ValueError:
        return None

    return "/visuals/" + relative_path.as_posix()


def ingest_file(
    path: str | Path,
    workspace_id: str = "default",
    original_filename: str | None = None,
) -> dict:
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    workspace_id = _validate_workspace(workspace_id)
    display_name = original_filename or path.name
    visual_dir = _visual_directory(workspace_id, path.name)
    is_pdf = path.suffix.lower() == ".pdf"

    try:
        extracted = extract_text(
            path,
            figures_dir=visual_dir if is_pdf else None,
        )

        text = extracted.get("text", "").strip()

        if not text:
            raise ValueError(
                f"No readable text could be extracted from {display_name}."
            )

        chunks = semantic_chunk(
            text,
            path.name,
            pages=extracted.get("pages", []),
        )

        if not chunks:
            raise ValueError(
                f"No searchable text chunks were created for {display_name}."
            )

        for chunk in chunks:
            chunk["workspace_id"] = workspace_id
            chunk["original_filename"] = display_name

        clip_vectors = embed_texts(
            [chunk["text"] for chunk in chunks],
            "clip",
        )
        minilm_vectors = embed_texts(
            [chunk["text"] for chunk in chunks],
            "minilm",
        )

        add_chunks(
            chunks,
            clip_vectors,
            minilm_vectors,
            workspace_id=workspace_id,
        )

        visual_metadata = []
        visual_paths = []

        for page_number, figure in enumerate(
            extracted.get("figures", []),
            start=1,
        ):
            image_path = Path(figure)

            if not image_path.is_file():
                continue

            try:
                image_path.resolve().relative_to(
                    VISUALS_DIR.resolve()
                )
            except ValueError:
                destination = visual_dir / image_path.name

                if image_path.resolve() != destination.resolve():
                    shutil.copy2(image_path, destination)

                image_path = destination

            url = _visual_url(image_path)

            if not url:
                continue

            visual_paths.append(str(image_path))
            visual_metadata.append({
                "source": path.name,
                "original_filename": display_name,
                "page": page_number,
                "path": str(image_path),
                "url": url,
                "workspace_id": workspace_id,
            })

        if visual_paths:
            visual_vectors = embed_images(visual_paths)
            add_visuals(
                visual_metadata,
                visual_vectors,
                workspace_id=workspace_id,
            )

        return {
            "filename": path.name,
            "original_filename": display_name,
            "workspace_id": workspace_id,
            "chunks_created": len(chunks),
            "visuals_created": len(visual_paths),
            "pages_processed": len(extracted.get("pages", [])),
            "status": "success",
        }

    except Exception:
        if is_pdf:
            shutil.rmtree(visual_dir, ignore_errors=True)
        raise


def query_rag(
    question: str,
    top_k: int = 5,
    workspace_id: str = "default",
) -> dict:
    question = str(question or "").strip()

    if not question:
        raise ValueError("Please enter a question.")

    top_k = _validate_top_k(top_k)
    workspace_id = _validate_workspace(workspace_id)

    chunks = all_chunks(workspace_id)

    if not chunks:
        raise ValueError(
            "This workspace has no indexed documents. Upload a document first."
        )

    ranked = rerank(
        question,
        top_k=top_k,
        workspace_id=workspace_id,
    )

    # Retrieve visuals before generation so the vision model can inspect them.
    query_vector = embed_texts([question], "clip")

    visual_results = search_visuals(
        query_vector,
        top_k=top_k,
        workspace_id=workspace_id,
    )

    accessible_visuals = []

    for visual in visual_results:
        item = dict(visual)
        url = item.get("url")

        if not url and item.get("path"):
            url = _visual_url(Path(item["path"]))

        if url:
            item["url"] = url
            accessible_visuals.append(item)

    # Both text and visual evidence are now available to the answer stage.
    result = generate_answer(
        question=question,
        chunks=ranked,
        visual_results=accessible_visuals,
    )

    result["visual_results"] = accessible_visuals
    result["retrieved_chunks"] = len(ranked)
    result["workspace_id"] = workspace_id
    result["evidence_status"] = (
        "retrieved"
        if ranked or accessible_visuals
        else "insufficient_evidence"
    )

    return result


def list_papers(workspace_id: str = "default") -> dict:
    workspace_id = _validate_workspace(workspace_id)
    chunks = all_chunks(workspace_id)
    papers = {}

    for chunk in chunks:
        source = chunk.get("source", "unknown")

        if source not in papers:
            papers[source] = {
                "filename": source,
                "original_filename": chunk.get(
                    "original_filename", source
                ),
                "chunks": 0,
                "pages": set(),
                "workspace_id": workspace_id,
            }

        papers[source]["chunks"] += 1

        if chunk.get("page") is not None:
            papers[source]["pages"].add(chunk["page"])

    paper_list = []

    for paper in papers.values():
        paper["pages"] = len(paper["pages"])
        paper_list.append(paper)

    paper_list.sort(
        key=lambda item: item["original_filename"].lower()
    )

    return {
        "papers": paper_list,
        "paper_count": len(paper_list),
        "total_chunks": len(chunks),
        "workspace_id": workspace_id,
    }


def delete_paper(
    filename: str,
    workspace_id: str = "default",
) -> dict:
    workspace_id = _validate_workspace(workspace_id)

    if not filename or Path(filename).name != filename:
        raise ValueError("Invalid document filename.")

    chunks = all_chunks(workspace_id)

    if not any(
        chunk.get("source") == filename for chunk in chunks
    ):
        raise ValueError(
            f"Document '{filename}' was not found in this workspace."
        )

    delete_document(filename, workspace_id=workspace_id)

    paper_path = _paper_directory(workspace_id) / filename

    if paper_path.is_file():
        paper_path.unlink()

    shutil.rmtree(
        _visual_directory(workspace_id, filename),
        ignore_errors=True,
    )

    return {
        "filename": filename,
        "workspace_id": workspace_id,
        "status": "deleted",
    }


def clear_all(workspace_id: str = "default") -> dict:
    workspace_id = _validate_workspace(workspace_id)

    clear(workspace_id=workspace_id)
    paper_dir = _paper_directory(workspace_id)

    if workspace_id == "default":
        for item in paper_dir.iterdir():
            if item.is_file():
                item.unlink()
    else:
        shutil.rmtree(paper_dir, ignore_errors=True)

    shutil.rmtree(
        VISUALS_DIR / workspace_id,
        ignore_errors=True,
    )

    return {
        "workspace_id": workspace_id,
        "status": "cleared",
    }


def reindex_visuals_only(
    path: str | Path,
    workspace_id: str = "default",
    original_filename: str | None = None,
) -> dict:
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Visual-only re-indexing currently supports PDFs."
        )

    workspace_id = _validate_workspace(workspace_id)
    display_name = original_filename or path.name
    visual_dir = _visual_directory(workspace_id, path.name)

    extracted = extract_text(path, figures_dir=visual_dir)

    visual_paths = []
    visual_metadata = []

    for page_number, figure in enumerate(
        extracted.get("figures", []),
        start=1,
    ):
        image_path = Path(figure)

        if not image_path.is_file():
            continue

        url = _visual_url(image_path)

        if not url:
            continue

        visual_paths.append(str(image_path))
        visual_metadata.append({
            "source": path.name,
            "original_filename": display_name,
            "page": page_number,
            "path": str(image_path),
            "url": url,
            "workspace_id": workspace_id,
        })

    if visual_paths:
        vectors = embed_images(visual_paths)
        add_visuals(
            visual_metadata,
            vectors,
            workspace_id=workspace_id,
        )

    return {
        "filename": path.name,
        "workspace_id": workspace_id,
        "visuals_created": len(visual_paths),
        "status": "success",
    }
