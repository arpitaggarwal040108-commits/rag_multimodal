
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



def _clip_text_for_chunk(text: str) -> str:
    """Truncate document text safely for CLIP's 77-token limit."""
    from app.services.embeddings import _truncate_clip_text, get_clip

    return _truncate_clip_text(str(text), get_clip())


def _figure_page_number(image_path: Path, fallback: int) -> int:
    """Infer a page number from common extractor filename conventions."""
    stem = image_path.stem
    patterns = (
        r"(?:page|pg|p)[_ -]?(\d+)",
        r"(?:^|[_ -])(\d+)(?:[_ -]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            try:
                number = int(match.group(1))
                if number > 0:
                    return number
            except (TypeError, ValueError):
                pass
    return fallback


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

        # CLIP is used for visual/cross-modal retrieval. Its text encoder
        # has a 77-token limit, so encode short representations of chunks.
        clip_texts = [
            _clip_text_for_chunk(chunk["text"])
            for chunk in chunks
        ]
        clip_vectors = embed_texts(clip_texts, "clip")

        # MiniLM receives the complete chunk text for semantic text retrieval.
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

        for figure_index, figure in enumerate(
            extracted.get("figures", []),
            start=1,
        ):
            image_path = Path(figure)
            page_number = _figure_page_number(image_path, figure_index)

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
            visual_metadata.append(
                {
                    "source": path.name,
                    "original_filename": display_name,
                    "page": page_number,
                    "path": str(image_path),
                    "url": url,
                    "workspace_id": workspace_id,
                }
            )

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
        # Do not remove visual_dir here: it may contain assets from an earlier
        # successful ingestion of the same document. Cleanup is intentionally
        # conservative so a failed re-index cannot delete existing assets.
        raise



def _build_retrieval_queries(question: str) -> list[str]:
    """Create focused subqueries for explicit comparison dimensions in a question."""
    question = re.sub(r"\s+", " ", str(question or "")).strip()
    queries = [question]
    lower = question.lower()

    # Preserve the user's wording while ensuring each explicitly named method
    # can retrieve its own evidence instead of competing in one broad query.
    method_patterns = [
        r"image[\s-]*based RAG",
        r"image\s*\+\s*text[\s-]*based RAG",
        r"image[\s-]*and[\s-]*text[\s-]*based RAG",
        r"page[\s-]*level retrieval",
        r"element[\s-]*level retrieval",
    ]
    methods = []
    for pattern in method_patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            value = match.group(0)
            if value.lower() not in {item.lower() for item in methods}:
                methods.append(value)

    metric_patterns = [
        r"retrieval accuracy",
        r"computational cost",
        r"latency",
        r"scalability",
        r"precision",
        r"recall",
        r"F1(?:[\s-]*score)?",
    ]
    metrics = []
    for pattern in metric_patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            metrics.append(match.group(0))

    for method in methods:
        queries.append(f"{method} definition advantages limitations evidence")
    for metric in metrics:
        queries.append(f"{metric} measured results comparison evidence")

    # Stable de-duplication; the original question always remains first.
    result = []
    seen = set()
    for query in queries:
        key = query.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(query)
    return result


def _merge_ranked_results(result_sets: list[list[dict]], limit: int) -> list[dict]:
    """Fuse query-specific rankings without assuming scores from different queries are calibrated."""
    fused = {}
    for result_set in result_sets:
        for rank, item in enumerate(result_set, start=1):
            identity = (
                item.get("workspace_id", "default"),
                item.get("source", item.get("original_filename", "unknown")),
                item.get("chunk_id", item.get("global_id", item.get("chunk_index", "unknown"))),
                item.get("page"),
                str(item.get("text", ""))[:300],
            )
            record = fused.setdefault(identity, {"item": dict(item), "rrf": 0.0, "best_rank": rank})
            record["rrf"] += 1.0 / (60 + rank)
            record["best_rank"] = min(record["best_rank"], rank)
            # Keep the strongest observed per-query score as diagnostic metadata.
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            record["item"]["best_query_score"] = max(
                float(record["item"].get("best_query_score", score)), score
            )
    
    ordered = sorted(
        fused.values(),
        key=lambda record: (
            record["rrf"],
            record["item"].get("best_query_score", 0.0),
        ),
        reverse=True,
    )

    results = []
    for record in ordered[:limit]:
        item = record["item"]
        item["multi_query_rrf_score"] = round(record["rrf"], 6)
        results.append(item)
    return results

def query_rag(
    question: str,
    top_k: int = 8,
    workspace_id: str = "default",
) -> dict:
    question = str(question or "").strip()

    if not question:
        raise ValueError("Please enter a question.")

    top_k = _validate_top_k(top_k)
    workspace_id = _validate_workspace(workspace_id)

    # The API may explicitly send a small top_k (for example, 3).
    # Keep a useful minimum for evidence-grounded answers without changing
    # the underlying indexes or any ingestion/embedding behavior.
    retrieval_k = max(top_k, 8)

    chunks = all_chunks(workspace_id)

    if not chunks:
        raise ValueError(
            "This workspace has no indexed documents. Upload a document first."
        )

    # Run the original query plus focused subqueries for explicitly requested
    # methods and metrics. This improves evidence coverage without touching indexes.
    retrieval_queries = _build_retrieval_queries(question)
    per_query_k = max(retrieval_k, 8)
    ranked_sets = [
        rerank(query, top_k=per_query_k, workspace_id=workspace_id)
        for query in retrieval_queries
    ]
    ranked = _merge_ranked_results(
        ranked_sets,
        limit=min(20, max(retrieval_k, 8)),
    )

    print("\n========== INSIGHTFORGE RETRIEVED TEXT ==========")
    print("Retrieval queries:", retrieval_queries)
    for i, chunk in enumerate(ranked, start=1):
        print(
            f"\n--- Chunk {i} | "
            f"Source: {chunk.get('original_filename', chunk.get('source'))} | "
            f"Page: {chunk.get('page')} | "
            f"Score: {chunk.get('score')} | "
            f"Multi-query RRF: {chunk.get('multi_query_rrf_score')} ---"
        )
        print(str(chunk.get("text", ""))[:2000])
        print("=================================================\n")

    # Search visuals using the same focused query set, then fuse the rankings.
    visual_sets = []
    for retrieval_query in retrieval_queries:
        query_vector = embed_texts([retrieval_query], "clip")
        visual_set = search_visuals(
            query_vector,
            top_k=per_query_k,
            workspace_id=workspace_id,
        )
        visual_sets.append(visual_set or [])

    visual_fused = {}
    for visual_set in visual_sets:
        for rank, visual in enumerate(visual_set, start=1):
            if not isinstance(visual, dict):
                continue
            identity = (
                visual.get("workspace_id", workspace_id),
                visual.get("source", visual.get("original_filename", "unknown")),
                visual.get("page"),
                visual.get("path", visual.get("url", "")),
            )
            record = visual_fused.setdefault(identity, {"item": dict(visual), "rrf": 0.0})
            record["rrf"] += 1.0 / (60 + rank)

    visual_results = [
        record["item"]
        for record in sorted(
            visual_fused.values(),
            key=lambda record: record["rrf"],
            reverse=True,
        )[:min(20, max(retrieval_k, 8))]
    ]

    accessible_visuals = []
    for visual in visual_results:
        item = dict(visual)
        url = item.get("url")
        if not url and item.get("path"):
            url = _visual_url(Path(item["path"]))
        if url:
            item["url"] = url
            accessible_visuals.append(item)

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

    for figure_index, figure in enumerate(
        extracted.get("figures", []),
        start=1,
    ):
        image_path = Path(figure)
        page_number = _figure_page_number(image_path, figure_index)

        if not image_path.is_file():
            continue

        url = _visual_url(image_path)

        if not url:
            continue

        visual_paths.append(str(image_path))
        visual_metadata.append(
            {
                "source": path.name,
                "original_filename": display_name,
                "page": page_number,
                "path": str(image_path),
                "url": url,
                "workspace_id": workspace_id,
            }
        )

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
