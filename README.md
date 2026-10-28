# Research Prism

**Research Prism** is a local multimodal research assistant for exploring scientific papers with text search, page/figure retrieval, OCR, table-aware chunks, mathematical-formula extraction (when MinerU is enabled), and local vision-language answering through Ollama.

## What changed in this upgrade

- Clean `backend/` and `frontend/` layout, matching Docker Compose paths.
- Qdrant persistent local vector store is the new default; the previous FAISS store remains available as `VECTOR_BACKEND=faiss`.
- BGE-M3 is the default text embedding model; CLIP remains for text-to-image and image-to-image retrieval.
- Hybrid candidate retrieval uses dense similarities, BM25 and reciprocal-rank fusion. A BGE cross-encoder can be enabled with `ENABLE_CROSS_ENCODER=true`.
- Chunking now defaults to 420 words with 60-word overlap, preserves page boundaries, and emits typed table/equation chunks.
- PDF extraction adds PyMuPDF native table extraction, OCR fallback, page renders and crops of embedded images. In `OCR_FIGURE_CROPS=auto` mode, EasyOCR also indexes text found in a bounded number of embedded figure crops when MinerU is unavailable; set it to `false` to reduce ingestion time.
- Optional MinerU 4.x integration adds richer document parsing, including equations, tables and reading order when the CLI is installed.
- Text-only queries can use the configured local Ollama model with source context; if the model is unavailable or returns a response that fails structure checks, the app falls back to extractive answers. Visual-only PDFs are allowed through ingestion if page images were extracted.

## Important notes before switching

1. **The original logo is preserved.** I copied the existing `insightforge-logo.svg` and `favicon.svg` unchanged from the previous project assets into `frontend/public/`. The UI still references `/insightforge-logo.svg`; only the surrounding product name changed to Research Prism.
2. **The embedding model changed.** Old MiniLM vectors cannot be queried with BGE-M3 vectors. Either re-index all source documents with the new settings (recommended) or temporarily use the legacy MiniLM model while using the FAISS-to-Qdrant migration utility.
3. **Source PDFs and indexes are not included in this source ZIP.** Back up your current `backend/data/` directory before replacing code. The `.gitignore` intentionally excludes uploaded papers, generated visuals, vector indexes and the Qdrant database.
4. MinerU is optional because it has a significant download/runtime footprint. Without it, Research Prism uses PyMuPDF, EasyOCR and native PDF table extraction. This fallback does not guarantee reliable extraction of every complex equation or scanned table.
5. Cross-encoder reranking is opt-in to control RAM and latency. Enable it only after the new Qdrant/BGE-M3 baseline is working.

## Prerequisites

- Python 3.11 (recommended for the base environment)
- Node.js and npm
- Docker Desktop (optional)
- Ollama with a vision-language model installed for local visual answering

## Run locally (PowerShell)

### 1. Backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

First run downloads BGE-M3 (larger than MiniLM) and CLIP weights. If PowerShell blocks `npm.ps1`, use `npm.cmd`.

### 2. Frontend (second terminal)

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open the Vite URL shown in the terminal, normally `http://localhost:5173`. API docs: `http://127.0.0.1:8000/docs`.

### 3. Ollama

For local backend execution, `backend/.env` uses `OLLAMA_URL=http://127.0.0.1:11434`. Confirm your desired model is installed with `ollama list`. The code's default remains `qwen2.5vl:3b` to avoid assuming a newer model is already pulled; set `VISION_MODEL` to another compatible Ollama vision model after installing it.

## Use Docker Compose

From the project root:

```powershell
Copy-Item backend/.env.example backend/.env
docker compose up --build
```

Docker Compose maps `backend/data` to `/app/data` and configures Ollama as `host.docker.internal:11434`. If MinerU is desired inside the image, set `INSTALL_MINERU=true` in a root `.env` file and rebuild; first-run model downloads can be large. This is an optional path, not required for starting the base stack.

## Optional structured document parsing (MinerU)

Install MinerU into the same Python environment as the backend:

```powershell
cd backend
python -m pip install -r requirements-document-ai.txt
```

Set in `backend/.env`:

```dotenv
DOCUMENT_PARSER=auto
MINERU_TIER=standard
```

`auto` uses `mineru-kit` when found and falls back to PyMuPDF + EasyOCR when unavailable or when parsing fails. Set `DOCUMENT_PARSER=mineru` to require MinerU and report its errors; set `DOCUMENT_PARSER=pymupdf` to disable MinerU. MinerU can add structured table/equation records from its Markdown output. For complex papers, review the rendered page/source in low-confidence cases rather than treating extraction as infallible.

Official MinerU setup and CLI reference: <https://opendatalab.github.io/MinerU/quick_start/> and <https://opendatalab.github.io/MinerU/usage/quick_usage/>.

## Re-index after changing embedding models or vector backends

**Do not delete the old data before making a backup.** The most reliable migration to the new BGE-M3 + Qdrant configuration is to re-embed the original source files:

```powershell
cd backend
python reindex_all_papers.py --workspace default --yes
```

This clears only the selected workspace's vector index, keeps source files in `data/papers`, and re-ingests supported files in that workspace's source directory. For non-default workspaces, use `--workspace <workspace_id>`; check that the corresponding files exist under `data/papers/<workspace_id>` first. Run the script once per workspace.

If you must preserve old FAISS vectors without re-embedding, set `TEXT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` and `TEXT_EMBEDDING_DIM=384`, then run:

```powershell
cd backend
python migrate_faiss_to_qdrant.py
```

That migrates the existing vectors but **does not upgrade their semantic quality**. Use re-indexing from source documents to switch to BGE-M3. Original FAISS index files are left untouched by the migration script.

## Configuration reference

| Setting | Default | Purpose |
|---|---|---|
| `VECTOR_BACKEND` | `qdrant` | `qdrant` or legacy `faiss` |
| `TEXT_EMBEDDING_MODEL` | `BAAI/bge-m3` | Dense text embedding model |
| `TEXT_EMBEDDING_DIM` | `1024` | Dense vector dimension for an empty-index response; must match the chosen model |
| `CLIP_MODEL` | `sentence-transformers/clip-ViT-B-32` | Text/image retrieval encoder |
| `CHUNK_TARGET_WORDS` | `420` | Approximate prose chunk size in words |
| `CHUNK_OVERLAP_WORDS` | `60` | Overlap between prose chunks; pages are not merged |
| `DOCUMENT_PARSER` | `auto` | Optional MinerU parser vs. PyMuPDF fallback |
| `ENABLE_CROSS_ENCODER` | `false` | Enable BGE Reranker V2 M3 after initial hybrid ranking |
| `VISION_MODEL` | `qwen2.5vl:3b` | Ollama model for visual-language and text answers |
| `ENABLE_TEXT_GENERATION` | `true` | Use local Ollama for text-only answers when available; otherwise use extractive fallback |

## Current limitations to be aware of

- The CLIP visual index includes rendered page images and embedded-image crops. It does not yet use a dedicated ColPali/ColQwen late-interaction page encoder.
- PyMuPDF native table detection can miss scanned tables. MinerU is the preferred optional structured parser; a further PaddleOCR PP-StructureV3 fallback could be added after benchmarking on the user's papers.
- Formula quality depends on document quality and parser mode; formula blocks are preserved in LaTeX-like Markdown when MinerU emits them, but all math cannot be guaranteed to parse perfectly.
- The cross-encoder is optional and has not been benchmarked against a labelled set for this project yet.
- New Qdrant data is stored under `backend/data/qdrant/`; uploaded sources, generated visuals and index stores must stay out of GitHub.

## Project layout

```text
Research-Prism/
├── backend/
│   ├── app/main.py
│   ├── app/services/
│   ├── data/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── requirements-document-ai.txt
│   ├── migrate_faiss_to_qdrant.py
│   └── reindex_all_papers.py
├── frontend/
│   ├── public/  # copy the original logo files here unchanged
│   ├── src/
│   ├── package.json
│   └── index.html
└── docker-compose.yml
```
