from pathlib import Path
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from app.services.pipeline import ingest_file, query_rag, list_papers, clear_all

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "data" / "papers"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".txt", ".md"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024

app = FastAPI(title="Hybrid Multimodal RAG Research Assistant", version="1.0.0")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)

@app.get("/")
def health():
    return {"status": "ok", "service": "Hybrid Multimodal RAG Research Assistant"}

@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Allowed: {sorted(ALLOWED)}")
    raw_name = Path(file.filename or "upload").name
    target = UPLOAD_DIR / raw_name
    if target.exists():
        target = UPLOAD_DIR / f"{target.stem}_{os.urandom(3).hex()}{target.suffix}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Maximum upload size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.")
    target.write_bytes(content)
    try:
        result = ingest_file(target)
        return {"message": "Ingestion completed", "filename": target.name, **result}
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

@app.post("/api/query")
def query(request: QueryRequest):
    try:
        return query_rag(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

@app.get("/api/papers")
def papers():
    return list_papers()

@app.delete("/api/clear")
def clear():
    return clear_all()
