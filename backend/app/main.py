from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import os
import re
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from app.services.pipeline import (
    ingest_file,
    query_rag,
    list_papers,
    delete_paper,
    clear_all,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "papers"
VISUALS_DIR = DATA_DIR / "visuals"
WORKSPACES_DIR = DATA_DIR / "indices" / "workspaces"
REGISTRY_PATH = DATA_DIR / "workspaces.json"

VISUALS_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".txt", ".md"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024


app = FastAPI(
    title="Research Prism — Multimodal Research Assistant",
    version="3.0.0",
)
app.mount(
    "/visuals",
    StaticFiles(directory=str(VISUALS_DIR)),
    name="visuals",
)

origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    workspace_id: str = Field(default="default", min_length=1, max_length=80)


class WorkspaceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


def safe_workspace_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", value):
        raise HTTPException(status_code=400, detail="Invalid workspace ID.")
    return value


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        try:
            data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("workspaces"), list):
                return data
        except (json.JSONDecodeError, OSError):
            pass

    data = {"workspaces": [{"id": "default", "name": "General Research"}]}
    save_registry(data)
    return data


def save_registry(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp = REGISTRY_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(REGISTRY_PATH)


def require_workspace(workspace_id: str) -> dict:
    workspace_id = safe_workspace_id(workspace_id)

    for workspace in load_registry()["workspaces"]:
        if workspace.get("id") == workspace_id:
            return workspace

    raise HTTPException(status_code=404, detail="Workspace not found.")


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Research Prism — Multimodal Research Assistant",
    }


@app.get("/api/workspaces")
def get_workspaces():
    result = []

    for workspace in load_registry()["workspaces"]:
        papers = list_papers(workspace["id"])
        result.append({
            **workspace,
            "paper_count": papers["paper_count"],
            "total_chunks": papers["total_chunks"],
        })

    return {"workspaces": result}


@app.post("/api/workspaces")
def create_workspace(request: WorkspaceRequest):
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Workspace name cannot be empty.")

    registry = load_registry()

    if any(w["name"].casefold() == name.casefold() for w in registry["workspaces"]):
        raise HTTPException(status_code=409, detail="A workspace with this name already exists.")

    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:55] or "research"
    workspace_id = f"{base}-{uuid.uuid4().hex[:8]}"
    workspace = {"id": workspace_id, "name": name}

    registry["workspaces"].append(workspace)
    save_registry(registry)

    (WORKSPACES_DIR / workspace_id).mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / workspace_id).mkdir(parents=True, exist_ok=True)

    return {"message": "Workspace created.", "workspace": workspace}


@app.patch("/api/workspaces/{workspace_id}")
def rename_workspace(workspace_id: str, request: WorkspaceRequest):
    workspace = require_workspace(workspace_id)
    name = request.name.strip()

    if not name:
        raise HTTPException(status_code=400, detail="Workspace name cannot be empty.")

    registry = load_registry()

    if any(
        w["id"] != workspace_id and w["name"].casefold() == name.casefold()
        for w in registry["workspaces"]
    ):
        raise HTTPException(status_code=409, detail="A workspace with this name already exists.")

    workspace["name"] = name
    save_registry(registry)

    return {"message": "Workspace renamed.", "workspace": workspace}


@app.delete("/api/workspaces/{workspace_id}")
def remove_workspace(workspace_id: str):
    workspace = require_workspace(workspace_id)

    if workspace_id == "default":
        raise HTTPException(status_code=400, detail="The default workspace cannot be deleted.")

    clear_all(workspace_id)

    shutil.rmtree(WORKSPACES_DIR / workspace_id, ignore_errors=True)
    shutil.rmtree(UPLOAD_DIR / workspace_id, ignore_errors=True)

    registry = load_registry()
    registry["workspaces"] = [
        w for w in registry["workspaces"] if w.get("id") != workspace_id
    ]
    save_registry(registry)

    return {"message": f"Workspace '{workspace['name']}' deleted."}


@app.post("/api/upload")
async def upload_paper(
    file: UploadFile = File(...),
    workspace_id: str = Form("default"),
):
    require_workspace(workspace_id)

    original_name = Path(file.filename or "upload").name
    suffix = Path(original_name).suffix.lower()

    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED)}",
        )

    directory = UPLOAD_DIR if workspace_id == "default" else UPLOAD_DIR / workspace_id
    directory.mkdir(parents=True, exist_ok=True)

    target = directory / original_name
    if target.exists():
        target = directory / (
            f"{target.stem}_{uuid.uuid4().hex[:8]}{target.suffix}"
        )

    # Read at most one byte beyond the configured limit.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    target.write_bytes(content)

    try:
        result = ingest_file(
            target,
            workspace_id=workspace_id,
            original_filename=original_name,
        )
        return {
            "message": "Ingestion completed",
            "filename": target.name,
            "original_filename": original_name,
            **result,
        }
    except Exception as exc:
        # Keep previous indexes intact; remove only this failed upload.
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        ) from exc
    finally:
        await file.close()


@app.post("/api/query")
def query(request: QueryRequest):
    require_workspace(request.workspace_id)

    try:
        return query_rag(
            request.question,
            request.top_k,
            workspace_id=request.workspace_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {exc}",
        ) from exc


@app.get("/api/papers")
def papers(workspace_id: str = "default"):
    require_workspace(workspace_id)
    return list_papers(workspace_id)


@app.delete("/api/papers/{filename}")
def remove_paper(filename: str, workspace_id: str = "default"):
    require_workspace(workspace_id)

    try:
        return delete_paper(filename, workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/clear")
def clear(workspace_id: str = "default"):
    require_workspace(workspace_id)
    return clear_all(workspace_id)
