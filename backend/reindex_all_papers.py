"""Rebuild Research Prism's indexes from uploaded papers after model/backend changes.

This script clears only vector indexes for the chosen workspace, not source files.
It reads source files from data/papers (default workspace) or data/papers/<workspace>.
"""
from pathlib import Path
import argparse
import json
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from app.services.pipeline import ingest_file
from app.services.vector_store import clear

DATA = ROOT / "data"
PAPERS = DATA / "papers"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--yes", action="store_true", help="Confirm clearing the selected workspace index")
    args = parser.parse_args()
    source_dir = PAPERS if args.workspace == "default" else PAPERS / args.workspace
    if not source_dir.exists():
        raise SystemExit(f"Paper directory does not exist: {source_dir}")
    files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".txt", ".md"})
    if not files:
        raise SystemExit(f"No supported source files found in {source_dir}")
    if not args.yes:
        raise SystemExit(f"Found {len(files)} files. Re-run with --yes to clear and rebuild the '{args.workspace}' index.")
    # Clear vectors only; source files remain in the papers directory.
    clear(workspace_id=args.workspace)
    results, failures = [], []
    for path in files:
        try:
            results.append(ingest_file(path, workspace_id=args.workspace, original_filename=path.name))
            print(f"OK: {path.name}")
        except Exception as exc:
            failures.append({"file": path.name, "error": str(exc)})
            print(f"FAILED: {path.name}: {exc}")
    print(json.dumps({"workspace": args.workspace, "indexed": len(results), "failed": failures}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
