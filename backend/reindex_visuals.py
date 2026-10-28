"""Re-index visual page/image assets for a single PDF without changing text chunks."""
from pathlib import Path
import argparse
import json
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from app.services.pipeline import reindex_visuals_only

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the PDF to re-index visually")
    parser.add_argument("--workspace", default="default", help="Workspace ID")
    parser.add_argument("--display-name", default=None, help="Original display filename")
    args = parser.parse_args()
    result = reindex_visuals_only(args.pdf, workspace_id=args.workspace, original_filename=args.display_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
