from pathlib import Path
import traceback

from app.services.pipeline import reindex_visuals_only
ROOT = Path(__file__).resolve().parent
PAPERS_DIR = Path.home() / "Downloads"

TARGET_FILES = [
    "mos pyq.pdf",
    "MS-299 end sem report 1.pdf",
]


def find_pdf(filename: str) -> Path | None:
    matches = [
        path
        for path in PAPERS_DIR.rglob(filename)
        if path.is_file()
    ]

    if not matches:
        return None

    # Prefer the PDF closest to the papers directory.
    matches.sort(
        key=lambda path: (
            len(path.relative_to(PAPERS_DIR).parts),
            str(path).lower(),
        )
    )
    return matches[0]


def main() -> None:
    successful = 0
    failed = 0

    for filename in TARGET_FILES:
        print(f"\n{'=' * 55}")
        print(f"Processing: {filename}")

        pdf_path = find_pdf(filename)

        if pdf_path is None:
            print(f"NOT FOUND: {filename}")
            failed += 1
            continue

        print(f"Found: {pdf_path}")

        try:
            result = reindex_visuals_only(
                path=pdf_path,
                workspace_id="default",
                original_filename=filename,
            )
            print("Result:", result)
            successful += 1

        except Exception as exc:
            print(f"FAILED: {filename}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"Completed: {successful}")
    print(f"Failed/not found: {failed}")
    print("Existing text chunks were not intentionally modified.")


if __name__ == "__main__":
    main()