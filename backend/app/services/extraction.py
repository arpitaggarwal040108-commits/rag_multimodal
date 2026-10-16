from pathlib import Path
import re
import fitz
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _ocr_image(image: Image.Image) -> str:
    """Run EasyOCR on an image, loading the reader only when needed."""
    import easyocr
    import numpy as np

    reader = _get_ocr_reader()
    results = reader.readtext(
        np.asarray(image.convert("RGB")),
        detail=0,
        paragraph=True,
    )
    return "\n".join(str(line).strip() for line in results if str(line).strip())


def _get_ocr_reader():
    global _ocr_reader

    try:
        return _ocr_reader
    except NameError:
        import easyocr
        import torch
        _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        return _ocr_reader


def extract_text(
    path: Path,
    figures_dir: Path | None = None,
) -> dict:
    """
    Extract text from PDF, image, TXT, or Markdown files.

    Returns:
        {
            "text": str,
            "pages": list[dict],
            "figures": list[str],
            "source": str
        }

    PDF pages are rendered separately for later visual embedding.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    pages = []
    figures = []

    if suffix == ".pdf":
        if figures_dir is not None:
            figures_dir = Path(figures_dir)
            figures_dir.mkdir(parents=True, exist_ok=True)

        with fitz.open(path) as document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")

            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()

                # OCR pages with little or no extractable text.
                if len(page_text) < 30:
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(1.5, 1.5),
                        alpha=False,
                    )
                    from io import BytesIO

                    image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
                    try:
                        ocr_text = _ocr_image(image)
                    finally:
                        image.close()

                    if ocr_text:
                        page_text = ocr_text

                pages.append({
                    "page": page_number,
                    "text": page_text,
                })

                if figures_dir is not None:
                    output_path = figures_dir / f"page_{page_number:04d}.png"
                    page.get_pixmap(
                        matrix=fitz.Matrix(1.5, 1.5),
                        alpha=False,
                    ).save(str(output_path))
                    figures.append(str(output_path))

        combined = "\n\n".join(
            f"[Page {item['page']}]\n{item['text']}"
            for item in pages
            if item["text"]
        )

        return {
            "text": combined,
            "pages": pages,
            "figures": figures,
            "source": path.name,
        }

    if suffix in IMAGE_EXTENSIONS:
        with Image.open(path) as image:
            text = _ocr_image(image)

        pages.append({"page": 1, "text": text})
        return {
            "text": f"[Page 1]\n{text}" if text else "",
            "pages": pages,
            "figures": [str(path.resolve())],
            "source": path.name,
        }

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace").strip()

        # Plain-text documents have no reliable page boundaries.
        pages.append({"page": 1, "text": text})
        return {
            "text": f"[Page 1]\n{text}" if text else "",
            "pages": pages,
            "figures": [],
            "source": path.name,
        }

    raise ValueError(f"Unsupported file type: {suffix}")
