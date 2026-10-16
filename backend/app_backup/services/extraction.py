from pathlib import Path
import fitz

_OCR_READER = None

def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("Install EasyOCR to process image files or scanned PDFs.") from exc
        _OCR_READER = easyocr.Reader(["en"], gpu=False)
    return _OCR_READER

def extract_text(path: Path) -> dict:
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                pages.append({"page": i + 1, "text": page.get_text("text").strip()})
        # OCR individual pages that have little/no embedded text.
        if any(len(p["text"]) < 20 for p in pages):
            with fitz.open(path) as doc:
                for i, page in enumerate(doc):
                    if len(pages[i]["text"]) >= 20:
                        continue
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmp:
                        image_path = Path(tmp) / f"page_{i+1}.png"
                        pix.save(image_path)
                        words = _get_ocr_reader().readtext(str(image_path), detail=0, paragraph=True)
                    pages[i]["text"] = "\n".join(words)
        combined = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages if p["text"].strip())
        return {"text": combined, "pages": pages, "source": path.name}
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
        words = _get_ocr_reader().readtext(str(path), detail=0, paragraph=True)
        return {"text": "\n".join(words), "pages": [], "source": path.name}
    if ext in {".txt", ".md"}:
        return {"text": path.read_text(encoding="utf-8", errors="replace"), "pages": [], "source": path.name}
    raise ValueError(f"Unsupported extension: {ext}")
