from __future__ import annotations

from io import BytesIO
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz
from PIL import Image, ImageEnhance, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_OCR_READER = None


def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        import torch
        _OCR_READER = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _OCR_READER


def _prepare_ocr_image(image: Image.Image) -> Image.Image:
    """Light preprocessing for weak scans before OCR; keep original for vision."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.15)
    return image


def _ocr_image(image: Image.Image) -> str:
    """OCR with EasyOCR and confidence filtering for noisy scan pages."""
    import numpy as np
    prepared = _prepare_ocr_image(image)
    try:
        results = _get_ocr_reader().readtext(
            np.asarray(prepared), detail=1, paragraph=False,
            batch_size=int(os.getenv("OCR_BATCH_SIZE", "1")),
        )
        valid = []
        for result in results:
            if len(result) < 3:
                continue
            box, text, confidence = result
            text = str(text).strip()
            if not text or float(confidence) < float(os.getenv("OCR_MIN_CONFIDENCE", "0.20")):
                continue
            try:
                ys = [float(point[1]) for point in box]
                xs = [float(point[0]) for point in box]
                y, x = sum(ys) / len(ys), min(xs)
            except Exception:
                y, x = 0.0, 0.0
            valid.append((round(y / 12), x, text))
        valid.sort(key=lambda item: (item[0], item[1]))
        return "\n".join(item[2] for item in valid)
    finally:
        prepared.close()


def _markdown_table(rows: list[list[object]]) -> str:
    normalized = [[re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", r"\|") for cell in row] for row in rows]
    width = max((len(row) for row in normalized), default=0)
    if not width:
        return ""
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    if len(normalized) == 1:
        normalized.append([""] * width)
    header = "| " + " | ".join(normalized[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in normalized[1:]]
    return "\n".join([header, separator, *body])


def _extract_pdf_tables(page: fitz.Page, page_number: int) -> list[dict]:
    """Use PyMuPDF table detection for native/vector tables as a first pass."""
    blocks = []
    try:
        finder = page.find_tables()
        tables = getattr(finder, "tables", []) or []
    except Exception:
        return blocks
    for index, table in enumerate(tables, start=1):
        try:
            rows = table.extract()
        except Exception:
            continue
        markdown = _markdown_table(rows or [])
        if markdown and len(markdown) > 10:
            blocks.append({
                "type": "table", "page": page_number,
                "text": f"Table {index} (page {page_number})\n{markdown}",
                "table_index": index,
            })
    return blocks


def _extract_embedded_image_crops(page: fitz.Page, page_number: int, figures_dir: Path) -> list[str]:
    """Crop embedded page images so visual search can target figures, not only whole pages."""
    saved = []
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        infos = []
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    crop_index = 0
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        rect = fitz.Rect(bbox) & page.rect
        if rect.is_empty or rect.width < 55 or rect.height < 45:
            continue
        area_ratio = (rect.width * rect.height) / page_area
        # Scanned pages often consist of one image covering the whole page; the
        # complete page render is already stored, so skip that duplicate crop.
        if area_ratio > 0.82:
            continue
        crop_index += 1
        output = figures_dir / f"figure_page_{page_number:04d}_{crop_index:02d}.png"
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=rect, alpha=False)
            pix.save(str(output))
            saved.append(str(output))
        except Exception:
            output.unlink(missing_ok=True)
    return saved


def _mineru_cli() -> str | None:
    # MinerU 4.x uses mineru-kit for stateless parsing.
    return shutil.which("mineru-kit") or shutil.which("mineru-kit.exe")


def _parse_mineru_markdown(markdown_text: str) -> list[dict]:
    """Extract typed table/equation blocks from MinerU Markdown/LaTeX output.

    Page is left unknown when MinerU output has no explicit page marker; we do
    not invent a citation page from the block's order.
    """
    blocks: list[dict] = []
    page = None
    page_re = re.compile(r"(?:PageNumber\s*[=:]\s*\"?(\d+)\"?|page_index\s*[=:]\s*(\d+)|\[Page\s+(\d+)\])", re.I)
    lines = markdown_text.splitlines()
    current_page = None
    i = 0
    seen = set()
    while i < len(lines):
        line = lines[i]
        pm = page_re.search(line)
        if pm:
            current_page = next((int(g) for g in pm.groups() if g), None)
        # Markdown tables: heading/paragraphs may precede the header.
        if line.strip().startswith("|") and i + 1 < len(lines) and re.search(r"\|\s*:?-{2,}", lines[i + 1]):
            tbl = [line.strip(), lines[i + 1].strip()]
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                tbl.append(lines[j].strip()); j += 1
            body = "\n".join(tbl)
            key = ("table", body)
            if key not in seen:
                blocks.append({"type": "table", "page": current_page, "text": body, "origin": "mineru"})
                seen.add(key)
            i = j
            continue
        # Display math from MinerU LaTeX.
        match = re.search(r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\begin\{(?:equation|align|gather|multline)\*?\}[\s\S]+?\\end\{(?:equation|align|gather|multline)\*?\})", line)
        if match:
            formula = match.group(1).strip()
            key = ("equation", formula)
            if key not in seen:
                blocks.append({"type": "equation", "page": current_page, "text": formula, "origin": "mineru"})
                seen.add(key)
        i += 1
    # Multi-line display equations spanning lines.
    full_re = re.compile(r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\begin\{(?:equation|align|gather|multline)\*?\}[\s\S]+?\\end\{(?:equation|align|gather|multline)\*?\})")
    for match in full_re.finditer(markdown_text):
        formula = match.group(1).strip()
        key = ("equation", formula)
        if key not in seen:
            blocks.append({"type": "equation", "page": None, "text": formula, "origin": "mineru"})
            seen.add(key)
    return blocks


def _run_mineru(path: Path, temp_dir: Path) -> tuple[str, list[dict]] | None:
    mode = os.getenv("DOCUMENT_PARSER", "auto").strip().lower()
    cli = _mineru_cli()
    if mode not in {"auto", "mineru", "pymupdf"}:
        raise ValueError("DOCUMENT_PARSER must be auto, mineru, or pymupdf.")
    if mode == "pymupdf":
        return None
    if not cli:
        if mode == "mineru":
            raise RuntimeError("DOCUMENT_PARSER=mineru but 'mineru-kit' was not found. Install requirements-document-ai.txt.")
        return None
    output = temp_dir / "mineru_output.md"
    timeout = max(60, int(os.getenv("MINERU_TIMEOUT_SECONDS", "900")))
    command = [cli, "parse", str(path), "-o", str(output), "--tier", os.getenv("MINERU_TIER", "standard")]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode != 0 or not output.is_file():
            message = (result.stderr or result.stdout or "No Markdown output was produced.")[-1200:]
            if mode == "mineru":
                raise RuntimeError(f"MinerU parsing failed: {message}")
            return None
        markdown_text = output.read_text(encoding="utf-8", errors="replace").strip()
        if not markdown_text:
            if mode == "mineru":
                raise RuntimeError("MinerU produced empty Markdown output.")
            return None
        return markdown_text, _parse_mineru_markdown(markdown_text)
    except subprocess.TimeoutExpired as exc:
        if mode == "mineru":
            raise RuntimeError(f"MinerU timed out after {timeout}s.") from exc
        return None


def extract_text(path: Path, figures_dir: Path | None = None) -> dict:
    """Extract PDF/text/image content and return text, pages, figures and typed blocks.

    MinerU is used when installed and DOCUMENT_PARSER=auto/mineru. If unavailable
    in auto mode, native PyMuPDF text + EasyOCR + native table extraction are used.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    pages: list[dict] = []
    figures: list[str] = []
    structured_blocks: list[dict] = []

    if suffix == ".pdf":
        if figures_dir is not None:
            figures_dir = Path(figures_dir)
            figures_dir.mkdir(parents=True, exist_ok=True)
        mineru_result = None
        if _mineru_cli() or os.getenv("DOCUMENT_PARSER", "auto").strip().lower() == "mineru":
            with tempfile.TemporaryDirectory(prefix="research_prism_mineru_") as temp:
                mineru_result = _run_mineru(path, Path(temp))
        if mineru_result:
            _, mineru_blocks = mineru_result
            structured_blocks.extend(mineru_blocks)

        mineru_table_pages = {
            int(block["page"])
            for block in structured_blocks
            if block.get("type") == "table" and block.get("page") is not None
        }
        with fitz.open(path) as document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text", sort=True).strip()
                if not mineru_result and len(page_text) < int(os.getenv("OCR_TRIGGER_CHARS", "35")):
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                    image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
                    try:
                        ocr_text = _ocr_image(image)
                    finally:
                        image.close()
                    if len(ocr_text.strip()) > len(page_text):
                        page_text = ocr_text.strip()
                pages.append({"page": page_number, "text": page_text})

                # Native table extraction remains a fallback on pages where MinerU
                # did not emit a table. This avoids exact duplicate table records.
                if page_number not in mineru_table_pages:
                    structured_blocks.extend(_extract_pdf_tables(page, page_number))

                if figures_dir is not None:
                    page_path = figures_dir / f"page_{page_number:04d}.png"
                    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(str(page_path))
                    figures.append(str(page_path))
                    crop_paths = _extract_embedded_image_crops(page, page_number, figures_dir)
                    figures.extend(crop_paths)

                    # When MinerU is unavailable, OCR a bounded number of figure
                    # crops as well as sparse page scans so text embedded inside a
                    # chart/screenshot can participate in text retrieval.
                    ocr_crops_setting = os.getenv("OCR_FIGURE_CROPS", "auto").strip().lower()
                    if ocr_crops_setting not in {"auto", "true", "false", "1", "0", "yes", "no"}:
                        raise ValueError("OCR_FIGURE_CROPS must be auto, true, or false.")
                    do_figure_ocr = ocr_crops_setting in {"true", "1", "yes"} or (
                        ocr_crops_setting == "auto" and not mineru_result
                    )
                    if do_figure_ocr:
                        max_crops = max(0, int(os.getenv("MAX_FIGURE_CROPS_OCR_PER_PAGE", "3")))
                        for crop_path in crop_paths[:max_crops]:
                            try:
                                with Image.open(crop_path) as crop_image:
                                    crop_text = _ocr_image(crop_image).strip()
                            except Exception as exc:
                                print(f"[Research Prism] Figure crop OCR skipped ({Path(crop_path).name}): {exc}")
                                continue
                            if len(crop_text) >= 8:
                                structured_blocks.append({
                                    "type": "figure_text",
                                    "page": page_number,
                                    "text": f"OCR text from figure on page {page_number}:\n{crop_text}",
                                    "path": str(crop_path),
                                    "origin": "easyocr_figure_crop",
                                })

        combined = "\n\n".join(f"[Page {item['page']}]\n{item['text']}" for item in pages if item["text"])
        # If native/OCR extraction is very sparse but MinerU produced useful
        # Markdown, use that Markdown as the chunkable text body. Keeping it as a
        # single typed block would accidentally create one huge unsplit chunk.
        mineru_md = mineru_result[0] if mineru_result else ""
        primary_text_from_mineru = bool(mineru_md and len(combined) < 80)
        if primary_text_from_mineru:
            combined = mineru_md
        return {
            "text": combined,
            "pages": pages,
            "figures": figures,
            "structured_blocks": structured_blocks,
            "source": path.name,
            "primary_text_from_mineru": primary_text_from_mineru,
        }

    if suffix in IMAGE_EXTENSIONS:
        with Image.open(path) as image:
            text = _ocr_image(image)
        pages.append({"page": 1, "text": text})
        return {"text": f"[Page 1]\n{text}" if text else "", "pages": pages, "figures": [str(path.resolve())], "structured_blocks": [], "source": path.name}

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        pages.append({"page": 1, "text": text})
        return {"text": f"[Page 1]\n{text}" if text else "", "pages": pages, "figures": [], "structured_blocks": [], "source": path.name}

    raise ValueError(f"Unsupported file type: {suffix}")
