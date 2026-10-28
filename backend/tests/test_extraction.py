import tempfile
import unittest
from pathlib import Path

import fitz

from app.services.extraction import _parse_mineru_markdown, extract_text


class ExtractionTests(unittest.TestCase):
    def test_mineru_markdown_detects_table_and_display_equation(self):
        markdown = """# Results

| Method | Accuracy |
| --- | --- |
| A | 0.91 |

$$ E = mc^2 $$
"""
        blocks = _parse_mineru_markdown(markdown)
        types = [block["type"] for block in blocks]
        self.assertIn("table", types)
        self.assertIn("equation", types)

    def test_native_pdf_text_and_page_render_are_extracted(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            pdf_path = root / "sample.pdf"
            visual_dir = root / "visuals"

            document = fitz.open()
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "This sample paper discusses multimodal research retrieval methods.",
            )
            document.save(pdf_path)
            document.close()

            result = extract_text(pdf_path, figures_dir=visual_dir)
            self.assertIn("multimodal research retrieval", result["text"])
            self.assertEqual(len(result["pages"]), 1)
            self.assertEqual(len(result["figures"]), 1)
            self.assertTrue(Path(result["figures"][0]).is_file())


if __name__ == "__main__":
    unittest.main()
