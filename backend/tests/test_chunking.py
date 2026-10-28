import unittest

from app.services.chunking import semantic_chunk, chunk_structured_blocks


class ChunkingTests(unittest.TestCase):
    def test_overlap_is_present_and_chunks_respect_target(self):
        sentences = [f"Sentence{i} has enough distinct token content for this chunking test." for i in range(75)]
        text = "[Page 1]\n" + " ".join(sentences)
        chunks = semantic_chunk(text, "paper.pdf", target_words=90, overlap_words=15)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["word_count"] <= 90 for chunk in chunks))
        self.assertTrue(all(chunk["page"] == 1 for chunk in chunks))
        self.assertTrue(all(chunk["chunk_type"] == "text" for chunk in chunks))
        # At least one neighboring chunk should share a full sentence due to overlap.
        self.assertTrue(any(set(a["text"].split()) & set(b["text"].split()) for a, b in zip(chunks, chunks[1:])))

    def test_table_chunks_repeat_headers(self):
        rows = [f"| {i} | value_{i} |" for i in range(35)]
        body = "Table 1\n| ID | Value |\n|---|---|\n" + "\n".join(rows)
        chunks = chunk_structured_blocks([{"type": "table", "page": 3, "text": body}], "paper.pdf", target_words=55, overlap_words=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("| ID | Value |", chunk["text"])
            self.assertEqual(chunk["chunk_type"], "table")
            self.assertEqual(chunk["page"], 3)


if __name__ == "__main__":
    unittest.main()
