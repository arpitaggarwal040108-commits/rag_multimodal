import os
import unittest
from unittest.mock import patch

from app.services.generation import generate_answer


class GenerationTests(unittest.TestCase):
    @patch.dict(os.environ, {"ENABLE_TEXT_GENERATION": "true"})
    @patch("app.services.generation.answer_research_text")
    def test_text_only_query_uses_local_model_when_available(self, mocked_generate):
        answer = (
            "## Direct answer\nThe retrieved passage describes the method.\n\n"
            "## Key findings\n- The method uses a retrieved passage.\n\n"
            "## Visual evidence\nNo visual asset was needed for this answer.\n\n"
            "## Limitations\nOnly the supplied passage was considered."
        )
        mocked_generate.return_value = {"available": True, "answer": answer, "model": "test-model"}
        result = generate_answer(
            question="What method does this passage describe?",
            chunks=[{
                "source": "paper.pdf", "original_filename": "paper.pdf",
                "chunk_id": 1, "page": 2, "text": "The method uses a retrieved passage to answer the question.",
                "score": 0.8,
            }],
            visual_results=[],
        )
        self.assertEqual(result["answer_mode"], "local_llm_text")
        self.assertEqual(result["vision_model"], "test-model")
        self.assertEqual(result["sources"][0]["page"], 2)


if __name__ == "__main__":
    unittest.main()
