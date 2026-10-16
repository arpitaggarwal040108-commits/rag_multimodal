from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_minilm():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_clip():
    # Text encoder for the CLIP model; uploaded image content currently enters via OCR text.
    return SentenceTransformer("sentence-transformers/clip-ViT-B-32")

def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = get_minilm() if model_name == "minilm" else get_clip()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.ascontiguousarray(vectors, dtype=np.float32)
