
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"


@lru_cache(maxsize=1)
def get_minilm():
    """Load the local text embedding model once."""
    return SentenceTransformer(MINILM_MODEL)


@lru_cache(maxsize=1)
def get_clip():
    """Load CLIP for compatible text and image embeddings."""
    return SentenceTransformer(CLIP_MODEL)


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Encode text into normalized float32 vectors."""
    if model_name not in {"minilm", "clip"}:
        raise ValueError(
            "Unsupported embedding model. Use 'minilm' or 'clip'."
        )

    if not texts:
        dimension = 384 if model_name == "minilm" else 512
        return np.empty((0, dimension), dtype=np.float32)

    cleaned = [str(text).strip() for text in texts]

    if any(not text for text in cleaned):
        raise ValueError("Embedding input contains empty text.")

    model = get_minilm() if model_name == "minilm" else get_clip()

    vectors = model.encode(
        cleaned,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(cleaned):
        raise RuntimeError("The embedding model returned an invalid shape.")

    if not np.isfinite(vectors).all():
        raise RuntimeError("The embedding model returned non-finite values.")

    return np.ascontiguousarray(vectors, dtype=np.float32)


def embed_images(image_paths: list[str]) -> np.ndarray:
    """Encode images with CLIP for visual retrieval."""
    if not image_paths:
        return np.empty((0, 512), dtype=np.float32)

    from PIL import Image

    images = []

    try:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())

        vectors = get_clip().encode(
            images,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] != len(images):
            raise RuntimeError(
                "The image embedding model returned an invalid shape."
            )

        if not np.isfinite(vectors).all():
            raise RuntimeError(
                "The image embedding model returned non-finite values."
            )

        return np.ascontiguousarray(vectors, dtype=np.float32)

    finally:
        for image in images:
            image.close()
