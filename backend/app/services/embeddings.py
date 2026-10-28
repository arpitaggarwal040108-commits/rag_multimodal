import os
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

# New installs default to BGE-M3 for dense research retrieval. Existing FAISS
# indexes created with MiniLM are NOT compatible and must be re-indexed.
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "BAAI/bge-m3")
TEXT_EMBEDDING_DIM = int(os.getenv("TEXT_EMBEDDING_DIM", "1024"))
CLIP_MODEL = os.getenv("CLIP_MODEL", "sentence-transformers/clip-ViT-B-32")
CLIP_DIM = 512
CLIP_MAX_TOKENS = 77


@lru_cache(maxsize=1)
def get_minilm() -> SentenceTransformer:
    """Compatibility-named loader for the configured dense text model."""
    return SentenceTransformer(TEXT_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_clip() -> SentenceTransformer:
    """Load the CLIP text/image model once."""
    return SentenceTransformer(CLIP_MODEL)


def _get_clip_tokenizer(model):
    """Load the CLIP tokenizer, falling back to its cached HF snapshot."""
    from pathlib import Path
    from transformers import CLIPTokenizer
    from huggingface_hub import snapshot_download

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode") and hasattr(tokenizer, "get_vocab"):
        return tokenizer

    snapshot = Path(snapshot_download(repo_id=CLIP_MODEL, local_files_only=True))
    vocab_files = list(snapshot.rglob("vocab.json"))
    for vocab_file in vocab_files:
        merges_file = vocab_file.parent / "merges.txt"
        if merges_file.is_file():
            return CLIPTokenizer(vocab_file=str(vocab_file), merges_file=str(merges_file))
    raise FileNotFoundError(f"Could not find CLIP vocab.json and merges.txt under {snapshot}")


def _truncate_clip_text(text: str, model) -> str:
    tokenizer = _get_clip_tokenizer(model)
    token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    if len(token_ids) > CLIP_MAX_TOKENS - 2:
        token_ids = token_ids[: CLIP_MAX_TOKENS - 2]
    shortened = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if not shortened:
        shortened = str(text).strip()[:200]
    if not shortened:
        raise ValueError("Could not create a valid CLIP text representation.")
    return shortened


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Encode text into normalized float32 vectors using the configured model."""
    if model_name not in {"minilm", "text", "clip"}:
        raise ValueError("Unsupported embedding model. Use 'text'/'minilm' or 'clip'.")
    dimension = CLIP_DIM if model_name == "clip" else TEXT_EMBEDDING_DIM
    if not texts:
        return np.empty((0, dimension), dtype=np.float32)
    cleaned = [str(text).strip() for text in texts]
    if any(not text for text in cleaned):
        raise ValueError("Embedding input contains empty text.")

    if model_name == "clip":
        model = get_clip()
        prepared_texts = [_truncate_clip_text(text, model) for text in cleaned]
        expected_dimension = CLIP_DIM
    else:
        model = get_minilm()
        prepared_texts = cleaned
        expected_dimension = model.get_sentence_embedding_dimension()

    vectors = model.encode(
        prepared_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(cleaned) or vectors.shape[1] != expected_dimension:
        raise RuntimeError(
            f"Invalid {model_name} embedding shape: {vectors.shape}; expected "
            f"({len(cleaned)}, {expected_dimension})."
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError(f"The {model_name} model returned non-finite values.")
    return np.ascontiguousarray(vectors, dtype=np.float32)


def embed_images(image_paths: list[str]) -> np.ndarray:
    """Encode extracted page/figure crops with CLIP for visual retrieval."""
    if not image_paths:
        return np.empty((0, CLIP_DIM), dtype=np.float32)
    from PIL import Image
    images = []
    try:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())
        vectors = get_clip().encode(
            images, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(images), CLIP_DIM):
            raise RuntimeError(f"Invalid CLIP image embedding shape: {vectors.shape}.")
        if not np.isfinite(vectors).all():
            raise RuntimeError("The CLIP image model returned non-finite values.")
        return np.ascontiguousarray(vectors, dtype=np.float32)
    finally:
        for image in images:
            image.close()
