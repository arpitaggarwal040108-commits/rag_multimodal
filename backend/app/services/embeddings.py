
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"

CLIP_MAX_TOKENS = 77


@lru_cache(maxsize=1)
def get_minilm() -> SentenceTransformer:
    """Load the MiniLM text embedding model once."""
    return SentenceTransformer(MINILM_MODEL)


@lru_cache(maxsize=1)
def get_clip() -> SentenceTransformer:
    """Load the CLIP embedding model once."""
    return SentenceTransformer(CLIP_MODEL)





def _get_clip_tokenizer(model):
    """Load the CLIP text tokenizer from its actual cached location."""
    from pathlib import Path
    from transformers import CLIPTokenizer
    from huggingface_hub import snapshot_download

    tokenizer = getattr(model, "tokenizer", None)

    if (
        tokenizer is not None
        and hasattr(tokenizer, "decode")
        and hasattr(tokenizer, "get_vocab")
    ):
        return tokenizer

    snapshot = Path(
        snapshot_download(
            repo_id=CLIP_MODEL,
            local_files_only=True,
        )
    )

    vocab_files = list(snapshot.rglob("vocab.json"))
    merges_files = list(snapshot.rglob("merges.txt"))

    if not vocab_files or not merges_files:
        raise FileNotFoundError(
            f"CLIP tokenizer files not found under {snapshot}"
        )

    # Match the vocab and merges files from the same directory.
    for vocab_file in vocab_files:
        merges_file = vocab_file.parent / "merges.txt"
        if merges_file.is_file():
            return CLIPTokenizer(
                vocab_file=str(vocab_file),
                merges_file=str(merges_file),
            )

    raise FileNotFoundError(
        f"Could not find matching vocab.json and merges.txt under {snapshot}"
    )


def _truncate_clip_text(text: str, model) -> str:
    """Truncate text to CLIP's 77-token limit before embedding."""
    tokenizer = _get_clip_tokenizer(model)

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=CLIP_MAX_TOKENS,
        padding=False,
    )

    token_ids = encoded["input_ids"]

    # Handle tokenizers that return a batch dimension.
    if len(token_ids) > 0 and isinstance(token_ids[0], (list, tuple)):
        token_ids = token_ids[0]

    shortened = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    if not shortened:
        raise ValueError("CLIP text is empty after tokenization.")

    # Verify that decoding and re-tokenization did not exceed the limit.
    verification = tokenizer(
        shortened,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    if len(verification["input_ids"]) > CLIP_MAX_TOKENS:
        token_ids = verification["input_ids"][:CLIP_MAX_TOKENS]
        shortened = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    if not shortened:
        raise ValueError("Could not create a valid CLIP text representation.")

    return shortened


def embed_texts(
    texts: list[str],
    model_name: str,
) -> np.ndarray:
    """Encode text into normalized float32 embedding vectors."""
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

    if model_name == "minilm":
        model = get_minilm()
        prepared_texts = cleaned
        expected_dimension = 384
    else:
        model = get_clip()
        prepared_texts = [
            _truncate_clip_text(text, model) for text in cleaned
        ]
        expected_dimension = 512

    vectors = model.encode(
        prepared_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(cleaned)
        or vectors.shape[1] != expected_dimension
    ):
        raise RuntimeError(
            f"Invalid {model_name} embedding shape: {vectors.shape}. "
            f"Expected ({len(cleaned)}, {expected_dimension})."
        )

    if not np.isfinite(vectors).all():
        raise RuntimeError(
            f"The {model_name} model returned non-finite values."
        )

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

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(images)
            or vectors.shape[1] != 512
        ):
            raise RuntimeError(
                f"Invalid CLIP image embedding shape: {vectors.shape}."
            )

        if not np.isfinite(vectors).all():
            raise RuntimeError(
                "The CLIP image model returned non-finite values."
            )

        return np.ascontiguousarray(vectors, dtype=np.float32)

    finally:
        for image in images:
            image.close()
