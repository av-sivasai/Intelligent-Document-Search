"""
embedder.py
-----------
Generates dense vector embeddings using Sentence Transformers.
Loads the model once and caches it for the session lifetime.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Model configuration ────────────────────────────────────────────
# "all-MiniLM-L6-v2" is fast (~80 MB) and very accurate for semantic search.
# Swap to "all-mpnet-base-v2" for higher quality (but slower) if needed.
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None   # module-level singleton


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def load_model() -> SentenceTransformer:
    """
    Return the cached SentenceTransformer model, loading it on first call.
    The model is stored as a module-level singleton to avoid reloading on
    every Streamlit rerun.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded successfully.")
    return _model


def embed_chunks(chunks: List[str],
                 batch_size: int = 64,
                 show_progress: bool = False) -> np.ndarray:
    """
    Encode a list of text chunks into L2-normalised embedding vectors.

    Args:
        chunks:        List of text strings to encode.
        batch_size:    Number of chunks processed per GPU/CPU batch.
        show_progress: Show tqdm progress bar (useful for large corpora).

    Returns:
        NumPy array of shape (len(chunks), embedding_dim), dtype float32.
    """
    if not chunks:
        return np.empty((0, get_embedding_dim()), dtype=np.float32)

    model = load_model()

    logger.debug("Encoding %d chunk(s) …", len(chunks))
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2-normalise for cosine similarity
        convert_to_numpy=True,
    )
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string into a normalised embedding vector.

    Returns:
        NumPy array of shape (1, embedding_dim), dtype float32.
    """
    return embed_chunks([query])


def get_embedding_dim() -> int:
    """Return the dimensionality of the embedding model's output vectors."""
    model = load_model()
    return model.get_sentence_embedding_dimension()
