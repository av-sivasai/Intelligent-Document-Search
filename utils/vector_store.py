"""
vector_store.py
---------------
FAISS-based vector index: build, persist, load, and search.

The index stores chunk-level embeddings.  Each position i in the index
maps directly to self.metadata[i], which holds the document name, chunk
index, and raw chunk text.
"""

import logging
import os
import pickle
from typing import List, Tuple, Dict, Any

import faiss
import numpy as np

from utils.embedder import get_embedding_dim

logger = logging.getLogger(__name__)

# ── Persistence paths ──────────────────────────────────────────────
DATA_DIR        = "data"
INDEX_FILE      = os.path.join(DATA_DIR, "faiss.index")
METADATA_FILE   = os.path.join(DATA_DIR, "metadata.pkl")


class VectorStore:
    """
    Wraps a FAISS IndexFlatIP (inner-product / cosine similarity) index.

    Because embeddings are L2-normalised by the embedder, inner-product
    is equivalent to cosine similarity, giving values in [0, 1].
    """

    def __init__(self):
        self.dim:       int                   = get_embedding_dim()
        self.index:     faiss.IndexFlatIP     = faiss.IndexFlatIP(self.dim)
        # Each entry: {"doc_name": str, "chunk_idx": int, "text": str,
        #              "file_type": str}
        self.metadata:  List[Dict[str, Any]]  = []

    # ── Build / Add ────────────────────────────────────────────────

    def add_document(self,
                     doc_name:  str,
                     file_type: str,
                     chunks:    List[str],
                     embeddings: np.ndarray) -> None:
        """
        Add all chunks of one document to the index.

        Args:
            doc_name:   Display name of the document.
            file_type:  Extension string, e.g. ".pdf".
            chunks:     List of text chunk strings.
            embeddings: Corresponding embedding array (n_chunks × dim).
        """
        if embeddings.shape[0] == 0:
            logger.warning("No embeddings to add for '%s'.", doc_name)
            return

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "doc_name":  doc_name,
                "file_type": file_type,
                "chunk_idx": i,
                "text":      chunk,
            })

        logger.info("Added %d chunk(s) from '%s'.", len(chunks), doc_name)

    def remove_document(self, doc_name: str) -> None:
        """
        Remove all chunks belonging to doc_name and rebuild the index.
        FAISS IndexFlatIP does not support partial deletion, so we rebuild.
        """
        keep_indices = [
            i for i, m in enumerate(self.metadata)
            if m["doc_name"] != doc_name
        ]

        if len(keep_indices) == len(self.metadata):
            logger.warning("'%s' not found in index.", doc_name)
            return

        # Reconstruct embeddings for kept chunks
        if keep_indices:
            kept_vectors = np.vstack([
                self.index.reconstruct(i) for i in keep_indices
            ]).astype(np.float32)
        else:
            kept_vectors = np.empty((0, self.dim), dtype=np.float32)

        kept_metadata = [self.metadata[i] for i in keep_indices]

        # Rebuild index
        self.index = faiss.IndexFlatIP(self.dim)
        if kept_vectors.shape[0] > 0:
            kept_vectors = np.ascontiguousarray(kept_vectors, dtype=np.float32)
            self.index.add(kept_vectors)
        self.metadata = kept_metadata
        logger.info("Removed '%s' from index. Remaining chunks: %d",
                    doc_name, len(self.metadata))

    # ── Search ─────────────────────────────────────────────────────

    def search(self,
               query_embedding: np.ndarray,
               top_k: int = 10,
               filter_doc: str | None = None,
               filter_type: str | None = None
               ) -> List[Dict[str, Any]]:
        """
        Return the top-k chunks most similar to query_embedding.

        Args:
            query_embedding: Shape (1, dim) float32 array.
            top_k:           Maximum number of results to return.
            filter_doc:      If set, restrict results to this document name.
            filter_type:     If set, restrict results to this file extension.

        Returns:
            List of result dicts sorted by score descending.
        """
        if self.index.ntotal == 0:
            return []

        # Retrieve more candidates to account for filtering
        fetch_k = min(self.index.ntotal, top_k * 10)
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        scores, indices = self.index.search(query_embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:          # FAISS sentinel for "no result"
                continue
            meta = self.metadata[idx]

            # Apply optional filters
            if filter_doc  and meta["doc_name"]  != filter_doc:
                continue
            if filter_type and meta["file_type"] != filter_type:
                continue

            results.append({
                "doc_name":  meta["doc_name"],
                "file_type": meta["file_type"],
                "chunk_idx": meta["chunk_idx"],
                "text":      meta["text"],
                "score":     float(score),
            })

            if len(results) >= top_k:
                break

        return results

    # ── Persistence ────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        os.makedirs(DATA_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Index saved  (%d vectors).", self.index.ntotal)

    def load(self) -> bool:
        """
        Load the FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False if no saved index exists.
        """
        if not (os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE)):
            return False
        try:
            self.index    = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info("Index loaded (%d vectors).", self.index.ntotal)
            return True
        except Exception as exc:
            logger.error("Failed to load index: %s", exc)
            return False

    def clear(self) -> None:
        """Wipe the in-memory index and delete persisted files."""
        self.index    = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        for path in (INDEX_FILE, METADATA_FILE):
            if os.path.exists(path):
                os.remove(path)
        logger.info("Index cleared.")

    # ── Accessors ──────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal

    @property
    def unique_documents(self) -> List[str]:
        return list({m["doc_name"] for m in self.metadata})

    @property
    def file_types(self) -> List[str]:
        return list({m["file_type"] for m in self.metadata})
