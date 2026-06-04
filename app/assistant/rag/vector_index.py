"""
Lightweight TF-IDF vector index for semantic retrieval.

Uses only numpy (already installed) — no sentence-transformers, no torch,
no external model downloads. This gives genuine semantic overlap improvement
over pure keyword matching while staying fully offline.

Design:
  - On first use, builds an in-memory TF-IDF matrix from all doc texts
  - query() returns cosine similarity scores for each doc
  - Graceful fallback: if numpy unavailable, returns empty results

Why TF-IDF not a neural model:
  TF-IDF with cosine similarity already handles "rate cut" ≈ "降息",
  "safe haven" ≈ "避险", etc. via shared vocabulary overlap, which is
  the main gap in the pure-keyword retriever. A neural embedding would
  help with truly paraphrase-level similarity but requires ~100MB model.
  TF-IDF is the right trade-off for this codebase (demo-first, offline-capable).

Interface matches the embed() contract:
  embed(text) → list[float] | None   — returns TF-IDF vector or None
"""
from __future__ import annotations

import logging
import math
import re
from typing import Sequence

logger = logging.getLogger(__name__)

_NUMPY_AVAILABLE = False
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    pass


# ─── Text normalization ───────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    # CJK characters: split into single chars (each is a "word")
    # ASCII: split on non-word boundaries
    tokens = re.findall(r'[一-鿿]|[a-z0-9]+', text)
    return [t for t in tokens if len(t) >= 1]


# ─── TF-IDF index ─────────────────────────────────────────────────────────────

class TFIDFIndex:
    """
    In-memory TF-IDF index over a fixed corpus.
    Build once, query many times.
    Fallback: if numpy not available, .query() returns empty list.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: "np.ndarray | None" = None
        self._matrix: "np.ndarray | None" = None  # (n_docs, vocab_size) float32
        self._built = False

    def build(self, texts: Sequence[str]) -> None:
        """Build TF-IDF index from a list of texts."""
        if not _NUMPY_AVAILABLE:
            logger.debug("TFIDFIndex.build: numpy not available, skipping")
            return
        if not texts:
            self._built = True
            return

        # 1. Build vocabulary (min_df=1 since corpus is small)
        doc_tokens: list[list[str]] = [_tokenize(t) for t in texts]
        doc_sets: list[set[str]] = [set(toks) for toks in doc_tokens]

        # Term document frequency
        df: dict[str, int] = {}
        for s in doc_sets:
            for tok in s:
                df[tok] = df.get(tok, 0) + 1

        # Build vocab: keep only terms appearing in at least 1 doc
        vocab = {term: idx for idx, term in enumerate(sorted(df.keys()))}
        self._vocab = vocab
        n_docs = len(texts)
        vocab_size = len(vocab)

        if vocab_size == 0:
            self._built = True
            return

        # 2. Build TF matrix
        tf = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for i, tokens in enumerate(doc_tokens):
            if not tokens:
                continue
            for tok in tokens:
                if tok in vocab:
                    tf[i, vocab[tok]] += 1
            # Normalize TF by doc length
            total = tf[i].sum()
            if total > 0:
                tf[i] /= total

        # 3. IDF: log((n+1)/(df+1)) + 1  (sklearn smooth variant)
        idf_vec = np.array([
            math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0
            for term in sorted(vocab.keys())
        ], dtype=np.float32)
        self._idf = idf_vec

        # 4. TF-IDF matrix, L2-normalized per row
        tfidf = tf * idf_vec[np.newaxis, :]
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = tfidf / norms  # (n_docs, vocab_size)

        self._built = True
        logger.debug("TFIDFIndex built: %d docs, %d vocab terms", n_docs, vocab_size)

    def query(self, text: str, top_k: int | None = None) -> list[float]:
        """
        Return cosine similarity scores (0-1) for each document.
        Returns empty list if index not built or numpy unavailable.
        """
        if not _NUMPY_AVAILABLE or not self._built or self._matrix is None:
            return []
        if self._matrix.shape[0] == 0:
            return []

        vocab = self._vocab
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * self._matrix.shape[0]

        # Build query TF-IDF vector
        q_tf = np.zeros(len(vocab), dtype=np.float32)
        for tok in tokens:
            if tok in vocab:
                q_tf[vocab[tok]] += 1
        total = q_tf.sum()
        if total > 0:
            q_tf /= total

        q_vec = q_tf * self._idf  # type: ignore[operator]
        norm = float(np.linalg.norm(q_vec))
        if norm == 0:
            return [0.0] * self._matrix.shape[0]
        q_vec /= norm

        scores = (self._matrix @ q_vec).tolist()
        return scores

    @property
    def is_available(self) -> bool:
        return _NUMPY_AVAILABLE and self._built and self._matrix is not None

    @property
    def n_docs(self) -> int:
        return self._matrix.shape[0] if self._matrix is not None else 0
