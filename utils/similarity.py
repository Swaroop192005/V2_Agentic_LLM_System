"""
utils/similarity.py
-------------------
Utility module for computing semantic similarity between text strings
using sentence-transformers and cosine similarity.

This is used by the pipeline to objectively score LLM responses
against the original query (relevance scoring).
"""

from __future__ import annotations
import os

# ── Prevent joblib/tokenizers from spawning subprocesses ─────────────────────
# Python 3.14 + macOS: forking inside an asyncio server causes SIGABRT crashes.
# Force everything to run in single-threaded mode.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
# ─────────────────────────────────────────────────────────────────────────────

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model once (cached globally to avoid reloading on every call)
# 'all-MiniLM-L6-v2' is lightweight and fast — ideal for local use
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily load the sentence transformer model (singleton pattern), on GPU when available."""
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SimilarityUtil] Loading sentence-transformer model on {device}...")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        print("[SimilarityUtil] Model loaded.")
    return _model


def compute_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two text strings.

    Args:
        text_a: First text (e.g., the user query)
        text_b: Second text (e.g., an LLM response)

    Returns:
        A float in range [0.0, 1.0] representing semantic similarity.
        1.0 = identical meaning, 0.0 = completely unrelated.
    """
    model = _get_model()
    embeddings = model.encode([text_a, text_b])
    sim_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    similarity: float = float(sim_matrix[0][0])
    # Clamp to [0, 1] (cosine can technically return tiny negatives)
    return float(np.clip(similarity, 0.0, 1.0))


def score_response(query: str, response: str) -> dict[str, float | int]:
    """
    Score an LLM response against the original query using multiple signals:
      - semantic_similarity  : cosine similarity of embeddings (0–1)
      - length_score         : normalised length score (0–1), penalises very short answers
      - composite_score      : weighted blend expressed out of 10

    Args:
        query:    The original user question.
        response: The LLM's generated answer.

    Returns:
        A dict with individual metrics and a composite_score (0–10).
    """
    # --- 1. Semantic similarity (query ↔ response) ---
    semantic_sim: float = compute_cosine_similarity(query, response)

    # --- 2. Length / completeness score ---
    # Assume a "good" answer is at least 150 words; cap benefit at 400 words
    word_count: int = len(response.split())
    length_score: float = float(np.clip(word_count / 150.0, 0.0, 1.0))

    # --- 3. Composite weighted score (out of 10) ---
    # Weights: semantic similarity carries more importance than raw length
    composite: float = round((0.70 * semantic_sim + 0.30 * length_score) * 10, 2)

    return {
        "semantic_similarity": round(semantic_sim, 4),
        "length_score": round(length_score, 4),
        "word_count": word_count,
        "composite_score": composite,   # 0–10
    }
