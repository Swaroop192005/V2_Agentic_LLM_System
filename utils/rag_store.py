"""
utils/rag_store.py
-------------------
Lightweight, zero-dependency local Vector Store for RAG (Retrieval-Augmented Generation).
Uses sentence-transformers ('all-MiniLM-L6-v2') and sqlite3 for vector storage and cosine search.

- Optional user document upload (PDF, TXT, MD)
- Pre-loaded domain knowledge base in data/knowledge/
- Zero external vector DB dependencies
"""

from __future__ import annotations
import os
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from utils.similarity import _get_model, compute_cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
RAG_DB_PATH = DATA_DIR / "rag_knowledge.db"

KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


def init_rag_db() -> None:
    """Initialize the SQLite vector storage table."""
    conn = sqlite3.connect(str(RAG_DB_PATH))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_name TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # RAG retrieval logging (RESEARCH_LOG.md Section 25): MIN_SIMILARITY
        # was never calibrated because nothing recorded retrieval decisions
        # anywhere persistent - the batch generator never even calls
        # retrieve_context(). This table logs EVERY candidate chunk's raw
        # similarity for every live-demo query, gated or not, so a future
        # threshold sweep (mirroring training/threshold_weight_sweep.py) can
        # replay "what would have passed at cutoff X" without needing to
        # regenerate anything - same raw-signals-first philosophy as
        # generate_training_data_v2.py (Section 5).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_retrieval_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                query TEXT,
                doc_name TEXT,
                chunk_index INTEGER,
                similarity REAL,
                min_similarity_at_time REAL,
                passed_gate INTEGER,
                final_rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # One row per full pipeline run that reaches a confidence score,
        # so retrieval quality (above) can eventually be correlated against
        # actual downstream answer quality, not just "did it pass the gate".
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_pipeline_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                query TEXT,
                use_rag INTEGER,
                rag_chunks_used INTEGER,
                weighted_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()

    # Check if default knowledge exists; if empty, populate initial domain knowledge
    _populate_default_knowledge_if_empty()


def log_pipeline_outcome(request_id: str, query: str, use_rag: bool, rag_chunks_used: int, weighted_score: float | None) -> None:
    """Best-effort: record the final confidence score for a pipeline run
    alongside whether/how much RAG context it used, for the future
    retrieval-quality-vs-outcome analysis described in RESEARCH_LOG Section 25."""
    try:
        conn = sqlite3.connect(str(RAG_DB_PATH))
        conn.execute(
            "INSERT INTO rag_pipeline_outcomes (request_id, query, use_rag, rag_chunks_used, weighted_score) VALUES (?, ?, ?, ?, ?)",
            (request_id, query, int(use_rag), rag_chunks_used, weighted_score),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[RAG Store Error] failed to log pipeline outcome: {exc}")


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 40) -> List[str]:
    """Split text into overlapping chunks of approx N words."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract plain text from binary PDF bytes using multi-pass pypdf & stream fallback."""
    import io
    import re
    pages_text = []

    # Pass 1 & 2: pypdf standard + layout mode
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            t = page.extract_text() or ""
            if not t.strip():
                try:
                    t = page.extract_text(extraction_mode="layout") or ""
                except Exception:
                    t = ""
            if t.strip():
                pages_text.append(t.strip())
    except Exception as exc:
        print(f"[RAG Store PDF pypdf Error] {exc}")

    # Pass 3: If pypdf extracted nothing (scanned or custom CID fonts), run raw stream extraction fallback
    if not pages_text:
        try:
            raw_data = pdf_bytes.decode("latin1", errors="ignore")
            # Extract text blocks inside Tj and TJ operators
            tj_matches = re.findall(r'\(([^()]{2,})\)\s*Tj', raw_data)
            if not tj_matches:
                # Fallback: extract text inside brackets [(text)] TJ
                tj_matches = re.findall(r'\[\s*\(([^()]{2,})\)\s*\]\s*TJ', raw_data)
            if not tj_matches:
                # Fallback: find any readable alphanumeric sequences of length >= 4
                tj_matches = re.findall(r'[A-Za-z0-9\s.,;:?!\'"-]{15,}', raw_data)
            
            clean_chunks = [m.strip() for m in tj_matches if len(m.strip()) > 3 and not m.startswith('/')]
            if clean_chunks:
                pages_text.append(" ".join(clean_chunks))
        except Exception as exc:
            print(f"[RAG Store PDF Fallback Error] {exc}")

    return "\n\n".join(pages_text)


def index_document(doc_name: str, content: str | bytes) -> int:
    """Chunk, embed, and store a document (PDF or Text) in the local RAG DB."""
    if isinstance(content, bytes) or doc_name.lower().endswith(".pdf"):
        if isinstance(content, str):
            content_bytes = content.encode("utf-8", errors="ignore")
        else:
            content_bytes = content
        text_content = extract_text_from_pdf_bytes(content_bytes)
    else:
        text_content = str(content)

    chunks = chunk_text(text_content)
    if not chunks:
        return 0

    model = _get_model()
    embeddings = model.encode(chunks)

    conn = sqlite3.connect(str(RAG_DB_PATH))
    try:
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            emb_json = str(emb.tolist())
            conn.execute(
                "INSERT INTO rag_documents (doc_name, chunk_index, content, embedding_json) VALUES (?, ?, ?, ?)",
                (doc_name, idx, chunk, emb_json)
            )
        conn.commit()
    finally:
        conn.close()

    return len(chunks)


# Below this cosine similarity, a chunk is treated as unrelated to the query
# and dropped rather than injected as "ground-truth context". Previously
# retrieve_context() always returned the top-K chunks regardless of how weak
# the match was, so unrelated queries (e.g. "what is ice cream?") would still
# pull in the preloaded RAG_and_Transformers_Overview.md / biomedical PDF
# chunks, which the Verifier then treated as required ground truth - actively
# penalizing correct answers for not discussing an unrelated document.
MIN_SIMILARITY = 0.35


def _log_retrieval(request_id: str | None, query: str, all_scored: List[Dict[str, Any]], min_similarity: float, kept: List[Dict[str, Any]]) -> None:
    """Best-effort: record every candidate chunk's raw similarity (not just
    the ones that passed the gate) so a future threshold sweep can replay
    'what would have passed at cutoff X' - see RESEARCH_LOG Section 25."""
    kept_keys = {(c["doc_name"], c["chunk_index"]): i + 1 for i, c in enumerate(kept)}
    try:
        conn = sqlite3.connect(str(RAG_DB_PATH))
        conn.executemany(
            """INSERT INTO rag_retrieval_log
               (request_id, query, doc_name, chunk_index, similarity, min_similarity_at_time, passed_gate, final_rank)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    request_id, query, c["doc_name"], c["chunk_index"], c["similarity"], min_similarity,
                    int(c["similarity"] >= min_similarity),
                    kept_keys.get((c["doc_name"], c["chunk_index"])),
                )
                for c in all_scored
            ],
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[RAG Store Error] failed to log retrieval: {exc}")


def retrieve_context(query: str, top_k: int = 3, min_similarity: float = MIN_SIMILARITY, request_id: str | None = None, log: bool = True) -> List[Dict[str, Any]]:
    """Retrieve up to top-K relevant chunks for a given query, dropping any
    below min_similarity so irrelevant context is never injected. Every
    candidate's raw similarity is logged (pass `log=False` to skip, e.g. for
    offline analysis scripts that shouldn't pollute the log) regardless of
    whether it clears the gate, so MIN_SIMILARITY can eventually be
    calibrated the way DEFAULT_THRESHOLD was (RESEARCH_LOG Section 25)."""
    if not RAG_DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(RAG_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT doc_name, chunk_index, content, embedding_json FROM rag_documents").fetchall()
        if not rows:
            return []

        model = _get_model()
        query_emb = model.encode([query])[0]

        scored_chunks = []
        for r in rows:
            doc_emb = np.array(eval(r["embedding_json"]))
            # Cosine similarity
            norm_q = np.linalg.norm(query_emb)
            norm_d = np.linalg.norm(doc_emb)
            sim = float(np.dot(query_emb, doc_emb) / (norm_q * norm_d + 1e-9))
            scored_chunks.append({
                "doc_name": r["doc_name"],
                "chunk_index": r["chunk_index"],
                "content": r["content"],
                "similarity": round(sim, 4)
            })

        passed = [c for c in scored_chunks if c["similarity"] >= min_similarity]
        passed.sort(key=lambda x: x["similarity"], reverse=True)
        kept = passed[:top_k]

        if log:
            _log_retrieval(request_id, query, scored_chunks, min_similarity, kept)

        return kept
    except Exception as exc:
        print(f"[RAG Store Error] {exc}")
        return []
    finally:
        conn.close()


def get_rag_stats() -> Dict[str, Any]:
    """Return total documents and chunk count."""
    if not RAG_DB_PATH.exists():
        return {"doc_count": 0, "chunk_count": 0, "documents": []}
    
    conn = sqlite3.connect(str(RAG_DB_PATH))
    try:
        chunk_count = conn.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]
        docs = [r[0] for r in conn.execute("SELECT DISTINCT doc_name FROM rag_documents").fetchall()]
        return {"doc_count": len(docs), "chunk_count": chunk_count, "documents": docs}
    finally:
        conn.close()


def _populate_default_knowledge_if_empty() -> None:
    """Pre-load default domain knowledge docs if database is brand new."""
    stats = get_rag_stats()
    if stats["chunk_count"] > 0:
        return

    # Add Default Project Knowledge
    default_doc_1 = (
        "Agentic Multi-LLM Validation System Architecture:\n"
        "The system uses a 6-stage pipeline to answer user questions with high factual accuracy. "
        "Stage 1 (LLaMA 3) and Stage 2 (Mistral) generate answers in parallel. "
        "Stage 3 (Verifier) fact-checks both answers for hallucinations. "
        "Stage 4 (Similarity Scorer) computes embedding similarity using Sentence-Transformers (all-MiniLM-L6-v2). "
        "Stage 5 (Judge) scores both answers across 5 criteria (Factual Accuracy, Completeness, Clarity, Relevance, Depth) "
        "and selects a single non-tied winner. "
        "Stage 6 (Combiner) synthesizes the final verified response. "
        "The system persists context in SQLite (context.db) and logs training data in judge_training.db."
    )
    
    default_doc_2 = (
        "RAG (Retrieval-Augmented Generation) & Transformer Deep Learning Mechanics:\n"
        "Transformers rely on multi-head self-attention mechanisms to weigh relationships between tokens across sequence length. "
        "Positional encodings (such as RoPE or Sinusoidal) preserve word order. "
        "Retrieval-Augmented Generation (RAG) reduces LLM hallucinations by retrieving external verified context chunks "
        "from a vector database before LLM inference, embedding both query and documents into dense vector spaces."
    )

    index_document("System_Architecture_Guide.md", default_doc_1)
    index_document("RAG_and_Transformers_Overview.md", default_doc_2)
    print("[RAG Store] Default domain knowledge pre-loaded successfully.")
