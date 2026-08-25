"""
utils/context_store.py
-----------------------
SQLite-backed Context Store for the Agentic Multi-LLM Validation System.

Stores metadata of the last 5 user queries and provides a clean interface
to inject that rolling context into new LLM prompts.

Schema (table: query_context):
  id              INTEGER  PRIMARY KEY AUTOINCREMENT
  timestamp       TEXT     ISO-8601 datetime of the query
  query           TEXT     The original user query
  answer_summary  TEXT     First 400 chars of the final combined answer
  word_count      INTEGER  Number of words in the query

Only the 5 most recent records are kept — older ones are auto-pruned.
"""

from __future__ import annotations

import sqlite3
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

# Store the DB next to this file (inside the project, persists across restarts)
_DB_PATH = Path(__file__).parent.parent / "context.db"
_MAX_CONTEXT = 5          # Rolling window size
_SUMMARY_LEN = 400        # Max chars of final answer to store as summary


# ── DB Initialisation ─────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Return a thread-safe SQLite connection with WAL mode for concurrency."""
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the context table if it doesn't already exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_context (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                query          TEXT    NOT NULL,
                answer_summary TEXT,
                word_count     INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()


# ── CRUD Operations ───────────────────────────────────────────────────────────

def save_query(query: str, final_answer: Optional[str] = None) -> None:
    """
    Persist a completed query + its answer summary to the DB.
    Automatically prunes records older than the last _MAX_CONTEXT entries.

    Args:
        query:        The raw user query string.
        final_answer: The final combined answer from the Combiner agent (optional).
    """
    summary = textwrap.shorten(final_answer or "", width=_SUMMARY_LEN, placeholder="…")
    word_count = len(query.split())
    ts = datetime.now(timezone.utc).isoformat()

    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO query_context (timestamp, query, answer_summary, word_count)
            VALUES (?, ?, ?, ?)
            """,
            (ts, query, summary, word_count),
        )
        # Prune: keep only the latest _MAX_CONTEXT rows
        conn.execute(
            """
            DELETE FROM query_context
            WHERE id NOT IN (
                SELECT id FROM query_context
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (_MAX_CONTEXT,),
        )
        conn.commit()


def get_recent_context() -> list[dict]:
    """
    Retrieve the last _MAX_CONTEXT query metadata records, newest first.

    Returns:
        List of dicts with keys: id, timestamp, query, answer_summary, word_count
    """
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, query, answer_summary, word_count
            FROM query_context
            ORDER BY id DESC
            LIMIT ?
            """,
            (_MAX_CONTEXT,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_context_records() -> list[dict]:
    """
    Retrieve ALL stored context records for the /api/context endpoint.
    Returns newest first.
    """
    return get_recent_context()  # Same cap of 5


def clear_context() -> int:
    """
    Delete all stored query-context records (the rolling last-5 history).
    Returns the number of rows deleted.
    """
    with _get_connection() as conn:
        cur = conn.execute("DELETE FROM query_context")
        conn.commit()
        return cur.rowcount


# ── Prompt Builder ────────────────────────────────────────────────────────────

# Below this cosine similarity to the current query, a prior query is treated
# as an unrelated topic switch and left out of the context block. Previously
# ALL of the last 5 queries were injected unconditionally, regardless of
# topic - so asking "what is ice cream?" right after questions about CRISPR,
# biomedical data, or UBI caused the LLM to narrate those unrelated topics
# ("In light of our previous conversations on CRISPR technology...") and the
# Verifier to penalize the answer for not connecting ice cream back to them.
MIN_RELEVANCE = 0.35


def build_context_prefix(current_query: str) -> str:
    """
    Build a natural-language context block from recent queries that are
    actually topically related to current_query. This is prepended to LLM
    prompts so agents are aware of conversation flow - but only when there
    IS a continuing conversation, not on every unrelated new topic.

    Args:
        current_query: The incoming query (not yet saved).

    Returns:
        A formatted string to prepend to any LLM prompt, or empty string if
        there is no prior context or none of it is relevant to this query.
    """
    records = get_recent_context()

    if not records:
        return ""  # No history yet — no prefix needed

    from utils.similarity import compute_cosine_similarity

    relevant = [r for r in records if compute_cosine_similarity(current_query, r["query"]) >= MIN_RELEVANCE]
    if not relevant:
        return ""  # Prior history exists but none of it relates to this new topic

    lines = ["[CONVERSATION CONTEXT — last related queries the user asked]"]
    for i, rec in enumerate(reversed(relevant), start=1):  # oldest → newest
        short_q = textwrap.shorten(rec["query"], width=120, placeholder="…")
        short_a = textwrap.shorten(rec["answer_summary"] or "(no summary)", width=200, placeholder="…")
        lines.append(f"  Q{i}: {short_q}")
        lines.append(f"  A{i} (summary): {short_a}")

    lines.append(
        "\n[Use the above context ONLY to understand the user's topic and intent. "
        "Do not repeat or rehash previous answers. Focus on the NEW question below.]\n"
    )
    return "\n".join(lines) + "\n"
