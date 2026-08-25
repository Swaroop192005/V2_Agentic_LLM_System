"""
utils/fact_check.py
--------------------
External factual verification against Wikipedia. Searches Wikipedia for the
article most relevant to the question, fetches its summary, and computes
semantic similarity between the pipeline's winning answer and that summary
as a 0-1 "does this answer agree with an independent, authoritative source"
signal - fills the `fact_check` slot in utils/scoring.py's weighted formula.

Network-dependent and best-effort: any failure (no connection, no matching
article, disambiguation page, timeout) returns None so the weighted scoring
system gracefully excludes it rather than breaking the pipeline.
"""

from __future__ import annotations

import re
import urllib.parse

import requests

from utils.similarity import compute_cosine_similarity

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
TIMEOUT = 6.0
HEADERS = {"User-Agent": "agentic-llm-system/1.0 (local research project)"}

# Below this question<->summary similarity, the "match" is treated as spurious
# and discarded (Wikipedia search returns SOME result for almost any query,
# often a wildly unrelated one - e.g. a compound engineering question like
# "security/privacy/architectural considerations for X" has no dedicated
# article, and blindly trusting the top hit would inject noise, not signal).
RELEVANCE_MIN = 0.35

# Strips a "how/what/explain ..." wrapper down to the actual topic before
# searching, since raw interrogative phrasing pollutes Wikipedia's search
# relevance (e.g. "What is the capital of France?" matches on the word
# "question" itself far more than on "France").
_QUESTION_PREFIXES = [
    r'^what (?:is|are|was|were|does|do|did)\s+(?:the\s+)?',
    r'^how (?:does|do|did|has|have)\s+',
    r'^why (?:does|do|did|is|are)\s+',
    r'^when (?:did|was|were|does|do)\s+',
    r'^where (?:is|are|was|were)\s+',
    r'^who (?:is|are|was|were)\s+',
    r'^which\s+',
    r'^explain\s+(?:how\s+|the\s+)?',
    r'^describe\s+(?:how\s+|the\s+)?',
    r'^compare and contrast\s+',
    r'^walk through[^.]*?breakdown of how\s+',
]


def clean_query(question: str) -> str:
    """Strip a leading interrogative phrase, leaving the core topic to search on."""
    q = question.strip().rstrip('?').strip()
    for pattern in _QUESTION_PREFIXES:
        new_q = re.sub(pattern, '', q, flags=re.IGNORECASE)
        if new_q != q:
            return new_q.strip()
    return q


def search_wikipedia_title(query: str) -> str | None:
    """Find the most relevant Wikipedia article title for a (cleaned) query."""
    try:
        resp = requests.get(
            SEARCH_URL,
            params={"action": "query", "list": "search", "srsearch": clean_query(query), "format": "json", "srlimit": 1},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        return results[0]["title"] if results else None
    except Exception:
        return None


def get_wikipedia_summary(title: str) -> str | None:
    """Fetch the plain-text summary (intro paragraph) of a Wikipedia article."""
    try:
        url = SUMMARY_URL.format(title=urllib.parse.quote(title))
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("type") == "disambiguation":
            return None
        extract = data.get("extract", "")
        return extract if extract.strip() else None
    except Exception:
        return None


def verify_against_wikipedia(question: str, answer: str) -> dict | None:
    """
    Check how well `answer` aligns with the most relevant Wikipedia article
    for `question`. Returns None if no article was found, the page was a
    disambiguation page, the match was too tangential (relevance gate), or a
    network error occurred - callers should treat None as "fact-check
    unavailable for this query", not "answer is wrong". Many of this
    pipeline's questions (compound engineering/trade-off analyses) simply
    have no dedicated Wikipedia article, and blindly scoring against the
    nearest unrelated hit would inject noise rather than signal.
    """
    title = search_wikipedia_title(question)
    if not title:
        return None
    summary = get_wikipedia_summary(title)
    if not summary:
        return None

    relevance = compute_cosine_similarity(question, summary)
    if relevance < RELEVANCE_MIN:
        return None

    score = compute_cosine_similarity(answer, summary)
    return {"score": score, "source_title": title, "source_summary": summary, "relevance": relevance}
