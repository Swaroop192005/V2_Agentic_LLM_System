"""
utils/wikidata_check.py
------------------------
Second external fact-check source, alongside utils/fact_check.py's Wikipedia
check (added per mentor request for an "Encyclopedia" cross-check). Uses
Wikidata's structured entity search - genuinely different in character from
Wikipedia's prose summaries: short, structured, fact-terse descriptions
(e.g. "capital and most populous city in France") rather than paragraphs.

Same design as fact_check.py: best-effort, gated by relevance, returns None
rather than a misleading score when nothing genuinely relevant is found.
"""

from __future__ import annotations

import re

import requests

from utils.fact_check import RELEVANCE_MIN
from utils.similarity import compute_cosine_similarity

SEARCH_URL = "https://www.wikidata.org/w/api.php"
TIMEOUT = 6.0
HEADERS = {"User-Agent": "agentic-llm-system/1.0 (local research project)"}

# Wikidata's wbsearchentities is a LABEL matcher (autocomplete-style over entity
# names/aliases), not a full-text search like Wikipedia's - searching the raw
# question ("capital of France") matches on literal label words and surfaces
# things like "list of capitals of France" instead of the entity "Paris".
# So instead we extract candidate proper-noun phrases from the ANSWER (which
# actually names entities, e.g. "...is Paris") and try those as search terms.
_SENTENCE_START = re.compile(r'(?:^|[.!?]\s+)([A-Z][a-z]+)')


def _extract_entity_candidates(text: str, limit: int = 5) -> list[str]:
    """Pull likely proper-noun phrases (1-3 consecutive capitalized words) out
    of `text`, skipping words that are merely capitalized because they start
    a sentence. Ordered by phrase length (longest/most specific first)."""
    sentence_starters = set(_SENTENCE_START.findall(text))
    candidates = re.findall(r'\b(?:[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,2})\b', text)
    seen, out = set(), []
    for c in candidates:
        first_word = c.split()[0]
        if len(c) < 3 or (first_word in sentence_starters and len(c.split()) == 1):
            continue
        if c not in seen:
            seen.add(c)
            out.append(c)
    out.sort(key=len, reverse=True)
    return out[:limit]


def _search_one(term: str) -> dict | None:
    try:
        resp = requests.get(
            SEARCH_URL,
            params={"action": "wbsearchentities", "search": term, "language": "en", "format": "json", "limit": 1},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("search", [])
        if not results:
            return None
        top = results[0]
        description = top.get("description", "")
        if not description.strip():
            return None
        return {"id": top.get("id", ""), "label": top.get("label", ""), "description": description}
    except Exception:
        return None


def search_wikidata_entity(question: str, answer: str) -> dict | None:
    """
    Try each candidate entity name extracted from the answer, then the
    question's own proper nouns, then the cleaned question phrase itself
    (covers common-noun topics like "photosynthesis" that are never
    capitalized in running prose, so the proper-noun extraction misses
    them) against Wikidata's label search. Returns the first hit whose
    description is actually relevant to the question, or None.
    """
    from utils.fact_check import clean_query

    candidates = (
        _extract_entity_candidates(answer)
        + _extract_entity_candidates(question)
        + [clean_query(question)]
    )
    seen = set()
    for term in candidates:
        if not term or term.lower() in seen:
            continue
        seen.add(term.lower())
        entity = _search_one(term)
        if not entity:
            continue
        if compute_cosine_similarity(question, entity["description"]) >= RELEVANCE_MIN:
            return entity
    return None


def verify_against_wikidata(question: str, answer: str) -> dict | None:
    """
    Check how well `answer` aligns with the most relevant Wikidata entity's
    description for `question`. Returns None if no entity was found or every
    candidate was too tangential (relevance gate) - same semantics as
    fact_check.verify_against_wikipedia, so both can be combined the same way.
    """
    entity = search_wikidata_entity(question, answer)
    if not entity:
        return None

    score = compute_cosine_similarity(answer, entity["description"])
    return {
        "score": score,
        "source_title": entity["label"],
        "source_summary": entity["description"],
    }
