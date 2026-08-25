"""
utils/rubric_parser.py
-----------------------
Parses structured 8-factor rubric output (agents/rubric.py) from EITHER the
Verifier's or the Judge's raw text - same format, same parser. The Judge's
output additionally has a winner declaration, handled separately.

Replaces the old 5-factor judge_parser.py (kept in generate_training_data.py
as its own frozen copy so that script isn't disturbed; server.py now uses
this module instead).
"""

from __future__ import annotations

import re

from agents.rubric import FACTORS, SLUGS, MAX_TOTAL


def _split_into_sections(text: str) -> tuple[str, str]:
    """Split rubric_text into (section_a, section_b) for Answer 1 / Answer 2."""
    sep = re.search(
        r'(?:^|\n)\s*#{0,3}\s*Answer\s*2(?:\s*\(Mistral\))?\s*(?:\n|:)',
        text, re.IGNORECASE | re.MULTILINE
    )
    if sep:
        return text[:sep.start()], text[sep.start():]
    mid = len(text) // 2
    return text[:mid], text[mid:]


def _extract_score(chunk: str, label: str) -> int:
    """Find 'Label: N/10' or similar in a text chunk. Flexible formatting."""
    escaped = re.escape(label)
    patterns = [
        rf"{escaped}[\s:*-]+(\d{{1,2}})/10",
        rf"{escaped}[\s:*-]+(\d{{1,2}})\s*out",
        rf"{escaped}[\s:*-]+Score[:\s]+(\d{{1,2}})",
        rf"{escaped}.*?(\d{{1,2}})/10",
    ]
    for p in patterns:
        m = re.search(p, chunk, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 10:
                return val
    return 0


def _extract_total(chunk: str) -> int:
    """
    Total is ALWAYS the sum of the 8 individually-parsed factor scores -
    NEVER the model's own self-reported "TOTAL: XX/80" line. Found via a
    real example (gemma2 test, 2026-08-18): a model can write per-criterion
    scores summing to 72 and then literally write "TOTAL: 54/80" - its own
    arithmetic, not a judgment. Trusting that line silently corrupted every
    score-distribution statistic computed so far (made a model look like it
    was grading stricter/less biased than it actually was at the per-
    criterion level, when the discrepancy was just a summation error).
    """
    return sum(_extract_score(chunk, name) for name, _ in FACTORS)


def parse_rubric_scores(text: str) -> dict:
    """
    Parse an 8-factor rubric text (from either Verifier or Judge) into a flat
    dict keyed by slug, e.g. {"factual_correctness_a": 8, ..., "total_a": 62,
    "factual_correctness_b": 7, ..., "total_b": 58}.
    """
    sec_a, sec_b = _split_into_sections(text)
    result: dict = {}
    for side, chunk in (('a', sec_a), ('b', sec_b)):
        for (name, _), slug in zip(FACTORS, SLUGS):
            result[f"{slug}_{side}"] = _extract_score(chunk, name)
        result[f"total_{side}"] = _extract_total(chunk)
    return result


def _extract_winner(text: str) -> str:
    """Find winner declaration - handles various phrasings."""
    patterns = [
        r"Winner[:\s*-]+Answer\s*(1|2)",
        r"Winner[:\s*-]+(?:Answer\s*)?(LLaMA|Llama|llama)",
        r"Winner[:\s*-]+(?:Answer\s*)?(Mistral|mistral)",
        r"(?:better|superior|wins)[\s\w]*?(?:Answer|LLM)\s*(1|2)",
        r"Answer\s*(1|2)\s+(?:is the winner|wins)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            raw = m.group(1) if m.lastindex else ''
            if raw in ('1', 'LLaMA', 'Llama', 'llama', ''):
                return 'A'
            if raw in ('2', 'Mistral', 'mistral'):
                return 'B'
    return 'unknown'


def resolve_winner_tie_break(winner: str, parsed: dict, word_count_a: int, word_count_b: int, question: str) -> str:
    """Apply strict, deterministic tie-breaking rules to guarantee a winner A or B."""
    total_a, total_b = parsed["total_a"], parsed["total_b"]
    if total_a > total_b:
        return 'A'
    if total_b > total_a:
        return 'B'
    # Tie-break in factor order: factual correctness first, then the rest in rubric order.
    for slug in SLUGS:
        a, b = parsed[f"{slug}_a"], parsed[f"{slug}_b"]
        if a != b:
            return 'A' if a > b else 'B'
    if word_count_a != word_count_b:
        if word_count_a == 0:
            return 'B'
        if word_count_b == 0:
            return 'A'
        return 'A' if word_count_a < word_count_b else 'B'
    import binascii
    h = binascii.crc32(question.encode('utf-8'))
    return 'A' if h % 2 == 0 else 'B'


def parse_judge_scores(judge_text: str, question: str = "", answer_a: str = "", answer_b: str = "") -> dict:
    """Parse the Judge's rubric scores + winner, resolving ties strictly."""
    parsed = parse_rubric_scores(judge_text)
    winner = _extract_winner(judge_text)
    word_count_a = len(answer_a.split()) if answer_a else 0
    word_count_b = len(answer_b.split()) if answer_b else 0
    parsed["winner"] = resolve_winner_tie_break(winner, parsed, word_count_a, word_count_b, question)
    return parsed


def remap_ab(parsed: dict, swap: bool) -> dict:
    """
    Swap every '<x>_a' / '<x>_b' key pair (and flip 'winner' if present).

    Used when an answer pair was fed to the Verifier/Judge in a randomized
    internal order (to avoid position/identity bias - see RESEARCH_LOG.md
    Section 17) but the caller wants results reported back in a fixed
    display order (e.g. server.py's UI always shows LLaMA 3 as "Answer 1").
    If swap=False, returns a shallow copy unchanged.
    """
    if not swap:
        return dict(parsed)
    out: dict = {}
    for k, v in parsed.items():
        if k.endswith("_a"):
            out[k[:-2] + "_b"] = v
        elif k.endswith("_b"):
            out[k[:-2] + "_a"] = v
        elif k == "winner":
            out[k] = "B" if v == "A" else ("A" if v == "B" else v)
        else:
            out[k] = v
    return out


def compute_agreement(verifier_parsed: dict, judge_parsed: dict) -> float:
    """
    Verifier-judge agreement: 1 - normalized mean absolute difference between
    their two independent total scores (both /MAX_TOTAL), for each answer,
    averaged. 1.0 = identical totals, 0.0 = maximally apart.
    """
    diff_a = abs(verifier_parsed["total_a"] - judge_parsed["total_a"])
    diff_b = abs(verifier_parsed["total_b"] - judge_parsed["total_b"])
    avg_diff = (diff_a + diff_b) / 2
    return max(0.0, 1.0 - (avg_diff / MAX_TOTAL))
