"""
utils/judge_parser.py
----------------------
Parses the Judge agent's raw text output into structured numeric scores.
Shared between the live server pipeline and the offline training-data
generator (generate_training_data.py keeps its own copy so this module can
evolve independently without risking that long-running script).
"""

from __future__ import annotations

import re


def _split_into_sections(text: str) -> tuple[str, str]:
    """
    Split judge_text into (section_a, section_b).
    Handles both '## Answer 1' and plain 'Answer 1' section headers,
    with or without LLaMA/Mistral labels in parentheses.
    """
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
    """Find TOTAL: XX/50 in a chunk, or compute sum of 5 criteria scores."""
    m = re.search(r"TOTAL[:\s*-]+(\d{1,2})/50", chunk, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 50:
            return val
    return sum([
        _extract_score(chunk, "Factual Accuracy"),
        _extract_score(chunk, "Completeness"),
        _extract_score(chunk, "Clarity"),
        _extract_score(chunk, "Relevance"),
        _extract_score(chunk, "Depth of Reasoning"),
    ])


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


def resolve_winner_tie_break(
    winner: str,
    total_a: int, total_b: int,
    factual_a: int, factual_b: int,
    depth_a: int, depth_b: int,
    completeness_a: int, completeness_b: int,
    clarity_a: int, clarity_b: int,
    relevance_a: int, relevance_b: int,
    word_count_a: int, word_count_b: int,
    question: str
) -> str:
    """Apply strict, deterministic tie-breaking rules to guarantee a winner A or B."""
    if total_a > total_b:
        return 'A'
    if total_b > total_a:
        return 'B'
    if factual_a != factual_b:
        return 'A' if factual_a > factual_b else 'B'
    if depth_a != depth_b:
        return 'A' if depth_a > depth_b else 'B'
    if completeness_a != completeness_b:
        return 'A' if completeness_a > completeness_b else 'B'
    if clarity_a != clarity_b:
        return 'A' if clarity_a > clarity_b else 'B'
    if relevance_a != relevance_b:
        return 'A' if relevance_a > relevance_b else 'B'
    if word_count_a != word_count_b:
        if word_count_a == 0:
            return 'B'
        if word_count_b == 0:
            return 'A'
        return 'A' if word_count_a < word_count_b else 'B'
    import binascii
    h = binascii.crc32(question.encode('utf-8'))
    return 'A' if h % 2 == 0 else 'B'


def parse_judge_scores(
    judge_text: str,
    question: str = "",
    answer_a: str = "",
    answer_b: str = ""
) -> dict:
    """Parse structured judge output into numeric fields, resolving ties strictly."""
    sec_a, sec_b = _split_into_sections(judge_text)

    result: dict = {}
    for side, chunk in (('a', sec_a), ('b', sec_b)):
        result[f"factual_{side}"]      = _extract_score(chunk, "Factual Accuracy")
        result[f"completeness_{side}"] = _extract_score(chunk, "Completeness")
        result[f"clarity_{side}"]      = _extract_score(chunk, "Clarity")
        result[f"relevance_{side}"]    = _extract_score(chunk, "Relevance")
        result[f"depth_{side}"]        = _extract_score(chunk, "Depth of Reasoning")
        result[f"total_{side}"]        = _extract_total(chunk)

    winner = _extract_winner(judge_text)

    word_count_a = len(answer_a.split()) if answer_a else 0
    word_count_b = len(answer_b.split()) if answer_b else 0

    result["winner"] = resolve_winner_tie_break(
        winner,
        result["total_a"], result["total_b"],
        result["factual_a"], result["factual_b"],
        result["depth_a"], result["depth_b"],
        result["completeness_a"], result["completeness_b"],
        result["clarity_a"], result["clarity_b"],
        result["relevance_a"], result["relevance_b"],
        word_count_a, word_count_b,
        question
    )
    return result
