"""
agents/rubric.py
-----------------
Shared 8-factor evaluation rubric used by BOTH the Verifier (phi3) and the
Judge (qwen2.5), so they score every answer independently on the same
criteria rather than the Judge simply trusting the Verifier's free-form
prose. Comparing their two independent score sets gives a verifier-judge
agreement signal, on top of the existing cross-model (answer_a vs answer_b)
agreement already used in the weighted confidence formula.

Replaces the old 5-factor judge-only rubric (factual/completeness/clarity/
relevance/depth, /50) with this 8-factor one (/80), used identically by
both agents.
"""

from __future__ import annotations

FACTORS: list[tuple[str, str]] = [
    ("Factual Correctness",
     "every discrete claim is independently verifiable and correct. A 9-10 means you "
     "checked each claim and found none you'd challenge; a 5 means at least one claim "
     "is unverifiable, imprecise, or slightly off; a 0-2 means a claim is flatly wrong"),
    ("Question Relevance",
     "every sentence serves the specific question asked. A 9-10 has zero padding or "
     "generic throat-clearing; a 5 answers the question but pads it with a generic "
     "preamble/conclusion or drifts into adjacent topics; a 0-2 answers a different "
     "question than the one asked"),
    ("Context Relevance",
     "claims track the retrieved reference documents where provided, without adding "
     "content the context doesn't support (score high if no context was given and the "
     "answer is still reasonable on general knowledge). A 5 uses the context loosely "
     "or adds unsupported extras; a 0-2 contradicts the context"),
    ("Faithfulness",
     "claims are grounded in the provided context or genuinely verifiable general "
     "knowledge, never fabricated to sound plausible. A 5 has at least one claim that "
     "is plausible-sounding but not something you can actually trace to a source"),
    ("Completeness",
     "every key sub-part of the question is covered at a depth proportional to its "
     "importance. A 5 covers the obvious parts but skips a sub-part, an edge case, or "
     "a caveat a subject-matter expert would expect; a 0-2 covers only a fraction"),
    ("Hallucination-Free",
     "free of fabricated, unsupported, or invented content, including invented "
     "specifics (numbers, names, dates, citations) that sound authoritative but "
     "aren't grounded in anything (10 = none found after actively checking every "
     "specific claim, 0 = severe hallucination)"),
    ("Clarity",
     "structure and phrasing minimize the reader's effort. A 5 is understandable but "
     "has at least one awkward transition, redundant sentence, or a structure that "
     "doesn't match the question's shape (e.g. no comparison structure for a "
     "compare/contrast question)"),
    ("Consistency",
     "no internal contradictions, including subtle ones (e.g. a claim in paragraph 1 "
     "that a later paragraph's framing quietly undercuts). A 5 is consistent but has "
     "at least one place where emphasis or framing wobbles"),
]
MAX_TOTAL = len(FACTORS) * 10  # 80

# v1 (75-80/80 clustering under qwen2.5, mean 76.3) led to the SCORING_GUIDE
# band rewrite below "v2". v2 alone was found to have ~no measurable effect
# on distribution while still on qwen2.5/general chat models (see
# RESEARCH_LOG.md Sections 6-9) - switching the Judge to CompassJudger-1 was
# the change that actually moved the numbers (N=40: mean 70.0/80, 0% >=75,
# vs qwen2.5's 71.9/80, 20% >=75 - Section 10). "v3" tightened the bands
# further (narrower widths, mandatory-flaw-search gate before any 7+).
# "v4" (this version) responds to a NEW pattern found after fixing the
# anchoring bug (Section 12): with the Judge finally scoring blind, its
# totals still clustered hard on specific numbers (e.g. 72 appeared as the
# Judge's total on 3 different, unrelated questions in a row) - suggesting
# a generic "default good answer" value rather than content-driven scoring.
# v4 adds two things v3 didn't have: (1) each FACTOR above now states a
# CONCRETE differentiator per band, specific to that criterion, instead of
# reusing the same generic 0-10 language for all 8 factors - vague criteria
# invite a memorized default number; (2) an explicit anti-default-value
# rule below, since a model that has "72" as a comfortable default will
# keep producing it regardless of content unless told to actively check.
SCORING_GUIDE = (
    "Scoring guide (strict - most answers should NOT score 7+; if your scores\n"
    "cluster above 6 across many answers you are being too lenient):\n"
    "  0-1: Severely flawed — major factual errors, off-topic, or unusable\n"
    "  2-3: Poor        — real errors or missing content a careful reader would flag\n"
    "  4-5: Adequate     — correct but unremarkable, with at least one identifiable\n"
    "                      gap, imprecision, or missed nuance. This is where most\n"
    "                      competent, ordinary answers belong.\n"
    "  6-7: Strong       — clearly above average; you can still name one minor\n"
    "                      flaw or omission, it is just small\n"
    "  8:   Excellent    — you searched specifically for a flaw and could not find\n"
    "                      one worth naming\n"
    "  9-10: Flawless    — truly exceptional and essentially unimprovable; must be\n"
    "                      rare (well under 5% of scores you give)\n\n"
    "Each of the 8 criteria below states what specifically separates a high score from\n"
    "a mid score FOR THAT CRITERION - use that specific test, not a generic impression\n"
    "of overall quality, to set each number.\n\n"
    "MANDATORY before scoring ANY criterion 7 or above: explicitly look for a\n"
    "specific weakness (a fact you can't verify, a missed angle, an awkward\n"
    "sentence, a claim stated with more confidence than it deserves). If you find\n"
    "one, that criterion is capped at 6. Only score 8+ if, after actively\n"
    "searching, you still cannot name a specific flaw - not because none occurred\n"
    "to you, but because you looked and there genuinely isn't one. Every answer\n"
    "has SOME room for improvement unless it is truly exceptional; assume that by\n"
    "default and let the answer prove otherwise, not the reverse.\n\n"
    "ANTI-DEFAULT RULE: do not let scores gravitate toward a comfortable habitual\n"
    "number regardless of content. Before finalizing each TOTAL, ask: \"could I \n"
    "point to a specific sentence in THIS answer that justifies THIS exact number,\n"
    "for THIS question - not a number that would fit almost any competent answer?\"\n"
    "If your total for this answer matches the total you gave a previous, unrelated\n"
    "answer, that is a signal to re-examine both rather than assume it's correct.\n"
    "Two different answers to two different questions coincidentally deserving the\n"
    "exact same total is rare - treat repeated identical totals as suspicious."
)

# Slugs used as DB column / dict-key suffixes, e.g. "factual_correctness_a"
SLUGS: list[str] = [
    "factual_correctness", "question_relevance", "context_relevance", "faithfulness",
    "completeness", "hallucination_free", "clarity", "consistency",
]


def format_rubric_block() -> str:
    """Numbered criteria definitions block for a system prompt."""
    return "\n".join(f"  {i + 1}. {name} — {desc}." for i, (name, desc) in enumerate(FACTORS))


def format_output_template(answer_label: str) -> str:
    """The exact per-answer scoring block agents must reproduce."""
    lines = [f"## {answer_label}"]
    for name, _ in FACTORS:
        lines.append(f"- {name}: X/10 | <one sentence justification>")
    lines.append(f"- TOTAL: XX/{MAX_TOTAL}")
    return "\n".join(lines)
