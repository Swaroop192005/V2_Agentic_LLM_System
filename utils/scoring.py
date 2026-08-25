"""
utils/scoring.py
-----------------
Weighted scoring system: combines multiple quality signals for the pipeline's
WINNING answer into a single 0-1 confidence score, used to decide whether the
answer is good enough to ship or should be regenerated.

Deliberately excludes any final_answer-referenced BERTScore/BLEU/ROUGE: those
were found to be systematically biased toward answer_a because the same
model (LLaMA 3) both writes answer_a and drafts final_answer as combiner -
see training/models/FINAL_CONFIG.md and the eval_metrics analysis for the
full writeup. Only signals that don't share that confound are used here.

Components (weights sum to 1.0 across whichever signals are actually
available - any missing signal has its weight redistributed proportionally
across the rest, so this keeps working as sources are added/unavailable):

    judge                (0.40) - the Judge's rubric total for the winning answer, /80
    similarity           (0.15) - cosine similarity of the winning answer to the question
    model_agreement      (0.10) - pairwise BERTScore between answer_a and answer_b:
                                   high agreement between two independently generated
                                   answers is weak corroborating evidence of correctness
    verifier_judge_agreement (0.10) - agreement between the Verifier's (phi3) and Judge's
                                   (qwen2.5) INDEPENDENT rubric totals for the winning
                                   answer - two independent evaluators agreeing is a
                                   stronger signal than either one alone
    wikipedia            (0.15) - external Wikipedia fact-check
    wikidata             (0.10) - external Wikidata fact-check (2nd, structurally
                                   different encyclopedia source, added per mentor request)

These starting weights are a reasonable prior, not a final answer - they are
the subject of the ongoing weight-sweep experiments (see training/ sweep
scripts). DEFAULT_THRESHOLD is similarly a starting point pending the
threshold-sweep phase once the new 8-factor dataset exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_WEIGHTS = {
    "judge": 0.40,
    "similarity": 0.15,
    "model_agreement": 0.10,
    "verifier_judge_agreement": 0.10,
    "wikipedia": 0.15,
    "wikidata": 0.10,
}

# Starting point pending the threshold-sweep phase (see project plan / reports).
DEFAULT_THRESHOLD = 0.68
MAX_REGENERATION_ATTEMPTS = 2


@dataclass
class ScoreBreakdown:
    weighted_score: float
    values: dict = field(default_factory=dict)     # per-signal 0-1 values actually used (None dropped)
    weights_used: dict = field(default_factory=dict)  # renormalized weights actually applied

    # Back-compat convenience accessors for the signals server.py/app.js already read
    @property
    def judge(self) -> float | None:
        return self.values.get("judge")

    @property
    def similarity(self) -> float | None:
        return self.values.get("similarity")

    @property
    def agreement(self) -> float | None:
        return self.values.get("model_agreement")

    @property
    def fact_check(self) -> float | None:
        """Combined encyclopedia signal for display: average of whichever of
        wikipedia/wikidata are available."""
        vals = [v for k, v in self.values.items() if k in ("wikipedia", "wikidata") and v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None


def compute_weighted_score(
    judge_total: float | None,
    similarity_0_1: float | None,
    model_agreement_0_1: float | None = None,
    verifier_judge_agreement_0_1: float | None = None,
    wikipedia_0_1: float | None = None,
    wikidata_0_1: float | None = None,
    weights: dict | None = None,
    judge_max_total: float = 80.0,
) -> ScoreBreakdown:
    """
    Combine available quality signals into one 0-1 confidence score.
    Any signal that is None is dropped and the remaining weights are
    renormalized proportionally, so this keeps working regardless of which
    signals happen to be available for a given attempt.
    """
    w = dict(weights or DEFAULT_WEIGHTS)

    values = {
        "judge": None if judge_total is None else max(0.0, min(1.0, judge_total / judge_max_total)),
        "similarity": None if similarity_0_1 is None else max(0.0, min(1.0, similarity_0_1)),
        "model_agreement": None if model_agreement_0_1 is None else max(0.0, min(1.0, model_agreement_0_1)),
        "verifier_judge_agreement": None if verifier_judge_agreement_0_1 is None else max(0.0, min(1.0, verifier_judge_agreement_0_1)),
        "wikipedia": None if wikipedia_0_1 is None else max(0.0, min(1.0, wikipedia_0_1)),
        "wikidata": None if wikidata_0_1 is None else max(0.0, min(1.0, wikidata_0_1)),
    }

    available = {k: v for k, v in values.items() if v is not None}
    if not available:
        raise ValueError("compute_weighted_score: at least one signal must be provided")

    weight_sum = sum(w[k] for k in available)
    normalized_weights = {k: w[k] / weight_sum for k in available}

    score = sum(available[k] * normalized_weights[k] for k in available)

    return ScoreBreakdown(
        weighted_score=round(score, 4),
        values=values,
        weights_used=normalized_weights,
    )


def needs_regeneration(breakdown: ScoreBreakdown, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return breakdown.weighted_score < threshold
