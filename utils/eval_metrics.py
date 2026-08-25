"""
utils/eval_metrics.py
----------------------
Automatic NLP evaluation metrics (BERTScore, BLEU, ROUGE-1/2/L) for scoring
a candidate answer against a reference text.

These are objective, model-free-ish signals (BERTScore still uses a frozen
LM internally, but it's not "asking an LLM to judge") meant to complement
the existing LLM-judge rubric scores in the weighted scoring system.

No single ground-truth reference exists in this pipeline, so callers
typically use the combiner's `final_answer` as the reference when scoring
`answer_a` / `answer_b`.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import Dict, List

_bertscorer = None  # lazy-loaded, reused across calls (loading the LM is the slow part)


def _get_bertscorer():
    global _bertscorer
    if _bertscorer is None:
        import torch
        from bert_score import BERTScorer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _bertscorer = BERTScorer(lang="en", device=device, rescale_with_baseline=True)
    return _bertscorer


def compute_bertscore_batch(candidates: List[str], references: List[str]) -> Dict[str, List[float]]:
    """Batched BERTScore — much faster than calling per-pair for large datasets."""
    scorer = _get_bertscorer()
    P, R, F1 = scorer.score(candidates, references, batch_size=32)
    return {"precision": P.tolist(), "recall": R.tolist(), "f1": F1.tolist()}


def compute_bleu(candidate: str, reference: str) -> float:
    """Sentence-level BLEU (sacrebleu), normalized to 0-1."""
    import sacrebleu

    if not candidate.strip() or not reference.strip():
        return 0.0
    result = sacrebleu.sentence_bleu(candidate, [reference])
    return result.score / 100.0


_rouge_scorer = None


def compute_rouge(candidate: str, reference: str) -> Dict[str, float]:
    """ROUGE-1 / ROUGE-2 / ROUGE-L F-measure."""
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer as rs

        _rouge_scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    if not candidate.strip() or not reference.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scores = _rouge_scorer.score(reference, candidate)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }
