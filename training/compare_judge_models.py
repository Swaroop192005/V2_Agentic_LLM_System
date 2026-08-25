"""
training/compare_judge_models.py
==================================
Validates a candidate judge model against qwen2.5 on the SAME set of real
(question, answer_a, answer_b) triples pulled from the archived dataset,
using the exact production rubric prompt. Reports distribution stats for
both models side by side.

Run:
    python training/compare_judge_models.py --n 20
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.base_agent import BaseAgent
from agents.rubric import format_rubric_block, format_output_template, SCORING_GUIDE, SLUGS
from agents.judge_agent import build_judge_prompt
from utils.rubric_parser import parse_judge_scores

ARCHIVE_DB = ROOT / "data" / "judge_training_v2_lenient_rubric_ARCHIVE.db"


def make_judge(model_name: str) -> BaseAgent:
    # Mirrors agents/judge_agent.py::create_judge_agent exactly (blind scoring,
    # same rubric v4 text, reasoning disabled for hybrid-reasoning models like
    # qwen3.5 - without this they can burn their whole token budget on
    # internal thinking and return empty output, see RESEARCH_LOG.md Section 15)
    # so results transfer directly to the live pipeline.
    return BaseAgent(
        role="Comparison Judge",
        model_name=model_name,
        temperature=0.1,
        num_predict=900,
        reasoning=(False if "qwen3.5" in model_name else None),
        system_prompt=(
            "You are an impartial, highly analytical AI evaluator. "
            "You evaluate answers on EIGHT criteria, each scored 0-10. "
            "For EVERY score you give, you MUST write a one-sentence justification "
            "referencing specific content from the answer. "
            "Do NOT give a score without explaining it. "
            "You are scoring BLIND - no other evaluator's scores are shown to you. Score purely from "
            "your own reading of the question and the two answers.\n\n"
            "Ties are strictly forbidden. You must select exactly one winner.\n\n"
            f"{SCORING_GUIDE}\n\n"
            f"Criteria definitions:\n{format_rubric_block()}\n\n"
            "Always use EXACTLY this format:\n"
            f"{format_output_template('Answer 1 (LLaMA 3)')}\n\n{format_output_template('Answer 2 (Mistral)')}\n\n"
            "## Verdict\nWinner: Answer 1 or Answer 2\nKey Strength of Winner: <one sentence>\n"
            "Key Weakness of Loser: <one sentence>\nOverall Reasoning: <2-3 sentences>"
        ),
    )


def is_flat(parsed: dict, side: str) -> bool:
    """True if all 8 per-criterion scores for this side are identical."""
    vals = [parsed[f"{slug}_{side}"] for slug in SLUGS]
    return len(set(vals)) == 1


def main(n: int, models: list[str]) -> None:
    con = sqlite3.connect(str(ARCHIVE_DB))
    rows = con.execute(
        "SELECT question, answer_a, answer_b FROM pipeline_runs "
        "WHERE is_final_attempt=1 ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()
    con.close()
    print(f"Loaded {len(rows)} real Q&A pairs for comparison\n", flush=True)

    results: dict[str, list[int]] = {m: [] for m in models}
    flat_counts: dict[str, list[bool]] = {m: [] for m in models}

    for model_name in models:
        print(f"=== Testing {model_name} ===", flush=True)
        judge = make_judge(model_name)
        for i, (q, a, b) in enumerate(rows, 1):
            t0 = time.time()
            try:
                raw = judge.run(build_judge_prompt(q, a, b))
                parsed = parse_judge_scores(raw, q, a, b)
                results[model_name].append(parsed["total_a"])
                results[model_name].append(parsed["total_b"])
                flat_a, flat_b = is_flat(parsed, "a"), is_flat(parsed, "b")
                flat_counts[model_name].append(flat_a)
                flat_counts[model_name].append(flat_b)
                elapsed = time.time() - t0
                flag = " [FLAT]" if (flat_a or flat_b) else ""
                print(f"  [{i}/{len(rows)}] {q[:50]}... a={parsed['total_a']}{'*' if flat_a else ''} b={parsed['total_b']}{'*' if flat_b else ''} ({elapsed:.0f}s){flag}", flush=True)
            except Exception as exc:
                print(f"  [{i}/{len(rows)}] FAILED: {exc}", flush=True)
        print(flush=True)

    print("=" * 70, flush=True)
    print("COMPARISON RESULTS", flush=True)
    print("=" * 70, flush=True)
    for model_name, scores in results.items():
        if not scores:
            print(f"{model_name}: no data", flush=True)
            continue
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0
        pct_at_ceiling = sum(1 for s in scores if s >= 75) / len(scores) * 100
        flats = flat_counts[model_name]
        pct_flat = sum(flats) / len(flats) * 100 if flats else 0
        print(
            f"{model_name}\n"
            f"  n={len(scores)}  mean={mean:.1f}/80 ({mean/80*100:.1f}%)  stdev={stdev:.1f}\n"
            f"  min={min(scores)}  max={max(scores)}  %-at-75+={pct_at_ceiling:.1f}%\n"
            f"  %-flat-all-8-identical={pct_flat:.1f}%  (* marks a flat score above)\n",
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--models", nargs="+", default=["qwen2.5", "hf.co/mradermacher/CompassJudger-1-7B-Instruct-GGUF:Q4_K_M"])
    args = p.parse_args()
    main(args.n, args.models)
