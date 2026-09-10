"""
ground_truth_study/analyze_ratings.py
======================================
Run AFTER humans have filled in rating_sheet.csv (one or more raters). Reads
their blind verdicts, joins them back to answer_key.json, and answers the
question the label-free methods couldn't (RESEARCH_LOG Sections 29-31):
when a human judges factual correctness, do they side with the Judge's
Mistral preference, or with the Verifier's "no real difference"?

Handles multiple raters: fill the same sheet multiple times (one copy per
rater) and pass all the CSVs as arguments, or keep a rater_initials column.
Each (item, rater) verdict is one data point.

Reports, per the three possible outcomes:
- Humans agree with the Judge (side with Mistral clearly above chance)
  -> the Mistral edge is real, the Verifier was under-sensitive on subtle gaps.
- Humans see a coin-flip (no side above chance)
  -> the Verifier was right; the Judge's edge is a model-specific artefact.
- Humans agree with neither / side with LLaMA3
  -> the Judge is actively miscalibrated on identity - the strongest finding.

Run:  python ground_truth_study/analyze_ratings.py [rating_sheet.csv ...]
      (defaults to the single rating_sheet.csv in this directory)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
KEY_PATH = HERE / "answer_key.json"


def norm(s: str) -> str:
    return (s or "").strip().lower()


def binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value for k successes in n trials."""
    if n == 0:
        return 1.0
    from math import comb
    def pmf(i):
        return comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def main():
    if not KEY_PATH.exists():
        print("answer_key.json not found - run build_study.py first.")
        return
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))["items"]

    csv_paths = [Path(a) for a in sys.argv[1:]] or [HERE / "rating_sheet.csv"]

    # Collect verdicts across all provided sheets
    human_picks_model = []   # the model the human judged more factually correct
    agree_with_judge = []    # bool: did human's pick match the Judge's pick
    ties = 0
    error_counts = {"mistral": 0, "llama3": 0}
    error_answers = 0
    total_verdicts = 0

    for path in csv_paths:
        if not path.exists():
            print(f"  (skipping missing {path})")
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                item = row.get("item_id", "").strip()
                if item not in key:
                    continue
                pick = norm(row.get("more_factually_correct(1/2/tie)", ""))
                k = key[item]
                if pick in ("1", "2"):
                    total_verdicts += 1
                    picked_model = k["answer1_model"] if pick == "1" else k["answer2_model"]
                    human_picks_model.append(picked_model)
                    agree_with_judge.append(pick == k["judge_picked_display"])
                elif pick == "tie":
                    total_verdicts += 1
                    ties += 1
                # factual-error tallies, mapped to true model
                for disp, col in (("1", "answer_1_has_factual_error(y/n)"),
                                  ("2", "answer_2_has_factual_error(y/n)")):
                    if norm(row.get(col, "")) == "y":
                        m = k["answer1_model"] if disp == "1" else k["answer2_model"]
                        error_counts[m] += 1
                        error_answers += 1

    if total_verdicts == 0:
        print("No filled verdicts found yet. Fill the 'more_factually_correct(1/2/tie)' column and re-run.")
        return

    decisive = len(human_picks_model)  # non-tie verdicts
    n_mistral = human_picks_model.count("mistral")
    n_llama = human_picks_model.count("llama3")

    print("=" * 68)
    print(f"HUMAN GROUND-TRUTH STUDY  ({total_verdicts} verdicts, {ties} ties, {decisive} decisive)")
    print("=" * 68)
    print(f"\nWhen a human picked a more factually correct answer:")
    print(f"  Mistral judged better: {n_mistral}/{decisive} ({n_mistral/decisive*100:.1f}%)")
    print(f"  LLaMA 3 judged better: {n_llama}/{decisive} ({n_llama/decisive*100:.1f}%)")
    p_pref = binom_two_sided_p(n_mistral, decisive)
    print(f"  Two-sided binomial vs 50/50: p = {p_pref:.4f}")

    agree = sum(agree_with_judge)
    print(f"\nHuman agreement with the JUDGE's pick (decisive items):")
    print(f"  Agreed: {agree}/{len(agree_with_judge)} ({agree/len(agree_with_judge)*100:.1f}%)")
    p_agree = binom_two_sided_p(agree, len(agree_with_judge))
    print(f"  Two-sided binomial vs 50/50: p = {p_agree:.4f}")

    print(f"\nOutright factual errors flagged: {error_answers} "
          f"(Mistral {error_counts['mistral']}, LLaMA 3 {error_counts['llama3']})")

    print("\n" + "-" * 68)
    print("INTERPRETATION")
    print("-" * 68)
    if p_pref < 0.05 and n_mistral > n_llama:
        print("Humans side with Mistral above chance -> the Judge's Mistral edge is")
        print("REAL; the Verifier was under-sensitive to a genuine but subtle gap.")
    elif p_pref < 0.05 and n_llama > n_mistral:
        print("Humans side with LLaMA 3 above chance -> the Judge is MISCALIBRATED on")
        print("identity (it preferred the model humans rate lower). Strongest finding.")
    else:
        print("Humans are at chance between the two models -> the Verifier was right:")
        print("the Judge's Mistral preference is a model-specific artefact, not a real")
        print("quality gap. (Consistent with Section 31's verifier-discrimination result.)")
    print(f"\nNote: n is small by design (a hand study). Treat a p just under 0.05 as")
    print(f"directional, and add raters/items if a firmer answer is needed.")


if __name__ == "__main__":
    main()
