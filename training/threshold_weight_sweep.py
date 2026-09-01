"""
training/threshold_weight_sweep.py
====================================
Searches for better weighted-confidence weights and regeneration threshold
than the DEFAULT_WEIGHTS / DEFAULT_THRESHOLD starting guess in
utils/scoring.py, using the pipeline_runs data collected so far.

There is no ground-truth "this answer is objectively correct" label
anywhere in this dataset, so "better" is defined by two label-free,
defensible objectives instead of accuracy against a target:

1. INTERNAL CONSISTENCY (generalization check)
   For each of the 6 signals, compute a composite score from the OTHER 5
   signals only (properly renormalized, mirroring compute_weighted_score's
   own missing-signal handling), then correlate that composite against the
   held-out signal's raw value. A weighting under which the signals predict
   each other well is more likely capturing real shared "answer quality"
   signal than one where a single high-weighted-but-noisy signal dominates
   and the rest are just along for the ride.

2. REGENERATION VALIDITY (does the threshold mean anything?)
   For every question that got a 2nd attempt (i.e. attempt 1 triggered a
   regeneration), check: excluding the Judge signal itself (to avoid
   circularity, since the Judge total heavily drives weighted_score),
   did the 2nd attempt actually score better on the OTHER signals on
   average? A meaningful threshold should mostly trigger regeneration
   when attempt 1 truly was weaker - if attempt 2 doesn't reliably improve,
   the threshold isn't cutting at an informative point.

Run:
    python training/threshold_weight_sweep.py
"""

from __future__ import annotations

import random
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.scoring import DEFAULT_WEIGHTS, DEFAULT_THRESHOLD

DB_PATH = ROOT / "data" / "judge_training_v2.db"
SIGNAL_KEYS = ["judge", "similarity", "model_agreement", "verifier_judge_agreement", "wikipedia", "wikidata"]


def load_rows() -> list[dict]:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT question_idx, attempt_number, winner,
                  judge_total_a, judge_total_b, sem_sim_a, sem_sim_b,
                  model_agreement, verifier_judge_agreement,
                  wikipedia_score, wikidata_score
           FROM pipeline_runs"""
    ).fetchall()
    con.close()

    out = []
    for r in rows:
        winner = r["winner"]
        judge_total = r["judge_total_a"] if winner == "A" else r["judge_total_b"]
        sem_sim = r["sem_sim_a"] if winner == "A" else r["sem_sim_b"]
        out.append({
            "question_idx": r["question_idx"],
            "attempt_number": r["attempt_number"],
            "judge": None if judge_total is None else max(0.0, min(1.0, judge_total / 80.0)),
            "similarity": None if sem_sim is None else max(0.0, min(1.0, sem_sim)),
            "model_agreement": r["model_agreement"],
            "verifier_judge_agreement": r["verifier_judge_agreement"],
            "wikipedia": r["wikipedia_score"],
            "wikidata": r["wikidata_score"],
        })
    return out


def weighted_composite(row: dict, weights: dict, exclude: str | None = None) -> float | None:
    """Mirrors compute_weighted_score's missing-signal renormalization,
    plus an optional extra excluded signal (for leave-one-out testing)."""
    available = {
        k: row[k] for k in SIGNAL_KEYS
        if row[k] is not None and k != exclude
    }
    if not available:
        return None
    wsum = sum(weights[k] for k in available)
    return sum(available[k] * (weights[k] / wsum) for k in available)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def internal_consistency_score(rows: list[dict], weights: dict) -> float:
    """Average leave-one-signal-out correlation across all 6 signals."""
    correlations = []
    for held_out in SIGNAL_KEYS:
        xs, ys = [], []
        for row in rows:
            if row[held_out] is None:
                continue
            composite = weighted_composite(row, weights, exclude=held_out)
            if composite is None:
                continue
            xs.append(composite)
            ys.append(row[held_out])
        if len(xs) >= 30:
            correlations.append(pearson(xs, ys))
    return statistics.mean(correlations) if correlations else 0.0


def random_weight_vector() -> dict:
    """Dirichlet-like random weights summing to 1 across the 6 signals."""
    raw = [random.random() ** 1.5 for _ in SIGNAL_KEYS]  # power skews toward sparser vectors
    total = sum(raw)
    return {k: v / total for k, v in zip(SIGNAL_KEYS, raw)}


def sweep_weights(rows: list[dict], n_candidates: int = 3000) -> list[tuple[float, dict]]:
    random.seed(42)
    candidates = [DEFAULT_WEIGHTS] + [random_weight_vector() for _ in range(n_candidates)]
    scored = [(internal_consistency_score(rows, w), w) for w in candidates]
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def regeneration_validity(rows: list[dict], weights: dict, threshold: float) -> dict:
    """
    For questions with a 2nd attempt, check whether attempt 1 scoring below
    `threshold` predicts attempt 2 improving on the OTHER 5 signals
    (judge excluded, since it's the dominant weight and checking "did the
    judge score go up" would be closer to circular).
    """
    by_q: dict[int, list[dict]] = {}
    for row in rows:
        by_q.setdefault(row["question_idx"], []).append(row)

    flagged_and_improved = 0
    flagged_total = 0
    not_flagged_total = 0
    not_flagged_improved = 0

    for qidx, attempts in by_q.items():
        attempts.sort(key=lambda r: r["attempt_number"])
        if len(attempts) < 2:
            continue
        a1, a2 = attempts[0], attempts[1]
        score1 = weighted_composite(a1, weights)
        if score1 is None:
            continue

        def other_signal_avg(row):
            vals = [row[k] for k in ("similarity", "model_agreement", "verifier_judge_agreement", "wikipedia", "wikidata") if row[k] is not None]
            return statistics.mean(vals) if vals else None

        o1, o2 = other_signal_avg(a1), other_signal_avg(a2)
        if o1 is None or o2 is None:
            continue
        improved = o2 > o1

        if score1 < threshold:
            flagged_total += 1
            if improved:
                flagged_and_improved += 1
        else:
            not_flagged_total += 1
            if improved:
                not_flagged_improved += 1

    return {
        "flagged_total": flagged_total,
        "flagged_improve_rate": flagged_and_improved / flagged_total if flagged_total else None,
        "not_flagged_total": not_flagged_total,
        "not_flagged_improve_rate": not_flagged_improved / not_flagged_total if not_flagged_total else None,
    }


def sweep_threshold(rows: list[dict], weights: dict) -> list[tuple[float, dict]]:
    results = []
    for t in [round(0.50 + 0.02 * i, 2) for i in range(19)]:  # 0.50 to 0.86
        results.append((t, regeneration_validity(rows, weights, t)))
    return results


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} attempt rows from {DB_PATH.name}\n", flush=True)

    for k in SIGNAL_KEYS:
        n_avail = sum(1 for r in rows if r[k] is not None)
        print(f"  {k}: {n_avail}/{len(rows)} available ({n_avail/len(rows)*100:.0f}%)", flush=True)
    print(flush=True)

    print("=" * 70)
    print("WEIGHT SWEEP (3000 random candidates + current default)")
    print("=" * 70, flush=True)
    ranked = sweep_weights(rows)
    default_score = internal_consistency_score(rows, DEFAULT_WEIGHTS)
    print(f"Current DEFAULT_WEIGHTS internal-consistency score: {default_score:.4f}")
    print(f"  {DEFAULT_WEIGHTS}\n")

    print("Top 5 candidates found:")
    for score, w in ranked[:5]:
        pretty = {k: round(v, 3) for k, v in w.items()}
        print(f"  score={score:.4f}  {pretty}")
    print(flush=True)

    best_score, best_weights = ranked[0]
    print(f"\nBest candidate improves internal consistency by {best_score - default_score:+.4f} over default.\n")

    print("=" * 70)
    print("THRESHOLD SWEEP (using current DEFAULT_WEIGHTS, for continuity)")
    print("=" * 70, flush=True)
    thresh_results = sweep_threshold(rows, DEFAULT_WEIGHTS)
    print(f"{'threshold':>10} {'n_flagged':>10} {'flagged_improve%':>18} {'n_ok':>8} {'ok_improve%':>14}")
    for t, res in thresh_results:
        fi = f"{res['flagged_improve_rate']*100:.1f}" if res['flagged_improve_rate'] is not None else "n/a"
        oi = f"{res['not_flagged_improve_rate']*100:.1f}" if res['not_flagged_improve_rate'] is not None else "n/a"
        print(f"{t:>10.2f} {res['flagged_total']:>10} {fi:>18} {res['not_flagged_total']:>8} {oi:>14}")

    print(f"\n(Current DEFAULT_THRESHOLD = {DEFAULT_THRESHOLD})")


if __name__ == "__main__":
    main()
