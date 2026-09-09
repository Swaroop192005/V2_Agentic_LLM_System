"""
training/optimal_threshold_roc.py
==================================
Proves (not just suggests) an optimal DEFAULT_THRESHOLD using standard
binary-classifier-cutoff methodology, instead of eyeballing two candidate
bands (as Sections 22-23 did).

Framing: attempt 1's weighted_score is a PREDICTOR. The thing it's trying
to predict is "would this answer actually improve if given a 2nd attempt?"
(measured on the OTHER 5 signals, judge excluded, to avoid circularity -
same definition as every prior section).

IMPORTANT CORRECTION (found while building this - see RESEARCH_LOG Section
24): a first version of this script computed sensitivity/specificity/AUC
directly from the raw labeled rows and got a nonsensical answer (AUC~0.51,
"optimal" cutoff BELOW the current threshold) that contradicted the clean,
significant band-comparison result from Section 23. The cause was a real
statistical error, not noise: this dataset is NOT a simple random sample.
Every sub-threshold attempt gets regenerated (100% census, sampling
probability 1.0), but only 8% of at-or-above-threshold attempts do
(CONTROL_REGEN_PROBABILITY, Section 21) - a classic case-control /
stratified-sampling design. Treating a control-sample row as equally
"representative" as a census row silently undercounts the true population
above threshold by ~12.5x, which corrupts sensitivity, specificity, and AUC
(all population-composition-dependent) even though it does NOT bias the
simple per-band conditional rates Section 23 used (those don't depend on
how many OTHER bands got sampled). Fixed with inverse-probability-of-
-sampling weighting: every low_confidence-derived row counts as weight 1.0,
every control_sample-derived row counts as weight 1/CONTROL_REGEN_PROBABILITY
(=12.5) - the standard correction for exactly this stratified-sampling
situation in diagnostic-test validation statistics.

Method:
1. ROC curve: at every candidate cutoff t, a "positive prediction" is
   score1 < t (i.e. "this needs a regen"). Sensitivity = weighted fraction
   of truly-improvable answers correctly caught; specificity = weighted
   fraction of truly-fine answers correctly left alone.
2. AUC: overall measure of whether the score is a meaningful predictor of
   "will this improve on retry" at all, independent of any specific cutoff.
3. Youden's J = sensitivity + specificity - 1, maximized at the cutoff that
   best separates the two groups - the standard, textbook way to pick an
   "optimal" threshold from a continuous score, not a subjective call.
4. Bootstrap (2000 resamples, weighted) for a confidence interval on WHERE
   the optimal cutoff falls - proof the answer is stable, not an artifact
   of this particular sample.

Run:
    python training/optimal_threshold_roc.py
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

from training.threshold_weight_sweep import DB_PATH, weighted_composite

OTHER_SIGNALS = ("similarity", "model_agreement", "verifier_judge_agreement", "wikipedia", "wikidata")
MIN_WEIGHTED_BUCKET = 15  # below this (in effective/weighted n), a cutoff is too noisy to trust
CENSUS_WEIGHT = 1.0

# CONTROL_REGEN_PROBABILITY changed mid-dataset (0.08 -> 0.60, RESEARCH_LOG
# Section 32) - importing the CURRENT value and applying it to every
# control_sample row would silently mis-weight the ~117 rows sampled under
# the OLD 8% rate (they'd be undercounted at 1.67x instead of their true
# 12.5x), diluting exactly the data that originally justified 0.75 and
# pulling the "optimal" cutoff back toward the low_confidence-dominated
# result the very first (uncorrected) version of this script produced.
# Boundary found empirically: the fraction of PASSING attempt-1 rows that
# were actually marked control_sample (undiluted by the failing
# population, unlike the raw per-question rate) sits at 0-20% noise before
# question_idx ~4650 and jumps to 33-100% after - consistent with 8% vs
# 60%. Rows are weighted by whichever rate was actually active when they
# were generated, not by re-reading today's constant.
CONTROL_PROBABILITY_CHANGE_AT_QIDX = 4650
CONTROL_WEIGHT_OLD = 1.0 / 0.08   # 12.5
CONTROL_WEIGHT_NEW = 1.0 / 0.60   # ~1.667


def control_weight_for(question_idx: int) -> float:
    return CONTROL_WEIGHT_NEW if question_idx > CONTROL_PROBABILITY_CHANGE_AT_QIDX else CONTROL_WEIGHT_OLD


def load_rows_with_reason() -> list[dict]:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT question_idx, attempt_number, winner, regen_reason,
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
            "regen_reason": r["regen_reason"],
            "judge": None if judge_total is None else max(0.0, min(1.0, judge_total / 80.0)),
            "similarity": None if sem_sim is None else max(0.0, min(1.0, sem_sim)),
            "model_agreement": r["model_agreement"],
            "verifier_judge_agreement": r["verifier_judge_agreement"],
            "wikipedia": r["wikipedia_score"],
            "wikidata": r["wikidata_score"],
        })
    return out


def other_signal_avg(row: dict) -> float | None:
    vals = [row[k] for k in OTHER_SIGNALS if row[k] is not None]
    return statistics.mean(vals) if vals else None


def build_labeled_set(rows: list[dict]) -> list[tuple[float, bool, float]]:
    """One (score1, improved, sampling_weight) triple per question that
    ever got a 2nd attempt. weight corrects for the fact that
    control_sample rows represent only an 8% draw of their true population,
    while low_confidence rows are a 100% census of theirs."""
    by_q: dict[int, list[dict]] = {}
    for row in rows:
        by_q.setdefault(row["question_idx"], []).append(row)

    labeled = []
    for qidx, attempts in by_q.items():
        attempts.sort(key=lambda r: r["attempt_number"])
        if len(attempts) < 2:
            continue
        a1, a2 = attempts[0], attempts[1]
        score1 = weighted_composite(a1, DEFAULT_WEIGHTS)
        o1, o2 = other_signal_avg(a1), other_signal_avg(a2)
        if score1 is None or o1 is None or o2 is None:
            continue
        weight = control_weight_for(qidx) if a1.get("regen_reason") == "control_sample" else CENSUS_WEIGHT
        labeled.append((score1, o2 > o1, weight))
    return labeled


def roc_point(labeled: list[tuple[float, bool, float]], t: float) -> dict | None:
    tp = fn = fp = tn = 0.0
    raw_flagged = raw_ok = 0
    for score, improved, w in labeled:
        flagged = score < t
        raw_flagged += flagged
        raw_ok += not flagged
        if improved and flagged:
            tp += w
        elif improved and not flagged:
            fn += w
        elif not improved and flagged:
            fp += w
        else:
            tn += w
    if (tp + fn) < MIN_WEIGHTED_BUCKET or (tn + fp) < MIN_WEIGHTED_BUCKET:
        return None
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "t": t, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sensitivity, "specificity": specificity,
        "youden_j": sensitivity + specificity - 1,
        "raw_flagged": raw_flagged, "raw_ok": raw_ok,
    }


def compute_auc(labeled: list[tuple[float, bool, float]], grid: list[float]) -> float:
    points = []
    for t in grid:
        p = roc_point(labeled, t)
        if p:
            points.append((round(1 - p["specificity"], 6), p["sensitivity"]))
    points = sorted(set(points))
    if len(points) < 2:
        return 0.5
    auc = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        auc += (x2 - x1) * (y1 + y2) / 2
    return auc


def find_optimal(labeled: list[tuple[float, bool, float]], grid: list[float]) -> dict | None:
    best = None
    for t in grid:
        p = roc_point(labeled, t)
        if p and (best is None or p["youden_j"] > best["youden_j"]):
            best = p
    return best


def bootstrap_optimal(labeled: list[tuple[float, bool, float]], grid: list[float], n_boot: int = 2000) -> list[float]:
    random.seed(7)
    n = len(labeled)
    optimal_ts = []
    for _ in range(n_boot):
        sample = [labeled[random.randrange(n)] for _ in range(n)]
        best = find_optimal(sample, grid)
        if best:
            optimal_ts.append(best["t"])
    return optimal_ts


def main():
    rows = load_rows_with_reason()
    labeled = build_labeled_set(rows)
    n_improved_w = sum(w for _, imp, w in labeled if imp)
    n_total_w = sum(w for _, _, w in labeled)
    print(f"Labeled pairs: {len(labeled)} raw questions -> {n_total_w:.0f} effective (sampling-corrected) population")
    print(f"  effective improved: {n_improved_w:.0f}  ({n_improved_w/n_total_w*100:.1f}%)")
    print(f"  control rows carry weight {CONTROL_WEIGHT_OLD:.2f}x (qidx<={CONTROL_PROBABILITY_CHANGE_AT_QIDX}, 1/0.08) "
          f"or {CONTROL_WEIGHT_NEW:.2f}x (qidx>{CONTROL_PROBABILITY_CHANGE_AT_QIDX}, 1/0.60); census rows carry 1.0x\n")

    grid = [round(0.50 + 0.01 * i, 2) for i in range(37)]  # 0.50 .. 0.86

    print("=" * 96)
    print("SAMPLING-CORRECTED ROC CURVE  (t = candidate cutoff; 'flagged' = score1 < t)")
    print("=" * 96)
    print(f"{'t':>5} {'raw n_flag':>10} {'raw n_ok':>9} {'sensitivity':>12} {'specificity':>12} {'youden J':>10}")
    for t in grid:
        p = roc_point(labeled, t)
        if p is None:
            continue
        print(f"{t:>5.2f} {p['raw_flagged']:>10} {p['raw_ok']:>9} {p['sensitivity']*100:>11.1f}% {p['specificity']*100:>11.1f}% {p['youden_j']:>10.4f}")

    auc = compute_auc(labeled, grid)
    print(f"\nAUC (sampling-corrected; 0.5=random / 1.0=perfect): {auc:.4f}")

    best = find_optimal(labeled, grid)
    print("\n" + "=" * 96)
    if best:
        print(f"OPTIMAL CUTOFF BY YOUDEN'S J (sampling-corrected):  t = {best['t']:.2f}")
        print(f"  sensitivity={best['sensitivity']*100:.1f}%  specificity={best['specificity']*100:.1f}%  J={best['youden_j']:.4f}")
    else:
        print("Not enough data at any cutoff to identify an optimum yet.")
    print("=" * 96)

    print("\nBootstrap stability check (2000 resamples of the labeled question set, weights preserved):")
    boot_ts = bootstrap_optimal(labeled, grid)
    if boot_ts:
        boot_ts.sort()
        lo = boot_ts[int(0.025 * len(boot_ts))]
        hi = boot_ts[int(0.975 * len(boot_ts))]
        mode = statistics.mode(boot_ts)
        print(f"  Across {len(boot_ts)} valid resamples: median optimal t = {statistics.median(boot_ts):.2f}, "
              f"mode = {mode:.2f}, 95% bootstrap CI = [{lo:.2f}, {hi:.2f}]")
    else:
        print("  Not enough data for a bootstrap estimate yet.")

    print(f"\n(Current DEFAULT_THRESHOLD = {DEFAULT_THRESHOLD})")


if __name__ == "__main__":
    main()
