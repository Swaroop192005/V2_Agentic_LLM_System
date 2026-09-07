"""
training/threshold_confidence_analysis.py
==========================================
Extends threshold_weight_sweep.py's regeneration-validity check with actual
statistical rigor: Wilson score confidence intervals on each band's
retry-improvement rate, and a two-proportion z-test between the two control
bands, instead of eyeballing raw percentages.

Rationale for why this is needed on top of the existing sweep: with only a
few dozen control samples per band, a 64.5% vs 40.9% split LOOKS like a big
gap, but small-n proportions are noisy - this script quantifies exactly how
noisy, so "does 0.75 beat 0.68" gets a defensible yes/no/not-yet answer
instead of a gut read.

Run:
    python training/threshold_confidence_analysis.py
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.scoring import DEFAULT_WEIGHTS, DEFAULT_THRESHOLD
from training.threshold_weight_sweep import load_rows, weighted_composite


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion - more reliable
    than the naive normal-approximation interval at small n (never goes
    below 0 or above 1, and doesn't collapse to a zero-width interval when
    the observed rate is 0% or 100%)."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def two_proportion_z_test(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    """Two-tailed z-test for a difference in two independent proportions.
    Returns (z, p_value). Uses the pooled proportion under the null that
    both groups share the same true improvement rate."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    # two-tailed p-value from the standard normal survival function
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return (z, p_value)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def other_signal_avg(row: dict) -> float | None:
    vals = [row[k] for k in ("similarity", "model_agreement", "verifier_judge_agreement", "wikipedia", "wikidata") if row[k] is not None]
    return statistics.mean(vals) if vals else None


def classify_band(score: float, edges: list[float]) -> int:
    """Which band index `score` falls into, given sorted band edges."""
    for i in range(len(edges) - 1):
        if edges[i] <= score < edges[i + 1]:
            return i
    return len(edges) - 2  # top band, inclusive of the final edge


def main():
    rows = load_rows()
    by_q: dict[int, list[dict]] = {}
    for row in rows:
        by_q.setdefault(row["question_idx"], []).append(row)

    # Bands: below-threshold get finer granularity since that's where most
    # of the (non-control) retry data lives; above-threshold bands match
    # the control-group read from RESEARCH_LOG.md Section 22 for continuity.
    edges = [0.0, 0.55, 0.62, 0.68, 0.75, 1.01]
    labels = ["< 0.55", "0.55 - 0.62", "0.62 - 0.68 (sub-threshold)", "0.68 - 0.75 (control)", "0.75+ (control)"]

    band_success = [0] * (len(edges) - 1)
    band_total = [0] * (len(edges) - 1)
    band_regen_reasons: list[set] = [set() for _ in range(len(edges) - 1)]

    for qidx, attempts in by_q.items():
        attempts.sort(key=lambda r: r["attempt_number"])
        if len(attempts) < 2:
            continue
        a1, a2 = attempts[0], attempts[1]
        score1 = weighted_composite(a1, DEFAULT_WEIGHTS)
        if score1 is None:
            continue
        o1, o2 = other_signal_avg(a1), other_signal_avg(a2)
        if o1 is None or o2 is None:
            continue
        improved = o2 > o1

        b = classify_band(score1, edges)
        band_total[b] += 1
        if improved:
            band_success[b] += 1
        band_regen_reasons[b].add(a1.get("regen_reason"))

    print("=" * 78)
    print(f"RETRY-IMPROVEMENT RATE BY ATTEMPT-1 SCORE BAND  (n={sum(band_total)} questions with a 2nd attempt)")
    print("=" * 78)
    print(f"{'band':<30} {'n':>5} {'improve%':>10} {'95% CI (Wilson)':>20}")
    results = []
    for i, label in enumerate(labels):
        n, s = band_total[i], band_success[i]
        rate = s / n if n else None
        lo, hi = wilson_ci(s, n)
        results.append((label, n, s, rate, lo, hi))
        rate_s = f"{rate*100:.1f}%" if rate is not None else "n/a"
        ci_s = f"[{lo*100:.1f}, {hi*100:.1f}]" if n else "n/a"
        print(f"{label:<30} {n:>5} {rate_s:>10} {ci_s:>20}")

    print()
    print("-" * 78)
    print("SIGNIFICANCE CHECK: is the 0.75+ band's rate really different from")
    print("the 0.68-0.75 band's rate, or could this be noise at this sample size?")
    print("-" * 78)
    # indices: 0.68-0.75 is index 3, 0.75+ is index 4 in `labels`/`results`
    _, n_mid, s_mid, rate_mid, *_ = results[3]
    _, n_hi, s_hi, rate_hi, *_ = results[4]
    if n_mid and n_hi:
        z, p = two_proportion_z_test(s_mid, n_mid, s_hi, n_hi)
        print(f"  0.68-0.75 band: {s_mid}/{n_mid} = {rate_mid*100:.1f}% improve")
        print(f"  0.75+     band: {s_hi}/{n_hi} = {rate_hi*100:.1f}% improve")
        print(f"  z = {z:.3f}, two-tailed p = {p:.4f}")
        if p < 0.05:
            print("  -> statistically significant at alpha=0.05: the drop-off at 0.75 is unlikely to be noise.")
        else:
            print("  -> NOT statistically significant at alpha=0.05: with this sample size, a gap this large")
            print("     is still plausibly noise. More control samples are needed before treating this as settled.")
    else:
        print("  Not enough control-band data yet to run this test.")

    print()
    print("-" * 78)
    print("REQUIRED SAMPLE SIZE CHECK")
    print("-" * 78)
    # Rough power-style estimate: how large would each control band need to be
    # for a gap of this observed magnitude to reliably clear p<0.05, assuming
    # the true rates are close to what's observed now (illustrative, not a
    # formal power analysis).
    if n_mid and n_hi and rate_mid is not None and rate_hi is not None:
        gap = abs(rate_mid - rate_hi)
        avg_p = (rate_mid + rate_hi) / 2
        # solve for n per group such that z ~= 1.96 given the current gap and pooled variance shape
        if gap > 0:
            se_per_unit_n = math.sqrt(2 * avg_p * (1 - avg_p))
            n_needed = ((1.96 * se_per_unit_n) / gap) ** 2
            print(f"  Observed gap: {gap*100:.1f} points. At this gap size, reaching significance would need")
            print(f"  roughly {n_needed:.0f} samples PER band (currently {n_mid} and {n_hi}).")
        else:
            print("  Observed gap is ~0 - no amount of additional data would show a difference here.")

    print()
    print(f"(Current DEFAULT_THRESHOLD = {DEFAULT_THRESHOLD})")


if __name__ == "__main__":
    main()
