"""
training/verifier_discrimination_test.py
==========================================
The decisive discrimination test from Section 9 (originally run against the
Judge, back when qwen2.5/CompassJudger-1 were under investigation), now run
against the Verifier (phi3) - which has never been tested this way. Motivated
by Section 29: the Verifier shows ~0 score gap between LLaMA3 and Mistral on
every rubric criterion, in flat contradiction to the Judge's real gap. Two
explanations were live: (a) the Judge's gap is a style-driven illusion, or
(b) the Verifier's known near-ceiling clustering (Section 18) makes it
unable to detect ANY real quality gap, model-identity or otherwise.

This test settles (b): give the Verifier one genuinely solid answer and one
genuinely, deliberately broken answer (factually wrong, self-contradictory,
irrelevant claims) for the same question, scored via the exact production
prompt/rubric. If the Verifier still can't tell them apart, its ~0 gap on
LLaMA3-vs-Mistral is explained by insensitivity, not genuine model parity -
if it discriminates clearly, the flat LLaMA3/Mistral result becomes a more
trustworthy "no real difference" reading instead.

Run:
    python training/verifier_discrimination_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.verifier_agent import create_verifier_agent, build_verifier_prompt
from utils.rubric_parser import parse_rubric_scores

CASES = [
    {
        "question": "What is the difference between machine learning and deep learning?",
        "solid": (
            "Machine learning (ML) is a broad field of AI where systems learn patterns from data "
            "instead of following explicitly programmed rules, using algorithms like decision trees, "
            "support vector machines, and linear regression. Deep learning (DL) is a subfield of ML "
            "that specifically uses multi-layered artificial neural networks to automatically learn "
            "hierarchical feature representations from raw data, such as pixels or audio waveforms. "
            "The key distinction is that classical ML often requires manual feature engineering by a "
            "human expert, while deep learning learns those features itself given enough data and "
            "compute. Deep learning tends to need much larger datasets and more computational power "
            "(often GPUs) than traditional ML methods, but can achieve state-of-the-art results on "
            "complex tasks like image recognition and natural language processing where manually "
            "designing features would be impractical."
        ),
        "broken": (
            "Machine learning was invented in 2015 by a major search engine company as a direct "
            "replacement for the internet itself. Deep learning is actually a type of database "
            "software used exclusively for storing digital photographs, unrelated to artificial "
            "intelligence. Neural networks are physical brain implants that have been surgically "
            "installed in human volunteers since the 1800s to improve memory. Every modern AI system "
            "runs exclusively on specialized quantum computers that are only manufactured in "
            "Switzerland and cost over one billion dollars each. Machine learning models require no "
            "data or training process whatsoever, since they operate purely through a form of "
            "telepathic communication with their programmers."
        ),
    },
    {
        "question": "What is CRISPR-Cas9 gene editing and how does it work?",
        "solid": (
            "CRISPR-Cas9 is a gene-editing technology adapted from a natural bacterial immune system "
            "that defends against viruses. It uses a guide RNA molecule to direct the Cas9 protein, "
            "an enzyme that acts like molecular scissors, to a specific target sequence in a genome. "
            "Once Cas9 binds the matching DNA sequence, it creates a precise double-strand break. The "
            "cell's own repair machinery then fixes this break, either by disrupting the gene "
            "(non-homologous end joining) or by inserting a new, desired sequence if a DNA template is "
            "supplied (homology-directed repair). This has made CRISPR-Cas9 a fast, relatively "
            "inexpensive tool for research, agriculture, and experimental gene therapies, though "
            "off-target edits and delivery into the right cells remain active challenges."
        ),
        "broken": (
            "CRISPR-Cas9 was discovered by a national space agency in 1995 as an unexpected byproduct "
            "of the Mars rover program. It functions as an oral vaccine that is swallowed to prevent "
            "the common cold. The term 'Cas9' stands for 'Computer Algorithm System 9' and refers to "
            "a piece of software, not a biological protein at all. Gene editing with this technology "
            "requires no DNA whatsoever - it works purely through targeted ultrasonic sound waves "
            "aimed at the patient's skin. Due to catastrophic safety failures, CRISPR-Cas9 has been "
            "completely banned in every country worldwide since 2010."
        ),
    },
]


def main():
    verifier = create_verifier_agent()
    print(f"Verifier model in use: {verifier.model_name if hasattr(verifier, 'model_name') else 'phi3 (see agents/verifier_agent.py)'}\n")

    results = []
    for i, case in enumerate(CASES, 1):
        print(f"--- Case {i}: {case['question'][:60]}... ---", flush=True)
        # Answer 1 = solid, Answer 2 = broken (fixed order here since this is
        # a discrimination probe, not a production run - no position bias
        # concern when both "sides" are deliberately unequal by construction).
        prompt = build_verifier_prompt(case["question"], case["solid"], case["broken"])
        raw = verifier.run(prompt)
        parsed = parse_rubric_scores(raw)
        solid_total, broken_total = parsed["total_a"], parsed["total_b"]
        gap = solid_total - broken_total
        results.append((case["question"], solid_total, broken_total, gap))
        print(f"  Solid answer total:  {solid_total}/80")
        print(f"  Broken answer total: {broken_total}/80")
        print(f"  Gap: {gap:+d}\n")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Question':<45}{'Solid':>8}{'Broken':>8}{'Gap':>8}")
    for q, s, b, g in results:
        print(f"{q[:43]:<45}{s:>8}{b:>8}{g:>+8}")

    avg_gap = sum(g for _, _, _, g in results) / len(results)
    print(f"\nAverage gap: {avg_gap:+.1f} / 80")
    if avg_gap < 10:
        print("-> Verifier does NOT meaningfully discriminate even a deliberately broken")
        print("   answer from a solid one. Its ~0 LLaMA3-vs-Mistral gap (Section 29) is")
        print("   explained by insensitivity, not genuine model parity.")
    else:
        print("-> Verifier DOES discriminate a genuinely broken answer from a solid one.")
        print("   Its ~0 LLaMA3-vs-Mistral gap is then a more trustworthy 'no real")
        print("   difference' reading, not a measurement-resolution artifact.")


if __name__ == "__main__":
    main()
