"""
training/rag_threshold_spotcheck.py
====================================
NOT a calibration - a sanity spot-check. Unlike DEFAULT_THRESHOLD (Sections
20-24), MIN_SIMILARITY (utils/rag_store.py, gates RAG chunk retrieval) has
zero usage-outcome data to calibrate against: the 5000-question batch
generator never calls retrieve_context() at all (rag_context defaults to
"" and generate_training_data_v2.py::process_question never overrides it),
so nothing in judge_training_v2.db reflects a RAG decision. The live demo
(server.py) does use it, but logs no retrieval outcome anywhere persistent.

This script substitutes the best available check given that gap: run real
embeddings against the actual, currently-populated knowledge base
(data/rag_knowledge.db) for a deliberately mixed set of on-topic and
off-topic test queries, and see whether MIN_SIMILARITY actually separates
them. It's a plausibility check on real content, not a statistical proof -
report it as such.

Run:
    python training/rag_threshold_spotcheck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.rag_store import get_rag_stats, retrieve_context, MIN_SIMILARITY, RAG_DB_PATH
from utils.similarity import _get_model
import sqlite3
import numpy as np

# One query per actual document in the knowledge base (should score HIGH -
# these are deliberately on-topic for real, currently-indexed content).
ON_TOPIC = [
    "What is Retrieval-Augmented Generation and how does it reduce hallucinations?",
    "Explain the 6-stage architecture of this multi-LLM validation system.",
    "What does quantum entanglement mean in the attached physics paper?",
    "Summarize the bioinformatics perspective document.",
    "What are the planets and structure of our solar system?",
]

# Queries with no matching document in the current knowledge base at all -
# should score LOW and ideally get dropped by the relevance gate.
OFF_TOPIC = [
    "What is the best recipe for chocolate chip cookies?",
    "How do I change a flat tire on a car?",
    "What's the plot of the movie Inception?",
    "Give me tips for training for a marathon.",
    "What's a good name for a pet goldfish?",
]


def main():
    stats = get_rag_stats()
    print(f"Knowledge base: {stats['doc_count']} documents, {stats['chunk_count']} chunks")
    print(f"Current MIN_SIMILARITY = {MIN_SIMILARITY}\n")

    print("=" * 78)
    print(f"{'query':<62} {'top sim':>8}  {'passes gate?'}")
    print("=" * 78)

    on_scores, off_scores = [], []
    for label, queries, bucket in [("ON-TOPIC", ON_TOPIC, on_scores), ("OFF-TOPIC", OFF_TOPIC, off_scores)]:
        print(f"\n-- {label} (should {'PASS' if label=='ON-TOPIC' else 'FAIL'} the gate) --")
        for q in queries:
            # retrieve with min_similarity=0.0 to see the RAW top score before gating
            hits = retrieve_context(q, top_k=1, min_similarity=0.0, log=False)
            top_sim = hits[0]["similarity"] if hits else 0.0
            bucket.append(top_sim)
            passes = "YES" if top_sim >= MIN_SIMILARITY else "no"
            print(f"{q[:60]:<62} {top_sim:>8.3f}  {passes}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    on_correct = sum(1 for s in on_scores if s >= MIN_SIMILARITY)
    off_correct = sum(1 for s in off_scores if s < MIN_SIMILARITY)
    print(f"On-topic queries correctly PASSED the gate:  {on_correct}/{len(on_scores)}  (scores: {[round(s,3) for s in on_scores]})")
    print(f"Off-topic queries correctly BLOCKED by gate:  {off_correct}/{len(off_scores)}  (scores: {[round(s,3) for s in off_scores]})")
    if on_scores and off_scores:
        gap = min(on_scores) - max(off_scores)
        print(f"\nGap between weakest on-topic score ({min(on_scores):.3f}) and strongest off-topic score ({max(off_scores):.3f}): {gap:+.3f}")
        if gap > 0:
            print(f"  -> On this spot-check, ANY threshold between {max(off_scores):.3f} and {min(on_scores):.3f} would cleanly separate the two groups.")
            print(f"     Current MIN_SIMILARITY={MIN_SIMILARITY} sits {'inside' if max(off_scores) < MIN_SIMILARITY < min(on_scores) else 'OUTSIDE'} that safe range.")
        else:
            print(f"  -> Overlap exists on this small sample - not clean separation, worth a larger check before trusting either direction.")

    print("\nCaveat: this is 10 hand-picked queries against 7 documents, not a")
    print("statistical calibration. It answers 'is 0.35 obviously broken?', not")
    print("'is 0.35 optimal?'. A real calibration needs the live demo to log")
    print("(query, retrieved chunk, similarity, was-it-actually-useful) so this")
    print("can be swept the way DEFAULT_THRESHOLD was - see recommendation below.")


if __name__ == "__main__":
    main()
