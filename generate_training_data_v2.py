"""
generate_training_data_v2.py
=============================
Fresh, clean training-dataset generator for the CURRENT pipeline design:
  - Verifier (phi3) and Judge (qwen2.5) both independently score all 8
    rubric factors (agents/rubric.py) — not the old 5-factor judge-only
    rubric, and not phi3 free-form prose.
  - Mirrors the live server.py pipeline EXACTLY, including the regeneration
    loop (retry up to MAX_REGENERATION_ATTEMPTS times when the weighted
    confidence score is below DEFAULT_THRESHOLD) — every attempt is logged,
    not just the final one, so the threshold/weight-sweep phase has real
    data to analyze ("did regenerating actually help?").
  - Records every raw signal (judge/similarity/model_agreement/
    verifier_judge_agreement/wikipedia/wikidata) separately, so the weighted
    formula can be recomputed with ANY weight combination later without
    regenerating anything.

Writes to a NEW, separate database (data/judge_training_v2.db) so this
dataset can never mix with the old (partially inconsistent-judge) one.

Resumable: already-fully-processed questions are skipped. Crash-safe: each
attempt is committed immediately.

Run:
    python generate_training_data_v2.py
    python generate_training_data_v2.py --limit 10      # smoke test
"""

from __future__ import annotations

import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass

import argparse
import asyncio
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agents.llm_agents import create_llama3_agent, create_mistral_agent
from agents.verifier_agent import create_verifier_agent, build_verifier_prompt
from agents.judge_agent import create_judge_agent, build_judge_prompt
from agents.combiner_agent import create_combiner_agent, build_combiner_prompt
from agents.rubric import SLUGS
from utils.similarity import score_response
from utils.rubric_parser import parse_rubric_scores, parse_judge_scores, compute_agreement
from utils.eval_metrics import compute_bertscore_batch
from utils.fact_check import verify_against_wikipedia
from utils.wikidata_check import verify_against_wikidata
from utils.scoring import compute_weighted_score, DEFAULT_THRESHOLD, MAX_REGENERATION_ATTEMPTS

# Reuse the existing 5000-question bank rather than duplicating it.
from generate_training_data import QUESTIONS  # noqa: E402

DB_PATH = ROOT / "data" / "judge_training_v2.db"
DB_PATH.parent.mkdir(exist_ok=True)
LOG_PATH = ROOT / "training_log_v2.txt"
# Lowered from 2 -> 1: each question needs up to 4 different ~6-9GB models
# (llama3/mistral/phi3/qwen2.5) cycling through a single 23GB GPU. Running
# 2 questions concurrently was over-subscribing VRAM, causing recurring
# orphaned-process stalls every 30-60 min (each costing several minutes of
# auto-recovery). Trading parallelism for stability - fewer/no stalls should
# net out faster than 2x parallel throughput interrupted every half hour.
CONCURRENCY = 1


def init_db(conn: sqlite3.Connection) -> None:
    factor_cols = []
    for prefix in ("verifier", "judge"):
        for slug in SLUGS:
            factor_cols.append(f"{prefix}_{slug}_a INTEGER")
            factor_cols.append(f"{prefix}_{slug}_b INTEGER")
        factor_cols.append(f"{prefix}_total_a INTEGER")
        factor_cols.append(f"{prefix}_total_b INTEGER")

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_idx INTEGER,
            question TEXT,
            attempt_number INTEGER,
            is_final_attempt INTEGER,
            answer_a TEXT, answer_b TEXT,
            verifier_raw TEXT, judge_raw TEXT,
            {", ".join(factor_cols)},
            winner TEXT,
            sem_sim_a REAL, sem_sim_b REAL,
            length_score_a REAL, length_score_b REAL,
            word_count_a INTEGER, word_count_b INTEGER,
            model_a TEXT, model_b TEXT,
            model_agreement REAL,
            verifier_judge_agreement REAL,
            wikipedia_score REAL, wikipedia_source TEXT,
            wikidata_score REAL, wikidata_source TEXT,
            weighted_score REAL,
            final_answer TEXT,
            elapsed_secs REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pr_question ON pipeline_runs(question)")
    conn.commit()


def already_done(conn: sqlite3.Connection, question: str) -> bool:
    """A question is done if it has a row marked as the final attempt."""
    row = conn.execute(
        "SELECT id FROM pipeline_runs WHERE question = ? AND is_final_attempt = 1", (question,)
    ).fetchone()
    return row is not None


async def run_one_attempt(query: str, rag_context: str = "") -> dict:
    """Run stages 1-5 once (one attempt) and return everything needed to
    score it and decide whether to regenerate. Mirrors server.py exactly."""
    loop = asyncio.get_event_loop()
    t0 = time.time()

    llama3_agent = create_llama3_agent()
    mistral_agent = create_mistral_agent()
    verifier_agent = create_verifier_agent()
    judge_agent = create_judge_agent()

    llama_prompt = (
        f"Answer the following question thoroughly and accurately.\n\nQuestion: {query}\n\n"
        f"Provide a well-structured, comprehensive answer of at least 150 words."
    )
    mistral_prompt = (
        f"Answer the following question thoroughly and accurately.\n\nQuestion: {query}\n\n"
        f"Provide a well-structured, comprehensive answer of at least 150 words. "
        f"Include practical examples and unique insights."
    )

    llama3_text, mistral_text = await asyncio.gather(
        loop.run_in_executor(None, llama3_agent.run, llama_prompt),
        loop.run_in_executor(None, mistral_agent.run, mistral_prompt),
    )

    # Randomize which physical model lands in position A vs B, per question.
    # Earlier data always had LLaMA 3 = position A and Mistral = position B,
    # AND revealed model identity to the Verifier/Judge in the prompt itself
    # - a full position+identity confound. Win rates differed by judge model
    # in a way that flipped direction (qwen2.5 favored A ~60-66%, CompassJudger
    # favored B ~58-72%), which is the signature of bias, not a genuine
    # LLaMA-vs-Mistral quality gap. See RESEARCH_LOG.md Section 17. Position
    # is tracked via model_a/model_b so true per-model stats stay recoverable.
    if random.random() < 0.5:
        answer_a, answer_b = llama3_text, mistral_text
        model_a, model_b = "llama3", "mistral"
    else:
        answer_a, answer_b = mistral_text, llama3_text
        model_a, model_b = "mistral", "llama3"

    verifier_prompt = build_verifier_prompt(query, answer_a, answer_b, rag_context=rag_context)
    (verification, (scores_a, scores_b)) = await asyncio.gather(
        loop.run_in_executor(None, verifier_agent.run, verifier_prompt),
        asyncio.gather(
            loop.run_in_executor(None, score_response, query, answer_a),
            loop.run_in_executor(None, score_response, query, answer_b),
        ),
    )

    # Judge no longer receives the Verifier's scores (scores blind - see
    # agents/judge_agent.py::build_judge_prompt) but still runs AFTER the
    # Verifier sequentially, not concurrently, to avoid loading both models'
    # weights in VRAM at once (see CONCURRENCY=1 note below - the project
    # already hit VRAM over-subscription stalls from simultaneous models once).
    judge_prompt = build_judge_prompt(query, answer_a, answer_b)
    judgement = await loop.run_in_executor(None, judge_agent.run, judge_prompt)

    verifier_parsed = parse_rubric_scores(verification)
    judge_parsed = parse_judge_scores(judgement, query, answer_a, answer_b)
    winner = judge_parsed["winner"]
    total_winner = judge_parsed["total_a"] if winner == "A" else judge_parsed["total_b"]
    sim_winner = scores_a["semantic_similarity"] if winner == "A" else scores_b["semantic_similarity"]
    verifier_judge_agreement = compute_agreement(verifier_parsed, judge_parsed)
    winner_answer = answer_a if winner == "A" else answer_b

    async def _safe(fn, *args):
        try:
            return await loop.run_in_executor(None, fn, *args)
        except Exception:
            return None

    bs_task = loop.run_in_executor(None, compute_bertscore_batch, [answer_a], [answer_b])
    wiki_task = _safe(verify_against_wikipedia, query, winner_answer)
    wikidata_task = _safe(verify_against_wikidata, query, winner_answer)

    try:
        bs = await bs_task
        model_agreement = bs["f1"][0]
    except Exception:
        model_agreement = None
    wikipedia_result = await wiki_task
    wikidata_result = await wikidata_task
    wikipedia_score = wikipedia_result["score"] if wikipedia_result else None
    wikidata_score = wikidata_result["score"] if wikidata_result else None

    breakdown = compute_weighted_score(
        judge_total=total_winner,
        similarity_0_1=sim_winner,
        model_agreement_0_1=model_agreement,
        verifier_judge_agreement_0_1=verifier_judge_agreement,
        wikipedia_0_1=wikipedia_score,
        wikidata_0_1=wikidata_score,
    )

    return {
        "answer_a": answer_a, "answer_b": answer_b,
        "model_a": model_a, "model_b": model_b,
        "verifier_raw": verification, "judge_raw": judgement,
        "verifier_parsed": verifier_parsed, "judge_parsed": judge_parsed,
        "winner": winner,
        "scores_a": scores_a, "scores_b": scores_b,
        "model_agreement": model_agreement,
        "verifier_judge_agreement": verifier_judge_agreement,
        "wikipedia_score": wikipedia_score, "wikipedia_source": wikipedia_result["source_title"] if wikipedia_result else None,
        "wikidata_score": wikidata_score, "wikidata_source": wikidata_result["source_title"] if wikidata_result else None,
        "weighted_score": breakdown.weighted_score,
        "elapsed_secs": time.time() - t0,
    }


def insert_attempt(conn: sqlite3.Connection, question_idx: int, question: str,
                    attempt_number: int, is_final: bool, r: dict, final_answer: str | None) -> None:
    vp, jp = r["verifier_parsed"], r["judge_parsed"]
    cols = ["question_idx", "question", "attempt_number", "is_final_attempt",
            "answer_a", "answer_b", "model_a", "model_b", "verifier_raw", "judge_raw"]
    vals = [question_idx, question, attempt_number, int(is_final),
            r["answer_a"], r["answer_b"], r["model_a"], r["model_b"], r["verifier_raw"], r["judge_raw"]]

    for prefix, parsed in (("verifier", vp), ("judge", jp)):
        for slug in SLUGS:
            cols += [f"{prefix}_{slug}_a", f"{prefix}_{slug}_b"]
            vals += [parsed[f"{slug}_a"], parsed[f"{slug}_b"]]
        cols += [f"{prefix}_total_a", f"{prefix}_total_b"]
        vals += [parsed["total_a"], parsed["total_b"]]

    cols += ["winner", "sem_sim_a", "sem_sim_b", "length_score_a", "length_score_b",
             "word_count_a", "word_count_b", "model_agreement", "verifier_judge_agreement",
             "wikipedia_score", "wikipedia_source", "wikidata_score", "wikidata_source",
             "weighted_score", "final_answer", "elapsed_secs", "created_at"]
    vals += [r["winner"], r["scores_a"]["semantic_similarity"], r["scores_b"]["semantic_similarity"],
              r["scores_a"]["length_score"], r["scores_b"]["length_score"],
              r["scores_a"]["word_count"], r["scores_b"]["word_count"],
              r["model_agreement"], r["verifier_judge_agreement"],
              r["wikipedia_score"], r["wikipedia_source"], r["wikidata_score"], r["wikidata_source"],
              r["weighted_score"], final_answer, r["elapsed_secs"], datetime.now().isoformat()]

    placeholders = ",".join(["?"] * len(vals))
    conn.execute(f"INSERT INTO pipeline_runs ({','.join(cols)}) VALUES ({placeholders})", vals)
    conn.commit()


async def process_question(idx: int, question: str, conn: sqlite3.Connection, sem: asyncio.Semaphore,
                            log_lines: list) -> None:
    async with sem:
        t_start = time.time()
        best = None
        best_attempt_number = None
        for attempt in range(1, MAX_REGENERATION_ATTEMPTS + 2):
            r = await run_one_attempt(question)
            if best is None or r["weighted_score"] > best["weighted_score"]:
                best = r
                best_attempt_number = attempt
            insert_attempt(conn, idx, question, attempt, is_final=False, r=r, final_answer=None)

            if r["weighted_score"] >= DEFAULT_THRESHOLD or attempt > MAX_REGENERATION_ATTEMPTS:
                break

        # Combiner runs once, on the best attempt
        combiner_agent = create_combiner_agent()
        loop = asyncio.get_event_loop()
        combiner_prompt = build_combiner_prompt(
            question, best["answer_a"], best["answer_b"], best["verifier_raw"], best["judge_raw"]
        )
        final_answer = await loop.run_in_executor(None, combiner_agent.run, combiner_prompt)

        # Mark the winning attempt's row as final - question_idx + attempt_number uniquely
        # identifies it even if the question text happens to repeat elsewhere in the bank.
        conn.execute(
            """UPDATE pipeline_runs SET is_final_attempt = 1, final_answer = ?
               WHERE question_idx = ? AND attempt_number = ?""",
            (final_answer, idx, best_attempt_number),
        )
        conn.commit()

        elapsed = time.time() - t_start
        msg = (f"  + [{idx}] {question[:55]}... ({elapsed:.0f}s | winner={best['winner']} | "
               f"score={best['weighted_score']:.3f} | judge={best['judge_parsed']['total_a']}/{best['judge_parsed']['total_b']})")
        print(msg, flush=True)
        log_lines.append(msg)
        with open(LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")


async def main(limit: int | None) -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    init_db(conn)

    questions = QUESTIONS[:limit] if limit else QUESTIONS
    pending = [(i, q) for i, q in enumerate(questions, start=1) if not already_done(conn, q)]
    print(f"Target: {len(questions)} questions | Already done: {len(questions) - len(pending)} | "
          f"Pending: {len(pending)} | DB: {DB_PATH}", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    log_lines: list = []
    tasks = [process_question(i, q, conn, sem, log_lines) for i, q in pending]
    await asyncio.gather(*tasks)

    conn.close()
    print(f"\nDone. {len(pending)} questions processed this run.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="only process the first N questions (for testing)")
    args = p.parse_args()
    asyncio.run(main(args.limit))
