"""
training/compute_eval_metrics.py
==================================
Batch-computes BERTScore, BLEU, and ROUGE-1/2/L for every row in
judge_training.db, scoring answer_a and answer_b against the combiner's
final_answer (the closest thing this pipeline has to a reference/gold
answer). Also computes a pairwise BERTScore between answer_a and answer_b
directly, as an agreement signal.

Results are written to a new `eval_metrics` table (one row per
training_samples row), resumable - already-scored ids are skipped.

Run:
    python training/compute_eval_metrics.py
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # so `from utils.eval_metrics import ...` resolves regardless of cwd
DB_PATH = ROOT / "data" / "judge_training.db"
BATCH_SIZE = 32


def init_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_metrics (
            id INTEGER PRIMARY KEY,
            bertscore_f1_a REAL, bertscore_p_a REAL, bertscore_r_a REAL,
            bertscore_f1_b REAL, bertscore_p_b REAL, bertscore_r_b REAL,
            bertscore_f1_ab REAL,
            bleu_a REAL, bleu_b REAL,
            rouge1_a REAL, rouge2_a REAL, rougeL_a REAL,
            rouge1_b REAL, rouge2_b REAL, rougeL_b REAL,
            computed_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def load_pending_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT t.id, t.answer_a, t.answer_b, t.final_answer
        FROM training_samples t
        LEFT JOIN eval_metrics e ON t.id = e.id
        WHERE e.id IS NULL
          AND t.final_answer IS NOT NULL AND TRIM(t.final_answer) != ''
        ORDER BY t.id
        """,
        conn,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="only score this many pending rows (for testing)")
    cli_args = p.parse_args()

    from utils.eval_metrics import compute_bertscore_batch, compute_bleu, compute_rouge

    conn = sqlite3.connect(str(DB_PATH), timeout=60.0)
    init_table(conn)

    df = load_pending_rows(conn)
    if cli_args.limit:
        df = df.head(cli_args.limit)
    total = len(df)
    print(f"Rows to score: {total}", flush=True)
    if total == 0:
        print("Nothing to do.", flush=True)
        return

    t_start = time.time()
    done = 0

    for start in range(0, total, BATCH_SIZE):
        batch = df.iloc[start : start + BATCH_SIZE]
        ids = batch["id"].tolist()
        ans_a = batch["answer_a"].tolist()
        ans_b = batch["answer_b"].tolist()
        refs = batch["final_answer"].tolist()

        # Batched BERTScore vs. final_answer (fast on GPU) + pairwise a-vs-b
        bs_a = compute_bertscore_batch(ans_a, refs)
        bs_b = compute_bertscore_batch(ans_b, refs)
        bs_ab = compute_bertscore_batch(ans_a, ans_b)

        rows_to_insert = []
        for i, rid in enumerate(ids):
            bleu_a = compute_bleu(ans_a[i], refs[i])
            bleu_b = compute_bleu(ans_b[i], refs[i])
            rouge_a = compute_rouge(ans_a[i], refs[i])
            rouge_b = compute_rouge(ans_b[i], refs[i])

            rows_to_insert.append(
                (
                    rid,
                    bs_a["f1"][i], bs_a["precision"][i], bs_a["recall"][i],
                    bs_b["f1"][i], bs_b["precision"][i], bs_b["recall"][i],
                    bs_ab["f1"][i],
                    bleu_a, bleu_b,
                    rouge_a["rouge1"], rouge_a["rouge2"], rouge_a["rougeL"],
                    rouge_b["rouge1"], rouge_b["rouge2"], rouge_b["rougeL"],
                )
            )

        conn.executemany(
            """
            INSERT INTO eval_metrics
                (id, bertscore_f1_a, bertscore_p_a, bertscore_r_a,
                 bertscore_f1_b, bertscore_p_b, bertscore_r_b, bertscore_f1_ab,
                 bleu_a, bleu_b, rouge1_a, rouge2_a, rougeL_a, rouge1_b, rouge2_b, rougeL_b)
            VALUES (?, ?,?,?, ?,?,?, ?, ?,?, ?,?,?, ?,?,?)
            """,
            rows_to_insert,
        )
        conn.commit()

        done += len(batch)
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  {done}/{total}  ({rate:.1f} rows/s, ETA {eta/60:.1f} min)", flush=True)

    conn.close()
    print(f"\nDone. Scored {done} rows in {(time.time()-t_start)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
