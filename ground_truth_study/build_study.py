"""
ground_truth_study/build_study.py
==================================
Builds a BLIND human-rating packet to establish the one thing the whole
project lacks (by necessity - see RESEARCH_LOG Sections 20-24): a small
anchor of real, human ground truth. Its specific job is to settle the
open Mistral-vs-LLaMA3 question (Sections 29-31): the Judge rates Mistral
higher on factual criteria, the Verifier sees no difference, and no
label-free method can break that tie. A human read can.

Design (mirrors the bias controls the pipeline itself uses):
- BALANCED: equal numbers of pairs the Judge awarded to Mistral vs LLaMA3,
  so the sample can't be skewed by the very preference it's testing.
- BLIND: the rater never sees which model wrote which answer, nor which
  one the Judge picked. Answer order within each item is independently
  re-randomized here, separate from the pipeline's own randomization.
- SEPARATED KEY: the mapping from displayed "Answer 1/2" back to model
  identity + Judge verdict is written to a separate answer_key.json that
  the rater never opens; only analyze_ratings.py reads it, afterwards.

Outputs (into this directory):
- rating_sheet.csv   -> what humans fill in (open in Excel/Sheets)
- rating_sheet.html  -> a friendlier read-only view of the same items
- answer_key.json    -> hidden mapping, for analysis only

Run:  python ground_truth_study/build_study.py [n_per_group]
"""

from __future__ import annotations

import csv
import html
import json
import random
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).parent
DB_PATH = HERE.parent / "data" / "judge_training_v2.db"
SEED = 20260910  # fixed so the sample is reproducible / auditable

N_PER_GROUP = int(sys.argv[1]) if len(sys.argv) > 1 else 15  # 15+15 = 30 items default


def true_winner(row) -> str:
    return row["model_a"] if row["winner"] == "A" else row["model_b"]


def main():
    random.seed(SEED)
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT id, question, answer_a, answer_b, model_a, model_b, winner,
                  judge_total_a, judge_total_b
           FROM pipeline_runs
           WHERE is_final_attempt=1 AND answer_a IS NOT NULL AND answer_b IS NOT NULL
                 AND judge_total_a != judge_total_b"""
    ).fetchall()
    con.close()

    mistral_won = [r for r in rows if true_winner(r) == "mistral"]
    llama_won = [r for r in rows if true_winner(r) == "llama3"]
    random.shuffle(mistral_won)
    random.shuffle(llama_won)

    n = min(N_PER_GROUP, len(mistral_won), len(llama_won))
    sample = mistral_won[:n] + llama_won[:n]
    random.shuffle(sample)  # interleave so raters can't infer groups from order

    key = {}
    sheet_rows = []
    for i, r in enumerate(sample, 1):
        item_id = f"Q{i:03d}"
        # Independently re-randomize which physical answer is shown as "1" vs "2"
        show_a_first = random.random() < 0.5
        if show_a_first:
            ans1, ans2 = r["answer_a"], r["answer_b"]
            model1, model2 = r["model_a"], r["model_b"]
            judge_pick_display = "1" if r["winner"] == "A" else "2"
        else:
            ans1, ans2 = r["answer_b"], r["answer_a"]
            model1, model2 = r["model_b"], r["model_a"]
            judge_pick_display = "1" if r["winner"] == "B" else "2"

        key[item_id] = {
            "db_id": r["id"],
            "answer1_model": model1,
            "answer2_model": model2,
            "judge_picked_display": judge_pick_display,   # "1" or "2" as shown
            "judge_picked_model": true_winner(r),
        }
        sheet_rows.append({
            "item_id": item_id,
            "question": r["question"],
            "answer_1": ans1,
            "answer_2": ans2,
        })

    # ---- rating_sheet.csv (the fillable artifact) ----
    csv_path = HERE / "rating_sheet.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "item_id", "question", "answer_1", "answer_2",
            "more_factually_correct(1/2/tie)",
            "answer_1_has_factual_error(y/n)",
            "answer_2_has_factual_error(y/n)",
            "rater_initials", "notes",
        ])
        for s in sheet_rows:
            w.writerow([s["item_id"], s["question"], s["answer_1"], s["answer_2"], "", "", "", "", ""])

    # ---- answer_key.json (kept separate; raters never open this) ----
    key_path = HERE / "answer_key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n_per_group": n, "items": key}, f, indent=2)

    # ---- rating_sheet.html (friendlier read-only view) ----
    html_path = HERE / "rating_sheet.html"
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Blind Rating Sheet</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:24px;line-height:1.5;color:#1a1a1a;background:#fff}"
        "h1{font-size:22px}.item{border:1px solid #ddd;border-radius:10px;padding:18px;margin:18px 0}"
        ".q{font-weight:700;margin-bottom:14px}.ans{background:#f6f6f6;border-radius:8px;padding:12px;margin:8px 0;white-space:pre-wrap}"
        ".lbl{font-family:monospace;font-size:12px;color:#666;text-transform:uppercase;letter-spacing:.05em}"
        ".id{font-family:monospace;color:#933;font-weight:700}</style>",
        "<h1>Blind rating sheet — factual correctness</h1>",
        "<p>For each item, decide which answer is <b>more factually correct</b> (or a genuine tie), "
        "and note any outright factual errors. You do <b>not</b> know which model wrote which answer, "
        "and that is intentional. Record your verdicts in <code>rating_sheet.csv</code>. "
        f"{len(sheet_rows)} items total.</p>",
    ]
    for s in sheet_rows:
        parts.append(
            f"<div class='item'><div><span class='id'>{s['item_id']}</span></div>"
            f"<div class='q'>{html.escape(s['question'])}</div>"
            f"<div class='lbl'>Answer 1</div><div class='ans'>{html.escape(s['answer_1'])}</div>"
            f"<div class='lbl'>Answer 2</div><div class='ans'>{html.escape(s['answer_2'])}</div></div>"
        )
    html_path.write_text("\n".join(parts), encoding="utf-8")

    print(f"Built blind study: {len(sheet_rows)} items ({n} Mistral-won + {n} LLaMA3-won, interleaved)")
    print(f"  rating sheet (fill this): {csv_path}")
    print(f"  readable view:            {html_path}")
    print(f"  hidden key (do NOT open): {key_path}")
    print(f"\nBlinding check - the CSV/HTML contain NO model names or judge verdicts.")


if __name__ == "__main__":
    main()
