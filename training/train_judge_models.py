"""
training/train_judge_models.py
===============================
Trains and compares 4 candidate replacements for the slow LLM judge,
all evaluated on the identical held-out test split for a fair comparison:

  A. Sklearn baseline    - gradient boosting on the cheap features already
                            stored per row (sem_sim, length_score, word_count).
  B. Embedding + MLP      - sentence-transformer embeddings feed a small
                            regressor that predicts all 5 rubric scores per side.
  C. Pairwise ranking     - Bradley-Terry linear scorer on embedding deltas:
                            P(A wins) = sigmoid(w.embed_a - w.embed_b).
  D. Fine-tuned DistilBERT - cross-encoder sequence classifier over
                            (question, answer_a, answer_b) -> winner.

Metric used for comparison: winner-prediction accuracy on the test set.
Run:
    python training/train_judge_models.py
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "judge_training.db"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
EMB_CACHE = MODELS_DIR / "embeddings_cache.npz"
RESULTS_PATH = MODELS_DIR / "comparison_results.json"

RESULTS: dict = {}


MIN_ID = 1600      # rows below this were scored by the old judge (Mistral-based, ~69% A-win-rate);
                   # rows from here on use the current judge (phi3 verifier + qwen2.5 auditor, ~43%
                   # A-win-rate). Mixing both regimes trains against two disagreeing labeling
                   # standards, so we only train/evaluate on the consistent, current-judge portion.
MIN_MARGIN = 2     # drop rows where |total_a - total_b| <= this: near-ties are close to a coin
                   # flip in the judge's own scoring and are effectively unlearnable label noise.
                   # Keeps ~59% of the consistent-judge subset (rows with a real quality gap).
                   # Overridable via --min-margin.
TRANSFORMER_MODEL = "distilbert-base-uncased"  # step D's base model; overridable via --model


def load_data() -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        f"""
        SELECT id, question, answer_a, answer_b,
               sem_sim_a, length_score_a, composite_a, word_count_a,
               sem_sim_b, length_score_b, composite_b, word_count_b,
               factual_a, completeness_a, clarity_a, relevance_a, depth_a, total_a,
               factual_b, completeness_b, clarity_b, relevance_b, depth_b, total_b,
               winner
        FROM training_samples
        WHERE winner IN ('A', 'B') AND id >= {MIN_ID}
          AND ABS(total_a - total_b) > {int(MIN_MARGIN)}
        ORDER BY id
        """,
        con,
    )
    con.close()
    df["label"] = (df["winner"] == "A").astype(int)  # 1 = A wins, 0 = B wins
    return df


def make_split(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Data: {len(df)} rows -> train={len(train_df)}  test={len(test_df)}", flush=True)
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# A. Sklearn baseline on cheap features
# ─────────────────────────────────────────────────────────────────────────────
def run_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n[A] Sklearn baseline (gradient boosting on cheap features)...", flush=True)

    def feats(d: pd.DataFrame) -> pd.DataFrame:
        f = d[
            [
                "sem_sim_a", "sem_sim_b", "length_score_a", "length_score_b",
                "word_count_a", "word_count_b", "composite_a", "composite_b",
            ]
        ].copy()
        f["composite_diff"] = f["composite_a"] - f["composite_b"]
        f["sem_sim_diff"] = f["sem_sim_a"] - f["sem_sim_b"]
        f["word_count_diff"] = f["word_count_a"] - f["word_count_b"]
        return f

    Xtr, Xte = feats(train_df), feats(test_df)
    clf = HistGradientBoostingClassifier(random_state=42)

    t0 = time.time()
    clf.fit(Xtr, train_df["label"])
    train_time = time.time() - t0

    preds = clf.predict(Xte)
    acc = accuracy_score(test_df["label"], preds)

    joblib.dump(clf, MODELS_DIR / "A_sklearn_baseline.joblib")
    RESULTS["A_sklearn_baseline"] = {
        "description": "HistGradientBoosting on sem_sim/length/word_count features",
        "accuracy": round(float(acc), 4),
        "train_time_s": round(train_time, 2),
    }
    print(f"[A] done. accuracy={acc:.4f}  train_time={train_time:.1f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: sentence-transformer embeddings (cached, reused by B and C)
# ─────────────────────────────────────────────────────────────────────────────
def get_embeddings(df: pd.DataFrame):
    if EMB_CACHE.exists():
        data = np.load(EMB_CACHE)
        if len(data["ids"]) == len(df) and (data["ids"] == df["id"].to_numpy()).all():
            print("[embeddings] loaded from cache", flush=True)
            return data["q"], data["a"], data["b"]

    print("[embeddings] encoding with all-MiniLM-L6-v2 (first run only)...", flush=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q = model.encode(df["question"].tolist(), batch_size=64, show_progress_bar=False)
    a = model.encode(df["answer_a"].tolist(), batch_size=64, show_progress_bar=False)
    b = model.encode(df["answer_b"].tolist(), batch_size=64, show_progress_bar=False)
    np.savez(EMB_CACHE, ids=df["id"].to_numpy(), q=q, a=a, b=b)
    print("[embeddings] done and cached", flush=True)
    return q, a, b


# ─────────────────────────────────────────────────────────────────────────────
# B. Embedding + MLP regressor (predicts all 5 rubric scores per side)
# ─────────────────────────────────────────────────────────────────────────────
CRITERIA = ["factual", "completeness", "clarity", "relevance", "depth"]


def run_embedding_mlp(df: pd.DataFrame, train_idx, test_idx, q_emb, a_emb, b_emb) -> None:
    print("\n[B] Embedding + MLP regressor (5 rubric scores per side)...", flush=True)

    X = np.hstack([q_emb, a_emb, b_emb, a_emb - b_emb])
    y_cols = [f"{c}_a" for c in CRITERIA] + [f"{c}_b" for c in CRITERIA]
    y = df[y_cols].to_numpy(dtype=float)

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    reg = MLPRegressor(
        hidden_layer_sizes=(256, 64),
        max_iter=400,
        early_stopping=True,
        random_state=42,
    )
    t0 = time.time()
    reg.fit(Xtr, ytr)
    train_time = time.time() - t0

    pred = reg.predict(Xte)
    mae = mean_absolute_error(yte, pred)

    total_a_true = yte[:, :5].sum(axis=1)
    total_b_true = yte[:, 5:].sum(axis=1)
    total_a_pred = pred[:, :5].sum(axis=1)
    total_b_pred = pred[:, 5:].sum(axis=1)
    winner_true = (total_a_true > total_b_true).astype(int)
    winner_pred = (total_a_pred > total_b_pred).astype(int)
    acc = accuracy_score(winner_true, winner_pred)

    joblib.dump(reg, MODELS_DIR / "B_embedding_mlp.joblib")
    RESULTS["B_embedding_mlp"] = {
        "description": "MLPRegressor over MiniLM embeddings, predicts 5 rubric scores/side",
        "accuracy": round(float(acc), 4),
        "score_mae": round(float(mae), 3),
        "train_time_s": round(train_time, 2),
    }
    print(f"[B] done. accuracy={acc:.4f}  score_MAE={mae:.3f}  train_time={train_time:.1f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# C. Pairwise ranking (Bradley-Terry linear scorer on embedding deltas)
# ─────────────────────────────────────────────────────────────────────────────
def run_pairwise_ranking(df: pd.DataFrame, train_idx, test_idx, a_emb, b_emb) -> None:
    print("\n[C] Pairwise ranking (Bradley-Terry logistic model)...", flush=True)

    # X = embed_a - embed_b  =>  logistic(w.X) = sigmoid(s(a) - s(b))  with s(x) = w.x
    # This is exactly a linear Bradley-Terry scorer trained via pairwise logistic loss.
    X = a_emb - b_emb
    y = df["label"].to_numpy()

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    clf = LogisticRegression(max_iter=2000, C=1.0)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    train_time = time.time() - t0

    preds = clf.predict(Xte)
    acc = accuracy_score(yte, preds)

    joblib.dump(clf, MODELS_DIR / "C_pairwise_bt.joblib")
    RESULTS["C_pairwise_bt"] = {
        "description": "Linear Bradley-Terry scorer (logistic regression on embed_a - embed_b)",
        "accuracy": round(float(acc), 4),
        "train_time_s": round(train_time, 2),
    }
    print(f"[C] done. accuracy={acc:.4f}  train_time={train_time:.1f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# D. Fine-tuned DistilBERT cross-encoder
# ─────────────────────────────────────────────────────────────────────────────
def run_transformer(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    MODEL_NAME = TRANSFORMER_MODEL
    print(f"\n[D] Fine-tuning {MODEL_NAME} cross-encoder (this is the slow one)...", flush=True)
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    MAX_LEN = 512  # BERT-family hard position-embedding limit (RoBERTa/BERT/DistilBERT all use 512)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Truncate question/answer_a/answer_b to independent TOKEN budgets before
    # concatenating, so a long answer_a can't crowd out answer_b (or vice versa) -
    # naive string-level truncation always keeps A intact and chops B first.
    Q_BUDGET, ANS_BUDGET = 40, 210  # 40 + 210 + 210 + special/marker tokens <= 512

    def encode_pair(question: str, answer_a: str, answer_b: str) -> dict:
        q_ids = tokenizer.encode(question, add_special_tokens=False, truncation=True, max_length=Q_BUDGET)
        a_ids = tokenizer.encode(answer_a, add_special_tokens=False, truncation=True, max_length=ANS_BUDGET)
        b_ids = tokenizer.encode(answer_b, add_special_tokens=False, truncation=True, max_length=ANS_BUDGET)

        marker_a = tokenizer.encode("Answer A:", add_special_tokens=False)
        marker_b = tokenizer.encode("Answer B:", add_special_tokens=False)

        cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
        ids = [cls] + q_ids + [sep] + marker_a + a_ids + [sep] + marker_b + b_ids + [sep]
        ids = ids[:MAX_LEN]
        attn = [1] * len(ids)
        pad_len = MAX_LEN - len(ids)
        if pad_len > 0:
            ids = ids + [tokenizer.pad_token_id] * pad_len
            attn = attn + [0] * pad_len
        return {"input_ids": ids, "attention_mask": attn}

    class PairDataset(Dataset):
        def __init__(self, d: pd.DataFrame):
            self.rows = d[["question", "answer_a", "answer_b"]].to_dict("records")
            self.labels = d["label"].tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            r = self.rows[idx]
            enc = encode_pair(r["question"], r["answer_a"], r["answer_b"])
            enc["labels"] = self.labels[idx]
            return {k: torch.tensor(v) for k, v in enc.items()}

    train_ds = PairDataset(train_df)
    test_ds = PairDataset(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    model_tag = MODEL_NAME.replace("/", "_")
    run_tag = f"{model_tag}_margin{MIN_MARGIN}"
    out_dir = MODELS_DIR / f"D_{run_tag}_run"
    steps_per_epoch = max(1, len(train_ds) // 8)
    warmup_steps = int(0.1 * steps_per_epoch * 20)
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=20,
        learning_rate=2e-5,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        disable_tqdm=True,
        use_cpu=not torch.cuda.is_available(),
    )

    from transformers import EarlyStoppingCallback

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    eval_metrics = trainer.evaluate()
    acc = eval_metrics["eval_accuracy"]

    final_dir = MODELS_DIR / f"D_{run_tag}_final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    RESULTS[f"D_{run_tag}"] = {
        "description": f"Fine-tuned {MODEL_NAME} cross-encoder over (question, answer_a, answer_b)",
        "accuracy": round(float(acc), 4),
        "train_time_s": round(train_time, 2),
    }
    print(f"[D:{MODEL_NAME}] done. accuracy={acc:.4f}  train_time={train_time:.1f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--min-margin", type=int, default=MIN_MARGIN,
                    help="drop rows with |total_a-total_b| <= this (default: %(default)s)")
    p.add_argument("--model", type=str, default=TRANSFORMER_MODEL,
                    help="HF model name for step D, e.g. roberta-base, bert-base-uncased (default: %(default)s)")
    return p.parse_args()


def main() -> None:
    global MIN_MARGIN, TRANSFORMER_MODEL
    cli_args = parse_args()
    MIN_MARGIN = cli_args.min_margin
    TRANSFORMER_MODEL = cli_args.model
    run_tag = f"{TRANSFORMER_MODEL.replace('/', '_')}_margin{MIN_MARGIN}"
    print(f"=== Run config: model={TRANSFORMER_MODEL}  min_margin={MIN_MARGIN} ===", flush=True)

    t_start = time.time()
    df = load_data()
    train_df, test_df = make_split(df)

    # A: cheap features, no embeddings needed
    run_baseline(train_df, test_df)

    # Shared embeddings for B and C
    q_emb, a_emb, b_emb = get_embeddings(df)
    # Map each split's row ids back to positions in the id-ordered df used for embeddings
    id_to_pos = {rid: i for i, rid in enumerate(df["id"].tolist())}
    train_idx = np.array([id_to_pos[i] for i in train_df["id"]])
    test_idx = np.array([id_to_pos[i] for i in test_df["id"]])

    run_embedding_mlp(df, train_idx, test_idx, q_emb, a_emb, b_emb)
    run_pairwise_ranking(df, train_idx, test_idx, a_emb, b_emb)

    # D: slow fine-tune, run last
    run_transformer(train_df, test_df)

    total_time = time.time() - t_start

    print("\n" + "=" * 72, flush=True)
    print("COMPARISON RESULTS (winner-prediction accuracy on held-out test set)", flush=True)
    print("=" * 72, flush=True)
    for name, r in sorted(RESULTS.items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"  {name:22s}  accuracy={r['accuracy']:.4f}  train_time={r['train_time_s']:.1f}s", flush=True)
    print("=" * 72, flush=True)
    print(f"Total wall time: {total_time / 60:.1f} min", flush=True)

    RESULTS["_meta"] = {
        "total_time_s": round(total_time, 1),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "min_margin": MIN_MARGIN,
        "transformer_model": TRANSFORMER_MODEL,
    }
    out_path = MODELS_DIR / f"comparison_results_{run_tag}.json"
    out_path.write_text(json.dumps(RESULTS, indent=2))
    RESULTS_PATH.write_text(json.dumps(RESULTS, indent=2))  # also keep as "latest"
    print(f"\nSaved comparison to {out_path}", flush=True)


if __name__ == "__main__":
    main()
