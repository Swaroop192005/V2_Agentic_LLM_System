"""
training/cross_validate.py
===========================
5-fold cross-validation for the embedding-based models (B: MLP regressor,
C: Bradley-Terry logistic) at different margin thresholds, to get a
statistically trustworthy accuracy estimate instead of relying on a single
(possibly small) held-out test split.

Run:
    python training/cross_validate.py --min-margin 4
    python training/cross_validate.py --min-margin 6
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "judge_training.db"
MIN_ID = 1600
CRITERIA = ["factual", "completeness", "clarity", "relevance", "depth"]


def load_data(min_margin: int) -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        f"""
        SELECT id, question, answer_a, answer_b,
               factual_a, completeness_a, clarity_a, relevance_a, depth_a, total_a,
               factual_b, completeness_b, clarity_b, relevance_b, depth_b, total_b,
               winner
        FROM training_samples
        WHERE winner IN ('A', 'B') AND id >= {MIN_ID}
          AND ABS(total_a - total_b) > {int(min_margin)}
        ORDER BY id
        """,
        con,
    )
    con.close()
    df["label"] = (df["winner"] == "A").astype(int)
    return df


def get_embeddings(df: pd.DataFrame):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q = model.encode(df["question"].tolist(), batch_size=64, show_progress_bar=False)
    a = model.encode(df["answer_a"].tolist(), batch_size=64, show_progress_bar=False)
    b = model.encode(df["answer_b"].tolist(), batch_size=64, show_progress_bar=False)
    return q, a, b


def cv_pairwise_bt(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
    return np.array(accs)


def cv_embedding_mlp(X, y_scores, y_label, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y_label):
        reg = MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=400, early_stopping=True, random_state=42)
        reg.fit(X[train_idx], y_scores[train_idx])
        pred = reg.predict(X[test_idx])
        total_a_pred = pred[:, :5].sum(axis=1)
        total_b_pred = pred[:, 5:].sum(axis=1)
        winner_pred = (total_a_pred > total_b_pred).astype(int)
        accs.append(accuracy_score(y_label[test_idx], winner_pred))
    return np.array(accs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-margin", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    df = load_data(args.min_margin)
    print(f"min_margin={args.min_margin}  rows={len(df)}", flush=True)

    q_emb, a_emb, b_emb = get_embeddings(df)
    y = df["label"].to_numpy()

    # C: Bradley-Terry
    X_bt = a_emb - b_emb
    accs_c = cv_pairwise_bt(X_bt, y, args.folds)
    print(f"[C pairwise-BT]   {args.folds}-fold acc = {accs_c.mean():.4f} +/- {accs_c.std():.4f}   folds={np.round(accs_c,3)}", flush=True)

    # B: embedding + MLP
    X_b = np.hstack([q_emb, a_emb, b_emb, a_emb - b_emb])
    y_cols = [f"{c}_a" for c in CRITERIA] + [f"{c}_b" for c in CRITERIA]
    y_scores = df[y_cols].to_numpy(dtype=float)
    accs_b = cv_embedding_mlp(X_b, y_scores, y, args.folds)
    print(f"[B embedding-MLP] {args.folds}-fold acc = {accs_b.mean():.4f} +/- {accs_b.std():.4f}   folds={np.round(accs_b,3)}", flush=True)


if __name__ == "__main__":
    main()
