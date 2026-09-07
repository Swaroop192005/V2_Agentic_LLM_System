#!/usr/bin/env python3
"""
Lightweight HTTP server for the v2 dataset generation progress viewer.
Serves data_viewer_v2.html and exposes /api/progress + /api/recent so the
frontend can poll live updates from data/judge_training_v2.db.

Usage:
    python viewer_server_v2.py
Then open: http://localhost:7790
"""

import sqlite3
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PORT = 7790
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "judge_training_v2.db")
HTML = os.path.join(os.path.dirname(__file__), "data_viewer_v2.html")
TARGET_QUESTIONS = 5000


def query_db(sql, params=()):
    con = sqlite3.connect(DB_PATH, timeout=30.0)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def get_progress():
    row = query_db(
        """
        SELECT
          COUNT(DISTINCT CASE WHEN is_final_attempt=1 THEN question_idx END) AS done,
          COUNT(*)                                                            AS attempts_total,
          ROUND(AVG(CASE WHEN is_final_attempt=1 THEN weighted_score END), 4) AS avg_weighted_score,
          ROUND(AVG(CASE WHEN is_final_attempt=1 THEN judge_total_a END), 2)  AS avg_judge_total_a,
          ROUND(AVG(CASE WHEN is_final_attempt=1 THEN judge_total_b END), 2)  AS avg_judge_total_b,
          ROUND(AVG(CASE WHEN is_final_attempt=1 THEN verifier_total_a END), 2) AS avg_verifier_total_a,
          ROUND(AVG(CASE WHEN is_final_attempt=1 THEN verifier_total_b END), 2) AS avg_verifier_total_b,
          ROUND(AVG(CASE WHEN is_final_attempt=1 THEN verifier_judge_agreement END), 4) AS avg_vj_agreement,
          ROUND(AVG(elapsed_secs), 1)                                         AS avg_elapsed,
          SUM(CASE WHEN is_final_attempt=1 AND winner='A' THEN 1 ELSE 0 END)  AS wins_a,
          SUM(CASE WHEN is_final_attempt=1 AND winner='B' THEN 1 ELSE 0 END)  AS wins_b,
          MAX(created_at)                                                    AS last_at
        FROM pipeline_runs
        """
    )[0]

    regen_row = query_db(
        """SELECT COUNT(*) AS c FROM (
             SELECT question_idx FROM pipeline_runs GROUP BY question_idx HAVING COUNT(*) > 1
           )"""
    )[0]

    recent_times = query_db(
        """SELECT created_at FROM pipeline_runs WHERE is_final_attempt=1
           ORDER BY id DESC LIMIT 25"""
    )

    row["target"] = TARGET_QUESTIONS
    row["pending"] = max(0, TARGET_QUESTIONS - (row["done"] or 0))
    row["regenerated_count"] = regen_row["c"]
    row["recent_completion_times"] = [r["created_at"] for r in recent_times]
    return row


def get_recent(limit=30):
    return query_db(
        """SELECT id, question_idx, question, attempt_number, winner, weighted_score,
                  judge_total_a, judge_total_b, verifier_judge_agreement,
                  wikipedia_score, wikidata_score, elapsed_secs, created_at
           FROM pipeline_runs
           WHERE is_final_attempt=1
           ORDER BY id DESC
           LIMIT ?""",
        (limit,),
    )


def get_detail(row_id):
    rows = query_db("SELECT * FROM pipeline_runs WHERE id=?", (row_id,))
    return rows[0] if rows else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if "40" in str(args) or "50" in str(args):
            super().log_message(fmt, *args)

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, path):
        try:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self.send_html(HTML)
        elif parsed.path == "/api/progress":
            try:
                self.send_json(get_progress())
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        elif parsed.path == "/api/recent":
            limit = int(qs.get("limit", ["30"])[0])
            try:
                self.send_json({"rows": get_recent(limit)})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        elif parsed.path == "/api/detail":
            row_id = qs.get("id", [None])[0]
            if row_id is None:
                self.send_json({"error": "missing id"}, status=400)
                return
            try:
                detail = get_detail(int(row_id))
                if detail is None:
                    self.send_json({"error": "not found"}, status=404)
                else:
                    self.send_json(detail)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()


def main():
    if not os.path.exists(DB_PATH):
        print(f"Database not found yet: {DB_PATH} (will appear once generation starts writing)")

    # ThreadingHTTPServer (not plain HTTPServer): the generator writes to the
    # SQLite DB constantly, so any single query that has to wait on a write
    # lock would otherwise block every other request behind it - a public
    # tunnel makes this worse by adding more concurrent hits. Each request
    # now gets its own thread/connection so one slow query can't wedge the
    # whole viewer.
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\n  v2 Dataset Generation Progress Viewer")
    print(f"  -----------------------------------------")
    print(f"  [OK] Server running -> {url}")
    print(f"  [DB] DB             -> {DB_PATH}")
    print(f"  -----------------------------------------\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
