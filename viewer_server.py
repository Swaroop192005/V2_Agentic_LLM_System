#!/usr/bin/env python3
"""
Lightweight HTTP server for the Judge Training Dataset Viewer.
Serves data_viewer_live.html and exposes /api/data + /api/stats
so the frontend can poll for live updates from judge_training.db.

Usage:
    python3 viewer_server.py
Then open: http://localhost:7788
"""

import sqlite3
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# ── Config ────────────────────────────────────────────────────────────────────
PORT    = 7788
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "judge_training.db")
HTML    = os.path.join(os.path.dirname(__file__), "data_viewer_live.html")
# ─────────────────────────────────────────────────────────────────────────────


def query_db(sql, params=()):
    """Run a SQL query and return rows as list-of-dicts."""
    con = sqlite3.connect(DB_PATH, timeout=30.0)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def get_data(since_id=0, limit=500):
    rows = query_db(
        """SELECT id, question, answer_a, answer_b, judge_raw, verifier_text,
                  sem_sim_a, sem_sim_b, length_score_a, length_score_b,
                  composite_a, composite_b, word_count_a, word_count_b,
                  factual_a, completeness_a, clarity_a, relevance_a, depth_a, total_a,
                  factual_b, completeness_b, clarity_b, relevance_b, depth_b, total_b,
                  winner, elapsed_secs, created_at
           FROM training_samples
           WHERE id > ?
           ORDER BY id DESC
           LIMIT ?""",
        (since_id, limit),
    )
    return rows


def get_stats():
    rows = query_db(
        """SELECT
             COUNT(*)                                    AS total,
             SUM(winner='A')                             AS wins_a,
             SUM(winner='B')                             AS wins_b,
             SUM(winner='unknown')                       AS ties,
             ROUND(AVG(total_a),2)                       AS avg_score_a,
             ROUND(AVG(total_b),2)                       AS avg_score_b,
             ROUND(AVG(composite_a),3)                   AS avg_composite_a,
             ROUND(AVG(composite_b),3)                   AS avg_composite_b,
             ROUND(AVG(elapsed_secs),1)                  AS avg_elapsed,
             MIN(created_at)                             AS first_at,
             MAX(created_at)                             AS last_at,
             MAX(id)                                     AS max_id
           FROM training_samples"""
    )
    return rows[0] if rows else {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # quiet logging – only print errors
        if "40" in str(args) or "50" in str(args):
            super().log_message(fmt, *args)

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, path):
        try:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_GET(self):
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self.send_html(HTML)

        elif parsed.path == "/api/data":
            since = int(qs.get("since_id", ["0"])[0])
            rows  = get_data(since_id=since)
            self.send_json({"rows": rows, "count": len(rows)})

        elif parsed.path == "/api/stats":
            self.send_json(get_stats())

        elif parsed.path == "/api/all":
            rows = get_data(since_id=0, limit=5000)
            self.send_json({"rows": rows, "count": len(rows)})

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()


def main():
    if not os.path.exists(DB_PATH):
        print(f"❌  Database not found: {DB_PATH}")
        sys.exit(1)

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url    = f"http://localhost:{PORT}"

    print(f"")
    print(f"  Judge Dataset Viewer")
    print(f"  -----------------------------------------")
    print(f"  [OK] Server running -> {url}")
    print(f"  [DB] DB             -> {DB_PATH}")
    print(f"  [REFRESH] Auto-refresh -> every 5 s")
    print(f"  -----------------------------------------")
    print(f"  Press Ctrl+C to stop.")
    print(f"")

    # Auto-open browser
    import subprocess, platform
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", url])
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", url])
        else:
            import webbrowser; webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  🛑  Server stopped.")


if __name__ == "__main__":
    main()
