#!/bin/bash
# watchdog_v2.sh — keeps generate_training_data_v2.py, keep_awake.py,
# server.py, and viewer_server_v2.py alive during the multi-day dataset
# regeneration. Auto-restarts any that die.
#
# Also auto-detects STALLS: recurring incident this run was orphaned
# llama-server.exe subprocesses (children of ollama.exe serve) that
# survive a plain process restart and silently eat VRAM until a verifier/
# judge call wedges for 20+ minutes. If no new question completes within
# STALL_THRESHOLD seconds, this script now performs the full recovery
# automatically: kill the generator, kill every llama-server.exe
# directly (not just the ollama parent - that's the part a plain restart
# missed), restart ollama's serve process, then relaunch the generator.
#
# The generator itself is crash-safe/resumable (commits per attempt), so
# this is a best-effort layer on top of that, not the only thing standing
# between a crash and lost work.

PROJECT_DIR="/c/Users/23102A0001/Downloads/agentic_llm_system_windows/agentic-llm-system"
PROJECT_DIR_WIN="C:\\Users\\23102A0001\\Downloads\\agentic_llm_system_windows\\agentic-llm-system"
PY="/c/ProgramData/anaconda3/python.exe"
INTERVAL=60
STALL_THRESHOLD=480   # 8 minutes with zero new completions -> auto-recover
                      # (tightened from 15 min now that CONCURRENCY=1 means a
                      # normal single-question worst case is well under this)

is_running() {
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -like '*$1*'}).ProcessId" 2>/dev/null | grep -q .
}

done_count() {
  "$PY" -c "
import sqlite3
try:
    con = sqlite3.connect(r'$PROJECT_DIR_WIN\\data\\judge_training_v2.db', timeout=5)
    print(con.execute('SELECT COUNT(DISTINCT question_idx) FROM pipeline_runs WHERE is_final_attempt=1').fetchone()[0])
except Exception:
    print(-1)
" 2>/dev/null
}

progress_snapshot() {
  "$PY" -c "
import sqlite3
try:
    con = sqlite3.connect(r'$PROJECT_DIR_WIN\\data\\judge_training_v2.db', timeout=5)
    total = con.execute('SELECT COUNT(DISTINCT question_idx) FROM pipeline_runs WHERE is_final_attempt=1').fetchone()[0]
    attempts = con.execute('SELECT COUNT(*) FROM pipeline_runs').fetchone()[0]
    print(f'questions_done={total} total_attempts_logged={attempts}')
except Exception as e:
    print(f'progress check failed: {e}')
" 2>/dev/null
}

kill_generator() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -like '*generate_training_data_v2*'} | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" 2>/dev/null
}

kill_orphaned_llama_servers() {
  powershell -NoProfile -Command "Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.Id -Force -ErrorAction SilentlyContinue }" 2>/dev/null
}

restart_ollama_serve() {
  powershell -NoProfile -Command "\$o = Get-CimInstance Win32_Process -Filter \"Name='ollama.exe'\" | Where-Object {\$_.CommandLine -like '*serve*'}; if (\$o) { Stop-Process -Id \$o.ProcessId -Force -ErrorAction SilentlyContinue }" 2>/dev/null
  sleep 5   # ollama's tray app auto-relaunches "ollama serve"
}

start_generator() {
  cd "$PROJECT_DIR" && KMP_DUPLICATE_LIB_OK=TRUE "$PY" -u "generate_training_data_v2.py" >> "generate_v2_stdout.txt" 2>&1 &
  disown
  sleep 8
}

full_recovery() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] STALL DETECTED (no new completion in ${STALL_THRESHOLD}s) — running full recovery"
  kill_generator
  sleep 3
  kill_orphaned_llama_servers
  restart_ollama_serve
  sleep 3
  kill_orphaned_llama_servers   # second pass - a request in flight during the ollama restart can respawn one
  start_generator
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] full recovery complete, generator relaunched"
}

echo "watchdog_v2 started: checking every ${INTERVAL}s, stall threshold ${STALL_THRESHOLD}s"
last_done=-1
last_progress_ts=$(date +%s)

while true; do
  sleep "$INTERVAL"

  if ! is_running "generate_training_data_v2"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] generator DOWN — restarting"
    start_generator
  fi

  if ! is_running "keep_awake.py"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] keep_awake DOWN — restarting"
    cd "$PROJECT_DIR" && KMP_DUPLICATE_LIB_OK=TRUE "$PY" -u "keep_awake.py" >> "keep_awake_log.txt" 2>&1 &
    disown
  fi

  if ! is_running "server.py"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] server.py DOWN — restarting"
    cd "$PROJECT_DIR" && KMP_DUPLICATE_LIB_OK=TRUE "$PY" -u "server.py" >> "server_run_log.txt" 2>&1 &
    disown
    sleep 6
  fi

  if ! is_running "viewer_server_v2.py"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] viewer_server_v2 DOWN — restarting"
    cd "$PROJECT_DIR" && "$PY" -u "viewer_server_v2.py" >> "viewer_v2_log.txt" 2>&1 &
    disown
  fi

  # ── Stall detection ──────────────────────────────────────────────────
  current_done=$(done_count)
  now_ts=$(date +%s)
  if [ "$current_done" != "-1" ] && [ "$current_done" != "$last_done" ]; then
    last_done="$current_done"
    last_progress_ts="$now_ts"
  elif [ "$current_done" != "-1" ] && [ $(( now_ts - last_progress_ts )) -ge "$STALL_THRESHOLD" ]; then
    full_recovery
    last_progress_ts=$(date +%s)   # reset the clock after recovering
  fi

  # Periodic progress line (every ~10 min = every 10th check at 60s interval)
  ts=$(date +%s)
  if [ $(( (ts / INTERVAL) % 10 )) -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $(progress_snapshot)"
  fi
done
