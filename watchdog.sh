#!/bin/bash
# watchdog.sh — keeps server.py and the cloudflared tunnel alive.
# Checks every 30s; relaunches whichever is down; reports a NEW URL line
# whenever cloudflared restarts (its quick-tunnel subdomain is not stable).

PROJECT_DIR="/c/Users/23102A0001/Downloads/agentic_llm_system_windows/agentic-llm-system"
CLOUDFLARED="/c/Users/23102A0001/cloudflared/cloudflared.exe"
URL_FILE="$PROJECT_DIR/current_tunnel_url.txt"
CF_LOG="$PROJECT_DIR/cloudflared_log.txt"
SERVER_LOG="$PROJECT_DIR/server_run_log.txt"
INTERVAL=30

is_server_up() {
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://localhost:8000/)
  [ "$code" = "200" ]
}

is_cloudflared_up() {
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='cloudflared.exe'\").ProcessId" 2>/dev/null | grep -q .
}

restart_server() {
  echo "[$(date '+%H:%M:%S')] server.py DOWN — restarting"
  cd "$PROJECT_DIR" && KMP_DUPLICATE_LIB_OK=TRUE "/c/ProgramData/anaconda3/python.exe" -u "server.py" > "$SERVER_LOG" 2>&1 &
  disown
  sleep 6
}

restart_cloudflared() {
  echo "[$(date '+%H:%M:%S')] cloudflared DOWN — restarting"
  : > "$CF_LOG"
  "$CLOUDFLARED" tunnel --url http://localhost:8000 > "$CF_LOG" 2>&1 &
  disown
  # wait for the new URL to appear in the log (up to 20s)
  for i in $(seq 1 20); do
    sleep 1
    NEW_URL=$(grep -oE "https://[a-zA-Z0-9.-]+\.trycloudflare\.com" "$CF_LOG" | head -1)
    [ -n "$NEW_URL" ] && break
  done
  if [ -n "$NEW_URL" ]; then
    echo "$NEW_URL" > "$URL_FILE"
    echo "[$(date '+%H:%M:%S')] NEW URL: $NEW_URL"
  else
    echo "[$(date '+%H:%M:%S')] WARNING: cloudflared restarted but no URL found in log yet"
  fi
}

echo "watchdog started: checking every ${INTERVAL}s indefinitely"
while true; do
  sleep "$INTERVAL"

  if ! is_server_up; then
    restart_server
  fi

  if ! is_cloudflared_up; then
    restart_cloudflared
  fi
done
