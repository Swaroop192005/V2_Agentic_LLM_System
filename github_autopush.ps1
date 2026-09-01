# github_autopush.ps1
# ---------------------
# Runs every 2 hours via Windows Task Scheduler (task: AgenticLLM_GitHubSync),
# independent of any terminal or Claude Code session - same durability
# reasoning as heartbeat.ps1 (see RESEARCH_LOG.md Section 19). Commits and
# pushes the current project state (code + RESEARCH_LOG.md + LFS-tracked
# databases) so GitHub stays a running snapshot of progress without needing
# a live session to trigger it.

$root = "C:\Users\23102A0001\Downloads\agentic_llm_system_windows\agentic-llm-system"
$log = Join-Path $root "github_autopush_log.txt"
Set-Location $root

# Excludes the actively-growing live database: GitHub's free LFS tier is
# only 1GB storage + 1GB bandwidth/month, and LFS re-uploads the FULL file
# on every change (no binary diffing) - pushing this every 2 hours for the
# remaining multi-day run would exceed that quota within days. Pushed
# manually at checkpoints instead (see RESEARCH_LOG.md).
$EXCLUDE_PATHSPEC = ":!data/judge_training_v2.db"

$status = git status --porcelain -- . $EXCLUDE_PATHSPEC 2>&1
if ([string]::IsNullOrWhiteSpace($status)) {
    Add-Content $log "$(Get-Date -Format o)  no changes, skipped"
    exit 0
}

git add -A -- . $EXCLUDE_PATHSPEC 2>&1 | Out-Null

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$commitMsg = "Automated progress snapshot - $timestamp"
git commit -m $commitMsg 2>&1 | Out-Null

$pushResult = git push 2>&1
$pushExit = $LASTEXITCODE

if ($pushExit -eq 0) {
    Add-Content $log "$(Get-Date -Format o)  committed and pushed: $commitMsg"
} else {
    Add-Content $log "$(Get-Date -Format o)  PUSH FAILED: $pushResult"
}
