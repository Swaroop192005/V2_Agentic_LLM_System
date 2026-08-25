# heartbeat.ps1
# ---------------
# Runs every 5 minutes via Windows Task Scheduler (task: AgenticLLM_Heartbeat),
# independent of any terminal or Claude Code session. Restarts any of the four
# core pipeline processes if it isn't running. This exists because the
# in-session bash watchdog (watchdog_v2.sh, launched via the Monitor tool)
# does NOT survive terminal closure, session teardown, or the machine
# sleeping - which caused three separate multi-hour/multi-day generation gaps
# during this project (see RESEARCH_LOG.md). Task Scheduler survives all of
# that as long as the user is logged in.

$root = "C:\Users\23102A0001\Downloads\agentic_llm_system_windows\agentic-llm-system"
$py = "C:\ProgramData\anaconda3\python.exe"
Set-Location $root

$log = Join-Path $root "heartbeat_log.txt"

function Test-ProcRunning($pattern) {
    $match = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like $pattern }
    return $null -ne $match
}

function Start-Proc($script, $logBase) {
    Start-Process -FilePath $py `
        -ArgumentList "-u `"$script`"" `
        -WorkingDirectory $root `
        -RedirectStandardOutput "$root\$logBase.txt" `
        -RedirectStandardError "$root\$logBase.err.txt" `
        -WindowStyle Hidden
}

$restarted = @()

if (-not (Test-ProcRunning "*keep_awake.py*")) {
    Start-Proc "keep_awake.py" "keep_awake_log"
    $restarted += "keep_awake.py"
}
if (-not (Test-ProcRunning "*viewer_server_v2.py*")) {
    Start-Proc "viewer_server_v2.py" "viewer_v2_log"
    $restarted += "viewer_server_v2.py"
}
if (-not (Test-ProcRunning "*server.py*")) {
    Start-Proc "server.py" "server_log"
    $restarted += "server.py"
}
if (-not (Test-ProcRunning "*generate_training_data_v2.py*")) {
    Start-Proc "generate_training_data_v2.py" "generate_v2_stdout"
    $restarted += "generate_training_data_v2.py"
}

if ($restarted.Count -gt 0) {
    Add-Content $log "$(Get-Date -Format o)  RESTARTED: $($restarted -join ', ')"
} else {
    Add-Content $log "$(Get-Date -Format o)  ok (all 4 processes running)"
}
