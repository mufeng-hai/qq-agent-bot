# QQ-Opencode Bridge startup script
# Uses opencode's free models to power the QQ bot

$ProjectDir = 'D:\HermesWork\astrbot-napcat'
$BridgePy = Join-Path $ProjectDir 'qq-bridge.py'
$BridgeLog = Join-Path $ProjectDir 'bridge.log'
$NapCatExe = 'C:\Program Files\NapCatQQ Desktop\NapCatQQ-Desktop.exe'
$NapCatLog = Join-Path $ProjectDir 'napcat-autostart.log'

# 1. Start NapCatQQ Desktop first
if (Test-Path $NapCatExe) {
    $napcatRunning = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'NapCat*' }
    if (-not $napcatRunning) {
        Add-Content -Path $NapCatLog -Encoding UTF8 -Value "$(Get-Date -Format s) starting NapCatQQ Desktop"
        Start-Process -FilePath $NapCatExe -WorkingDirectory (Split-Path $NapCatExe) -WindowStyle Minimized
        Start-Sleep -Seconds 15
    }
}

# 2. Start the Bridge in direct mode (no AstrBot needed)
# NapCat is configured to connect to us via reverse WebSocket on port 6199

# Kill existing bridge and AstrBot (both would try to use port 6199)
Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq 'astrbot' } | Stop-Process -Force
Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'qq-bridge' } | Stop-Process -Force
Start-Sleep -Seconds 2

$env:BRIDGE_MODE = "direct"
$env:WS_HOST = "0.0.0.0"
$env:WS_PORT = "6199"
$env:LLM_PROVIDER = "opencode"
$env:OPENCODE_MODEL = "opencode/deepseek-v4-flash-free"
$env:CLAUDE_CODE_HEADLESS = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Start bridge with logging
Start-Process -FilePath "python" `
    -ArgumentList "-u", "qq-bridge.py" `
    -WorkingDirectory $ProjectDir `
    -RedirectStandardOutput $BridgeLog `
    -RedirectStandardError "${BridgeLog}.err" `
    -WindowStyle Hidden
Start-Sleep -Seconds 3

$now = Get-Date -Format s
Add-Content -Path $BridgeLog -Encoding UTF8 -Value "$now Bridge started (direct mode, opencode free model)"
Write-Output ""
Write-Output "QQ-Opencode Bridge started."
Write-Output "  Model: opencode/deepseek-v4-flash-free (FREE)"
Write-Output "  Mode:  direct (NapCat connects to port 6199)"
Write-Output ""
Write-Output "查看日志:  Get-Content $BridgeLog -Wait -Tail 20"
