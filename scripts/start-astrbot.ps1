# AstrBot QQ robot autostart script
$ErrorActionPreference = 'Stop'
$ProjectDir = 'D:\HermesWork\astrbot-napcat'
$AstrBotExe = Join-Path $ProjectDir '.venv\Scripts\astrbot.exe'
$OutLog = Join-Path $ProjectDir 'astrbot.out.log'
$ErrLog = Join-Path $ProjectDir 'astrbot.err.log'

$NapCatExe = 'C:\Program Files\NapCatQQ Desktop\NapCatQQ-Desktop.exe'
$NapCatLog = Join-Path $ProjectDir 'napcat-autostart.log'

# Start NapCat/QQ first; AstrBot needs its OneBot WebSocket connection.
if (Test-Path $NapCatExe) {
  $napcatRunning = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'NapCat*' -or ($_.Path -eq $NapCatExe) }
  if (-not $napcatRunning) {
    Add-Content -Path $NapCatLog -Encoding UTF8 -Value "$(Get-Date -Format s) starting NapCatQQ Desktop"
    Start-Process -FilePath $NapCatExe -WorkingDirectory (Split-Path $NapCatExe) -WindowStyle Minimized
    Start-Sleep -Seconds 15
  }
}

# Avoid GBK console encoding issues in redirected logs
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

# Do not start another instance only when both AstrBot ports are already listening.
$existingPorts = @(Get-NetTCPConnection -LocalPort 6185,6199 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty LocalPort -Unique)
if (($existingPorts -contains 6185) -and ($existingPorts -contains 6199)) { exit 0 }

# If AstrBot is half-started/stale, stop it and start cleanly.
Get-Process astrbot,python -ErrorAction SilentlyContinue | Where-Object { $_.Path -match 'astrbot-napcat|Python312' } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Start-Process -FilePath $AstrBotExe `
  -ArgumentList 'run' `
  -WorkingDirectory $ProjectDir `
  -RedirectStandardOutput $OutLog `
  -RedirectStandardError $ErrLog `
  -WindowStyle Hidden

# Wait briefly and log whether the two expected AstrBot ports came up.
Start-Sleep -Seconds 8
$startedPorts = @(Get-NetTCPConnection -LocalPort 6185,6199 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty LocalPort -Unique)
Add-Content -Path (Join-Path $ProjectDir 'autostart-health.log') -Encoding UTF8 -Value "$(Get-Date -Format s) AstrBot ports: $($startedPorts -join ',')"

