$base = "C:\Program Files\DataBridge AI"
Write-Host "Installed files:"
if (Test-Path $base) { Get-ChildItem $base -Recurse -Depth 5 | Select-Object FullName, Length | Format-Table -AutoSize }
$logDir = Join-Path $env:LOCALAPPDATA "DataBridgeAI"
Write-Host "`nLogs in $logDir"
if (Test-Path $logDir) {
  Get-ChildItem $logDir
  Write-Host "`n--- tauri_launcher.log ---"
  Get-Content (Join-Path $logDir "tauri_launcher.log") -ErrorAction SilentlyContinue
  Write-Host "`n--- streamlit_launcher.log ---"
  Get-Content (Join-Path $logDir "streamlit_launcher.log") -ErrorAction SilentlyContinue
  Write-Host "`n--- streamlit_runtime.log tail ---"
  Get-Content (Join-Path $logDir "streamlit_runtime.log") -ErrorAction SilentlyContinue -Tail 100
}


Write-Host "`nPossible per-user install locations:"
$paths = @(
  (Join-Path $env:LOCALAPPDATA "Programs\DataBridge AI"),
  (Join-Path $env:LOCALAPPDATA "DataBridge AI")
)
foreach ($p in $paths) {
  Write-Host $p
  if (Test-Path $p) {
    Get-ChildItem $p -Recurse -Depth 4 | Select-Object FullName, Length | Format-Table -AutoSize
  }
}
