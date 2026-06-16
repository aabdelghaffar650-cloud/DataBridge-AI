$logDir = Join-Path $env:LOCALAPPDATA "DataBridgeAI"
Write-Host "DataBridgeAI user data dir:" $logDir
if (Test-Path $logDir) {
  Get-ChildItem $logDir -Recurse -Depth 3 | Select-Object FullName, Length | Format-Table -AutoSize
} else {
  Write-Host "Not found."
}
Write-Host "`nInstalled app:"
$base = "C:\Program Files\DataBridge AI"
if (Test-Path $base) {
  Get-ChildItem $base -Recurse -Depth 4 | Select-Object FullName, Length | Format-Table -AutoSize
}
