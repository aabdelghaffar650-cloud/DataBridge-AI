[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$KeepDownload
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeDir = Join-Path $ProjectRoot "resources\python"
$CacheDir = Join-Path $ProjectRoot ".build-cache"
$ArchiveName = "python-3.12.10-embeddable-amd64.zip"
$ArchivePath = Join-Path $CacheDir $ArchiveName
$PythonUrl = "https://www.python.org/ftp/python/3.12.10/$ArchiveName"
$ExpectedSha256 = "156c7eea90d58cd7e91a23f28a0056616b13e9f4cf4901b7b99b837b7848c6da"
$Requirements = Join-Path $ProjectRoot "requirements-release.txt"

function Fail([string]$Message) {
    throw "Stage 13 portable runtime: $Message"
}

if ($env:OS -ne "Windows_NT") {
    Fail "This builder must run on Windows."
}
if (-not (Test-Path $Requirements)) {
    Fail "requirements-release.txt was not found."
}

$BuilderProbe = & py -3.12 -c "import platform,sys; print(sys.executable); print(platform.architecture()[0]); print(sys.version_info[:2])" 2>&1
if ($LASTEXITCODE -ne 0) {
    Fail "Install a 64-bit CPython 3.12 build and ensure the py launcher is available."
}
if (($BuilderProbe | Out-String) -notmatch "64bit") {
    Fail "The Python 3.12 builder must be 64-bit."
}

New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
if ($Force -and (Test-Path $ArchivePath)) {
    Remove-Item $ArchivePath -Force
}
if (-not (Test-Path $ArchivePath)) {
    Write-Host "Downloading verified CPython portable runtime..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $PythonUrl -OutFile $ArchivePath -UseBasicParsing
}

$ActualHash = (Get-FileHash $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualHash -ne $ExpectedSha256) {
    Remove-Item $ArchivePath -Force -ErrorAction SilentlyContinue
    Fail "CPython archive SHA-256 mismatch. Download deleted."
}

if (Test-Path $RuntimeDir) {
    Remove-Item $RuntimeDir -Recurse -Force
}
New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
Expand-Archive -Path $ArchivePath -DestinationPath $RuntimeDir -Force

$PthFile = Get-ChildItem $RuntimeDir -Filter "python*._pth" -File | Select-Object -First 1
if (-not $PthFile) {
    Fail "The embedded Python ._pth file was not found."
}
$PthLines = @(
    "python312.zip",
    ".",
    "Lib\site-packages",
    "import site"
)
Set-Content -Path $PthFile.FullName -Value $PthLines -Encoding ASCII

$SitePackages = Join-Path $RuntimeDir "Lib\site-packages"
New-Item -ItemType Directory -Path $SitePackages -Force | Out-Null

Write-Host "Installing the pinned Windows runtime dependencies..." -ForegroundColor Cyan
& py -3.12 -m pip install `
    --disable-pip-version-check `
    --no-input `
    --only-binary=:all: `
    --upgrade `
    --target $SitePackages `
    -r $Requirements
if ($LASTEXITCODE -ne 0) {
    Fail "Dependency installation failed."
}

Get-ChildItem $RuntimeDir -Directory -Recurse -Force |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") } |
    Sort-Object FullName -Descending |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $RuntimeDir -Filter "*.pyc" -File -Recurse -Force |
    Remove-Item -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $RuntimeDir "pyvenv.cfg") -Force -ErrorAction SilentlyContinue

$RuntimePython = Join-Path $RuntimeDir "python.exe"
$RuntimePythonw = Join-Path $RuntimeDir "pythonw.exe"
if (-not (Test-Path $RuntimePython) -or -not (Test-Path $RuntimePythonw)) {
    Fail "python.exe/pythonw.exe are missing from the portable runtime."
}

$VerifyCode = @'
import importlib.metadata as md
import json
from pathlib import Path
required = {
    "streamlit": "1.58.0",
    "pandas": "3.0.3",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "plotly": "6.8.0",
    "openpyxl": "3.1.5",
    "xlrd": "2.0.2",
    "pyarrow": "24.0.0",
    "scikit-learn": "1.9.0",
    "requests": "2.34.2",
    "anthropic": "0.108.0",
    "SQLAlchemy": "2.0.50",
    "PyMySQL": "1.2.0",
    "psycopg2-binary": "2.9.12",
    "numexpr": "2.14.1",
    "keyring": "25.6.0",
    "cryptography": "46.0.4",
    "xgboost-cpu": "3.4.0",
}
actual = {name: md.version(name) for name in required}
import xgboost
if xgboost.__version__ != required["xgboost-cpu"]:
    raise SystemExit(f"XGBoost import/version mismatch: {xgboost.__version__}")
errors = {name: {"expected": expected, "actual": actual[name]} for name, expected in required.items() if actual[name] != expected}
if errors:
    raise SystemExit("Pinned runtime mismatch: " + json.dumps(errors, sort_keys=True))
manifest = {
    "runtime": "CPython 3.12.10 embeddable amd64",
    "python_archive_sha256": "156c7eea90d58cd7e91a23f28a0056616b13e9f4cf4901b7b99b837b7848c6da",
    "packages": actual,
}
Path("resources/runtime_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("Portable runtime verification passed.")
'@
$VerifyScript = Join-Path $CacheDir "verify_portable_runtime.py"
Set-Content -Path $VerifyScript -Value $VerifyCode -Encoding ASCII

Push-Location $ProjectRoot
try {
    & $RuntimePython -I $VerifyScript
    if ($LASTEXITCODE -ne 0) {
        Fail "Portable runtime import/version verification failed."
    }
}
finally {
    Pop-Location
    Remove-Item $VerifyScript -Force -ErrorAction SilentlyContinue
}

if (-not $KeepDownload) {
    Remove-Item $ArchivePath -Force -ErrorAction SilentlyContinue
}

Write-Host "PASS: portable CPython runtime prepared at $RuntimeDir" -ForegroundColor Green
