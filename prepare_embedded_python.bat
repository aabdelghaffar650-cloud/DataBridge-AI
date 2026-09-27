@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo DataBridge AI 1.13.0 - Prepare Portable Python
echo ==================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_portable_python.ps1" %*
if errorlevel 1 (
  echo Portable Python preparation failed.
  exit /b 1
)
echo Portable Python runtime is ready.
