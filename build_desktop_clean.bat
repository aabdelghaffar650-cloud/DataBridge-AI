@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo DataBridge AI 1.13.0 - Production Windows Build
echo ==================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0BUILD_STAGE13_WINDOWS_RELEASE.ps1" %*
if errorlevel 1 (
  echo Production build failed.
  exit /b 1
)
