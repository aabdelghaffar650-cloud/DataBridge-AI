@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo DataBridge AI - Build Tauri Desktop App
echo ==================================================
if not exist resources\python\Scripts\python.exe (
  echo Embedded Python not found. Run prepare_embedded_python.bat first.
  pause
  exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is not installed or not in PATH.
  pause
  exit /b 1
)
where cargo >nul 2>nul
if errorlevel 1 (
  echo Rust/Cargo is not installed or not in PATH.
  pause
  exit /b 1
)
cd desktop
if not exist node_modules npm install
npm run tauri build
echo Build finished. Check desktop\src-tauri\target\release\bundle
pause
