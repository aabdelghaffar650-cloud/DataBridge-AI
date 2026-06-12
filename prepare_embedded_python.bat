@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo DataBridge AI - Prepare Embedded Python
echo ==================================================
if not exist resources mkdir resources
if not exist resources\python (
  py -3 -m venv resources\python
  if errorlevel 1 (
    echo Failed to create venv. Install Python 3.10+ first.
    pause
    exit /b 1
  )
)
resources\python\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
resources\python\Scripts\python.exe -m pip install -r requirements.txt
resources\python\Scripts\python.exe -m pip install streamlit
echo Embedded Python is ready.
pause
