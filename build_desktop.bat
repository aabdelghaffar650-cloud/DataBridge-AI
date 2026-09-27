@echo off
setlocal
cd /d "%~dp0"
call "%~dp0build_desktop_clean.bat" %*
exit /b %errorlevel%
