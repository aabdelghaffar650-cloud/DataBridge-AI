# No-console startup fix

Changes:
- Tauri now prefers pythonw.exe over python.exe.
- Windows CREATE_NO_WINDOW is used for the Python child process.
- Loading screen is clearer during slow startup.
- Streamlit startup timeout increased to 120 seconds.
- Runtime logs remain under %LOCALAPPDATA%\DataBridgeAI.

Build:
1. prepare_embedded_python.bat
2. build_desktop_clean.bat
