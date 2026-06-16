# DataBridge AI Desktop Build

This package adds a Tauri desktop shell around the current DataBridge AI Streamlit app.

## Build on Windows

1. Install prerequisites:
- Python 3.10+
- Node.js LTS
- Rust via rustup
- Microsoft Visual Studio Build Tools with C++ workload

2. Run:
prepare_embedded_python.bat

3. Run:
build_desktop.bat

4. Output:
desktop\src-tauri\target\release\bundle

The desktop app starts bundled Python, launches Streamlit locally, then opens it inside a Tauri window.
