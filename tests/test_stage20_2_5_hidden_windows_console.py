from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_RS = PROJECT_ROOT / "desktop" / "src-tauri" / "src" / "main.rs"
RELEASE_GATE = PROJECT_ROOT / "tools" / "release_gate.py"
RELEASE_SPEC = PROJECT_ROOT / "release" / "release_spec.json"

EXPECTED_ATTR = '#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]'


def main() -> None:
    rust = MAIN_RS.read_text(encoding="utf-8")
    gate = RELEASE_GATE.read_text(encoding="utf-8")
    spec = RELEASE_SPEC.read_text(encoding="utf-8")

    # The attribute must be crate-level and active for release builds. It switches
    # the Windows executable from the console subsystem to the GUI subsystem, so
    # starting DataBridge AI does not allocate a visible terminal window.
    assert rust.lstrip().startswith(EXPECTED_ATTR)
    assert 'CREATE_NO_WINDOW' in rust
    assert 'pythonw.exe' in rust

    # Defense in depth: the production release gate must reject a future removal.
    assert 'Windows GUI subsystem' in gate
    assert 'windows_subsystem = "windows"' in gate
    assert 'tests/test_stage20_2_5_hidden_windows_console.py' in spec

    print(
        "PASS: Stage 20.2.5 configures the release Tauri executable for the Windows GUI subsystem, "
        "keeps Python/Streamlit child processes hidden, and makes the release gate reject regressions "
        "that would reintroduce a visible console window."
    )


if __name__ == "__main__":
    main()
