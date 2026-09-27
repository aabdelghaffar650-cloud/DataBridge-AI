"""DataBridge AI Stage 18.1 Streamlit/Pandas/PyArrow compatibility checks."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_compat_module():
    module_path = PROJECT_ROOT / "ui" / "streamlit_compat.py"
    spec = importlib.util.spec_from_file_location("databridge_streamlit_compat_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mixed_object_columns_are_display_safe_without_source_mutation() -> None:
    compat = _load_compat_module()
    source = pd.DataFrame(
        {
            "Video Views": ["1200", 3400, None, "5100"],
            "Stable Numeric": [1, 2, 3, 4],
            "Metadata": [{"source": "web"}, {"source": "app"}, None, {"source": "api"}],
        }
    )
    original = source.copy(deep=True)

    prepared = compat.prepare_dataframe_for_display(source)

    assert source.equals(original)
    assert source is not prepared
    assert str(prepared["Video Views"].dtype).startswith("string")
    assert str(prepared["Metadata"].dtype).startswith("string")
    assert prepared["Stable Numeric"].dtype == source["Stable Numeric"].dtype
    assert prepared.loc[1, "Video Views"] == "3400"

    # This is the serialization step that emitted the original Streamlit warning.
    table = pa.Table.from_pandas(prepared, preserve_index=True)
    assert table.num_rows == len(source)


def test_streamlit_deprecations_and_accessibility_warning_are_removed() -> None:
    roots = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "_pages",
        PROJECT_ROOT / "ui",
        PROJECT_ROOT / "core",
        PROJECT_ROOT / "modules",
    ]
    source_files: list[Path] = []
    for root in roots:
        if root.is_file():
            source_files.append(root)
        else:
            source_files.extend(root.rglob("*.py"))

    offenders = []
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        if "use_container_width" in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, f"Deprecated use_container_width remains in: {offenders}"

    app_text = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert 'st.file_uploader("Upload dataset file"' in app_text
    assert not re.search(r"st\.file_uploader\(\s*[\"']\s*[\"']", app_text)


def test_manual_browser_mode_and_desktop_headless_mode_are_separated() -> None:
    config_text = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    launcher = (PROJECT_ROOT / "launch_streamlit.py").read_text(encoding="utf-8")

    assert 'address = "127.0.0.1"' in config_text
    assert "headless = false" in config_text
    assert 'os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")' in launcher
    assert '"--server.headless=true"' in launcher
    assert 'APP_HOST = "127.0.0.1"' in launcher


def test_release_gate_includes_stage18_1() -> None:
    spec = json.loads((PROJECT_ROOT / "release" / "release_spec.json").read_text(encoding="utf-8"))
    assert "tests/test_stage18_1_streamlit_compat.py" in spec["stage_tests"]


def main() -> None:
    test_mixed_object_columns_are_display_safe_without_source_mutation()
    test_streamlit_deprecations_and_accessibility_warning_are_removed()
    test_manual_browser_mode_and_desktop_headless_mode_are_separated()
    test_release_gate_includes_stage18_1()
    print(
        "PASS: Stage 18.1 separates manual browser and desktop headless startup, removes deprecated Streamlit width calls and empty widget labels, and renders mixed-type DataFrames through a non-mutating Arrow-safe display layer."
    )


if __name__ == "__main__":
    main()
