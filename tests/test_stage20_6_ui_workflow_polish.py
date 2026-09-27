"""DataBridge AI Stage 20.6 UI/workflow polish regression verification.

Run from project root:
    python tests/test_stage20_6_ui_workflow_polish.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_nested_tabs_are_visually_button_like() -> None:
    styles = (PROJECT_ROOT / "ui" / "styles.py").read_text(encoding="utf-8")
    required = [
        'background: var(--surface-2) !important;',
        'border: 1px solid var(--border-soft) !important;',
        'border-radius: 11px !important;',
        '[data-baseweb="tab"][aria-selected="true"]',
        'border-color: var(--primary) !important;',
        '[data-baseweb="tab-highlight"]',
    ]
    for marker in required:
        assert marker in styles, marker


def test_feature_engineering_surfaces_replayable_derivations() -> None:
    page = (PROJECT_ROOT / "_pages" / "feature_engineering.py").read_text(encoding="utf-8")
    builder = (PROJECT_ROOT / "ui" / "feature_derivation_builder.py").read_text(encoding="utf-8")

    assert "render_replayable_feature_builder" in page
    assert '#### 2. Assign every source column to a safe transformation' in page
    assert '#### 3. Configure train-only transformers' in page
    assert '#### 4. Validate and save' in page

    required = [
        "Computed Column",
        "Regex Extract",
        "Value Mapping",
        "build_formula_derivation_step",
        "build_regex_extract_derivation_step",
        "build_value_map_derivation_step",
        "append_derivation_step",
        "apply_dataset_change",
        'context_updates={"feature_derivation_recipe": next_recipe}',
        "protected Raw stays unchanged",
    ]
    for marker in required:
        assert marker in builder, marker

    # Backward compatibility: the original Stage 20.5 controls remain reachable
    # from Replace Values as well.
    replace_page = (PROJECT_ROOT / "_pages" / "replace_values.py").read_text(encoding="utf-8")
    assert "Text / Regex Feature" in replace_page
    assert "Add Computed Column" in replace_page
    assert "Add Regex Extract Feature" in replace_page


def test_active_dataset_can_be_removed_without_deleting_source() -> None:
    dataset = (PROJECT_ROOT / "core" / "dataset.py").read_text(encoding="utf-8")
    data_sources = (PROJECT_ROOT / "_pages" / "data_sources.py").read_text(encoding="utf-8")

    assert "def deactivate_dataset()" in dataset
    deactivate_body = dataset.split("def deactivate_dataset()", 1)[1].split("def get_working_dataframe", 1)[0]
    assert "reset_file_state()" in deactivate_body
    assert 'st.session_state["pre_dataset_page"] = "upload"' in deactivate_body
    assert "source file" in deactivate_body.lower()
    assert "unlink(" not in deactivate_body
    assert "remove(" not in deactivate_body

    assert "Remove active dataset" in data_sources
    assert "Confirm remove" in data_sources
    assert "deactivate_dataset()" in data_sources
    assert "The original source file is not deleted" in data_sources
    assert "_data_sources_remove_pending" in data_sources


def test_desktop_downloads_use_native_save_as_dialog() -> None:
    config = json.loads(
        (PROJECT_ROOT / "desktop" / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
    )
    cargo = (PROJECT_ROOT / "desktop" / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8")
    rust = (PROJECT_ROOT / "desktop" / "src-tauri" / "src" / "main.rs").read_text(encoding="utf-8")
    gate = (PROJECT_ROOT / "tools" / "release_gate.py").read_text(encoding="utf-8")

    assert config["app"]["windows"][0]["create"] is False
    assert 'rfd = "=0.17.2"' in cargo
    assert "WebviewWindowBuilder::from_config" in rust
    assert ".on_download(" in rust
    assert "DownloadEvent::Requested" in rust
    assert "rfd::FileDialog::new()" in rust
    assert '.set_title("Save DataBridge AI download as")' in rust
    assert ".save_file()" in rust
    assert "*destination = chosen_path" in rust
    assert "None =>" in rust and "false" in rust
    assert "Desktop Save As download interception is missing" in gate


def test_release_spec_contains_stage20_6_regression() -> None:
    spec = json.loads((PROJECT_ROOT / "release" / "release_spec.json").read_text(encoding="utf-8"))
    assert "tests/test_stage20_6_ui_workflow_polish.py" in spec["stage_tests"]


def main() -> None:
    test_nested_tabs_are_visually_button_like()
    test_feature_engineering_surfaces_replayable_derivations()
    test_active_dataset_can_be_removed_without_deleting_source()
    test_desktop_downloads_use_native_save_as_dialog()
    test_release_spec_contains_stage20_6_regression()
    print(
        "PASS: Stage 20.6 makes nested tabs visibly clickable, adds a native desktop Save As flow, "
        "adds safe active-dataset removal, and surfaces replayable Computed/Regex/Value-Mapping "
        "feature derivations directly inside Feature Engineering without removing Stage 20.5 compatibility controls."
    )


if __name__ == "__main__":
    main()
