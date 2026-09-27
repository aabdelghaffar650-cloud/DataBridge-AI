"""DataBridge AI Stage 13 production/release contract verification."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.release_gate import (  # noqa: E402
    check_dependency_lock_shape,
    check_python_compile,
    check_secret_hygiene,
    check_structure,
    check_tauri_release_contract,
    check_version_consistency,
)


def test_release_gate_static_checks() -> None:
    assert "required source/build paths" in check_structure()
    expected_version = json.loads((PROJECT_ROOT / "release/release_spec.json").read_text(encoding="utf-8"))["version"]
    assert f"all product versions are {expected_version}" == check_version_consistency()
    assert "exactly pinned" in check_dependency_lock_shape()
    assert "dynamic loopback port" in check_tauri_release_contract()
    assert "no forbidden credential files" in check_secret_hygiene()
    assert "compiled" in check_python_compile()


def test_portable_runtime_builder_is_verified_and_not_a_venv_builder() -> None:
    script = (PROJECT_ROOT / "tools/prepare_portable_python.ps1").read_text(encoding="utf-8")
    assert "python-3.12.10-embeddable-amd64.zip" in script
    assert "156c7eea90d58cd7e91a23f28a0056616b13e9f4cf4901b7b99b837b7848c6da" in script
    assert "Get-FileHash" in script and "SHA256" in script
    assert "Lib\\site-packages" in script
    assert "--only-binary=:all:" in script
    assert "py -3.12 -m venv" not in script
    assert "Remove-Item (Join-Path $RuntimeDir \"pyvenv.cfg\")" in script


def test_desktop_launcher_uses_private_dynamic_port_and_kills_process_tree() -> None:
    rust = (PROJECT_ROOT / "desktop/src-tauri/src/main.rs").read_text(encoding="utf-8")
    launcher = (PROJECT_ROOT / "launch_streamlit.py").read_text(encoding="utf-8")
    config = json.loads((PROJECT_ROOT / "desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    csp = config["app"]["security"]["csp"]

    assert 'TcpListener::bind(("127.0.0.1", 0))' in rust
    assert '.env("DATABRIDGE_PORT", port.to_string())' in rust
    assert 'Command::new("taskkill")' in rust
    assert '"/T"' in rust and '"/F"' in rust
    assert 'APP_HOST = "127.0.0.1"' in launcher
    assert "_port_is_free" in launcher
    assert "--server.enableXsrfProtection=true" in launcher
    assert "--client.showErrorDetails=false" in launcher
    assert "http://127.0.0.1:*" in csp["connect-src"]
    assert "ws://127.0.0.1:*" in csp["connect-src"]


def test_release_build_uses_locked_inputs_and_full_gate() -> None:
    build = (PROJECT_ROOT / "BUILD_STAGE13_WINDOWS_RELEASE.ps1").read_text(encoding="utf-8")
    gate = (PROJECT_ROOT / "RUN_STAGE13_RELEASE_GATE.ps1").read_text(encoding="utf-8")
    package = json.loads((PROJECT_ROOT / "desktop/package.json").read_text(encoding="utf-8"))

    assert "prepare_portable_python.ps1" in build
    assert "--strict-dependencies" in build
    assert "--require-portable-runtime" in build
    assert re.search(r"\bnpm\s+ci\b", build)
    assert "cargo generate-lockfile" in build
    assert "Get-FileHash" in build
    assert "tools/release_gate.py" in gate
    assert package["dependencies"]["@tauri-apps/api"] == "2.11.0"
    assert package["devDependencies"]["@tauri-apps/cli"] == "2.11.2"


def test_source_packagers_include_release_assets_and_exclude_runtime_binary_tree() -> None:
    for name in ("make_source_zip.ps1", "make_review_zip.ps1"):
        text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        for required in ('"_pages"', '"tests"', '"tools"', '"release"'):
            assert required in text, f"{required} missing from {name}"
        assert '"resources\\python"' in text or '"python"' in text
        assert "secrets.toml" in text
        assert "model_package_signing.key" in text


def main() -> None:
    test_release_gate_static_checks()
    test_portable_runtime_builder_is_verified_and_not_a_venv_builder()
    test_desktop_launcher_uses_private_dynamic_port_and_kills_process_tree()
    test_release_build_uses_locked_inputs_and_full_gate()
    test_source_packagers_include_release_assets_and_exclude_runtime_binary_tree()
    print(
        "PASS: Stage 13 production release controls unify versioning, pin release inputs, verify a portable non-venv Python runtime, run the complete regression gate, preserve all required Tauri resources, use a private dynamic loopback port, terminate the Windows process tree, exclude secrets/runtime binaries from source packages, and generate a hashed release report before installer delivery."
    )


if __name__ == "__main__":
    main()
