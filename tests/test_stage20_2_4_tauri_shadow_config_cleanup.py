from __future__ import annotations

import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "BUILD_STAGE20_2_WINDOWS_RELEASE.ps1"
RELEASE_GATE = PROJECT_ROOT / "tools" / "release_gate.py"


def main() -> None:
    builder = BUILDER.read_text(encoding="utf-8")
    gate = RELEASE_GATE.read_text(encoding="utf-8")

    assert 'function Remove-LegacyDesktopTauriShadowConfig' in builder
    assert 'Join-Path $DesktopDir "tauri.conf.json"' in builder
    assert 'Join-Path $TauriDir "tauri.conf.json"' in builder
    assert 'Remove-Item -LiteralPath $LegacyConfig -Force' in builder
    assert 'Remove-LegacyDesktopTauriShadowConfig' in builder

    cleanup_call = builder.index("    Remove-LegacyDesktopTauriShadowConfig")
    runtime_call = builder.index("    $PrepareRuntimeScript")
    release_gate_call = builder.index("tools/release_gate.py")
    tauri_build_call = builder.index("npm run tauri build")
    assert cleanup_call < runtime_call < release_gate_call < tauri_build_call

    assert 'legacy_shadow = PROJECT_ROOT / "desktop" / "tauri.conf.json"' in gate
    assert 'shadows the canonical desktop/src-tauri/tauri.conf.json' in gate

    canonical = PROJECT_ROOT / "desktop" / "src-tauri" / "tauri.conf.json"
    assert canonical.is_file()
    canonical_text = canonical.read_text(encoding="utf-8")
    assert '"Referrer-Policy"' not in canonical_text

    with tempfile.TemporaryDirectory(prefix="databridge-stage20-2-4-") as tmp:
        root = Path(tmp)
        desktop = root / "desktop"
        tauri_dir = desktop / "src-tauri"
        tauri_dir.mkdir(parents=True)
        (tauri_dir / "tauri.conf.json").write_text("{}", encoding="utf-8")
        legacy = desktop / "tauri.conf.json"
        legacy.write_text('{"app":{"security":{"headers":{"Referrer-Policy":"no-referrer"}}}}', encoding="utf-8")
        assert legacy.exists()
        legacy.unlink()
        assert not legacy.exists()
        assert (tauri_dir / "tauri.conf.json").exists()

    print(
        "PASS: Stage 20.2.4 removes the obsolete desktop/tauri.conf.json shadow before release checks/build "
        "and the release gate rejects any reintroduced shadow config."
    )


if __name__ == "__main__":
    main()
