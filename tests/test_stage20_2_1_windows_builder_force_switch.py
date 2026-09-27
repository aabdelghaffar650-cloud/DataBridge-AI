from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "BUILD_STAGE20_2_WINDOWS_RELEASE.ps1"
PREPARE = PROJECT_ROOT / "tools" / "prepare_portable_python.ps1"


def main() -> int:
    builder = BUILDER.read_text(encoding="utf-8")
    prepare = PREPARE.read_text(encoding="utf-8")

    assert "[switch]$ForceRuntime" in builder, "Builder lost the ForceRuntime switch contract"
    assert "[switch]$Force" in prepare, "Portable runtime builder lost its Force switch contract"

    # Passing a SwitchParameter through powershell.exe as -Force:$ForceRuntime turns
    # the value into a string argument under Windows PowerShell. The child script
    # must receive the switch token only when the caller explicitly requested it.
    assert "-Force:$ForceRuntime" not in builder, "Unsafe SwitchParameter forwarding is still present"
    assert '$PrepareRuntimeArgs += "-Force"' in builder, "Conditional -Force token forwarding is missing"
    assert "if ($ForceRuntime)" in builder, "-Force must only be appended when ForceRuntime is enabled"
    assert "& powershell @PrepareRuntimeArgs" in builder, "Portable runtime child process must use the argument array"

    # Keep the surrounding production-build safety controls intact.
    assert "-AllowUnsigned" in builder or "[switch]$AllowUnsigned" in builder
    assert "--strict-dependencies" in builder
    assert "--require-portable-runtime" in builder
    assert "npm ci" in builder
    assert "npm run tauri build" in builder

    print("PASS: Stage 20.2.1 forwards the portable-runtime Force switch safely without string coercion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
