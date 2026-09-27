from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARE = PROJECT_ROOT / "tools" / "prepare_portable_python.ps1"
SPEC = PROJECT_ROOT / "release" / "release_spec.json"


def main() -> int:
    script = PREPARE.read_text(encoding="utf-8")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))

    # Windows PowerShell 5.1 can mangle embedded quotes when a multiline script is
    # passed as a native-process `-c` argument. Verification must execute from a
    # temporary .py file instead so Python receives the source byte-for-byte.
    assert '& $RuntimePython -I -c $VerifyCode' not in script
    assert '$VerifyScript = Join-Path $CacheDir "verify_portable_runtime.py"' in script
    assert 'Set-Content -Path $VerifyScript -Value $VerifyCode -Encoding ASCII' in script
    assert '& $RuntimePython -I $VerifyScript' in script
    assert 'Remove-Item $VerifyScript -Force -ErrorAction SilentlyContinue' in script

    # Preserve the pinned-runtime and manifest verification contract.
    for expected in (
        '"streamlit": "1.58.0"',
        '"pandas": "3.0.3"',
        '"scikit-learn": "1.9.0"',
        'Path("resources/runtime_manifest.json")',
        'Portable runtime verification passed.',
    ):
        assert expected in script, f"Missing runtime verification contract: {expected}"

    assert "tests/test_stage20_2_2_portable_runtime_verify_script.py" in spec["stage_tests"]

    print(
        "PASS: Stage 20.2.2 executes portable-runtime verification from a temporary Python file, "
        "avoiding Windows PowerShell native-argument quote corruption while preserving pinned-version checks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
