from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config_path = PROJECT_ROOT / "desktop" / "src-tauri" / "tauri.conf.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    security = config["app"]["security"]
    csp = security.get("csp")
    assert isinstance(csp, dict) and csp, f"CSP is missing or disabled in {config_path}"
    assert csp.get("object-src") == "'none'"
    assert csp.get("base-uri") == "'none'"
    assert csp.get("frame-ancestors") == "'none'"
    assert security.get("dangerousDisableAssetCspModification") is False
    headers = security.get("headers") or {}
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert "Referrer-Policy" not in headers
    assert isinstance(config.get("version"), str) and config["version"]
    print(
        "PASS: Stage 12.1 force-enables a non-null Tauri CSP, blocks objects/base/frame ancestors, "
        "keeps Tauri CSP asset injection enabled, and uses supported security headers."
    )


if __name__ == "__main__":
    main()
