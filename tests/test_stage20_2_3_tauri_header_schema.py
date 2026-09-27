from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAURI_DIR = PROJECT_ROOT / "desktop" / "src-tauri"

# Tauri v2 app.security.headers names documented by Tauri. Keep
# Tauri-Custom-Header out of production configuration even though the schema
# accepts it for testing/development scenarios.
SUPPORTED_PRODUCTION_HEADERS = {
    "Access-Control-Allow-Credentials",
    "Access-Control-Allow-Headers",
    "Access-Control-Allow-Methods",
    "Access-Control-Expose-Headers",
    "Access-Control-Max-Age",
    "Cross-Origin-Embedder-Policy",
    "Cross-Origin-Opener-Policy",
    "Cross-Origin-Resource-Policy",
    "Permissions-Policy",
    "Service-Worker-Allowed",
    "Timing-Allow-Origin",
    "X-Content-Type-Options",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _headers(config: dict) -> dict:
    return (((config.get("app") or {}).get("security") or {}).get("headers") or {})


def main() -> None:
    config_files = [TAURI_DIR / "tauri.conf.json"]
    config_files.extend(sorted(TAURI_DIR.glob("tauri.*.conf.json")))

    checked = []
    for path in config_files:
        if not path.is_file():
            continue
        config = _load(path)
        headers = _headers(config)
        assert isinstance(headers, dict), f"Tauri headers must be an object: {path}"
        unsupported = sorted(set(headers) - SUPPORTED_PRODUCTION_HEADERS)
        assert not unsupported, (
            f"Unsupported Tauri app.security.headers in {path.name}: {unsupported}. "
            "Referrer-Policy is not supported by the Tauri v2 header schema."
        )
        checked.append(path.name)

    main_config = _load(TAURI_DIR / "tauri.conf.json")
    main_headers = _headers(main_config)
    assert main_headers.get("X-Content-Type-Options") == "nosniff"
    assert main_headers.get("Permissions-Policy") == "camera=(), microphone=(), geolocation=(), payment=()"
    assert "Referrer-Policy" not in main_headers

    print(
        "PASS: Stage 20.2.3 validates Tauri security headers across base/platform configs "
        f"and blocks unsupported header names ({', '.join(checked)})."
    )


if __name__ == "__main__":
    main()
