from __future__ import annotations

import argparse
import compileall
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "release" / "release_spec.json"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-release.txt"

EXCLUDED_TREE_PARTS = {
    ".git",
    ".build-cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "release-output",
    "target",
    "venv",
}

RUNTIME_SOURCE_ROOTS = (
    "app.py",
    "launch_streamlit.py",
    "_pages",
    "ai",
    "config",
    "core",
    "modules",
    "ui",
)

FORBIDDEN_SECRET_NAMES = {
    ".env",
    "auth.json",
    "credentials.json",
    "model_package_signing.key",
    "secrets.toml",
}

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    seconds: float
    details: str


class ReleaseGate:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []
        self.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def run(self, name: str, check: Callable[[], str]) -> None:
        started = time.perf_counter()
        try:
            details = check()
            passed = True
        except Exception as exc:  # release report must include every failure
            details = f"{type(exc).__name__}: {exc}"
            passed = False
        self.results.append(
            CheckResult(
                name=name,
                passed=passed,
                seconds=round(time.perf_counter() - started, 3),
                details=details,
            )
        )
        marker = "PASS" if passed else "FAIL"
        print(f"[{marker}] {name}: {details}")

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def report(self) -> dict[str, object]:
        return {
            "product": self.spec["product"],
            "version": self.spec["version"],
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python": sys.version,
            "passed": self.passed,
            "checks": [asdict(result) for result in self.results],
        }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _release_user_data_dir() -> Path:
    base = (
        os.environ.get("DATABRIDGE_USER_DATA_DIR")
        or os.environ.get("LOCALAPPDATA")
        or os.environ.get("APPDATA")
        or str(Path.home())
    )
    path = Path(base).expanduser()
    if path.name.casefold() != "databridgeai":
        path = path / "DataBridgeAI"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_legacy_project_auth(project_root: Path | None = None, target: Path | None = None) -> str:
    root = (project_root or PROJECT_ROOT).resolve()
    legacy = root / ".databridge" / "auth.json"
    destination = (target or (_release_user_data_dir() / "auth.json")).expanduser().resolve()
    if not legacy.exists():
        return "no legacy project auth file found"
    _assert(legacy.resolve() != destination, "Legacy auth path resolves to active auth path")

    try:
        raw = legacy.read_bytes()
        legacy_data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise AssertionError(f"Legacy auth file is invalid and was not deleted: {legacy}") from exc
    _assert(
        isinstance(legacy_data, dict) and bool(str(legacy_data.get("password_hash") or "").strip()),
        f"Legacy auth file is missing password_hash and was not deleted: {legacy}",
    )

    if destination.exists():
        try:
            current = json.loads(destination.read_text(encoding="utf-8"))
        except Exception as exc:
            raise AssertionError(f"Per-user auth file is invalid; refusing legacy cleanup: {destination}") from exc
        _assert(
            isinstance(current, dict) and bool(str(current.get("password_hash") or "").strip()),
            f"Per-user auth file is missing password_hash; refusing legacy cleanup: {destination}",
        )
        action = "removed stale project auth after validating per-user auth"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + ".migration.tmp")
        try:
            temp.write_bytes(raw)
            try:
                os.chmod(temp, 0o600)
            except OSError:
                pass
            os.replace(temp, destination)
            try:
                os.chmod(destination, 0o600)
            except OSError:
                pass
        finally:
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass
        action = "migrated project auth to per-user storage"

    try:
        legacy.unlink()
    except OSError as exc:
        raise AssertionError(f"Legacy project auth could not be removed after safe migration: {legacy}") from exc
    try:
        legacy.parent.rmdir()
    except OSError:
        pass
    return action


def _read_version_from_constants() -> str:
    text = (PROJECT_ROOT / "config" / "constants.py").read_text(encoding="utf-8")
    match = re.search(r'^APP_VERSION\s*=\s*["\']([^"\']+)["\']', text, flags=re.MULTILINE)
    _assert(match is not None, "APP_VERSION is missing from config/constants.py")
    return match.group(1)


def _read_version_from_cargo() -> str:
    text = (PROJECT_ROOT / "desktop" / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    _assert(match is not None, "Cargo package version is missing")
    return match.group(1)


def check_structure() -> str:
    required_files = [
        "app.py",
        "launch_streamlit.py",
        "modules/deployment_api.py",
        "_pages/deployment_api.py",
        "modules/remote_model_registry.py",
        "_pages/remote_model_registry.py",
        "requirements-release.txt",
        ".streamlit/config.toml",
        "desktop/package.json",
        "desktop/package-lock.json",
        "desktop/src-tauri/Cargo.toml",
        "desktop/src-tauri/tauri.conf.json",
        "desktop/src-tauri/src/main.rs",
        "tools/prepare_portable_python.ps1",
        "tools/release_gate.py",
        "release/release_spec.json",
    ]
    missing = [path for path in required_files if not (PROJECT_ROOT / path).is_file()]
    missing.extend(
        path for path in ReleaseGate().spec["required_runtime_directories"] if not (PROJECT_ROOT / path).is_dir()
    )
    _assert(not missing, "Missing required paths: " + ", ".join(sorted(missing)))

    page_files = list((PROJECT_ROOT / "_pages").glob("*.py"))
    _assert(len(page_files) >= 20, f"Expected at least 20 page modules, found {len(page_files)}")
    return f"required source/build paths exist; {len(page_files)} page modules found"


def check_version_consistency() -> str:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    tauri = json.loads((PROJECT_ROOT / "desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    package = json.loads((PROJECT_ROOT / "desktop/package.json").read_text(encoding="utf-8"))
    versions = {
        "release_spec": spec["version"],
        "python_constants": _read_version_from_constants(),
        "tauri": tauri["version"],
        "cargo": _read_version_from_cargo(),
        "npm": package["version"],
    }
    _assert(len(set(versions.values())) == 1, f"Version mismatch: {versions}")
    return f"all product versions are {spec['version']}"


def check_dependency_lock_shape() -> str:
    text = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    entries = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    _assert(len(entries) >= 18, "requirements-release.txt is unexpectedly small")
    bad = [line for line in entries if "==" not in line]
    _assert(not bad, "Unpinned direct dependencies: " + ", ".join(bad))

    _assert(any(line.startswith("xgboost-cpu==3.4.0") for line in entries), "Pinned xgboost-cpu==3.4.0 is required for the Windows production runtime")

    package = json.loads((PROJECT_ROOT / "desktop/package.json").read_text(encoding="utf-8"))
    node_specs = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
    floating = {name: version for name, version in node_specs.items() if re.search(r"[~^*><= ]", str(version))}
    _assert(not floating, f"Floating npm dependencies: {floating}")

    lock = json.loads((PROJECT_ROOT / "desktop/package-lock.json").read_text(encoding="utf-8"))
    root = lock.get("packages", {}).get("", {})
    _assert(root.get("version") == package["version"], "package-lock root version does not match package.json")
    _assert(root.get("dependencies") == package.get("dependencies"), "package-lock runtime dependencies differ")
    _assert(root.get("devDependencies") == package.get("devDependencies"), "package-lock dev dependencies differ")
    return f"{len(entries)} Python direct dependencies and npm dependencies are exactly pinned"


def check_tauri_release_contract() -> str:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    legacy_shadow = PROJECT_ROOT / "desktop" / "tauri.conf.json"
    _assert(
        not legacy_shadow.exists(),
        "Obsolete desktop/tauri.conf.json shadows the canonical desktop/src-tauri/tauri.conf.json",
    )
    config = json.loads((PROJECT_ROOT / "desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    resources = config["bundle"]["resources"]
    missing = [resource for resource in spec["required_tauri_resources"] if resource not in resources]
    _assert(not missing, "Tauri resources missing: " + ", ".join(missing))
    _assert(config["bundle"]["targets"] == ["nsis"], "Only the NSIS release target is allowed")
    _assert(config["bundle"]["windows"]["nsis"]["installMode"] == "currentUser", "Installer must remain per-user")

    security = config["app"]["security"]
    csp = security.get("csp") or {}
    _assert(csp.get("object-src") == "'none'", "CSP object-src must be none")
    _assert(csp.get("frame-ancestors") == "'none'", "CSP frame-ancestors must be none")
    _assert("http://127.0.0.1:*" in csp.get("connect-src", ""), "Dynamic loopback port is not allowed by CSP")
    _assert("ws://127.0.0.1:*" in csp.get("connect-src", ""), "Dynamic Streamlit WebSocket port is not allowed")

    rust = (PROJECT_ROOT / "desktop/src-tauri/src/main.rs").read_text(encoding="utf-8")
    _assert(
        rust.lstrip().startswith('#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]'),
        "Release desktop binary is not configured for the Windows GUI subsystem; a console window may appear",
    )
    _assert('TcpListener::bind(("127.0.0.1", 0))' in rust, "Desktop launcher does not reserve a dynamic loopback port")
    _assert('env("DATABRIDGE_PORT"' in rust, "Desktop launcher does not pass the selected port")
    _assert('"taskkill"' in rust and '"/T"' in rust, "Windows process-tree termination is missing")
    _assert(
        config["app"]["windows"][0].get("create") is False,
        "The main Tauri window must be created manually so its download handler can be attached",
    )
    _assert("WebviewWindowBuilder::from_config" in rust, "Manual main-window creation is missing")
    _assert(".on_download(" in rust, "Desktop Save As download interception is missing")
    _assert("rfd::FileDialog" in rust and ".save_file()" in rust, "Native Save As dialog is missing")
    return "resource bundle, CSP, per-user installer, dynamic loopback port, process cleanup, and native Save As downloads are configured"


def _iter_runtime_text_files() -> Iterable[Path]:
    allowed_suffixes = {".py", ".rs", ".json", ".toml", ".html", ".js", ".ts"}
    for root_name in RUNTIME_SOURCE_ROOTS:
        root = PROJECT_ROOT / root_name
        if root.is_file():
            yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            if any(part in EXCLUDED_TREE_PARTS for part in path.parts):
                continue
            yield path


def check_secret_hygiene() -> str:
    # A pre-Stage-12 install may still contain .databridge/auth.json even though
    # current releases store credentials in the per-user data directory. Remove
    # only that known legacy file after validating/migrating it. All other secret
    # files remain blocking release errors.
    cleanup_legacy_project_auth(PROJECT_ROOT)

    forbidden: list[str] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDED_TREE_PARTS for part in relative.parts):
            continue
        if relative.parts[:2] == ("resources", "python"):
            continue
        lower_name = path.name.lower()
        if lower_name in FORBIDDEN_SECRET_NAMES or path.suffix.lower() in {".pem", ".pfx", ".key"}:
            forbidden.append(str(relative))
    _assert(not forbidden, "Forbidden secret files found: " + ", ".join(sorted(forbidden)))

    pattern_hits: list[str] = []
    for path in _iter_runtime_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            pattern_hits.append(str(path.relative_to(PROJECT_ROOT)))
    _assert(not pattern_hits, "Potential plaintext secrets found in runtime source: " + ", ".join(pattern_hits))
    return "no forbidden credential files or high-confidence plaintext secret patterns found"


def check_python_compile() -> str:
    files: list[Path] = []
    for root_name in (*RUNTIME_SOURCE_ROOTS, "tools", "tests"):
        root = PROJECT_ROOT / root_name
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    failures: list[str] = []
    for path in sorted(set(files)):
        result = compileall.compile_file(str(path), quiet=1, force=True)
        if not result:
            failures.append(str(path.relative_to(PROJECT_ROOT)))
    _assert(not failures, "Python compile failures: " + ", ".join(failures))
    return f"compiled {len(files)} Python files"


def _read_locked_requirements() -> dict[str, str]:
    locked: dict[str, str] = {}
    for raw_line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement, _, marker = line.partition(";")
        marker_lower = marker.lower()
        if marker:
            is_windows = os.name == "nt"
            if "platform_system" in marker_lower:
                if "!= \"windows\"" in marker_lower and is_windows:
                    continue
                if "== \"windows\"" in marker_lower and not is_windows:
                    continue
        name, version = requirement.strip().split("==", 1)
        locked[name.strip()] = version.strip()
    return locked


def check_installed_dependencies() -> str:
    locked = _read_locked_requirements()
    mismatches: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for name, expected in locked.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    _assert(not missing, "Missing locked dependencies: " + ", ".join(sorted(missing)))
    _assert(not mismatches, "Dependency version mismatches: " + json.dumps(mismatches, sort_keys=True))
    return f"{len(locked)} installed direct dependencies match the production lock"


def check_portable_runtime() -> str:
    runtime = PROJECT_ROOT / "resources" / "python"
    required = [
        runtime / "python.exe",
        runtime / "pythonw.exe",
        runtime / "python312._pth",
        PROJECT_ROOT / "resources" / "runtime_manifest.json",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    _assert(not missing, "Portable runtime missing: " + ", ".join(missing))
    _assert(not (runtime / "pyvenv.cfg").exists(), "resources/python is still a machine-bound venv")
    pth = (runtime / "python312._pth").read_text(encoding="utf-8", errors="ignore")
    _assert("Lib\\site-packages" in pth or "Lib/site-packages" in pth, "portable runtime does not expose site-packages")
    _assert("import site" in pth, "portable runtime does not enable site initialization")

    command = [str(runtime / "python.exe"), "-I", "-c", "import streamlit,pandas,sklearn,pyarrow,keyring,cryptography,xgboost; print('ok')"]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=60, check=False)
    _assert(completed.returncode == 0, "Portable runtime import probe failed: " + (completed.stderr or completed.stdout).strip())
    return "portable CPython runtime exists, is not a venv, and imports production dependencies"


def check_stage_tests() -> str:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    tests: Sequence[str] = spec["stage_tests"]
    failures: list[str] = []
    for relative in tests:
        path = PROJECT_ROOT / relative
        _assert(path.is_file(), f"Required test is missing: {relative}")
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=240,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"},
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        if completed.returncode != 0 or "PASS:" not in completed.stdout:
            failures.append(f"{relative}: {output[-1600:]}")
        else:
            print(f"    PASS {relative}")
    _assert(not failures, "Stage test failures:\n" + "\n".join(failures))
    return f"all {len(tests)} stage tests passed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DataBridge AI Stage 20 production release gate")
    parser.add_argument("--skip-stage-tests", action="store_true", help="Skip the complete Stage 1-20 regression suite")
    parser.add_argument("--strict-dependencies", action="store_true", help="Require exact installed production dependency versions")
    parser.add_argument("--require-portable-runtime", action="store_true", help="Require and probe resources/python portable runtime")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "release-output" / "stage20_release_report.json",
        help="JSON report output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate = ReleaseGate()
    gate.run("Project structure", check_structure)
    gate.run("Version consistency", check_version_consistency)
    gate.run("Dependency lock shape", check_dependency_lock_shape)
    gate.run("Tauri release contract", check_tauri_release_contract)
    gate.run("Secret hygiene", check_secret_hygiene)
    gate.run("Python compile", check_python_compile)
    if args.strict_dependencies:
        gate.run("Installed dependency lock", check_installed_dependencies)
    if args.require_portable_runtime:
        gate.run("Portable Windows runtime", check_portable_runtime)
    if not args.skip_stage_tests:
        gate.run("Stage 1-20 regression suite", check_stage_tests)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(gate.report(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Release report: {args.report}")
    if gate.passed:
        print("PASS: Stage 20 production release gate completed successfully.")
        return 0
    print("FAIL: Stage 20 production release gate found blocking issues.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
