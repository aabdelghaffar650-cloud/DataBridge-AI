from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Optional

APP_HOST = "127.0.0.1"
APP_PORT = int(os.environ.get("DATABRIDGE_PORT", "8501"))
_STARTUP_TIMEOUT_SECONDS = int(os.environ.get("DATABRIDGE_STARTUP_TIMEOUT", "120"))

_process: Optional[subprocess.Popen] = None
_runtime_log_handle: Optional[IO[str]] = None


def _user_data_dir() -> Path:
    configured = os.environ.get("DATABRIDGE_USER_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    return (base / "DataBridgeAI").resolve()


def log(message: str) -> None:
    try:
        directory = _user_data_dir()
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "streamlit_launcher.log").open("a", encoding="utf-8") as handle:
            handle.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")
    except Exception:
        pass


def _port_is_free() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((APP_HOST, APP_PORT))
        except OSError:
            return False
    return True


def _wait_until_ready(timeout: int = _STARTUP_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _process is not None and _process.poll() is not None:
            return False
        try:
            with socket.create_connection((APP_HOST, APP_PORT), timeout=1):
                return True
        except OSError:
            time.sleep(0.35)
    return False


def _terminate_child() -> None:
    global _process, _runtime_log_handle
    proc = _process
    _process = None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
    if _runtime_log_handle is not None:
        try:
            _runtime_log_handle.close()
        except Exception:
            pass
        _runtime_log_handle = None


def _handle_signal(signum: int, _frame) -> None:
    log(f"received signal={signum}; stopping Streamlit")
    _terminate_child()
    raise SystemExit(0)


def main() -> int:
    global _process, _runtime_log_handle

    root = Path(__file__).resolve().parent
    app_py = root / "app.py"
    log(f"root={root}")
    log(f"python={sys.executable}")
    log(f"app_py={app_py}")
    log(f"endpoint=http://{APP_HOST}:{APP_PORT}")

    if not app_py.is_file():
        log("ERROR app.py not found")
        return 2
    if not (1 <= APP_PORT <= 65535):
        log(f"ERROR invalid port {APP_PORT}")
        return 3
    if not _port_is_free():
        log(f"ERROR port {APP_PORT} is already in use")
        return 4

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("DATABRIDGE_USER_DATA_DIR", str(_user_data_dir()))

    log_dir = _user_data_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    runtime_log = log_dir / "streamlit_runtime.log"

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_py),
        "--server.headless=true",
        "--global.developmentMode=false",
        f"--server.address={APP_HOST}",
        f"--server.port={APP_PORT}",
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
        "--client.showErrorDetails=false",
        "--client.toolbarMode=minimal",
    ]
    log("cmd=" + " ".join(command))

    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    _runtime_log_handle = runtime_log.open("a", encoding="utf-8", errors="ignore")
    try:
        _process = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=_runtime_log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            env=os.environ.copy(),
        )
    except Exception as exc:
        log(f"ERROR failed to start Streamlit: {type(exc).__name__}: {exc}")
        _terminate_child()
        return 5

    if not _wait_until_ready():
        return_code = _process.poll() if _process is not None else None
        log(f"ERROR server not ready; child_return_code={return_code}")
        _terminate_child()
        return 6

    log("READY")
    return_code = _process.wait()
    log(f"EXIT return_code={return_code}")
    _terminate_child()
    return int(return_code or 0)


atexit.register(_terminate_child)
for _signal_name in ("SIGTERM", "SIGINT", "SIGBREAK"):
    _signal_value = getattr(signal, _signal_name, None)
    if _signal_value is not None:
        try:
            signal.signal(_signal_value, _handle_signal)
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
