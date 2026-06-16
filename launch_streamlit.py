import os, sys, time, socket, subprocess
from pathlib import Path

APP_PORT = int(os.environ.get("DATABRIDGE_PORT", "8501"))
APP_HOST = "127.0.0.1"

def log(msg):
    try:
        d = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "DataBridgeAI"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "streamlit_launcher.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except Exception:
        pass

def wait_port(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((APP_HOST, APP_PORT), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def main():
    root = Path(__file__).resolve().parent
    app_py = root / "app.py"
    log(f"root={root}")
    log(f"python={sys.executable}")
    log(f"app_py={app_py}")
    if not app_py.exists():
        log("ERROR app.py not found")
        return 2

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("DATABRIDGE_USER_DATA_DIR", str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "DataBridgeAI"))

    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "DataBridgeAI"
    log_dir.mkdir(parents=True, exist_ok=True)
    runtime_log = log_dir / "streamlit_runtime.log"

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_py),
           "--server.headless=true",
           "--global.developmentMode=false", f"--server.address={APP_HOST}",
           f"--server.port={APP_PORT}", "--browser.gatherUsageStats=false"]
    log("cmd=" + " ".join(cmd))
    f = open(runtime_log, "a", encoding="utf-8", errors="ignore")
    proc = subprocess.Popen(cmd, cwd=str(root), stdout=f, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if wait_port():
        log("READY")
    else:
        log("ERROR server not ready")
    return proc.wait()

if __name__ == "__main__":
    raise SystemExit(main())
