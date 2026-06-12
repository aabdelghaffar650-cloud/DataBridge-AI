import os
from pathlib import Path

def ensure_user_data_writable() -> None:
    base = Path(
        os.environ.get("DATABRIDGE_USER_DATA_DIR")
        or os.environ.get("LOCALAPPDATA")
        or os.environ.get("APPDATA")
        or str(Path.home())
    ) / "DataBridgeAI"
    base.mkdir(parents=True, exist_ok=True)
    test = base / ".write_test"
    test.write_text("ok", encoding="utf-8")
    try:
        test.unlink()
    except Exception:
        pass
