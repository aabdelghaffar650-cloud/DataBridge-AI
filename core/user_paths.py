from pathlib import Path
import os
import tempfile

APP_DIR_NAME = "DataBridgeAI"

def user_data_dir() -> Path:
    base = (
        os.environ.get("DATABRIDGE_USER_DATA_DIR")
        or os.environ.get("LOCALAPPDATA")
        or os.environ.get("APPDATA")
        or str(Path.home())
    )
    path = Path(base) / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path

def logs_dir() -> Path:
    path = user_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def temp_dir() -> Path:
    path = Path(tempfile.gettempdir()) / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path

def auth_file() -> Path:
    return user_data_dir() / "auth.json"

def settings_file() -> Path:
    return user_data_dir() / "settings.json"
