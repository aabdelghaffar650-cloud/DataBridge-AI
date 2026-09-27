from __future__ import annotations

import os
import tempfile
from pathlib import Path

APP_DIR_NAME = "DataBridgeAI"


def _normalise_app_dir(base: str | Path) -> Path:
    path = Path(base).expanduser()
    if path.name.casefold() != APP_DIR_NAME.casefold():
        path = path / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_data_dir() -> Path:
    base = (
        os.environ.get("DATABRIDGE_USER_DATA_DIR")
        or os.environ.get("LOCALAPPDATA")
        or os.environ.get("APPDATA")
        or str(Path.home())
    )
    return _normalise_app_dir(base)


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


def signing_key_file() -> Path:
    return user_data_dir() / "model_package_signing.key"


def model_registry_dir() -> Path:
    path = user_data_dir() / "model_registry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def monitoring_reports_dir() -> Path:
    path = user_data_dir() / "monitoring_reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def scheduled_monitoring_dir() -> Path:
    path = user_data_dir() / "scheduled_monitoring"
    path.mkdir(parents=True, exist_ok=True)
    return path


def scheduled_monitoring_reports_dir() -> Path:
    path = monitoring_reports_dir() / "scheduled"
    path.mkdir(parents=True, exist_ok=True)
    return path


def deployment_api_dir() -> Path:
    path = user_data_dir() / "deployment_api"
    path.mkdir(parents=True, exist_ok=True)
    return path


def deployment_api_logs_dir() -> Path:
    path = deployment_api_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def remote_model_registry_dir() -> Path:
    path = user_data_dir() / "remote_model_registry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def team_auth_db() -> Path:
    """Stage 20 local multi-user identity database."""
    return user_data_dir() / "team_auth.db"
