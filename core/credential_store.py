"""Secure secret storage for DataBridge AI.

Persistent secrets are stored through the operating-system credential backend
provided by ``keyring``. No plaintext file fallback is used.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


SERVICE_NAME = "DataBridgeAI"
SUPPORTED_SECRETS = {
    "anthropic_api_key": ("DATABRIDGE_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    "gemini_api_key": ("DATABRIDGE_GEMINI_API_KEY", "GEMINI_API_KEY"),
    "deployment_api_token": ("DATABRIDGE_DEPLOYMENT_API_TOKEN",),
    "remote_registry_token": ("DATABRIDGE_REMOTE_REGISTRY_TOKEN",),
}


class SecretStoreError(RuntimeError):
    """Raised when the secure credential backend cannot complete an operation."""


@dataclass(frozen=True)
class SecretStatus:
    name: str
    configured: bool
    source: str
    backend: str


class CredentialStore:
    """Small, testable wrapper around the active OS keyring backend."""

    def __init__(self, service_name: str = SERVICE_NAME, backend: Any | None = None):
        self.service_name = str(service_name or SERVICE_NAME)
        self._backend = backend

    @staticmethod
    def _validate_name(name: str) -> str:
        clean = str(name or "").strip()
        if clean not in SUPPORTED_SECRETS:
            raise SecretStoreError("Unsupported secret identifier.")
        return clean

    def _keyring(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            import keyring  # type: ignore
        except Exception as exc:
            raise SecretStoreError(
                "Secure credential storage is unavailable. Install the keyring dependency."
            ) from exc
        return keyring

    def backend_name(self) -> str:
        try:
            backend = self._keyring()
            if hasattr(backend, "get_keyring"):
                active = backend.get_keyring()
                priority = getattr(active, "priority", 0)
                if priority is not None and float(priority) <= 0:
                    return "Unavailable"
                return f"{active.__class__.__module__}.{active.__class__.__name__}"
            return backend.__class__.__name__
        except Exception:
            return "Unavailable"

    def is_available(self) -> bool:
        return self.backend_name() != "Unavailable"

    def get_secret(self, name: str) -> tuple[Optional[str], str]:
        clean = self._validate_name(name)
        for env_name in SUPPORTED_SECRETS[clean]:
            value = os.getenv(env_name, "").strip()
            if value:
                return value, f"environment:{env_name}"
        try:
            value = self._keyring().get_password(self.service_name, clean)
        except Exception as exc:
            raise SecretStoreError("The secure credential backend could not read the secret.") from exc
        return (str(value), "credential_manager") if value else (None, "not_configured")

    def set_secret(self, name: str, value: str) -> None:
        clean = self._validate_name(name)
        secret = str(value or "").strip()
        if len(secret) < 8:
            raise SecretStoreError("The secret is empty or too short to save safely.")
        try:
            self._keyring().set_password(self.service_name, clean, secret)
        except Exception as exc:
            raise SecretStoreError("The secure credential backend could not save the secret.") from exc

    def delete_secret(self, name: str) -> bool:
        clean = self._validate_name(name)
        try:
            existing = self._keyring().get_password(self.service_name, clean)
            if not existing:
                return False
            self._keyring().delete_password(self.service_name, clean)
            return True
        except Exception as exc:
            raise SecretStoreError("The secure credential backend could not delete the secret.") from exc

    def status(self, name: str) -> SecretStatus:
        clean = self._validate_name(name)
        try:
            value, source = self.get_secret(clean)
            return SecretStatus(
                name=clean,
                configured=bool(value),
                source=source,
                backend=self.backend_name(),
            )
        except SecretStoreError:
            return SecretStatus(
                name=clean,
                configured=False,
                source="unavailable",
                backend="Unavailable",
            )


def get_credential_store() -> CredentialStore:
    return CredentialStore()
