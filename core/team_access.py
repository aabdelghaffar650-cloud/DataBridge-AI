"""Role-based access control for DataBridge AI team mode.

Stage 20 keeps permissions explicit and centrally enforced. Roles are convenience
bundles; application code should check permissions, not role names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


ROLE_ADMIN = "admin"
ROLE_MANAGER = "manager"
ROLE_DATA_SCIENTIST = "data_scientist"
ROLE_ANALYST = "analyst"
ROLE_VIEWER = "viewer"
VALID_ROLES = (
    ROLE_ADMIN,
    ROLE_MANAGER,
    ROLE_DATA_SCIENTIST,
    ROLE_ANALYST,
    ROLE_VIEWER,
)

ROLE_LABELS = {
    ROLE_ADMIN: "Admin",
    ROLE_MANAGER: "Manager / Approver",
    ROLE_DATA_SCIENTIST: "Data Scientist",
    ROLE_ANALYST: "Analyst",
    ROLE_VIEWER: "Viewer",
}

PERM_ACCOUNT_SELF = "account.self"
PERM_DATA_READ = "data.read"
PERM_DATA_IMPORT = "data.import"
PERM_DATA_TRANSFORM = "data.transform"
PERM_DATA_EXPORT = "data.export"
PERM_ML_TRAIN = "ml.train"
PERM_ML_EXPLAIN = "ml.explain"
PERM_MODEL_PREDICT = "model.predict"
PERM_MODEL_MONITOR = "model.monitor"
PERM_MONITOR_SCHEDULE = "monitor.schedule"
PERM_MODEL_RETRAIN = "model.retrain"
PERM_MODEL_SUBMIT = "model.governance.submit"
PERM_MODEL_APPROVE = "model.governance.approve"
PERM_DEPLOYMENT_MANAGE = "deployment.manage"
PERM_REMOTE_REGISTRY_MANAGE = "registry.remote.manage"
PERM_AI_USE = "ai.use"
PERM_SECRETS_MANAGE = "secrets.manage"
PERM_SIGNING_KEY_MANAGE = "signing_key.manage"
PERM_AUDIT_VIEW = "audit.view"
PERM_TEAM_MANAGE = "team.manage"

ALL_PERMISSIONS = frozenset(
    {
        PERM_ACCOUNT_SELF,
        PERM_DATA_READ,
        PERM_DATA_IMPORT,
        PERM_DATA_TRANSFORM,
        PERM_DATA_EXPORT,
        PERM_ML_TRAIN,
        PERM_ML_EXPLAIN,
        PERM_MODEL_PREDICT,
        PERM_MODEL_MONITOR,
        PERM_MONITOR_SCHEDULE,
        PERM_MODEL_RETRAIN,
        PERM_MODEL_SUBMIT,
        PERM_MODEL_APPROVE,
        PERM_DEPLOYMENT_MANAGE,
        PERM_REMOTE_REGISTRY_MANAGE,
        PERM_AI_USE,
        PERM_SECRETS_MANAGE,
        PERM_SIGNING_KEY_MANAGE,
        PERM_AUDIT_VIEW,
        PERM_TEAM_MANAGE,
    }
)

ROLE_PERMISSIONS = {
    ROLE_ADMIN: ALL_PERMISSIONS,
    ROLE_MANAGER: frozenset(
        {
            PERM_ACCOUNT_SELF,
            PERM_DATA_READ,
            PERM_DATA_EXPORT,
            PERM_MODEL_PREDICT,
            PERM_MODEL_MONITOR,
            PERM_MONITOR_SCHEDULE,
            PERM_MODEL_APPROVE,
            PERM_AI_USE,
            PERM_AUDIT_VIEW,
        }
    ),
    ROLE_DATA_SCIENTIST: frozenset(
        {
            PERM_ACCOUNT_SELF,
            PERM_DATA_READ,
            PERM_DATA_IMPORT,
            PERM_DATA_TRANSFORM,
            PERM_DATA_EXPORT,
            PERM_ML_TRAIN,
            PERM_ML_EXPLAIN,
            PERM_MODEL_PREDICT,
            PERM_MODEL_MONITOR,
            PERM_MODEL_RETRAIN,
            PERM_MODEL_SUBMIT,
            PERM_AI_USE,
        }
    ),
    ROLE_ANALYST: frozenset(
        {
            PERM_ACCOUNT_SELF,
            PERM_DATA_READ,
            PERM_DATA_IMPORT,
            PERM_DATA_TRANSFORM,
            PERM_DATA_EXPORT,
            PERM_MODEL_PREDICT,
            PERM_MODEL_MONITOR,
            PERM_AI_USE,
        }
    ),
    ROLE_VIEWER: frozenset({PERM_ACCOUNT_SELF, PERM_DATA_READ}),
}

# A page can be opened when the current identity has at least one permission in
# the tuple. Fine-grained actions inside mixed-purpose pages have separate checks.
PAGE_ANY_PERMISSIONS: Mapping[str, tuple[str, ...]] = {
    "overview": (PERM_DATA_READ,),
    "data_sources": (PERM_DATA_IMPORT,),
    "data_mapper": (PERM_DATA_TRANSFORM,),
    "quality_engine": (PERM_DATA_TRANSFORM,),
    "kpi_tracker": (PERM_DATA_READ,),
    "filter_search": (PERM_DATA_TRANSFORM,),
    "cleaning": (PERM_DATA_TRANSFORM,),
    "data_types": (PERM_DATA_TRANSFORM,),
    "replace_values": (PERM_DATA_TRANSFORM,),
    "feature_engineering": (PERM_ML_TRAIN,),
    "visualization": (PERM_DATA_READ,),
    "outlier_detection": (PERM_DATA_TRANSFORM,),
    "ml_studio": (PERM_ML_TRAIN,),
    "explainability_studio": (PERM_ML_EXPLAIN,),
    "prediction_studio": (PERM_MODEL_PREDICT,),
    "model_monitoring": (PERM_MODEL_MONITOR,),
    "monitoring_scheduler": (PERM_MONITOR_SCHEDULE,),
    "deployment_api": (PERM_DEPLOYMENT_MANAGE,),
    "remote_model_registry": (PERM_REMOTE_REGISTRY_MANAGE,),
    "retraining_workflow": (PERM_MODEL_RETRAIN,),
    "model_governance": (PERM_MODEL_SUBMIT, PERM_MODEL_APPROVE),
    "delete_dedupe": (PERM_DATA_TRANSFORM,),
    "export": (PERM_DATA_EXPORT,),
    "settings": (PERM_ACCOUNT_SELF,),
    "team_admin": (PERM_TEAM_MANAGE,),
    "ai_assistant": (PERM_AI_USE,),
}


class AccessDenied(PermissionError):
    """Raised when a user attempts an operation outside their permission set."""


@dataclass(frozen=True)
class Identity:
    user_id: str
    username: str
    display_name: str
    role: str
    permissions: frozenset[str]
    revision: int = 0
    env_managed: bool = False

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)


def normalize_role(role: str) -> str:
    clean = str(role or "").strip().lower().replace("-", "_").replace(" ", "_")
    if clean not in VALID_ROLES:
        raise AccessDenied("Unknown team role.")
    return clean


def permissions_for_role(role: str) -> frozenset[str]:
    return ROLE_PERMISSIONS[normalize_role(role)]


def identity_from_session(session_state) -> Identity:
    role = str(session_state.get("current_role") or ROLE_VIEWER)
    try:
        role = normalize_role(role)
    except AccessDenied:
        role = ROLE_VIEWER
    explicit = session_state.get("current_permissions")
    permissions = (
        frozenset(str(value) for value in explicit)
        if isinstance(explicit, (list, tuple, set, frozenset)) and explicit
        else permissions_for_role(role)
    )
    return Identity(
        user_id=str(session_state.get("current_user_id") or ""),
        username=str(session_state.get("current_user") or ""),
        display_name=str(session_state.get("current_display_name") or session_state.get("current_user") or ""),
        role=role,
        permissions=permissions,
        revision=int(session_state.get("current_user_revision") or 0),
        env_managed=bool(session_state.get("current_user_env_managed", False)),
    )


def has_permission(permission: str, *, session_state=None) -> bool:
    if session_state is None:
        try:
            import streamlit as st
            session_state = st.session_state
        except Exception:
            return False
    return str(permission) in identity_from_session(session_state).permissions


def require_permission(permission: str, *, session_state=None) -> None:
    if not has_permission(permission, session_state=session_state):
        raise AccessDenied(f"Permission required: {permission}")


def can_access_page(page_key: str, *, session_state=None) -> bool:
    required = PAGE_ANY_PERMISSIONS.get(str(page_key), (PERM_DATA_READ,))
    return any(has_permission(permission, session_state=session_state) for permission in required)


def first_accessible_page(page_keys: Sequence[str], *, session_state=None) -> str:
    for page_key in page_keys:
        if can_access_page(page_key, session_state=session_state):
            return page_key
    return "settings"


def visible_page_keys(page_keys: Iterable[str], *, session_state=None) -> list[str]:
    return [str(key) for key in page_keys if can_access_page(str(key), session_state=session_state)]
