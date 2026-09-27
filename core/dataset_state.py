# ════════════════════════════════════════════════════════
#  DataBridge AI — Unified Dataset State
#  Stage 11: unified data, ML, explainability, and quality-policy context
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

import pandas as pd


class DatasetStateError(RuntimeError):
    """Raised when the active dataset state is invalid or unsafe."""


class StaleDatasetRevisionError(DatasetStateError):
    """Raised when a page tries to change an older dataset revision."""


class DatasetMutationError(DatasetStateError):
    """Raised when an atomic dataset mutation cannot be completed safely."""


@dataclass
class DatasetState:
    """
    Authoritative state for the active dataset.

    Streamlit session keys remain compatibility mirrors. All internal mutation
    code must update this object through the dataset API.
    """

    raw_df: Optional[pd.DataFrame] = None
    working_df: Optional[pd.DataFrame] = None
    file_name: Optional[str] = None

    raw_shape: Optional[tuple[int, int]] = None
    initial_working_shape: Optional[tuple[int, int]] = None
    raw_fingerprint: Optional[str] = None
    working_fingerprint: Optional[str] = None
    revision: int = 0
    last_action: str = ""

    mapper_approved: bool = False
    column_mappings: Dict[str, Any] = field(default_factory=dict)
    mapping_confidence: Dict[str, Any] = field(default_factory=dict)
    semantic_profiles: Dict[str, Any] = field(default_factory=dict)
    ml_readiness_report: Dict[str, Any] = field(default_factory=dict)
    feature_derivation_recipe: list[Dict[str, Any]] = field(default_factory=list)
    feature_pipeline_spec: Dict[str, Any] = field(default_factory=dict)
    feature_pipeline_report: Dict[str, Any] = field(default_factory=dict)
    ml_experiment_report: Dict[str, Any] = field(default_factory=dict)
    explainability_report: Dict[str, Any] = field(default_factory=dict)
    quality_report: Dict[str, Any] = field(default_factory=dict)
    quality_policy: Dict[str, Any] = field(default_factory=dict)
    quality_baseline_report: Dict[str, Any] = field(default_factory=dict)
    quality_previous_report: Dict[str, Any] = field(default_factory=dict)
    kpi_targets: Dict[str, Any] = field(default_factory=dict)
    data_clean_report: Dict[str, Any] = field(default_factory=dict)
    show_import_report: bool = False

    @property
    def is_loaded(self) -> bool:
        return isinstance(self.working_df, pd.DataFrame)

    @property
    def shape(self) -> Optional[tuple[int, int]]:
        if not self.is_loaded:
            return None
        return tuple(self.working_df.shape)


@dataclass(frozen=True)
class DatasetChangeResult:
    changed: bool
    action: str
    revision: int
    before_shape: tuple[int, int]
    after_shape: tuple[int, int]
    before_fingerprint: str
    after_fingerprint: str
    snapshot_id: str = ""


_CONTEXT_FIELDS = {
    "mapper_approved",
    "column_mappings",
    "mapping_confidence",
    "semantic_profiles",
    "ml_readiness_report",
    "feature_derivation_recipe",
    "feature_pipeline_spec",
    "feature_pipeline_report",
    "ml_experiment_report",
    "explainability_report",
    "quality_report",
    "quality_policy",
    "quality_baseline_report",
    "quality_previous_report",
    "kpi_targets",
    "data_clean_report",
    "show_import_report",
}


def _deepcopy_or_default(value: Any, default: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return copy.deepcopy(default)


def state_from_session(session: Mapping[str, Any]) -> DatasetState:
    """Create a unified state from compatibility keys during startup/migration."""
    raw_df = session.get("raw_df")
    working_df = session.get("df")
    return DatasetState(
        raw_df=raw_df if isinstance(raw_df, pd.DataFrame) else None,
        working_df=working_df if isinstance(working_df, pd.DataFrame) else None,
        file_name=session.get("file_name"),
        raw_shape=session.get("raw_shape"),
        initial_working_shape=session.get("initial_working_shape"),
        raw_fingerprint=session.get("raw_fingerprint"),
        working_fingerprint=(
            session.get("working_fingerprint")
            or session.get("df_fingerprint")
        ),
        revision=int(session.get("dataset_revision", 0) or 0),
        last_action=str(session.get("last_dataset_action", "") or ""),
        mapper_approved=bool(session.get("mapper_approved", False)),
        column_mappings=_deepcopy_or_default(session.get("column_mappings", {}), {}),
        mapping_confidence=_deepcopy_or_default(session.get("mapping_confidence", {}), {}),
        semantic_profiles=_deepcopy_or_default(session.get("semantic_profiles", {}), {}),
        ml_readiness_report=_deepcopy_or_default(session.get("ml_readiness_report", {}), {}),
        feature_derivation_recipe=_deepcopy_or_default(session.get("feature_derivation_recipe", []), []),
        feature_pipeline_spec=_deepcopy_or_default(session.get("feature_pipeline_spec", {}), {}),
        feature_pipeline_report=_deepcopy_or_default(session.get("feature_pipeline_report", {}), {}),
        ml_experiment_report=_deepcopy_or_default(session.get("ml_experiment_report", {}), {}),
        explainability_report=_deepcopy_or_default(session.get("explainability_report", {}), {}),
        quality_report=_deepcopy_or_default(session.get("quality_report", {}), {}),
        quality_policy=_deepcopy_or_default(session.get("quality_policy", {}), {}),
        quality_baseline_report=_deepcopy_or_default(session.get("quality_baseline_report", {}), {}),
        quality_previous_report=_deepcopy_or_default(session.get("quality_previous_report", {}), {}),
        kpi_targets=_deepcopy_or_default(session.get("kpi_targets", {}), {}),
        data_clean_report=_deepcopy_or_default(session.get("data_clean_report", {}), {}),
        show_import_report=bool(session.get("show_import_report", False)),
    )


def ensure_dataset_state(session: MutableMapping[str, Any]) -> DatasetState:
    state = session.get("dataset_state")
    if not isinstance(state, DatasetState):
        state = state_from_session(session)
        session["dataset_state"] = state
    return state


def sync_state_to_session(
    session: MutableMapping[str, Any],
    state: Optional[DatasetState] = None,
) -> DatasetState:
    """Publish compatibility mirrors from the authoritative DatasetState."""
    state = state or ensure_dataset_state(session)
    session["dataset_state"] = state

    session["raw_df"] = state.raw_df
    session["df"] = state.working_df
    session["hdf"] = state.working_df
    session["standard_df"] = None
    session["file_name"] = state.file_name

    session["raw_shape"] = state.raw_shape
    session["initial_working_shape"] = state.initial_working_shape
    session["raw_fingerprint"] = state.raw_fingerprint
    session["working_fingerprint"] = state.working_fingerprint
    session["df_fingerprint"] = state.working_fingerprint
    session["dataset_revision"] = int(state.revision)
    session["last_dataset_action"] = state.last_action

    session["mapper_approved"] = state.mapper_approved
    session["column_mappings"] = state.column_mappings
    session["mapping_confidence"] = state.mapping_confidence
    session["semantic_profiles"] = state.semantic_profiles
    session["ml_readiness_report"] = state.ml_readiness_report
    session["feature_derivation_recipe"] = state.feature_derivation_recipe
    session["feature_pipeline_spec"] = state.feature_pipeline_spec
    session["feature_pipeline_report"] = state.feature_pipeline_report
    session["ml_experiment_report"] = state.ml_experiment_report
    session["explainability_report"] = state.explainability_report
    session["quality_report"] = state.quality_report
    session["quality_policy"] = state.quality_policy
    session["quality_baseline_report"] = state.quality_baseline_report
    session["quality_previous_report"] = state.quality_previous_report
    session["kpi_targets"] = state.kpi_targets
    session["data_clean_report"] = state.data_clean_report
    session["show_import_report"] = state.show_import_report
    return state


def update_state_context(
    state: DatasetState,
    updates: Mapping[str, Any],
) -> None:
    unknown = set(updates) - _CONTEXT_FIELDS
    if unknown:
        raise DatasetStateError(
            "Unsupported dataset context field(s): " + ", ".join(sorted(unknown))
        )
    for key, value in updates.items():
        try:
            copied = copy.deepcopy(value)
        except Exception as exc:
            raise DatasetStateError(
                f"Could not copy dataset context field '{key}' safely: {exc}"
            ) from exc
        setattr(state, key, copied)
