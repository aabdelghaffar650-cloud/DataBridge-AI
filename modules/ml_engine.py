# ════════════════════════════════════════════════════════
#  DataBridge AI — Leakage-Safe ML Experiment Engine
#  Stage 8: supervised model comparison, honest holdout evaluation,
#           split-aware cross-validation, and safe clustering diagnostics
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import LinearSVC

from modules.feature_pipeline import (
    FeaturePipelineError,
    build_model_pipeline,
    feature_pipeline_spec_fingerprint,
    normalise_feature_pipeline_spec,
    validate_feature_pipeline_spec,
)
from modules.model_monitoring import build_monitoring_reference


ML_EXPERIMENT_VERSION = 1
DEFAULT_RANDOM_STATE = 42
DEFAULT_HOLDOUT_SIZE = 0.20
DEFAULT_CV_FOLDS = 5
MAX_EXPERIMENT_ROWS = 250_000
MAX_CLUSTER_ROWS = 100_000

CLASSIFICATION = "classification"
REGRESSION = "regression"

SPLIT_RANDOM = "random"
SPLIT_STRATIFIED = "stratified"
SPLIT_TIME = "time"
SPLIT_GROUP = "group"
SUPPORTED_SPLITS = {SPLIT_RANDOM, SPLIT_STRATIFIED, SPLIT_TIME, SPLIT_GROUP}


class MLExperimentError(ValueError):
    """Raised when an ML experiment cannot be executed safely."""


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    estimator: BaseEstimator
    baseline: bool = False
    tuning_space: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    note: str = ""


@dataclass
class SupervisedExperimentResult:
    experiment_id: str
    created_at: str
    task: str
    target: str
    dataset_revision: int
    dataset_fingerprint: str
    pipeline_spec_fingerprint: str
    effective_pipeline_spec_fingerprint: str
    split_strategy: str
    split_column: str
    random_state: int
    requested_cv_folds: int
    actual_cv_folds: int
    source_rows: int
    modelling_rows: int
    sampled: bool
    train_rows: int
    holdout_rows: int
    selected_model: str
    baseline_model: str
    primary_metric: str
    primary_direction: str
    leaderboard: pd.DataFrame
    holdout_metrics: Dict[str, Optional[float]]
    fitted_pipeline: Pipeline
    target_encoder: Optional[LabelEncoder]
    classes: list[str]
    confusion: Optional[np.ndarray]
    classification_report_df: Optional[pd.DataFrame]
    predictions: pd.DataFrame
    feature_names: list[str]
    warnings: list[str]
    failures: Dict[str, str]
    tuned: bool
    tuning_iterations: int
    best_params: Dict[str, Any]
    baseline_cv_value: Optional[float]
    selected_cv_value: Optional[float]
    improvement_vs_baseline: Optional[float]
    split_summary: Dict[str, Any]
    monitoring_reference: Dict[str, Any] = field(default_factory=dict)
    experiment_config: Dict[str, Any] = field(default_factory=dict)

    def report(self) -> Dict[str, Any]:
        """Return metadata only; no fitted estimator or row-level predictions."""
        return {
            "report_version": ML_EXPERIMENT_VERSION,
            "status": "Completed",
            "stale": False,
            "artifact_available": True,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "task": self.task,
            "target": self.target,
            "dataset_revision": int(self.dataset_revision),
            "dataset_fingerprint": self.dataset_fingerprint,
            "pipeline_spec_fingerprint": self.pipeline_spec_fingerprint,
            "effective_pipeline_spec_fingerprint": self.effective_pipeline_spec_fingerprint,
            "split_strategy": self.split_strategy,
            "split_column": self.split_column,
            "random_state": int(self.random_state),
            "requested_cv_folds": int(self.requested_cv_folds),
            "actual_cv_folds": int(self.actual_cv_folds),
            "source_rows": int(self.source_rows),
            "modelling_rows": int(self.modelling_rows),
            "sampled": bool(self.sampled),
            "train_rows": int(self.train_rows),
            "holdout_rows": int(self.holdout_rows),
            "selected_model": self.selected_model,
            "baseline_model": self.baseline_model,
            "primary_metric": self.primary_metric,
            "primary_direction": self.primary_direction,
            "holdout_metrics": _json_safe(self.holdout_metrics),
            "tuned": bool(self.tuned),
            "tuning_iterations": int(self.tuning_iterations),
            "best_params": _json_safe(self.best_params),
            "baseline_cv_value": _optional_float(self.baseline_cv_value),
            "selected_cv_value": _optional_float(self.selected_cv_value),
            "improvement_vs_baseline": _optional_float(self.improvement_vs_baseline),
            "class_labels": list(self.classes),
            "feature_count": int(len(self.feature_names)),
            "warnings": list(self.warnings),
            "failures": dict(self.failures),
            "split_summary": _json_safe(self.split_summary),
            "experiment_config": _json_safe(self.experiment_config),
            "leaderboard": _json_safe(self.leaderboard.to_dict(orient="records")),
        }


@dataclass
class ClusteringExperimentResult:
    algorithm: str
    rows_used: int
    source_rows: int
    sampled: bool
    feature_columns: list[str]
    labels: np.ndarray
    fitted_pipeline: Pipeline
    projection: pd.DataFrame
    cluster_sizes: pd.DataFrame
    metrics: Dict[str, Optional[float]]
    warnings: list[str]
    source_indices: pd.Index


@dataclass(frozen=True)
class _PreparedData:
    frame: pd.DataFrame
    target: np.ndarray
    target_encoder: Optional[LabelEncoder]
    class_labels: list[str]
    source_rows: int
    sampled: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _SplitData:
    train: pd.DataFrame
    holdout: pd.DataFrame
    y_train: np.ndarray
    y_holdout: np.ndarray
    groups_train: Optional[np.ndarray]
    groups_holdout: Optional[np.ndarray]
    summary: Dict[str, Any]
    warnings: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _optional_float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return _optional_float(value)
    if value is pd.NA:
        return None
    return value


def _stable_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_safe(config), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def infer_task_from_target(series: pd.Series) -> str:
    clean = series.dropna()
    if clean.empty:
        return CLASSIFICATION
    unique = int(clean.nunique(dropna=True))
    if not pd.api.types.is_numeric_dtype(clean.dtype):
        return CLASSIFICATION
    threshold = max(20, int(len(clean) * 0.05))
    return CLASSIFICATION if unique <= threshold else REGRESSION


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def _classification_models(
    random_state: int,
    *,
    class_weight: Optional[str],
    include_xgboost: bool,
) -> Dict[str, ModelDefinition]:
    definitions: Dict[str, ModelDefinition] = {
        "Baseline — Most Frequent": ModelDefinition(
            "Baseline — Most Frequent",
            DummyClassifier(strategy="most_frequent"),
            baseline=True,
            note="Reference model; a useful model must beat this baseline.",
        ),
        "Logistic Regression": ModelDefinition(
            "Logistic Regression",
            LogisticRegression(
                C=1.0,
                max_iter=2500,
                class_weight=class_weight,
                random_state=random_state,
                solver="lbfgs",
            ),
            tuning_space={
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        "Random Forest": ModelDefinition(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=1,
            ),
            tuning_space={
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [None, 8, 16, 30],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.7],
            },
        ),
        "Linear SVM": ModelDefinition(
            "Linear SVM",
            LinearSVC(
                C=1.0,
                class_weight=class_weight,
                random_state=random_state,
                dual="auto",
            ),
            tuning_space={
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
        ),
    }

    if include_xgboost and xgboost_available():
        try:
            from xgboost import XGBClassifier

            definitions["XGBoost"] = ModelDefinition(
                "XGBoost",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=random_state,
                    n_jobs=1,
                    device="cpu",
                    tree_method="hist",
                    eval_metric="logloss",
                    verbosity=0,
                ),
                tuning_space={
                    "model__n_estimators": [150, 250, 400],
                    "model__max_depth": [3, 5, 7, 9],
                    "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
                    "model__subsample": [0.7, 0.85, 1.0],
                    "model__colsample_bytree": [0.7, 0.85, 1.0],
                },
            )
        except Exception:
            pass
    return definitions


def _regression_models(
    random_state: int,
    *,
    include_xgboost: bool,
) -> Dict[str, ModelDefinition]:
    definitions: Dict[str, ModelDefinition] = {
        "Baseline — Median": ModelDefinition(
            "Baseline — Median",
            DummyRegressor(strategy="median"),
            baseline=True,
            note="Reference model; a useful model must reduce error below this baseline.",
        ),
        "Ridge Regression": ModelDefinition(
            "Ridge Regression",
            Ridge(alpha=1.0),
            tuning_space={
                "model__alpha": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
            },
        ),
        "Random Forest": ModelDefinition(
            "Random Forest",
            RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            ),
            tuning_space={
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [None, 8, 16, 30],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.7, 1.0],
            },
        ),
        "Extra Trees": ModelDefinition(
            "Extra Trees",
            ExtraTreesRegressor(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            ),
            tuning_space={
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [None, 8, 16, 30],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.7, 1.0],
            },
        ),
    }

    if include_xgboost and xgboost_available():
        try:
            from xgboost import XGBRegressor

            definitions["XGBoost"] = ModelDefinition(
                "XGBoost",
                XGBRegressor(
                    n_estimators=250,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=random_state,
                    n_jobs=1,
                    device="cpu",
                    tree_method="hist",
                    objective="reg:squarederror",
                    verbosity=0,
                ),
                tuning_space={
                    "model__n_estimators": [150, 250, 400],
                    "model__max_depth": [3, 5, 7, 9],
                    "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
                    "model__subsample": [0.7, 0.85, 1.0],
                    "model__colsample_bytree": [0.7, 0.85, 1.0],
                },
            )
        except Exception:
            pass
    return definitions


def available_model_names(
    task: str,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    class_weight: Optional[str] = None,
    include_xgboost: bool = True,
) -> list[str]:
    if task == CLASSIFICATION:
        return list(
            _classification_models(
                random_state,
                class_weight=class_weight,
                include_xgboost=include_xgboost,
            ).keys()
        )
    if task == REGRESSION:
        return list(
            _regression_models(
                random_state,
                include_xgboost=include_xgboost,
            ).keys()
        )
    raise MLExperimentError("Task must be classification or regression.")


def _model_definitions(
    task: str,
    *,
    random_state: int,
    class_weight: Optional[str],
    include_xgboost: bool,
) -> Dict[str, ModelDefinition]:
    if task == CLASSIFICATION:
        return _classification_models(
            random_state,
            class_weight=class_weight,
            include_xgboost=include_xgboost,
        )
    if task == REGRESSION:
        return _regression_models(random_state, include_xgboost=include_xgboost)
    raise MLExperimentError("Task must be classification or regression.")


def _safe_datetime(series: pd.Series, *, dayfirst: bool = True) -> pd.Series:
    """Parse split-control dates without reinterpreting ISO year-first values.

    Pandas releases that do not support ``format="mixed"`` fall back to the
    generic parser. Combining that fallback with ``dayfirst=True`` can
    reinterpret ISO strings such as ``2023-02-01`` as 2 January instead of
    1 February, which breaks chronological train/holdout separation. ISO-like
    values are therefore parsed first with explicit year-first semantics, then
    any remaining values are parsed using the requested day-first preference.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return pd.to_datetime(series, errors="coerce", utc=True)

        text = series.astype("string").str.strip()
        result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

        # Plain ISO calendar dates are unambiguous and can be parsed with one
        # fixed format on every pandas version supported by DataBridge AI.
        iso_date_mask = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$", na=False)
        if bool(iso_date_mask.any()):
            result.loc[iso_date_mask] = pd.to_datetime(
                text.loc[iso_date_mask],
                errors="coerce",
                format="%Y-%m-%d",
                dayfirst=False,
                yearfirst=True,
                utc=True,
            )

        # ISO timestamps remain year-first even when the user's preference for
        # non-ISO dates is day-first. ``ISO8601`` is preferred where available.
        iso_timestamp_mask = text.str.match(
            r"^\d{4}-\d{1,2}-\d{1,2}[Tt\s].+$",
            na=False,
        )
        if bool(iso_timestamp_mask.any()):
            timestamp_values = text.loc[iso_timestamp_mask]
            try:
                parsed_timestamps = pd.to_datetime(
                    timestamp_values,
                    errors="coerce",
                    format="ISO8601",
                    dayfirst=False,
                    yearfirst=True,
                    utc=True,
                )
            except (TypeError, ValueError):
                try:
                    parsed_timestamps = pd.to_datetime(
                        timestamp_values,
                        errors="coerce",
                        format="mixed",
                        dayfirst=False,
                        yearfirst=True,
                        utc=True,
                    )
                except (TypeError, ValueError):
                    parsed_timestamps = pd.to_datetime(
                        timestamp_values,
                        errors="coerce",
                        dayfirst=False,
                        yearfirst=True,
                        utc=True,
                    )
            result.loc[iso_timestamp_mask] = parsed_timestamps

        iso_mask = iso_date_mask | iso_timestamp_mask
        remaining = (~iso_mask) & text.notna() & text.ne("")
        if bool(remaining.any()):
            remaining_values = text.loc[remaining]
            try:
                parsed_remaining = pd.to_datetime(
                    remaining_values,
                    errors="coerce",
                    format="mixed",
                    dayfirst=dayfirst,
                    utc=True,
                )
            except (TypeError, ValueError):
                parsed_remaining = pd.to_datetime(
                    remaining_values,
                    errors="coerce",
                    dayfirst=dayfirst,
                    utc=True,
                )
            result.loc[remaining] = parsed_remaining

        return result


def _safe_stratify(y: np.ndarray, requested_rows: int, total_rows: int) -> bool:
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2 or int(counts.min()) < 2:
        return False
    other_rows = total_rows - requested_rows
    return requested_rows >= len(values) and other_rows >= len(values)


def _cap_modelling_rows(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    split_strategy: str,
    split_column: str,
    max_rows: Optional[int],
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, bool, list[str]]:
    warnings_list: list[str] = []
    if max_rows is None or int(max_rows) <= 0 or len(frame) <= int(max_rows):
        return frame, y, False, warnings_list

    limit = max(100, min(int(max_rows), MAX_EXPERIMENT_ROWS))
    if limit >= len(frame):
        return frame, y, False, warnings_list

    if split_strategy == SPLIT_TIME:
        parsed = _safe_datetime(frame[split_column])
        order = np.argsort(parsed.astype("int64").to_numpy(), kind="stable")
        chosen_positions = order[-limit:]
    elif split_strategy == SPLIT_GROUP:
        groups = frame[split_column].astype("string").fillna("__MISSING_GROUP__")
        unique_groups = groups.drop_duplicates().tolist()
        rng = np.random.default_rng(random_state)
        rng.shuffle(unique_groups)
        chosen: list[int] = []
        for group in unique_groups:
            positions = np.flatnonzero((groups == group).to_numpy())
            if len(positions) > limit and not chosen:
                raise MLExperimentError(
                    "One group is larger than the row safety cap. Increase the cap so complete groups remain intact."
                )
            if len(chosen) + len(positions) > limit:
                continue
            chosen.extend(positions.tolist())
            if len(chosen) >= limit:
                break
        if len(chosen) < 100:
            raise MLExperimentError(
                "The row safety cap is too small to preserve complete groups safely."
            )
        chosen_positions = np.asarray(sorted(chosen), dtype=int)
    elif task == CLASSIFICATION and _safe_stratify(y, limit, len(frame)):
        positions = np.arange(len(frame))
        chosen_positions, _ = train_test_split(
            positions,
            train_size=limit,
            random_state=random_state,
            stratify=y,
        )
        chosen_positions = np.asarray(sorted(chosen_positions), dtype=int)
    else:
        rng = np.random.default_rng(random_state)
        chosen_positions = np.sort(rng.choice(len(frame), size=limit, replace=False))

    warnings_list.append(
        f"The experiment used a deterministic safety sample of {len(chosen_positions):,} "
        f"from {len(frame):,} target-valid rows."
    )
    return (
        frame.iloc[chosen_positions].copy(),
        np.asarray(y)[chosen_positions],
        True,
        warnings_list,
    )


def _prepare_supervised_data(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    split_strategy: str,
    split_column: str,
    max_rows: Optional[int],
    random_state: int,
) -> _PreparedData:
    clean = normalise_feature_pipeline_spec(spec)
    target = clean["target"]
    required = list(clean["feature_columns"]) + [target]
    if split_strategy in {SPLIT_TIME, SPLIT_GROUP} and split_column not in required:
        required.append(split_column)
    frame = df[required].copy(deep=True)
    source_rows = len(frame)

    target_valid = frame[target].notna()
    removed_missing = int((~target_valid).sum())
    frame = frame.loc[target_valid].copy()
    warnings_list: list[str] = []
    if removed_missing:
        warnings_list.append(
            f"Excluded {removed_missing:,} rows with missing target values before splitting."
        )

    if clean["task"] == CLASSIFICATION:
        raw = frame[target].astype("string")
        encoder = LabelEncoder()
        y = encoder.fit_transform(raw.astype(str))
        labels = [str(value) for value in encoder.classes_]
        if len(labels) < 2:
            raise MLExperimentError("Classification requires at least two target classes.")
    else:
        numeric = pd.to_numeric(frame[target], errors="coerce")
        invalid = numeric.isna()
        if invalid.any():
            raise MLExperimentError(
                f"Regression target contains {int(invalid.sum()):,} non-numeric values."
            )
        encoder = None
        y = numeric.to_numpy(dtype=float)
        labels = []

    frame, y, sampled, cap_warnings = _cap_modelling_rows(
        frame,
        y,
        task=clean["task"],
        split_strategy=split_strategy,
        split_column=split_column,
        max_rows=max_rows,
        random_state=random_state,
    )
    warnings_list.extend(cap_warnings)
    if len(frame) < 20:
        raise MLExperimentError("At least 20 target-valid rows are required for ML Studio V2.")

    return _PreparedData(
        frame=frame,
        target=np.asarray(y),
        target_encoder=encoder,
        class_labels=labels,
        source_rows=source_rows,
        sampled=sampled,
        warnings=tuple(warnings_list),
    )


def _split_supervised_data(
    prepared: _PreparedData,
    *,
    task: str,
    target: str,
    split_strategy: str,
    split_column: str,
    holdout_size: float,
    random_state: int,
) -> _SplitData:
    frame = prepared.frame
    y = prepared.target
    warnings_list: list[str] = []
    holdout_size = float(holdout_size)
    if not 0.10 <= holdout_size <= 0.40:
        raise MLExperimentError("Holdout size must be between 10% and 40%.")

    positions = np.arange(len(frame))
    groups_train: Optional[np.ndarray] = None
    groups_holdout: Optional[np.ndarray] = None
    summary: Dict[str, Any] = {
        "strategy": split_strategy,
        "holdout_fraction": holdout_size,
    }

    if split_strategy in {SPLIT_RANDOM, SPLIT_STRATIFIED}:
        stratify = None
        if split_strategy == SPLIT_STRATIFIED:
            if task != CLASSIFICATION:
                raise MLExperimentError("Stratified holdout is only available for classification.")
            requested_test = int(math.ceil(len(frame) * holdout_size))
            if not _safe_stratify(y, requested_test, len(frame)):
                raise MLExperimentError(
                    "The class distribution is too small for the requested stratified holdout."
                )
            stratify = y
        train_pos, holdout_pos = train_test_split(
            positions,
            test_size=holdout_size,
            random_state=random_state,
            stratify=stratify,
        )
    elif split_strategy == SPLIT_TIME:
        if not split_column or split_column not in frame.columns:
            raise MLExperimentError("Select a valid time column for time-ordered splitting.")
        parsed = _safe_datetime(frame[split_column])
        invalid = int(parsed.isna().sum())
        if invalid:
            raise MLExperimentError(
                f"Time split column '{split_column}' contains {invalid:,} invalid or missing dates."
            )
        order = np.argsort(parsed.astype("int64").to_numpy(), kind="stable")
        holdout_rows = max(1, int(math.ceil(len(frame) * holdout_size)))
        split_at = len(frame) - holdout_rows
        if split_at < 10:
            raise MLExperimentError("Time split leaves fewer than ten training rows.")
        train_pos = order[:split_at]
        holdout_pos = order[split_at:]
        summary.update(
            {
                "time_column": split_column,
                "train_time_end": parsed.iloc[train_pos].max().isoformat(),
                "holdout_time_start": parsed.iloc[holdout_pos].min().isoformat(),
            }
        )
        if task == CLASSIFICATION:
            train_classes = set(np.unique(y[train_pos]).tolist())
            holdout_classes = set(np.unique(y[holdout_pos]).tolist())
            if len(train_classes) < 2:
                raise MLExperimentError(
                    "The time-ordered training period contains fewer than two classes."
                )
            missing_holdout = train_classes - holdout_classes
            if missing_holdout:
                warnings_list.append(
                    "The time holdout does not contain every training class; some holdout metrics may be less informative."
                )
    elif split_strategy == SPLIT_GROUP:
        if not split_column or split_column not in frame.columns:
            raise MLExperimentError("Select a valid group column for group holdout.")
        groups = frame[split_column].astype("string").fillna("__MISSING_GROUP__").to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) < 3:
            raise MLExperimentError("Group holdout requires at least three distinct groups.")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=holdout_size,
            random_state=random_state,
        )
        train_pos, holdout_pos = next(splitter.split(positions, y, groups=groups))
        groups_train = groups[train_pos]
        groups_holdout = groups[holdout_pos]
        overlap = set(groups_train.tolist()) & set(groups_holdout.tolist())
        if overlap:
            raise MLExperimentError("Group leakage detected between training and holdout.")
        summary.update(
            {
                "group_column": split_column,
                "training_groups": int(len(np.unique(groups_train))),
                "holdout_groups": int(len(np.unique(groups_holdout))),
                "group_overlap": 0,
            }
        )
    else:
        raise MLExperimentError("Unsupported split strategy.")

    train = frame.iloc[np.asarray(train_pos)].copy()
    holdout = frame.iloc[np.asarray(holdout_pos)].copy()
    y_train = y[np.asarray(train_pos)]
    y_holdout = y[np.asarray(holdout_pos)]
    if len(train) < 10 or len(holdout) < 2:
        raise MLExperimentError("The requested split is too small for reliable training and evaluation.")

    summary.update(
        {
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "train_index_min": str(train.index.min()),
            "train_index_max": str(train.index.max()),
            "holdout_index_min": str(holdout.index.min()),
            "holdout_index_max": str(holdout.index.max()),
        }
    )
    return _SplitData(
        train=train,
        holdout=holdout,
        y_train=np.asarray(y_train),
        y_holdout=np.asarray(y_holdout),
        groups_train=groups_train,
        groups_holdout=groups_holdout,
        summary=summary,
        warnings=tuple(warnings_list),
    )


def _build_cv(
    *,
    task: str,
    split_strategy: str,
    y_train: np.ndarray,
    groups_train: Optional[np.ndarray],
    requested_folds: int,
    random_state: int,
) -> tuple[Any, Optional[np.ndarray], int, list[str]]:
    warnings_list: list[str] = []
    requested_folds = max(2, min(10, int(requested_folds)))

    if split_strategy == SPLIT_TIME:
        actual = min(requested_folds, max(2, len(y_train) // 8))
        if actual >= len(y_train):
            actual = len(y_train) - 1
        if actual < 2:
            raise MLExperimentError("Not enough training rows for time-series cross-validation.")
        cv = TimeSeriesSplit(n_splits=actual)
        if task == CLASSIFICATION:
            for fold, (train_idx, _) in enumerate(cv.split(np.arange(len(y_train))), start=1):
                if len(np.unique(y_train[train_idx])) < 2:
                    raise MLExperimentError(
                        f"Time-series CV fold {fold} has only one training class. Use a later time range or another split strategy."
                    )
        if actual != requested_folds:
            warnings_list.append(
                f"Cross-validation folds were reduced from {requested_folds} to {actual} for the available time-ordered rows."
            )
        return cv, None, actual, warnings_list

    if split_strategy == SPLIT_GROUP:
        if groups_train is None:
            raise MLExperimentError("Group values are required for group cross-validation.")
        unique_groups = np.unique(groups_train)
        actual = min(requested_folds, len(unique_groups))
        if actual < 2:
            raise MLExperimentError("At least two training groups are required for group cross-validation.")
        cv = GroupKFold(n_splits=actual)
        if task == CLASSIFICATION:
            for fold, (train_idx, _) in enumerate(
                cv.split(np.arange(len(y_train)), y_train, groups_train), start=1
            ):
                if len(np.unique(y_train[train_idx])) < 2:
                    raise MLExperimentError(
                        f"Group CV fold {fold} has only one training class. Choose another group column or split strategy."
                    )
        if actual != requested_folds:
            warnings_list.append(
                f"Cross-validation folds were reduced from {requested_folds} to {actual} because only {len(unique_groups)} training groups are available."
            )
        return cv, groups_train, actual, warnings_list

    if task == CLASSIFICATION:
        _, counts = np.unique(y_train, return_counts=True)
        if len(counts) < 2:
            raise MLExperimentError("Training data contains fewer than two classes.")
        actual = min(requested_folds, int(counts.min()))
        if actual < 2:
            raise MLExperimentError("Every class needs at least two training rows for cross-validation.")
        if actual != requested_folds:
            warnings_list.append(
                f"Cross-validation folds were reduced from {requested_folds} to {actual} because of the smallest class."
            )
        return (
            StratifiedKFold(n_splits=actual, shuffle=True, random_state=random_state),
            None,
            actual,
            warnings_list,
        )

    actual = min(requested_folds, max(2, len(y_train) // 5))
    if actual >= len(y_train):
        actual = len(y_train) - 1
    if actual < 2:
        raise MLExperimentError("Not enough training rows for cross-validation.")
    if actual != requested_folds:
        warnings_list.append(
            f"Cross-validation folds were reduced from {requested_folds} to {actual}."
        )
    return (
        KFold(n_splits=actual, shuffle=True, random_state=random_state),
        None,
        actual,
        warnings_list,
    )


def _classification_scoring() -> Dict[str, Any]:
    from sklearn.metrics import make_scorer

    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_weighted": make_scorer(
            precision_score, average="weighted", zero_division=0
        ),
        "recall_weighted": make_scorer(
            recall_score, average="weighted", zero_division=0
        ),
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }


def _regression_scoring() -> Dict[str, Any]:
    return {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }


def _mean_std(values: Iterable[Any]) -> tuple[Optional[float], Optional[float]]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(arr.mean()), float(arr.std(ddof=0))


def _evaluate_candidate_cv(
    definition: ModelDefinition,
    *,
    spec: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: Any,
    cv_groups: Optional[np.ndarray],
    task: str,
) -> Dict[str, Any]:
    pipeline = build_model_pipeline(spec, definition.estimator)
    scoring = _classification_scoring() if task == CLASSIFICATION else _regression_scoring()
    started = time.perf_counter()
    result = cross_validate(
        pipeline,
        X_train,
        y_train,
        groups=cv_groups,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    elapsed = time.perf_counter() - started

    row: Dict[str, Any] = {
        "Model": definition.name,
        "Baseline": bool(definition.baseline),
        "Status": "OK",
        "Total CV Seconds": round(float(elapsed), 4),
    }
    fit_mean, fit_std = _mean_std(result.get("fit_time", []))
    score_mean, score_std = _mean_std(result.get("score_time", []))
    row["Fit Seconds Mean"] = fit_mean
    row["Fit Seconds Std"] = fit_std
    row["Score Seconds Mean"] = score_mean
    row["Score Seconds Std"] = score_std

    if task == CLASSIFICATION:
        for key, label in (
            ("accuracy", "CV Accuracy"),
            ("balanced_accuracy", "CV Balanced Accuracy"),
            ("precision_weighted", "CV Precision Weighted"),
            ("recall_weighted", "CV Recall Weighted"),
            ("f1_weighted", "CV F1 Weighted"),
            ("f1_macro", "CV F1 Macro"),
        ):
            mean, std = _mean_std(result[f"test_{key}"])
            row[f"{label} Mean"] = mean
            row[f"{label} Std"] = std
        row["Selection Score"] = row["CV F1 Weighted Mean"]
        row["Primary Value"] = row["CV F1 Weighted Mean"]
    else:
        mae_mean, mae_std = _mean_std(-np.asarray(result["test_mae"], dtype=float))
        rmse_mean, rmse_std = _mean_std(-np.asarray(result["test_rmse"], dtype=float))
        r2_mean, r2_std = _mean_std(result["test_r2"])
        row.update(
            {
                "CV MAE Mean": mae_mean,
                "CV MAE Std": mae_std,
                "CV RMSE Mean": rmse_mean,
                "CV RMSE Std": rmse_std,
                "CV R² Mean": r2_mean,
                "CV R² Std": r2_std,
                "Selection Score": -rmse_mean if rmse_mean is not None else None,
                "Primary Value": rmse_mean,
            }
        )
    return row


def _select_winner(leaderboard: pd.DataFrame) -> str:
    successful = leaderboard.loc[
        (leaderboard["Status"] == "OK") & leaderboard["Selection Score"].notna()
    ].copy()
    if successful.empty:
        raise MLExperimentError("Every selected model failed during cross-validation.")
    non_baseline = successful.loc[~successful["Baseline"]]
    pool = non_baseline if not non_baseline.empty else successful
    return str(pool.sort_values("Selection Score", ascending=False).iloc[0]["Model"])


def _tune_selected_model(
    definition: ModelDefinition,
    *,
    spec: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: Any,
    cv_groups: Optional[np.ndarray],
    task: str,
    random_state: int,
    tuning_iterations: int,
) -> tuple[Pipeline, Dict[str, Any], int]:
    if definition.baseline or not definition.tuning_space:
        pipeline = build_model_pipeline(spec, definition.estimator)
        pipeline.fit(X_train, y_train)
        return pipeline, {}, 0

    iterations = max(2, min(30, int(tuning_iterations)))
    pipeline = build_model_pipeline(spec, definition.estimator)
    scoring = "f1_weighted" if task == CLASSIFICATION else "neg_root_mean_squared_error"
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=dict(definition.tuning_space),
        n_iter=iterations,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    fit_params: Dict[str, Any] = {}
    if cv_groups is not None:
        fit_params["groups"] = cv_groups
    search.fit(X_train, y_train, **fit_params)
    return search.best_estimator_, dict(search.best_params_), iterations


def _fit_selected_model(
    definition: ModelDefinition,
    *,
    spec: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Pipeline:
    pipeline = build_model_pipeline(spec, definition.estimator)
    pipeline.fit(X_train, y_train)
    return pipeline


def _prediction_scores(pipeline: Pipeline, X: pd.DataFrame) -> tuple[Optional[np.ndarray], bool]:
    if hasattr(pipeline, "predict_proba"):
        try:
            return np.asarray(pipeline.predict_proba(X)), True
        except Exception:
            pass
    if hasattr(pipeline, "decision_function"):
        try:
            return np.asarray(pipeline.decision_function(X)), False
        except Exception:
            pass
    return None, False


def _classification_holdout_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray],
    *,
    probabilities: bool,
    class_count: int,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision Weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall Weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1 Weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1 Macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ROC AUC": None,
        "PR AUC": None,
        "Log Loss": None,
    }
    if scores is None:
        return metrics

    try:
        if class_count == 2:
            positive = scores[:, 1] if scores.ndim == 2 and scores.shape[1] >= 2 else scores.reshape(-1)
            metrics["ROC AUC"] = float(roc_auc_score(y_true, positive))
            metrics["PR AUC"] = float(average_precision_score(y_true, positive))
        elif scores.ndim == 2 and scores.shape[1] == class_count:
            metrics["ROC AUC"] = float(
                roc_auc_score(y_true, scores, multi_class="ovr", average="weighted")
            )
            binary = label_binarize(y_true, classes=np.arange(class_count))
            metrics["PR AUC"] = float(
                average_precision_score(binary, scores, average="weighted")
            )
    except Exception:
        pass

    if probabilities:
        try:
            metrics["Log Loss"] = float(
                log_loss(y_true, scores, labels=np.arange(class_count))
            )
        except Exception:
            pass
    return metrics


def _regression_holdout_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Optional[float]]:
    nonzero = np.abs(y_true) > np.finfo(float).eps
    mape: Optional[float]
    if nonzero.any():
        mape = float(mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero]))
    else:
        mape = None
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": float(r2_score(y_true, y_pred)),
        "Median Absolute Error": float(median_absolute_error(y_true, y_pred)),
        "MAPE": mape,
    }


def _feature_names_from_pipeline(pipeline: Pipeline, X: pd.DataFrame) -> list[str]:
    preprocessor = pipeline.named_steps.get("features")
    if preprocessor is None:
        return []
    try:
        return [str(value) for value in preprocessor.get_feature_names_out()]
    except Exception:
        transformed = preprocessor.transform(X.head(1))
        return [f"feature_{index}" for index in range(int(transformed.shape[1]))]


def _baseline_and_improvement(
    leaderboard: pd.DataFrame,
    selected_model: str,
    *,
    task: str,
) -> tuple[str, Optional[float], Optional[float], Optional[float]]:
    baseline_rows = leaderboard.loc[
        (leaderboard["Status"] == "OK") & leaderboard["Baseline"]
    ]
    selected_rows = leaderboard.loc[
        (leaderboard["Status"] == "OK") & (leaderboard["Model"] == selected_model)
    ]
    baseline_name = str(baseline_rows.iloc[0]["Model"]) if not baseline_rows.empty else ""
    baseline = _optional_float(baseline_rows.iloc[0]["Primary Value"]) if not baseline_rows.empty else None
    selected = _optional_float(selected_rows.iloc[0]["Primary Value"]) if not selected_rows.empty else None
    improvement: Optional[float] = None
    if baseline is not None and selected is not None:
        improvement = selected - baseline if task == CLASSIFICATION else baseline - selected
    return baseline_name, baseline, selected, improvement


def run_supervised_experiment(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    dataset_revision: int,
    dataset_fingerprint: str,
    split_strategy: str = SPLIT_STRATIFIED,
    split_column: str = "",
    holdout_size: float = DEFAULT_HOLDOUT_SIZE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
    model_names: Optional[Sequence[str]] = None,
    class_weight_mode: str = "none",
    tune_best: bool = False,
    tuning_iterations: int = 8,
    max_rows: Optional[int] = 100_000,
    include_xgboost: bool = True,
) -> SupervisedExperimentResult:
    """
    Compare models using cross-validation on training rows only, select by CV,
    then evaluate the selected model once on an untouched holdout.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise MLExperimentError("A non-empty DataFrame is required.")
    clean = normalise_feature_pipeline_spec(spec)
    report = validate_feature_pipeline_spec(
        df,
        clean,
        require_target=True,
        current_revision=dataset_revision,
        current_fingerprint=dataset_fingerprint,
    )
    if not report["valid"]:
        raise MLExperimentError("; ".join(report["blockers"]))
    if report.get("stale"):
        raise MLExperimentError(
            "The feature pipeline is stale. Review and save it again before training."
        )

    task = clean["task"]
    target = clean["target"]
    split_strategy = str(split_strategy).lower().strip()
    if split_strategy not in SUPPORTED_SPLITS:
        raise MLExperimentError("Unsupported split strategy.")
    if task == REGRESSION and split_strategy == SPLIT_STRATIFIED:
        split_strategy = SPLIT_RANDOM
    if split_strategy in {SPLIT_TIME, SPLIT_GROUP}:
        if not split_column or split_column not in df.columns:
            raise MLExperimentError("The selected split column does not exist.")
        if split_column == target:
            raise MLExperimentError("The target column cannot be used as a split-control column.")

    experiment_spec = copy.deepcopy(clean)
    control_excluded: list[str] = []
    if split_strategy == SPLIT_GROUP and split_column in experiment_spec["feature_columns"]:
        for columns in experiment_spec["groups"].values():
            if split_column in columns:
                columns.remove(split_column)
        experiment_spec = normalise_feature_pipeline_spec(experiment_spec)
        control_excluded.append(split_column)
        if not experiment_spec["feature_columns"]:
            raise MLExperimentError(
                "Excluding the group-control column leaves no model features."
            )

    prepared = _prepare_supervised_data(
        df,
        experiment_spec,
        split_strategy=split_strategy,
        split_column=split_column,
        max_rows=max_rows,
        random_state=int(random_state),
    )
    split = _split_supervised_data(
        prepared,
        task=task,
        target=target,
        split_strategy=split_strategy,
        split_column=split_column,
        holdout_size=holdout_size,
        random_state=int(random_state),
    )

    X_train = split.train[list(experiment_spec["feature_columns"])]
    X_holdout = split.holdout[list(experiment_spec["feature_columns"])]
    y_train = split.y_train
    y_holdout = split.y_holdout

    distribution_warnings: list[str] = []
    if task == CLASSIFICATION:
        train_counts = np.bincount(y_train.astype(int), minlength=len(prepared.class_labels))
        holdout_counts = np.bincount(y_holdout.astype(int), minlength=len(prepared.class_labels))
        split.summary["training_class_distribution"] = {
            prepared.class_labels[index]: int(value)
            for index, value in enumerate(train_counts)
        }
        split.summary["holdout_class_distribution"] = {
            prepared.class_labels[index]: int(value)
            for index, value in enumerate(holdout_counts)
        }
        positive_counts = train_counts[train_counts > 0]
        imbalance_ratio = (
            float(positive_counts.max() / positive_counts.min())
            if len(positive_counts)
            else None
        )
        split.summary["training_imbalance_ratio"] = imbalance_ratio
        if imbalance_ratio is not None and imbalance_ratio >= 3.0:
            distribution_warnings.append(
                f"Training class imbalance ratio is {imbalance_ratio:.1f}:1. Review F1 Macro, Balanced Accuracy, and PR AUC instead of Accuracy alone."
            )
    else:
        split.summary["training_target_mean"] = float(np.mean(y_train))
        split.summary["training_target_std"] = float(np.std(y_train))
        split.summary["holdout_target_mean"] = float(np.mean(y_holdout))

    cv, cv_groups, actual_folds, cv_warnings = _build_cv(
        task=task,
        split_strategy=split_strategy,
        y_train=y_train,
        groups_train=split.groups_train,
        requested_folds=cv_folds,
        random_state=int(random_state),
    )

    class_weight: Optional[str] = None
    if task == CLASSIFICATION and str(class_weight_mode).lower() == "balanced":
        class_weight = "balanced"

    definitions = _model_definitions(
        task,
        random_state=int(random_state),
        class_weight=class_weight,
        include_xgboost=include_xgboost,
    )
    baseline_names = [name for name, definition in definitions.items() if definition.baseline]
    requested = list(dict.fromkeys(str(name) for name in (model_names or definitions.keys())))
    selected_names = baseline_names + [name for name in requested if name not in baseline_names]
    selected_names = [name for name in selected_names if name in definitions]
    if len(selected_names) < 2:
        raise MLExperimentError("Select at least one trainable model in addition to the baseline.")

    rows: list[Dict[str, Any]] = []
    failures: Dict[str, str] = {}
    for name in selected_names:
        definition = definitions[name]
        try:
            rows.append(
                _evaluate_candidate_cv(
                    definition,
                    spec=experiment_spec,
                    X_train=X_train,
                    y_train=y_train,
                    cv=cv,
                    cv_groups=cv_groups,
                    task=task,
                )
            )
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            failures[name] = message[:500]
            rows.append(
                {
                    "Model": name,
                    "Baseline": bool(definition.baseline),
                    "Status": "Failed",
                    "Selection Score": None,
                    "Primary Value": None,
                    "Error": message[:500],
                }
            )

    leaderboard = pd.DataFrame(rows)
    winner_name = _select_winner(leaderboard)
    winner = definitions[winner_name]

    tuned = False
    best_params: Dict[str, Any] = {}
    actual_tuning_iterations = 0
    if tune_best and not winner.baseline:
        try:
            fitted_pipeline, best_params, actual_tuning_iterations = _tune_selected_model(
                winner,
                spec=experiment_spec,
                X_train=X_train,
                y_train=y_train,
                cv=cv,
                cv_groups=cv_groups,
                task=task,
                random_state=int(random_state),
                tuning_iterations=tuning_iterations,
            )
            tuned = actual_tuning_iterations > 0
        except Exception as exc:
            failures[f"{winner_name} tuning"] = (str(exc).strip() or exc.__class__.__name__)[:500]
            fitted_pipeline = _fit_selected_model(
                winner,
                spec=experiment_spec,
                X_train=X_train,
                y_train=y_train,
            )
    else:
        fitted_pipeline = _fit_selected_model(
            winner,
            spec=experiment_spec,
            X_train=X_train,
            y_train=y_train,
        )

    y_pred = np.asarray(fitted_pipeline.predict(X_holdout))
    classes: list[str] = prepared.class_labels
    confusion: Optional[np.ndarray] = None
    report_df: Optional[pd.DataFrame] = None

    if task == CLASSIFICATION:
        scores, probabilities = _prediction_scores(fitted_pipeline, X_holdout)
        holdout_metrics = _classification_holdout_metrics(
            y_holdout,
            y_pred,
            scores,
            probabilities=probabilities,
            class_count=len(classes),
        )
        labels_numeric = np.arange(len(classes))
        confusion = confusion_matrix(y_holdout, y_pred, labels=labels_numeric)
        report_dict = classification_report(
            y_holdout,
            y_pred,
            labels=labels_numeric,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose()
        actual_labels = prepared.target_encoder.inverse_transform(y_holdout.astype(int))
        predicted_labels = prepared.target_encoder.inverse_transform(y_pred.astype(int))
        predictions = pd.DataFrame(
            {
                "Actual": actual_labels,
                "Predicted": predicted_labels,
                "Correct": actual_labels == predicted_labels,
            },
            index=split.holdout.index,
        )
        if scores is not None and probabilities and scores.ndim == 2:
            predictions["Confidence"] = np.max(scores, axis=1)
    else:
        holdout_metrics = _regression_holdout_metrics(y_holdout, y_pred)
        predictions = pd.DataFrame(
            {
                "Actual": y_holdout,
                "Predicted": y_pred,
                "Residual": y_holdout - y_pred,
                "Absolute Error": np.abs(y_holdout - y_pred),
            },
            index=split.holdout.index,
        )

    feature_names = _feature_names_from_pipeline(fitted_pipeline, X_train)
    baseline_name, baseline_cv, selected_cv, improvement = _baseline_and_improvement(
        leaderboard, winner_name, task=task
    )

    if task == CLASSIFICATION:
        leaderboard = leaderboard.sort_values(
            ["Status", "Selection Score"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
        primary_metric = "CV F1 Weighted"
        primary_direction = "higher"
    else:
        leaderboard = leaderboard.sort_values(
            ["Status", "Selection Score"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
        primary_metric = "CV RMSE"
        primary_direction = "lower"

    warnings_out = (
        list(prepared.warnings)
        + list(split.warnings)
        + list(cv_warnings)
        + distribution_warnings
    )
    warnings_out.extend(report.get("warnings", []))
    if control_excluded:
        warnings_out.append(
            "Group-control columns were excluded from model features to prevent group identity leakage: "
            + ", ".join(control_excluded)
        )
    if class_weight == "balanced":
        warnings_out.append(
            "Balanced class weights were applied only to models that support class_weight."
        )
    if improvement is not None and improvement <= 0:
        warnings_out.append(
            "The selected model did not beat the baseline in cross-validation. Treat the result as not production-ready."
        )

    experiment_config = {
        "split_strategy": split_strategy,
        "split_column": str(split_column or ""),
        "holdout_size": float(holdout_size),
        "cv_folds": int(cv_folds),
        "actual_cv_folds": int(actual_folds),
        "random_state": int(random_state),
        "candidate_models": [name for name in selected_names if not definitions[name].baseline],
        "class_weight_mode": "balanced" if class_weight == "balanced" else "none",
        "tune_best": bool(tune_best),
        "tuning_iterations": int(tuning_iterations),
        "actual_tuning_iterations": int(actual_tuning_iterations),
        "max_rows": None if max_rows is None else int(max_rows),
        "include_xgboost": bool(include_xgboost),
    }
    config = {
        "task": task,
        "target": target,
        "dataset_fingerprint": dataset_fingerprint,
        "pipeline_spec_fingerprint": feature_pipeline_spec_fingerprint(clean),
        "effective_pipeline_spec_fingerprint": feature_pipeline_spec_fingerprint(experiment_spec),
        **experiment_config,
        "best_params": best_params,
    }
    experiment_id = f"ml-{uuid.uuid4().hex[:10]}-{_stable_config_hash(config)[:8]}"
    monitoring_reference = build_monitoring_reference(X_train, experiment_spec)

    return SupervisedExperimentResult(
        experiment_id=experiment_id,
        created_at=_utc_now(),
        task=task,
        target=target,
        dataset_revision=int(dataset_revision),
        dataset_fingerprint=str(dataset_fingerprint),
        pipeline_spec_fingerprint=feature_pipeline_spec_fingerprint(clean),
        effective_pipeline_spec_fingerprint=feature_pipeline_spec_fingerprint(experiment_spec),
        split_strategy=split_strategy,
        split_column=str(split_column or ""),
        random_state=int(random_state),
        requested_cv_folds=int(cv_folds),
        actual_cv_folds=int(actual_folds),
        source_rows=int(prepared.source_rows),
        modelling_rows=int(len(prepared.frame)),
        sampled=bool(prepared.sampled),
        train_rows=int(len(split.train)),
        holdout_rows=int(len(split.holdout)),
        selected_model=winner_name,
        baseline_model=baseline_name,
        primary_metric=primary_metric,
        primary_direction=primary_direction,
        leaderboard=leaderboard,
        holdout_metrics=holdout_metrics,
        fitted_pipeline=fitted_pipeline,
        target_encoder=prepared.target_encoder,
        classes=classes,
        confusion=confusion,
        classification_report_df=report_df,
        predictions=predictions,
        feature_names=feature_names,
        warnings=list(dict.fromkeys(warnings_out)),
        failures=failures,
        tuned=tuned,
        tuning_iterations=actual_tuning_iterations,
        best_params=best_params,
        baseline_cv_value=baseline_cv,
        selected_cv_value=selected_cv,
        improvement_vs_baseline=improvement,
        split_summary=split.summary,
        monitoring_reference=monitoring_reference,
        experiment_config=experiment_config,
    )


def reconcile_ml_experiment_report(
    report: Mapping[str, Any] | None,
    *,
    current_revision: int,
    current_fingerprint: str,
    current_pipeline_spec_fingerprint: str = "",
    artifact_available: bool = False,
) -> Dict[str, Any]:
    if not report:
        return {
            "status": "Pending",
            "stale": False,
            "artifact_available": False,
        }
    result = copy.deepcopy(dict(report))
    dataset_matches = str(result.get("dataset_fingerprint", "")) == str(
        current_fingerprint or ""
    )
    pipeline_matches = True
    expected_pipeline = str(result.get("pipeline_spec_fingerprint", ""))
    if expected_pipeline and current_pipeline_spec_fingerprint:
        pipeline_matches = expected_pipeline == current_pipeline_spec_fingerprint
    stale = not dataset_matches or not pipeline_matches
    result["stale"] = stale
    result["status"] = "Stale" if stale else "Completed"
    result["artifact_available"] = bool(artifact_available and not stale)
    result["current_revision"] = int(current_revision)
    if stale:
        reasons: list[str] = []
        if not dataset_matches:
            reasons.append("dataset changed")
        if not pipeline_matches:
            reasons.append("feature pipeline changed")
        result["stale_reason"] = ", ".join(reasons)
    else:
        result.pop("stale_reason", None)
    return result


def _cluster_projection(matrix: Any, random_state: int) -> np.ndarray:
    if matrix.shape[1] == 1:
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        return np.column_stack([dense[:, 0], np.zeros(len(dense))])
    if sparse.issparse(matrix):
        reducer = TruncatedSVD(n_components=2, random_state=random_state)
    else:
        reducer = PCA(n_components=2, random_state=random_state)
    return np.asarray(reducer.fit_transform(matrix), dtype=float)


def run_clustering_experiment(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    algorithm: str = "kmeans",
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_rows: int = 50_000,
) -> ClusteringExperimentResult:
    columns = list(dict.fromkeys(str(column) for column in feature_columns))
    if len(columns) < 2:
        raise MLExperimentError("Clustering requires at least two numeric features.")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise MLExperimentError("Missing clustering features: " + ", ".join(missing))
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise MLExperimentError(
            "Clustering V2 accepts numeric source columns only: " + ", ".join(non_numeric)
        )

    frame = df[columns].copy(deep=True)
    source_rows = len(frame)
    if source_rows < 5:
        raise MLExperimentError("Clustering requires at least five rows.")
    sampled = False
    warnings_list: list[str] = []
    limit = max(100, min(int(max_rows), MAX_CLUSTER_ROWS))
    if len(frame) > limit:
        frame = frame.sample(n=limit, random_state=random_state).sort_index()
        sampled = True
        warnings_list.append(
            f"Clustering used a deterministic sample of {len(frame):,} from {source_rows:,} rows."
        )

    preprocessor = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    matrix = preprocessor.fit_transform(frame)
    algorithm_key = str(algorithm).lower().strip()
    if algorithm_key == "kmeans":
        n_clusters = max(2, min(20, int(n_clusters)))
        if n_clusters >= len(frame):
            raise MLExperimentError("Number of clusters must be smaller than the number of rows.")
        estimator: BaseEstimator = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        algorithm_label = "K-Means"
    elif algorithm_key == "dbscan":
        estimator = DBSCAN(eps=float(eps), min_samples=max(2, int(min_samples)))
        algorithm_label = "DBSCAN"
    else:
        raise MLExperimentError("Algorithm must be kmeans or dbscan.")

    labels = np.asarray(estimator.fit_predict(matrix), dtype=int)
    unique_clusters = sorted(value for value in np.unique(labels).tolist() if value != -1)
    noise_count = int((labels == -1).sum())
    if len(unique_clusters) < 1:
        raise MLExperimentError("The clustering algorithm did not produce any usable cluster.")

    evaluation_mask = labels != -1
    eval_matrix = matrix[evaluation_mask]
    eval_labels = labels[evaluation_mask]
    silhouette: Optional[float] = None
    db_index: Optional[float] = None
    if len(np.unique(eval_labels)) >= 2 and len(eval_labels) > len(np.unique(eval_labels)):
        try:
            silhouette = float(silhouette_score(eval_matrix, eval_labels))
        except Exception:
            pass
        try:
            dense_eval = eval_matrix.toarray() if sparse.issparse(eval_matrix) else np.asarray(eval_matrix)
            db_index = float(davies_bouldin_score(dense_eval, eval_labels))
        except Exception:
            pass
    else:
        warnings_list.append(
            "Silhouette and Davies–Bouldin scores require at least two non-noise clusters."
        )

    projection_values = _cluster_projection(matrix, random_state)
    projection = pd.DataFrame(
        {
            "Component 1": projection_values[:, 0],
            "Component 2": projection_values[:, 1],
            "Cluster": labels.astype(str),
        },
        index=frame.index,
    )
    counts = pd.Series(labels, name="Cluster").value_counts().sort_index()
    cluster_sizes = counts.rename_axis("Cluster").reset_index(name="Count")
    cluster_sizes["Cluster"] = cluster_sizes["Cluster"].astype(str)

    pipeline = Pipeline([("features", preprocessor), ("model", estimator)])
    # The estimator and preprocessor are already fitted separately; replace the
    # pipeline steps with those fitted instances for Stage 9 packaging.
    pipeline.steps = [("features", preprocessor), ("model", estimator)]

    return ClusteringExperimentResult(
        algorithm=algorithm_label,
        rows_used=int(len(frame)),
        source_rows=int(source_rows),
        sampled=sampled,
        feature_columns=columns,
        labels=labels,
        fitted_pipeline=pipeline,
        projection=projection,
        cluster_sizes=cluster_sizes,
        metrics={
            "Clusters": float(len(unique_clusters)),
            "Noise Points": float(noise_count),
            "Silhouette": silhouette,
            "Davies-Bouldin": db_index,
        },
        warnings=warnings_list,
        source_indices=frame.index.copy(),
    )
