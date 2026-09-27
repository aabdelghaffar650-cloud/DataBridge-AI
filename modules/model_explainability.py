# ════════════════════════════════════════════════════════
#  DataBridge AI — Model Explainability Engine
#  Stage 10: model-agnostic global importance, native transformed-feature
#            importance, local sensitivity, optional SHAP, and error analysis
# ════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from modules.feature_pipeline import (
    feature_pipeline_spec_fingerprint,
    normalise_feature_pipeline_spec,
)
from modules.ml_engine import CLASSIFICATION, REGRESSION, SupervisedExperimentResult


EXPLAINABILITY_REPORT_VERSION = 1
DEFAULT_RANDOM_STATE = 42
DEFAULT_PERMUTATION_REPEATS = 5
MAX_PERMUTATION_REPEATS = 20
MAX_EXPLAIN_ROWS = 5_000
MAX_SOURCE_FEATURES = 250
MAX_NATIVE_FEATURES = 20_000
MAX_LOCAL_BACKGROUND_ROWS = 1_000
MAX_SHAP_BACKGROUND_ROWS = 100
MAX_SHAP_DENSE_CELLS = 2_000_000


class ExplainabilityError(ValueError):
    """Raised when a model explanation cannot be produced safely."""


@dataclass
class GlobalExplainabilityResult:
    explanation_id: str
    created_at: str
    experiment_id: str
    task: str
    target: str
    selected_model: str
    dataset_revision: int
    dataset_fingerprint: str
    pipeline_spec_fingerprint: str
    rows_available: int
    rows_used: int
    sampled: bool
    permutation_metric: str
    permutation_repeats: int
    source_importance: pd.DataFrame
    native_importance: pd.DataFrame
    native_method: str
    warnings: list[str]

    def report(self) -> Dict[str, Any]:
        return {
            "report_version": EXPLAINABILITY_REPORT_VERSION,
            "status": "Completed",
            "stale": False,
            "artifact_available": True,
            "explanation_id": self.explanation_id,
            "created_at": self.created_at,
            "experiment_id": self.experiment_id,
            "task": self.task,
            "target": self.target,
            "selected_model": self.selected_model,
            "dataset_revision": int(self.dataset_revision),
            "dataset_fingerprint": self.dataset_fingerprint,
            "pipeline_spec_fingerprint": self.pipeline_spec_fingerprint,
            "rows_available": int(self.rows_available),
            "rows_used": int(self.rows_used),
            "sampled": bool(self.sampled),
            "permutation_metric": self.permutation_metric,
            "permutation_repeats": int(self.permutation_repeats),
            "source_feature_count": int(len(self.source_importance)),
            "native_feature_count": int(len(self.native_importance)),
            "native_method": self.native_method,
            "warnings": list(self.warnings),
        }


@dataclass
class LocalExplanationResult:
    experiment_id: str
    source_index: Any
    task: str
    target: str
    predicted_label: str
    predicted_value: float
    score_name: str
    score_value: float
    actual_value: Any
    sensitivity: pd.DataFrame
    reference_values: Dict[str, Any]
    warnings: list[str]


@dataclass
class ShapExplanationResult:
    available: bool
    method: str
    feature_values: pd.DataFrame
    base_value: Optional[float]
    output_value: Optional[float]
    warning: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_active_result(
    df: pd.DataFrame,
    result: SupervisedExperimentResult,
    spec: Mapping[str, Any],
    *,
    current_revision: int,
    current_fingerprint: str,
) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ExplainabilityError("A non-empty working dataset is required.")
    if not isinstance(result, SupervisedExperimentResult):
        raise ExplainabilityError("Run a supervised ML Studio V2 experiment first.")
    if result.task not in {CLASSIFICATION, REGRESSION}:
        raise ExplainabilityError("Only supervised classification/regression experiments can be explained.")
    if result.dataset_revision != int(current_revision):
        raise ExplainabilityError("The model belongs to an older dataset revision. Train it again before explaining it.")
    if str(result.dataset_fingerprint) != str(current_fingerprint):
        raise ExplainabilityError("The model dataset fingerprint does not match the active working dataset.")

    clean = normalise_feature_pipeline_spec(spec)
    if not clean:
        raise ExplainabilityError("The fitted Feature Engineering Pipeline specification is unavailable.")
    spec_fp = feature_pipeline_spec_fingerprint(clean)
    if str(result.pipeline_spec_fingerprint) != str(spec_fp):
        raise ExplainabilityError("The Feature Engineering Pipeline changed after this model was trained.")
    if result.target not in df.columns:
        raise ExplainabilityError(f"Target column '{result.target}' is missing from the working dataset.")
    missing = [column for column in clean.get("feature_columns", []) if column not in df.columns]
    if missing:
        raise ExplainabilityError("Model feature columns are missing: " + ", ".join(missing[:10]))
    return clean



def _fitted_feature_columns(
    result: SupervisedExperimentResult,
    spec: Mapping[str, Any],
) -> list[str]:
    fitted = getattr(result.fitted_pipeline, "feature_names_in_", None)
    if fitted is not None and len(fitted):
        return list(dict.fromkeys(map(str, fitted)))
    clean = normalise_feature_pipeline_spec(spec)
    return list(dict.fromkeys(map(str, clean.get("feature_columns", []) or [])))

def _holdout_frame(
    df: pd.DataFrame,
    result: SupervisedExperimentResult,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    predictions = result.predictions
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ExplainabilityError("The experiment has no holdout predictions.")
    if not df.index.is_unique:
        raise ExplainabilityError(
            "Explainability requires a unique source index. Reset or create a unique row identifier before training."
        )
    missing_indices = predictions.index.difference(df.index)
    if len(missing_indices):
        raise ExplainabilityError("Some holdout source rows are no longer present in the working dataset.")

    X = df.loc[predictions.index, list(feature_columns)].copy(deep=True)
    if result.task == CLASSIFICATION:
        if result.target_encoder is None:
            raise ExplainabilityError("The classification target encoder is unavailable.")
        actual = predictions["Actual"].astype(str)
        try:
            y = result.target_encoder.transform(actual)
        except Exception as exc:
            raise ExplainabilityError(f"Could not reconstruct holdout class labels: {exc}") from exc
    else:
        y = pd.to_numeric(predictions["Actual"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ExplainabilityError("Regression holdout targets contain non-finite values.")
    return X, np.asarray(y), predictions.copy(deep=True)


def _stratified_sample_positions(
    y: np.ndarray,
    max_rows: int,
    random_state: int,
) -> np.ndarray:
    if len(y) <= max_rows:
        return np.arange(len(y), dtype=int)
    rng = np.random.default_rng(int(random_state))
    classes, inverse = np.unique(y, return_inverse=True)
    selected: list[int] = []
    for class_index in range(len(classes)):
        positions = np.flatnonzero(inverse == class_index)
        allocation = max(1, int(round(max_rows * len(positions) / len(y))))
        allocation = min(allocation, len(positions))
        selected.extend(rng.choice(positions, size=allocation, replace=False).tolist())
    if len(selected) > max_rows:
        selected = rng.choice(np.asarray(selected), size=max_rows, replace=False).tolist()
    elif len(selected) < max_rows:
        remaining = np.setdiff1d(np.arange(len(y)), np.asarray(selected, dtype=int))
        add = min(max_rows - len(selected), len(remaining))
        if add:
            selected.extend(rng.choice(remaining, size=add, replace=False).tolist())
    return np.asarray(sorted(set(selected)), dtype=int)


def _sample_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, bool]:
    cap = max(50, min(int(max_rows), MAX_EXPLAIN_ROWS))
    if len(X) <= cap:
        return X.copy(deep=True), np.asarray(y).copy(), False
    if task == CLASSIFICATION:
        positions = _stratified_sample_positions(np.asarray(y), cap, random_state)
    else:
        rng = np.random.default_rng(int(random_state))
        positions = np.sort(rng.choice(len(X), size=cap, replace=False))
    return X.iloc[positions].copy(deep=True), np.asarray(y)[positions].copy(), True


def _source_from_transformed_name(name: str, spec: Mapping[str, Any]) -> str:
    clean = normalise_feature_pipeline_spec(spec)
    source_columns = list(map(str, clean.get("feature_columns", []) or []))
    transformer, separator, suffix = str(name).partition("__")
    payload = suffix if separator else transformer

    if transformer.startswith("text_"):
        for column in sorted(source_columns, key=len, reverse=True):
            if transformer.endswith("_" + column) or transformer == column:
                return column

    for column in sorted(source_columns, key=len, reverse=True):
        if payload == column:
            return column
        if payload.startswith(column + "_"):
            return column
        if payload.startswith("missingindicator_" + column):
            return column
        if payload.endswith("_" + column):
            return column
    return "Derived / Unmapped"


def _native_feature_importance(
    result: SupervisedExperimentResult,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, str, list[str]]:
    pipeline = result.fitted_pipeline
    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
    if model is None:
        return pd.DataFrame(), "Unavailable", ["The fitted model step is unavailable."]

    method = ""
    values: Optional[np.ndarray] = None
    if hasattr(model, "feature_importances_"):
        try:
            values = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
            method = "Native tree feature importance"
        except Exception:
            values = None
    if values is None and hasattr(model, "coef_"):
        try:
            coef = np.asarray(model.coef_, dtype=float)
            values = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
            values = values.reshape(-1)
            method = "Absolute model coefficients"
        except Exception:
            values = None

    if values is None:
        return pd.DataFrame(), "Unavailable", [
            "This estimator does not expose stable native coefficients or feature_importances_."
        ]

    names = list(map(str, result.feature_names))
    if len(values) != len(names):
        return pd.DataFrame(), "Unavailable", [
            "Native importance width did not match the transformed feature contract."
        ]
    if len(values) > MAX_NATIVE_FEATURES:
        return pd.DataFrame(), "Unavailable", [
            f"Native transformed-feature importance was skipped above {MAX_NATIVE_FEATURES:,} features."
        ]

    frame = pd.DataFrame(
        {
            "Transformed Feature": names,
            "Source Feature": [_source_from_transformed_name(name, spec) for name in names],
            "Importance": np.abs(values),
        }
    )
    total = float(frame["Importance"].sum())
    frame["Normalized Importance"] = (
        frame["Importance"] / total if total > 0 else 0.0
    )
    frame = frame.sort_values("Importance", ascending=False).reset_index(drop=True)
    return frame, method, []


def run_global_explainability(
    df: pd.DataFrame,
    result: SupervisedExperimentResult,
    spec: Mapping[str, Any],
    *,
    current_revision: int,
    current_fingerprint: str,
    max_rows: int = 2_000,
    n_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> GlobalExplainabilityResult:
    """
    Explain an already-fitted model without refitting it.

    Permutation importance is computed on untouched holdout rows at the original
    source-column level. Preprocessing stays inside the fitted sklearn Pipeline.
    """
    clean = _validate_active_result(
        df,
        result,
        spec,
        current_revision=current_revision,
        current_fingerprint=current_fingerprint,
    )
    feature_columns = _fitted_feature_columns(result, clean)
    if len(feature_columns) > MAX_SOURCE_FEATURES:
        raise ExplainabilityError(
            f"Source feature count ({len(feature_columns):,}) exceeds the explainability safety limit of {MAX_SOURCE_FEATURES:,}."
        )

    X_holdout, y_holdout, _ = _holdout_frame(df, result, feature_columns)
    X_used, y_used, sampled = _sample_holdout(
        X_holdout,
        y_holdout,
        task=result.task,
        max_rows=max_rows,
        random_state=random_state,
    )
    repeats = max(2, min(int(n_repeats), MAX_PERMUTATION_REPEATS))
    scoring = "f1_weighted" if result.task == CLASSIFICATION else "neg_root_mean_squared_error"

    try:
        perm = permutation_importance(
            result.fitted_pipeline,
            X_used,
            y_used,
            scoring=scoring,
            n_repeats=repeats,
            random_state=int(random_state),
            n_jobs=1,
        )
    except Exception as exc:
        raise ExplainabilityError(f"Permutation importance failed safely: {exc}") from exc

    source = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance Mean": np.asarray(perm.importances_mean, dtype=float),
            "Importance Std": np.asarray(perm.importances_std, dtype=float),
        }
    )
    source["Absolute Importance"] = source["Importance Mean"].abs()
    positive_total = float(source["Absolute Importance"].sum())
    source["Normalized Importance"] = (
        source["Absolute Importance"] / positive_total if positive_total > 0 else 0.0
    )
    source["Direction"] = np.where(
        source["Importance Mean"] > 0,
        "Helpful",
        np.where(source["Importance Mean"] < 0, "Potentially harmful / noise", "Neutral"),
    )
    source = source.sort_values("Absolute Importance", ascending=False).reset_index(drop=True)

    native, native_method, native_warnings = _native_feature_importance(result, clean)
    warnings_out: list[str] = list(native_warnings)
    if sampled:
        warnings_out.append(
            f"Global importance used a deterministic sample of {len(X_used):,} from {len(X_holdout):,} holdout rows."
        )
    if (source["Importance Mean"] < 0).any():
        warnings_out.append(
            "Negative permutation importance can indicate noise, instability, or correlated substitutes; it is not proof that a feature is harmful."
        )
    warnings_out.append(
        "Feature importance and local sensitivity describe model behavior, not causal relationships."
    )

    config = {
        "experiment_id": result.experiment_id,
        "dataset_fingerprint": current_fingerprint,
        "pipeline_spec_fingerprint": feature_pipeline_spec_fingerprint(clean),
        "rows_used": len(X_used),
        "repeats": repeats,
        "random_state": int(random_state),
    }
    explanation_id = f"exp-{uuid.uuid4().hex[:10]}-{_stable_hash(config)[:8]}"
    return GlobalExplainabilityResult(
        explanation_id=explanation_id,
        created_at=_utc_now(),
        experiment_id=result.experiment_id,
        task=result.task,
        target=result.target,
        selected_model=result.selected_model,
        dataset_revision=int(current_revision),
        dataset_fingerprint=str(current_fingerprint),
        pipeline_spec_fingerprint=feature_pipeline_spec_fingerprint(clean),
        rows_available=int(len(X_holdout)),
        rows_used=int(len(X_used)),
        sampled=sampled,
        permutation_metric=("Weighted F1 decrease" if result.task == CLASSIFICATION else "RMSE increase"),
        permutation_repeats=repeats,
        source_importance=source,
        native_importance=native,
        native_method=native_method,
        warnings=list(dict.fromkeys(warnings_out)),
    )


def _reference_value(series: pd.Series) -> Any:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    if pd.api.types.is_numeric_dtype(clean.dtype):
        value = pd.to_numeric(clean, errors="coerce").median()
        return float(value) if pd.notna(value) else np.nan
    if pd.api.types.is_datetime64_any_dtype(clean.dtype):
        ordered = clean.sort_values()
        return ordered.iloc[len(ordered) // 2]
    mode = clean.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else clean.iloc[0]


def _classification_score(
    pipeline: Any,
    row: pd.DataFrame,
    predicted_class: int,
) -> tuple[float, str]:
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = np.asarray(pipeline.predict_proba(row), dtype=float)
            if probs.ndim == 2 and probs.shape[1] > predicted_class:
                return float(probs[0, predicted_class]), "Predicted-class probability"
        except Exception:
            pass
    if hasattr(pipeline, "decision_function"):
        try:
            score = np.asarray(pipeline.decision_function(row), dtype=float)
            if score.ndim == 1:
                raw = float(score[0])
                return (raw if predicted_class == 1 else -raw), "Predicted-class decision score"
            if score.ndim == 2 and score.shape[1] > predicted_class:
                return float(score[0, predicted_class]), "Predicted-class decision score"
        except Exception:
            pass
    predicted = int(np.asarray(pipeline.predict(row)).reshape(-1)[0])
    return (1.0 if predicted == predicted_class else 0.0), "Predicted-class match indicator"


def run_local_sensitivity(
    df: pd.DataFrame,
    result: SupervisedExperimentResult,
    spec: Mapping[str, Any],
    *,
    source_index: Any,
    current_revision: int,
    current_fingerprint: str,
    background_rows: int = 500,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LocalExplanationResult:
    """
    Measure one-row prediction sensitivity by replacing one source feature at a
    time with a holdout reference value. This is model-agnostic and non-additive.
    """
    clean = _validate_active_result(
        df,
        result,
        spec,
        current_revision=current_revision,
        current_fingerprint=current_fingerprint,
    )
    feature_columns = _fitted_feature_columns(result, clean)
    X_holdout, _, predictions = _holdout_frame(df, result, feature_columns)
    if source_index not in X_holdout.index:
        raise ExplainabilityError("The selected row is not part of the untouched holdout.")

    row = X_holdout.loc[[source_index]].copy(deep=True)
    if len(row) != 1:
        raise ExplainabilityError("The selected source index is not unique.")
    cap = max(20, min(int(background_rows), MAX_LOCAL_BACKGROUND_ROWS, len(X_holdout)))
    background = X_holdout.sample(n=cap, random_state=int(random_state)) if len(X_holdout) > cap else X_holdout
    references = {column: _reference_value(background[column]) for column in feature_columns}

    predicted_raw = np.asarray(result.fitted_pipeline.predict(row)).reshape(-1)[0]
    if result.task == CLASSIFICATION:
        predicted_class = int(predicted_raw)
        score_value, score_name = _classification_score(
            result.fitted_pipeline,
            row,
            predicted_class,
        )
        if result.target_encoder is None:
            raise ExplainabilityError("The target encoder is unavailable.")
        predicted_label = str(result.target_encoder.inverse_transform([predicted_class])[0])
        predicted_value = float(predicted_class)
    else:
        predicted_value = float(predicted_raw)
        predicted_label = f"{predicted_value:.6g}"
        score_value = predicted_value
        score_name = "Predicted value"

    rows: list[Dict[str, Any]] = []
    for column in feature_columns:
        altered = row.copy(deep=True)
        altered[column] = altered[column].astype("object")
        altered.at[source_index, column] = references[column]
        if result.task == CLASSIFICATION:
            altered_score, _ = _classification_score(
                result.fitted_pipeline,
                altered,
                int(predicted_raw),
            )
        else:
            altered_score = float(np.asarray(result.fitted_pipeline.predict(altered)).reshape(-1)[0])
        impact = float(score_value - altered_score)
        rows.append(
            {
                "Feature": column,
                "Observed Value": _json_safe(row.iloc[0][column]),
                "Reference Value": _json_safe(references[column]),
                "Original Score": float(score_value),
                "Score After Replacement": float(altered_score),
                "Sensitivity": impact,
                "Absolute Sensitivity": abs(impact),
                "Effect": "Supports prediction" if impact > 0 else "Suppresses prediction" if impact < 0 else "No measured change",
            }
        )
    sensitivity = pd.DataFrame(rows).sort_values("Absolute Sensitivity", ascending=False).reset_index(drop=True)
    actual_value = predictions.loc[source_index, "Actual"]
    warnings_out = [
        "Local sensitivity changes one feature at a time and is not an additive decomposition or causal estimate.",
        "Correlated features can substitute for one another and reduce measured sensitivity.",
    ]
    return LocalExplanationResult(
        experiment_id=result.experiment_id,
        source_index=source_index,
        task=result.task,
        target=result.target,
        predicted_label=predicted_label,
        predicted_value=predicted_value,
        score_name=score_name,
        score_value=float(score_value),
        actual_value=actual_value,
        sensitivity=sensitivity,
        reference_values=references,
        warnings=warnings_out,
    )


def _dense_safely(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        cells = int(matrix.shape[0]) * int(matrix.shape[1])
        if cells > MAX_SHAP_DENSE_CELLS:
            raise ExplainabilityError(
                f"Optional SHAP would require a dense matrix with {cells:,} cells, above the safety limit."
            )
        return matrix.toarray()
    array = np.asarray(matrix)
    cells = int(array.shape[0]) * int(array.shape[1]) if array.ndim == 2 else int(array.size)
    if cells > MAX_SHAP_DENSE_CELLS:
        raise ExplainabilityError("Optional SHAP transformed matrix exceeds the safety limit.")
    return array


def run_optional_shap(
    df: pd.DataFrame,
    result: SupervisedExperimentResult,
    spec: Mapping[str, Any],
    *,
    source_index: Any,
    current_revision: int,
    current_fingerprint: str,
    background_rows: int = 50,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ShapExplanationResult:
    """Attempt an optional SHAP explanation. The core Stage 10 path never requires SHAP."""
    try:
        import shap  # type: ignore
    except Exception:
        return ShapExplanationResult(
            available=False,
            method="Unavailable",
            feature_values=pd.DataFrame(),
            base_value=None,
            output_value=None,
            warning="Optional SHAP is not installed. Global permutation importance and local sensitivity remain fully available.",
        )

    clean = _validate_active_result(
        df,
        result,
        spec,
        current_revision=current_revision,
        current_fingerprint=current_fingerprint,
    )
    feature_columns = _fitted_feature_columns(result, clean)
    X_holdout, _, _ = _holdout_frame(df, result, feature_columns)
    if source_index not in X_holdout.index:
        raise ExplainabilityError("The selected row is not in the holdout.")
    row = X_holdout.loc[[source_index]].copy(deep=True)
    cap = max(10, min(int(background_rows), MAX_SHAP_BACKGROUND_ROWS, len(X_holdout)))
    background = X_holdout.sample(n=cap, random_state=int(random_state)) if len(X_holdout) > cap else X_holdout

    pipeline = result.fitted_pipeline
    preprocessor = pipeline.named_steps.get("features")
    model = pipeline.named_steps.get("model")
    if preprocessor is None or model is None:
        raise ExplainabilityError("The fitted preprocessing/model steps are unavailable.")
    background_t = _dense_safely(preprocessor.transform(background))
    row_t = _dense_safely(preprocessor.transform(row))
    names = list(map(str, result.feature_names))
    if row_t.shape[1] != len(names):
        raise ExplainabilityError("SHAP transformed width does not match the feature-name contract.")

    try:
        explainer = shap.Explainer(model, background_t, feature_names=names)
        explanation = explainer(row_t)
    except Exception as exc:
        return ShapExplanationResult(
            available=False,
            method="Unavailable for this estimator",
            feature_values=pd.DataFrame(),
            base_value=None,
            output_value=None,
            warning=f"Optional SHAP could not explain this fitted estimator safely: {exc}",
        )

    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    predicted_raw = int(np.asarray(pipeline.predict(row)).reshape(-1)[0]) if result.task == CLASSIFICATION else 0
    if values.ndim == 3:
        output_index = min(predicted_raw, values.shape[2] - 1)
        selected_values = values[0, :, output_index]
        selected_base = float(np.asarray(base).reshape(-1)[output_index]) if base.size > output_index else None
    elif values.ndim == 2:
        selected_values = values[0]
        selected_base = float(np.asarray(base).reshape(-1)[0]) if base.size else None
    elif values.ndim == 1:
        selected_values = values
        selected_base = float(np.asarray(base).reshape(-1)[0]) if base.size else None
    else:
        return ShapExplanationResult(
            available=False,
            method="Unsupported SHAP output",
            feature_values=pd.DataFrame(),
            base_value=None,
            output_value=None,
            warning="Optional SHAP returned an unsupported tensor shape.",
        )

    frame = pd.DataFrame(
        {
            "Transformed Feature": names,
            "Source Feature": [_source_from_transformed_name(name, clean) for name in names],
            "SHAP Value": np.asarray(selected_values, dtype=float),
        }
    )
    frame["Absolute SHAP"] = frame["SHAP Value"].abs()
    frame = frame.sort_values("Absolute SHAP", ascending=False).reset_index(drop=True)
    output_value = float(selected_base + frame["SHAP Value"].sum()) if selected_base is not None else None
    return ShapExplanationResult(
        available=True,
        method="SHAP on transformed fitted features",
        feature_values=frame,
        base_value=selected_base,
        output_value=output_value,
        warning="SHAP explains model output in transformed feature space; it does not establish causality.",
    )


def build_error_analysis(
    result: SupervisedExperimentResult,
    *,
    limit: int = 100,
) -> pd.DataFrame:
    if not isinstance(result, SupervisedExperimentResult):
        raise ExplainabilityError("A supervised experiment result is required.")
    predictions = result.predictions.copy(deep=True)
    predictions = predictions.reset_index(names="Source Index")
    if result.task == CLASSIFICATION:
        errors = predictions.loc[~predictions["Correct"].astype(bool)].copy()
        if "Confidence" in errors.columns:
            errors = errors.sort_values("Confidence", ascending=False)
    else:
        errors = predictions.sort_values("Absolute Error", ascending=False)
    return errors.head(max(1, min(int(limit), 5_000))).reset_index(drop=True)


def compute_segment_performance(
    df: pd.DataFrame,
    result: SupervisedExperimentResult,
    *,
    segment_column: str,
    min_group_rows: int = 3,
    max_groups: int = 50,
) -> pd.DataFrame:
    if segment_column not in df.columns:
        raise ExplainabilityError("The selected segment column does not exist.")
    if not df.index.is_unique:
        raise ExplainabilityError("Segment analysis requires a unique source index.")
    predictions = result.predictions.copy(deep=True)
    if len(predictions.index.difference(df.index)):
        raise ExplainabilityError("Some holdout source rows are unavailable for segment analysis.")
    joined = predictions.join(df[[segment_column]], how="left")
    joined[segment_column] = joined[segment_column].astype("string").fillna("__MISSING__")
    counts = joined[segment_column].value_counts(dropna=False)
    keep = counts[counts >= max(1, int(min_group_rows))].head(max(1, int(max_groups))).index
    joined = joined.loc[joined[segment_column].isin(keep)]
    rows: list[Dict[str, Any]] = []
    for value, group in joined.groupby(segment_column, dropna=False, sort=False):
        row: Dict[str, Any] = {"Segment": str(value), "Rows": int(len(group))}
        if result.task == CLASSIFICATION:
            row["Accuracy"] = float(accuracy_score(group["Actual"], group["Predicted"]))
            row["F1 Weighted"] = float(
                f1_score(group["Actual"], group["Predicted"], average="weighted", zero_division=0)
            )
            row["Error Rate"] = 1.0 - row["Accuracy"]
        else:
            actual = pd.to_numeric(group["Actual"], errors="coerce").to_numpy(dtype=float)
            predicted = pd.to_numeric(group["Predicted"], errors="coerce").to_numpy(dtype=float)
            row["MAE"] = float(mean_absolute_error(actual, predicted))
            row["RMSE"] = float(math.sqrt(mean_squared_error(actual, predicted)))
            row["R²"] = float(r2_score(actual, predicted)) if len(group) >= 2 else None
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_column = "Error Rate" if result.task == CLASSIFICATION else "RMSE"
    return frame.sort_values(sort_column, ascending=False, na_position="last").reset_index(drop=True)


def reconcile_explainability_report(
    report: Mapping[str, Any] | None,
    *,
    current_revision: int,
    current_fingerprint: str,
    current_experiment_id: str,
    current_pipeline_spec_fingerprint: str,
    artifact_available: bool,
) -> Dict[str, Any]:
    source = dict(report or {})
    if not source:
        return {
            "status": "Pending",
            "stale": False,
            "artifact_available": False,
        }
    stale_reasons: list[str] = []
    if int(source.get("dataset_revision", -1)) != int(current_revision):
        stale_reasons.append("dataset revision changed")
    if str(source.get("dataset_fingerprint", "")) != str(current_fingerprint):
        stale_reasons.append("dataset fingerprint changed")
    if str(source.get("experiment_id", "")) != str(current_experiment_id or ""):
        stale_reasons.append("ML experiment changed")
    if str(source.get("pipeline_spec_fingerprint", "")) != str(current_pipeline_spec_fingerprint or ""):
        stale_reasons.append("feature pipeline changed")
    if not artifact_available:
        stale_reasons.append("explanation artifact is unavailable")

    source["artifact_available"] = bool(artifact_available)
    source["stale"] = bool(stale_reasons)
    source["status"] = "Stale" if stale_reasons else "Completed"
    source["stale_reasons"] = list(dict.fromkeys(stale_reasons))
    return source
