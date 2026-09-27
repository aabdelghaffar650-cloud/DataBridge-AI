# ════════════════════════════════════════════════════════
# DataBridge AI — Safe Auto Retraining Workflow
# Stage 16: trigger assessment, contract-locked replay, candidate creation
#           and mandatory Champion/Challenger human promotion gate.
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from modules.feature_pipeline import (
    normalise_feature_pipeline_spec,
    validate_feature_pipeline_spec,
)
from modules.ml_engine import SupervisedExperimentResult, run_supervised_experiment
from modules.model_governance import (
    STATUS_CHAMPION,
    governance_status_for_package,
    submit_as_challenger,
)
from modules.model_package import (
    LoadedModelPackage,
    ModelPackageBuild,
    ModelPackageError,
    create_signed_model_package,
    prepare_model_input_frame,
)
from modules.model_registry import (
    ModelRegistryError,
    delete_registered_package,
    register_signed_package,
)


RETRAINING_CONTRACT_VERSION = 1
RECOMMEND_STATUSES = {"Drifted", "Critical"}
VALID_MONITOR_STATUSES = {"Stable", "Watch", "Drifted", "Critical"}
DEFAULT_CLASSIFICATION_DROP = 0.05
DEFAULT_REGRESSION_RMSE_INCREASE = 0.15
MIN_RETRAIN_ROWS = 30


class RetrainingWorkflowError(ValueError):
    """Raised when retraining cannot proceed without weakening the saved contract."""


@dataclass(frozen=True)
class RetrainingAssessment:
    package_id: str
    family_name: str
    monitoring_status: str
    recommended: bool
    eligible: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    reasons: tuple[str, ...]
    performance_signal: Dict[str, Any]
    contract_summary: Dict[str, Any]

    def report(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "family_name": self.family_name,
            "monitoring_status": self.monitoring_status,
            "recommended": bool(self.recommended),
            "eligible": bool(self.eligible),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
            "performance_signal": dict(self.performance_signal),
            "contract_summary": dict(self.contract_summary),
        }


@dataclass(frozen=True)
class RetrainingCandidateResult:
    experiment: SupervisedExperimentResult
    rebound_spec: Dict[str, Any]
    package_build: ModelPackageBuild
    registry_metadata: Dict[str, Any]
    governance: Dict[str, Any]
    assessment: RetrainingAssessment


# ── Contract helpers ─────────────────────────────────────────────────────────
def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _logic_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    clean = normalise_feature_pipeline_spec(spec)
    clean["configured_revision"] = 0
    clean["configured_fingerprint"] = ""
    return clean


def feature_logic_fingerprint(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(_logic_spec(spec))).hexdigest()


def build_retraining_contract(
    result: SupervisedExperimentResult,
    feature_pipeline_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the replay contract stored only inside the authenticated model payload."""
    if not isinstance(result, SupervisedExperimentResult):
        raise RetrainingWorkflowError("A completed supervised experiment is required.")
    clean = normalise_feature_pipeline_spec(feature_pipeline_spec)
    if clean.get("task") != result.task or clean.get("target") != result.target:
        raise RetrainingWorkflowError("Feature pipeline task/target does not match the trained model.")

    config = dict(getattr(result, "experiment_config", {}) or {})
    candidate_models = list(config.get("candidate_models") or [])
    if not candidate_models:
        candidate_models = [
            str(row.get("Model"))
            for row in result.leaderboard.to_dict(orient="records")
            if not bool(row.get("Baseline")) and str(row.get("Status")) == "OK"
        ]
    candidate_models = list(dict.fromkeys(name for name in candidate_models if name))
    if not candidate_models:
        candidate_models = [result.selected_model]

    split_summary = dict(result.split_summary or {})
    holdout_size = float(
        config.get("holdout_size")
        or split_summary.get("holdout_fraction")
        or (result.holdout_rows / max(result.modelling_rows, 1))
    )
    contract = {
        "contract_version": RETRAINING_CONTRACT_VERSION,
        "task": result.task,
        "target": result.target,
        "feature_pipeline_spec": clean,
        "feature_logic_fingerprint": feature_logic_fingerprint(clean),
        "experiment": {
            "source_experiment_id": result.experiment_id,
            "split_strategy": str(config.get("split_strategy") or result.split_strategy),
            "split_column": str(config.get("split_column") or result.split_column or ""),
            "holdout_size": holdout_size,
            "cv_folds": int(config.get("cv_folds") or result.requested_cv_folds or result.actual_cv_folds),
            "random_state": int(config.get("random_state") or result.random_state),
            "candidate_models": candidate_models,
            "class_weight_mode": str(config.get("class_weight_mode") or "none"),
            "tune_best": bool(config.get("tune_best", result.tuned)),
            "tuning_iterations": int(config.get("tuning_iterations") or max(result.tuning_iterations, 8)),
            "max_rows": config.get("max_rows", 100_000),
            "include_xgboost": bool(config.get("include_xgboost", True)),
        },
        "safety": {
            "rebind_only_revision_and_fingerprint": True,
            "holdout_used_for_selection": False,
            "auto_promotion_allowed": False,
            "governance_target_status": "Challenger",
        },
    }
    return contract


def _contract(package: LoadedModelPackage) -> Dict[str, Any]:
    if not isinstance(package, LoadedModelPackage) or not package.trusted:
        raise RetrainingWorkflowError("A verified signed Champion package is required.")
    contract = getattr(package, "retraining_contract", None)
    if not isinstance(contract, Mapping) or not contract:
        raise RetrainingWorkflowError(
            "This package has no Stage 16 retraining contract. Rebuild and re-register the model package after installing Stage 16."
        )
    contract = copy.deepcopy(dict(contract))
    if int(contract.get("contract_version", 0)) != RETRAINING_CONTRACT_VERSION:
        raise RetrainingWorkflowError("Unsupported retraining contract version.")
    if str(contract.get("task")) != package.task or str(contract.get("target")) != package.target:
        raise RetrainingWorkflowError("Retraining contract task/target does not match the signed model.")
    spec = contract.get("feature_pipeline_spec")
    if not isinstance(spec, Mapping):
        raise RetrainingWorkflowError("Retraining feature contract is missing.")
    signed_fp = str(contract.get("feature_logic_fingerprint", ""))
    if not signed_fp or signed_fp != feature_logic_fingerprint(spec):
        raise RetrainingWorkflowError("Retraining feature contract integrity check failed.")
    experiment = contract.get("experiment")
    if not isinstance(experiment, Mapping):
        raise RetrainingWorkflowError("Retraining experiment contract is missing.")
    return contract


def retraining_contract_summary(package: LoadedModelPackage) -> Dict[str, Any]:
    contract = _contract(package)
    spec = normalise_feature_pipeline_spec(contract["feature_pipeline_spec"])
    experiment = dict(contract["experiment"])
    return {
        "contract_version": int(contract["contract_version"]),
        "task": package.task,
        "target": package.target,
        "feature_logic_fingerprint": str(contract["feature_logic_fingerprint"]),
        "feature_count": int(len(spec.get("feature_columns", []))),
        "split_strategy": str(experiment.get("split_strategy", "")),
        "split_column": str(experiment.get("split_column", "")),
        "holdout_size": float(experiment.get("holdout_size", 0.20)),
        "cv_folds": int(experiment.get("cv_folds", 5)),
        "random_state": int(experiment.get("random_state", 42)),
        "candidate_models": list(map(str, experiment.get("candidate_models", []) or [])),
        "class_weight_mode": str(experiment.get("class_weight_mode", "none")),
        "tune_best": bool(experiment.get("tune_best", False)),
        "tuning_iterations": int(experiment.get("tuning_iterations", 8)),
        "max_rows": experiment.get("max_rows", 100_000),
    }


# ── Trigger assessment ───────────────────────────────────────────────────────
def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _performance_signal(package: LoadedModelPackage, report: Mapping[str, Any]) -> Dict[str, Any]:
    observed = dict(report.get("observed_performance", {}) or {})
    if not observed.get("available"):
        return {"available": False, "triggered": False}
    current = dict(observed.get("metrics", {}) or {})
    signed = dict(observed.get("signed_holdout_metrics", {}) or {})

    if package.task == "classification":
        for metric in ("F1 Macro", "Balanced Accuracy", "Accuracy"):
            now = _number(current.get(metric))
            before = _number(signed.get(metric))
            if now is None or before is None:
                continue
            drop = before - now
            return {
                "available": True,
                "metric": metric,
                "signed_holdout": before,
                "current": now,
                "change": drop,
                "threshold": DEFAULT_CLASSIFICATION_DROP,
                "triggered": drop >= DEFAULT_CLASSIFICATION_DROP,
                "direction": "drop",
            }
    else:
        now = _number(current.get("RMSE"))
        before = _number(signed.get("RMSE"))
        if now is not None and before is not None:
            increase = (now - before) / max(abs(before), 1e-12)
            return {
                "available": True,
                "metric": "RMSE",
                "signed_holdout": before,
                "current": now,
                "change": increase,
                "threshold": DEFAULT_REGRESSION_RMSE_INCREASE,
                "triggered": increase >= DEFAULT_REGRESSION_RMSE_INCREASE,
                "direction": "relative_increase",
            }
    return {"available": False, "triggered": False}


def assess_retraining_need(
    package: LoadedModelPackage,
    frame: pd.DataFrame,
    *,
    monitoring_report: Optional[Mapping[str, Any]] = None,
    require_champion: bool = True,
) -> RetrainingAssessment:
    blockers: list[str] = []
    warnings: list[str] = []
    reasons: list[str] = []
    family_name = ""

    try:
        contract = _contract(package)
        summary = retraining_contract_summary(package)
    except RetrainingWorkflowError as exc:
        return RetrainingAssessment(
            package_id=getattr(package, "package_id", ""),
            family_name="",
            monitoring_status="Pending",
            recommended=False,
            eligible=False,
            blockers=(str(exc),),
            warnings=(),
            reasons=(),
            performance_signal={"available": False, "triggered": False},
            contract_summary={},
        )

    if require_champion:
        try:
            governance = governance_status_for_package(package.package_id)
            family_name = str(governance.get("family_name", ""))
            if str(governance.get("status")) != STATUS_CHAMPION:
                blockers.append("Safe auto retraining can start only from the active Champion of a governed model family.")
        except Exception as exc:
            blockers.append(f"Champion governance could not be verified: {exc}")

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        blockers.append("Retraining data must contain at least one row.")
    else:
        if len(frame) < MIN_RETRAIN_ROWS:
            blockers.append(f"Retraining requires at least {MIN_RETRAIN_ROWS} rows before task-specific validation.")
        required = list(package.required_columns) + [package.target]
        split_column = str(contract["experiment"].get("split_column", "") or "")
        if split_column:
            required.append(split_column)
        missing = [column for column in dict.fromkeys(required) if column not in frame.columns]
        if missing:
            blockers.append("Retraining data is missing contract columns: " + ", ".join(missing[:12]))

    report = dict(monitoring_report or {})
    status = str(report.get("overall_status", "Pending"))
    if status not in VALID_MONITOR_STATUSES:
        status = "Pending"
    if report:
        if str(report.get("package_id", "")) != package.package_id:
            blockers.append("Monitoring report belongs to a different model package.")
        if str(report.get("task", package.task)) != package.task or str(report.get("target", package.target)) != package.target:
            blockers.append("Monitoring report task/target does not match the Champion contract.")

    perf = _performance_signal(package, report)
    recommended = False
    if status in RECOMMEND_STATUSES:
        recommended = True
        reasons.append(f"Monitoring status is {status}.")
    if perf.get("triggered"):
        recommended = True
        metric = str(perf.get("metric", "performance"))
        reasons.append(f"Observed {metric} crossed the retraining degradation threshold.")
    if status == "Watch" and not perf.get("triggered"):
        warnings.append("Monitoring is in Watch state, but the configured retraining threshold has not been crossed.")
    if status in {"Stable", "Pending"} and not perf.get("triggered"):
        warnings.append("Automatic retraining is not currently recommended; a manual override remains possible after review.")

    # Schema failures that make the contract impossible to replay are blockers.
    schema = dict(report.get("schema", {}) or {})
    if report and schema and not bool(schema.get("valid", True)):
        missing_columns = schema.get("missing_columns", []) or []
        if missing_columns:
            blockers.append("Monitoring schema is missing model inputs required by the retraining contract.")
        else:
            warnings.append("Monitoring schema reported compatibility issues; retraining data will be validated independently.")

    return RetrainingAssessment(
        package_id=package.package_id,
        family_name=family_name,
        monitoring_status=status,
        recommended=recommended,
        eligible=not blockers,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        reasons=tuple(dict.fromkeys(reasons)),
        performance_signal=perf,
        contract_summary=summary,
    )


# ── Contract-locked replay ──────────────────────────────────────────────────
def rebind_retraining_spec(
    package: LoadedModelPackage,
    frame: pd.DataFrame,
    *,
    dataset_revision: int,
    dataset_fingerprint: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    contract = _contract(package)
    original = normalise_feature_pipeline_spec(contract["feature_pipeline_spec"])
    expected_logic = str(contract["feature_logic_fingerprint"])
    if feature_logic_fingerprint(original) != expected_logic:
        raise RetrainingWorkflowError("Stored retraining feature logic no longer matches its signed contract fingerprint.")

    rebound = copy.deepcopy(original)
    rebound["configured_revision"] = int(dataset_revision)
    rebound["configured_fingerprint"] = str(dataset_fingerprint or "")
    rebound = normalise_feature_pipeline_spec(rebound)
    if feature_logic_fingerprint(rebound) != expected_logic:
        raise RetrainingWorkflowError("Retraining attempted to alter feature-engineering logic; only dataset binding may change.")

    try:
        prepared_frame = prepare_model_input_frame(package, frame)
    except ModelPackageError as exc:
        raise RetrainingWorkflowError(f"Retraining derivation replay failed: {exc}") from exc

    validation = validate_feature_pipeline_spec(
        prepared_frame,
        rebound,
        require_target=True,
        current_revision=int(dataset_revision),
        current_fingerprint=str(dataset_fingerprint or ""),
    )
    if not validation.get("valid"):
        raise RetrainingWorkflowError("Retraining feature contract is invalid on the new data: " + "; ".join(validation.get("blockers", [])))
    if validation.get("stale"):
        raise RetrainingWorkflowError("Retraining contract unexpectedly remained stale after safe rebinding.")
    return rebound, validation


def run_safe_retraining_experiment(
    package: LoadedModelPackage,
    frame: pd.DataFrame,
    *,
    dataset_revision: int,
    dataset_fingerprint: str,
    monitoring_report: Optional[Mapping[str, Any]] = None,
    manual_override: bool = False,
) -> tuple[SupervisedExperimentResult, Dict[str, Any], RetrainingAssessment]:
    assessment = assess_retraining_need(
        package,
        frame,
        monitoring_report=monitoring_report,
        require_champion=True,
    )
    if not assessment.eligible:
        raise RetrainingWorkflowError("Retraining gate is blocked: " + "; ".join(assessment.blockers))
    if not assessment.recommended and not manual_override:
        raise RetrainingWorkflowError(
            "Retraining is not currently recommended. Use an explicit manual override only after reviewing the monitoring evidence."
        )

    contract = _contract(package)
    rebound, _ = rebind_retraining_spec(
        package,
        frame,
        dataset_revision=dataset_revision,
        dataset_fingerprint=dataset_fingerprint,
    )
    config = dict(contract["experiment"])
    candidate_models = list(map(str, config.get("candidate_models", []) or []))
    if not candidate_models:
        raise RetrainingWorkflowError("Retraining contract contains no candidate models.")

    try:
        prepared_frame = prepare_model_input_frame(package, frame)
    except ModelPackageError as exc:
        raise RetrainingWorkflowError(f"Retraining derivation replay failed: {exc}") from exc

    result = run_supervised_experiment(
        prepared_frame,
        rebound,
        dataset_revision=int(dataset_revision),
        dataset_fingerprint=str(dataset_fingerprint),
        split_strategy=str(config.get("split_strategy", "stratified")),
        split_column=str(config.get("split_column", "") or ""),
        holdout_size=float(config.get("holdout_size", 0.20)),
        cv_folds=int(config.get("cv_folds", 5)),
        random_state=int(config.get("random_state", 42)),
        model_names=candidate_models,
        class_weight_mode=str(config.get("class_weight_mode", "none")),
        tune_best=bool(config.get("tune_best", False)),
        tuning_iterations=int(config.get("tuning_iterations", 8)),
        max_rows=config.get("max_rows", 100_000),
        include_xgboost=bool(config.get("include_xgboost", True)),
    )
    # Verify that replay did not silently swap the training contract.
    if feature_logic_fingerprint(rebound) != str(contract["feature_logic_fingerprint"]):
        raise RetrainingWorkflowError("Completed retraining no longer matches the Champion feature-logic contract.")
    return result, rebound, assessment


def create_and_register_retraining_challenger(
    champion: LoadedModelPackage,
    frame: pd.DataFrame,
    result: SupervisedExperimentResult,
    rebound_spec: Mapping[str, Any],
    assessment: RetrainingAssessment,
    *,
    semantic_profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    actor: str = "local-user",
    note: str = "Stage 16 safe retraining candidate",
) -> RetrainingCandidateResult:
    """Create a new immutable package and stop at Challenger. Never promotes it."""
    if not assessment.eligible:
        raise RetrainingWorkflowError("Cannot register a retraining candidate from a blocked assessment.")
    governance = governance_status_for_package(champion.package_id)
    if str(governance.get("status")) != STATUS_CHAMPION:
        raise RetrainingWorkflowError("The source model is no longer the active Champion.")
    family_name = str(governance.get("family_name") or assessment.family_name or "")
    if not family_name:
        raise RetrainingWorkflowError("Champion model family could not be resolved.")

    metadata: Optional[Dict[str, Any]] = None
    try:
        build = create_signed_model_package(
            result,
            frame,
            rebound_spec,
            semantic_profiles=semantic_profiles,
            feature_derivation_recipe=champion.feature_derivation_recipe,
        )
        metadata = register_signed_package(
            build.package_bytes,
            label=f"Auto retrain from {champion.package_id}",
            model_family=family_name,
        )
        try:
            challenger = submit_as_challenger(
                str(metadata["package_id"]),
                actor=actor,
                note=(note or "Stage 16 safe retraining candidate")[:1000],
            )
        except Exception:
            # Avoid leaving a half-completed Candidate if the governance
            # transition fails after registry persistence. Candidate deletion is
            # itself governance-audited and active Champions remain protected.
            try:
                delete_registered_package(str(metadata["package_id"]))
            except Exception:
                pass
            raise
    except (ModelPackageError, ModelRegistryError) as exc:
        raise RetrainingWorkflowError(str(exc)) from exc
    except Exception as exc:
        raise RetrainingWorkflowError(f"Retraining candidate registration failed safely: {exc}") from exc

    if str(challenger.get("status")) != "Challenger":
        raise RetrainingWorkflowError("Retraining workflow did not stop at Challenger as required.")
    # Explicit invariant: Stage 16 has no path that calls promote_challenger.
    return RetrainingCandidateResult(
        experiment=result,
        rebound_spec=dict(rebound_spec),
        package_build=build,
        registry_metadata=dict(metadata),
        governance=dict(challenger),
        assessment=assessment,
    )
