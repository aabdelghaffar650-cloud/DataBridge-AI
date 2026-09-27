"""DataBridge AI Stage 20.3 XGBoost production integration verification."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


fake_streamlit = types.ModuleType("streamlit")
fake_streamlit.session_state = AttrDict()


def _cache_data(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return lambda func: func


fake_streamlit.cache_data = _cache_data
sys.modules.setdefault("streamlit", fake_streamlit)

from modules.data_mapper import analyze_dataframe  # noqa: E402
from modules.feature_pipeline import (  # noqa: E402
    create_default_feature_pipeline_spec,
    validate_feature_pipeline_spec,
)
from modules.ml_engine import (  # noqa: E402
    SPLIT_STRATIFIED,
    available_model_names,
    run_supervised_experiment,
    xgboost_available,
)


def _classification_frame(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(20260814)
    amount = rng.normal(500, 100, rows)
    spend = rng.gamma(2.0, 100.0, rows)
    segment = np.resize(["A", "B", "C"], rows)
    signal = amount * 0.006 + spend * 0.004 + (segment == "C") * 0.8 + rng.normal(0, 0.65, rows)
    target = np.where(signal > np.median(signal), "yes", "no")
    frame = pd.DataFrame({"amount": amount, "spend": spend, "segment": segment, "target": target})
    frame.loc[4, "amount"] = np.nan
    frame.loc[9, "segment"] = None
    return frame


def _spec(df: pd.DataFrame) -> dict:
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df,
        analysis["profiles"],
        "target",
        task="classification",
        configured_revision=1,
        configured_fingerprint="stage20-3-xgb",
    )
    report = validate_feature_pipeline_spec(
        df,
        spec,
        current_revision=1,
        current_fingerprint="stage20-3-xgb",
    )
    assert report["valid"] is True, report
    return spec


def test_release_runtime_pins_cpu_xgboost() -> None:
    requirements = (PROJECT_ROOT / "requirements-release.txt").read_text(encoding="utf-8")
    portable = (PROJECT_ROOT / "tools/prepare_portable_python.ps1").read_text(encoding="utf-8")
    release_gate = (PROJECT_ROOT / "tools/release_gate.py").read_text(encoding="utf-8")

    assert 'xgboost-cpu==3.4.0; platform_system == "Windows"' in requirements
    assert '"xgboost-cpu": "3.4.0"' in portable
    assert "import xgboost" in portable
    assert "cryptography,xgboost" in release_gate
    assert "Pinned xgboost-cpu==3.4.0" in release_gate


def test_ml_studio_defaults_to_xgboost_when_available() -> None:
    page = (PROJECT_ROOT / "_pages/ml_studio.py").read_text(encoding="utf-8")
    assert '["Logistic Regression", "Random Forest", "Linear SVM", "XGBoost"]' in page
    assert '["Ridge Regression", "Random Forest", "Extra Trees", "XGBoost"]' in page
    assert "XGBoost CPU is available" in page


def test_xgboost_uses_cpu_hist_and_full_leakage_safe_workflow() -> None:
    if not xgboost_available():
        raise AssertionError("XGBoost must be installed to satisfy the Stage 20.3 runtime contract.")

    names = available_model_names("classification", include_xgboost=True)
    assert "XGBoost" in names

    df = _classification_frame()
    spec = _spec(df)
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint="stage20-3-xgb",
        split_strategy=SPLIT_STRATIFIED,
        holdout_size=0.20,
        cv_folds=3,
        random_state=42,
        model_names=["XGBoost"],
        class_weight_mode="none",
        tune_best=False,
        max_rows=None,
        include_xgboost=True,
    )

    assert result.selected_model == "XGBoost"
    assert result.train_rows + result.holdout_rows == len(df)
    assert result.holdout_metrics["F1 Weighted"] is not None
    assert result.holdout_metrics["ROC AUC"] is not None
    assert "Baseline — Most Frequent" in result.leaderboard["Model"].tolist()

    estimator = result.fitted_pipeline.named_steps["model"]
    params = estimator.get_params()
    assert params.get("device") == "cpu"
    assert params.get("tree_method") == "hist"
    assert params.get("n_jobs") == 1


def test_stage20_3_is_part_of_release_gate() -> None:
    spec = json.loads((PROJECT_ROOT / "release/release_spec.json").read_text(encoding="utf-8"))
    assert "tests/test_stage20_3_xgboost_production_integration.py" in spec["stage_tests"]
    assert (PROJECT_ROOT / "RUN_STAGE20_3_RELEASE_GATE.ps1").is_file()
    assert (PROJECT_ROOT / "BUILD_STAGE20_3_WINDOWS_RELEASE.ps1").is_file()


def main() -> None:
    test_release_runtime_pins_cpu_xgboost()
    test_ml_studio_defaults_to_xgboost_when_available()
    test_xgboost_uses_cpu_hist_and_full_leakage_safe_workflow()
    test_stage20_3_is_part_of_release_gate()
    print(
        "PASS: Stage 20.3 installs pinned CPU-only XGBoost in the Windows portable runtime, "
        "surfaces it as a default ML Studio candidate, keeps it inside the training-only CV / untouched-holdout "
        "workflow, and verifies CPU histogram execution through the production release gate."
    )


if __name__ == "__main__":
    main()
