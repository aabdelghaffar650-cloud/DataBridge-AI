# ════════════════════════════════════════════════════════
#  DataBridge AI — KPI Tracker Module
# ════════════════════════════════════════════════════════
from typing import Dict, Any
import pandas as pd


AGG_FUNCS = {
    "Sum":   lambda s: float(s.sum()),
    "Mean":  lambda s: float(s.mean()),
    "Max":   lambda s: float(s.max()),
    "Min":   lambda s: float(s.min()),
    "Count": lambda s: float(s.count()),
}


def compute_kpi(
    df: pd.DataFrame,
    col: str,
    agg_method: str = "Sum",
) -> float:
    if col not in df.columns:
        return 0.0
    func = AGG_FUNCS.get(agg_method, AGG_FUNCS["Sum"])
    return func(df[col].dropna())


def compute_achievement(current: float, target: float) -> float:
    if target == 0:
        return 0.0
    return round(current / target * 100, 1)


def build_kpi_report(
    df: pd.DataFrame,
    kpi_targets: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Returns {col: {current, annual_target, quarterly_target,
                   annual_pct, quarterly_pct, agg, status}}
    """
    report = {}
    for col, cfg in kpi_targets.items():
        if col not in df.columns:
            continue
        agg      = cfg.get("agg", "Sum")
        current  = compute_kpi(df, col, agg)
        ann_tgt  = float(cfg.get("annual", 0))
        qtr_tgt  = float(cfg.get("quarterly", ann_tgt / 4 if ann_tgt else 0))
        ann_pct  = compute_achievement(current, ann_tgt)
        qtr_pct  = compute_achievement(current, qtr_tgt)
        status   = "✅ Met" if ann_pct >= 100 else ("⚠️ On Track" if ann_pct >= 70 else "🔴 Below")
        report[col] = {
            "current":           current,
            "annual_target":     ann_tgt,
            "quarterly_target":  qtr_tgt,
            "annual_pct":        ann_pct,
            "quarterly_pct":     qtr_pct,
            "agg":               agg,
            "status":            status,
        }
    return report
