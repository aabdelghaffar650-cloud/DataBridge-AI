# ════════════════════════════════════════════════════════
#  DataBridge AI — Card & Stat Components
# ════════════════════════════════════════════════════════
from typing import List, Tuple


def metric_card(label: str, value: str, sub: str = "", color: str = "#e0e0f0") -> str:
    return f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value" style="color:{color}">{value}</div>
  <div class="sub">{sub}</div>
</div>"""


def metrics_row(cards: List[Tuple[str, str, str, str]], cols: int = 4) -> str:
    """
    cards: list of (label, value, sub, color)
    """
    inner = "".join(metric_card(*c) for c in cards)
    return f'<div class="metrics-row" style="grid-template-columns:repeat({cols},1fr);">{inner}</div>'


def info_box(text: str) -> str:
    return f'<div class="info-box">{text}</div>'


def warning_box(text: str) -> str:
    return f'<div class="warning-box">⚠️ {text}</div>'


def success_box(text: str) -> str:
    return f'<div class="success-box">✅ {text}</div>'


def section_header(icon: str, title: str, badge: str = "") -> str:
    badge_html = (
        f'<div class="count">{badge}</div>' if badge else ""
    )
    return f"""
<div class="section-header">
  <div class="icon">{icon}</div>
  <h2>{title}</h2>
  {badge_html}
</div>"""
