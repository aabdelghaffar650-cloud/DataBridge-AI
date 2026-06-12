# ════════════════════════════════════════════════════════
#  DataBridge AI — Professional Flat UI Styles
# ════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg: #080810;
    --surface: #0c0c1e;
    --surface-2: #101025;
    --surface-3: #111127;
    --border: #13132a;
    --border-soft: #1a1a34;
    --text: #f4f4f7;
    --text-muted: #8b8ba7;
    --text-faint: #5f5f78;
    --primary: #7c6aff;
    --primary-soft: #171333;
    --success: #3ddc97;
    --warning: #f5b85b;
    --danger: #ff5c7a;
}

/* ── Global reset ───────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
.main .block-container,
[data-testid="stAppViewContainer"] .block-container {
    padding: 1.15rem 1.65rem 2.5rem 1.65rem !important;
    max-width: 100% !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }

/* ── Fixed sidebar shell ────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 312px !important;
    max-width: 312px !important;
    width: 312px !important;
}
section[data-testid="stSidebar"] > div {
    background: var(--bg) !important;
    padding: 1.05rem .95rem 1.2rem .95rem !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
section[data-testid="collapsedControl"],
button[aria-label*="sidebar" i],
button[title*="sidebar" i],
button[aria-label*="collapse" i],
button[title*="collapse" i] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavLink"],
[data-testid="stSidebarNav"],
section[data-testid="stSidebar"] nav {
    display: none !important;
}

/* ── Sidebar brand ──────────────────────────────────── */
.sidebar-brand {
    padding: .35rem .65rem .95rem .65rem;
    margin-bottom: .25rem;
    border-bottom: 1px solid var(--border);
}
.sidebar-logo {
    color: var(--text);
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: -.02em;
}
.sidebar-subtitle {
    color: var(--text-faint);
    font-size: .68rem;
    margin-top: .2rem;
    text-transform: uppercase;
    letter-spacing: .11em;
}
.sidebar-divider {
    height: 1px;
    background: var(--border);
    margin: 1.05rem .65rem;
}
.sidebar-section-title,
.nav-group-label {
    color: var(--text-faint);
    font-size: .63rem;
    line-height: 1;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin: 1rem .7rem .35rem .7rem;
}

/* ── Grouped navigation ─────────────────────────────── */
.st-key-sidebar_nav .stButton { margin-bottom: .12rem !important; }
.st-key-sidebar_nav .stButton > button {
    height: 34px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-left: 3px solid transparent !important;
    border-radius: 0 8px 8px 0 !important;
    color: var(--text-muted) !important;
    font-size: .83rem !important;
    font-weight: 500 !important;
    padding: .3rem .7rem .3rem 1rem !important;
    box-shadow: none !important;
    transition: background .12s ease, color .12s ease, border-color .12s ease !important;
}
.st-key-sidebar_nav .stButton > button:hover {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border-color: transparent !important;
    border-left-color: var(--border-soft) !important;
    box-shadow: none !important;
    transform: none !important;
}
.st-key-sidebar_nav .stButton > button:focus,
.st-key-sidebar_nav .stButton > button:active {
    border-left-color: var(--primary) !important;
    outline: none !important;
    box-shadow: none !important;
}
.nav-item {
    height: 34px;
    display: flex;
    align-items: center;
    margin: .12rem 0;
    padding: .3rem .7rem .3rem 1rem;
    border: 1px solid transparent;
    border-left: 3px solid transparent;
    border-radius: 0 8px 8px 0;
    color: var(--text-muted);
    font-size: .83rem;
    font-weight: 500;
    box-sizing: border-box;
}
.nav-item.active {
    background: var(--surface-2);
    border-color: var(--border-soft);
    border-left-color: var(--primary);
    color: var(--text);
    font-weight: 650;
}

/* ── Sidebar status blocks ──────────────────────────── */
.sidebar-status {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .75rem .8rem;
    margin: 1rem .15rem .75rem .15rem;
}
.status-label {
    color: var(--text-faint);
    font-size: .62rem;
    letter-spacing: .14em;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: .38rem;
}
.status-file {
    color: var(--text);
    font-size: .82rem;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-shape {
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem;
    margin-top: .2rem;
}
.status-chips { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .6rem; }
.mini-chip {
    display: inline-flex;
    align-items: center;
    border: 1px solid var(--border-soft);
    background: var(--surface-2);
    color: var(--text-muted);
    border-radius: 999px;
    padding: .16rem .45rem;
    font-size: .64rem;
    line-height: 1.3;
    font-weight: 650;
}
.mini-chip.ok { color: var(--success); border-color: rgba(61,220,151,.25); background: rgba(61,220,151,.06); }
.mini-chip.warn { color: var(--warning); border-color: rgba(245,184,91,.25); background: rgba(245,184,91,.06); }
.mini-chip.danger { color: var(--danger); border-color: rgba(255,92,122,.25); background: rgba(255,92,122,.06); }
.account-pill {
    margin: .7rem .15rem 0 .15rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: .45rem .7rem;
    color: var(--text-muted);
    font-size: .78rem;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.sidebar-footer {
    color: #3d3d55;
    font-size: .66rem;
    padding: .2rem .65rem;
}

/* ── Top header ─────────────────────────────────────── */
.top-header {
    display: grid;
    grid-template-columns: minmax(210px, 1.1fr) minmax(260px, 1.5fr) auto;
    gap: 1rem;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: .85rem 1rem;
    margin: 0 0 1.15rem 0;
}
.top-brand { min-width: 0; }
.logo {
    color: var(--text);
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: -.03em;
    line-height: 1;
}
.logo span { color: var(--primary); }
.subtitle {
    color: var(--text-faint);
    font-size: .68rem;
    letter-spacing: .06em;
    margin-top: .28rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.dataset-head {
    min-width: 0;
    border-left: 1px solid var(--border);
    padding-left: 1rem;
}
.dataset-title {
    color: var(--text);
    font-size: .88rem;
    font-weight: 700;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
.dataset-meta {
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem;
    margin-top: .18rem;
}
.top-chips {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: .4rem;
    flex-wrap: wrap;
}
.top-chip {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    background: var(--surface-2);
    border: 1px solid var(--border-soft);
    color: var(--text-muted);
    border-radius: 999px;
    padding: .22rem .55rem;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .02em;
}
.top-chip.ok { color: var(--success); border-color: rgba(61,220,151,.24); background: rgba(61,220,151,.06); }
.top-chip.warn { color: var(--warning); border-color: rgba(245,184,91,.25); background: rgba(245,184,91,.06); }
.top-chip.danger { color: var(--danger); border-color: rgba(255,92,122,.25); background: rgba(255,92,122,.06); }

/* ── Section headers/cards ──────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: .75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .85rem 1rem;
    margin-bottom: 1.15rem;
}
.section-header .icon {
    width: 30px;
    height: 30px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--surface-2);
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    font-size: 1rem;
    filter: grayscale(.25);
}
.section-header h2 {
    margin: 0;
    font-size: .98rem;
    color: var(--text);
    font-weight: 750;
    letter-spacing: -.01em;
}
.section-header .count {
    margin-left: auto;
    color: var(--text-muted);
    background: var(--surface-2);
    border: 1px solid var(--border-soft);
    border-radius: 999px;
    padding: .24rem .6rem;
    font-size: .65rem;
    font-weight: 750;
    letter-spacing: .08em;
    text-transform: uppercase;
}
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .85rem;
    margin-bottom: 1.15rem;
}
.metric-card,
.profile-item,
.login-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    box-shadow: none !important;
}
.metric-card:hover { border-color: var(--border-soft); }
.metric-card .label,
.profile-item .tag,
label,
[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: .1em;
}
.metric-card .label {
    color: var(--text-faint);
    font-size: .65rem;
    font-weight: 750;
    margin-bottom: .4rem;
}
.metric-card .value,
div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
}
.metric-card .value { font-size: 1.45rem; line-height: 1.15; }
.metric-card .sub { color: var(--text-faint); font-size: .7rem; margin-top: .28rem; }

/* ── Flat message boxes ─────────────────────────────── */
.info-box, .warning-box, .success-box {
    border-radius: 8px;
    padding: .8rem .95rem;
    margin-bottom: 1rem;
    font-size: .82rem;
    line-height: 1.6;
}
.info-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    color: var(--text-muted);
}
.warning-box {
    background: rgba(245,184,91,.055);
    border: 1px solid rgba(245,184,91,.18);
    border-left: 3px solid var(--warning);
    color: #d7ae72;
}
.success-box {
    background: rgba(61,220,151,.055);
    border: 1px solid rgba(61,220,151,.18);
    border-left: 3px solid var(--success);
    color: #82d7b5;
}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button,
button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-soft) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    font-weight: 650 !important;
    transition: background .12s ease, border-color .12s ease, color .12s ease !important;
}
.stButton > button:hover,
button[kind="secondary"]:hover {
    background: var(--surface-3) !important;
    border-color: rgba(124,106,255,.42) !important;
    color: var(--text) !important;
    transform: none !important;
    box-shadow: none !important;
}
button[kind="primary"],
.stButton > button[kind="primary"],
[data-testid="stDownloadButton"] button {
    background: var(--primary) !important;
    border: 1px solid var(--primary) !important;
    color: #ffffff !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #6e5cf0 !important;
    border-color: #6e5cf0 !important;
}

/* ── Inputs/focus ring ──────────────────────────────── */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox > div > div,
.stMultiSelect > div > div,
[data-baseweb="select"] > div,
[data-baseweb="input"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    box-shadow: none !important;
}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus,
[data-baseweb="input"] input:focus,
[data-baseweb="select"] > div:focus-within {
    border-color: rgba(124,106,255,.72) !important;
    box-shadow: 0 0 0 2px rgba(124,106,255,.14) !important;
    outline: none !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: var(--text-faint) !important;
    font-size: .66rem !important;
    font-weight: 750 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
}

/* ── Tables and dataframes ──────────────────────────── */
.stDataFrame,
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: var(--surface) !important;
}
[data-testid="stDataFrame"] * {
    font-size: .82rem !important;
}

/* ── Tabs / expanders ───────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: .25rem !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-faint) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-size: .8rem !important;
    font-weight: 700 !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}
.streamlit-expanderHeader,
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}

/* ── AI chat ────────────────────────────────────────── */
.ai-message-user,
.ai-message-bot {
    border-radius: 10px;
    padding: .85rem .95rem;
    margin: .55rem 0;
    font-size: .84rem;
    line-height: 1.6;
}
.ai-message-user {
    background: var(--surface-2);
    border: 1px solid var(--border-soft);
    border-left: 3px solid var(--primary);
    text-align: right;
}
.ai-message-bot {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border-soft);
}

/* ── Quality banner ─────────────────────────────────── */
.quality-score-banner {
    display: flex;
    align-items: center;
    gap: 1.3rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 1.15rem;
}


/* ── Settings panels ────────────────────────────────── */
.settings-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}
.settings-card-title {
    color: var(--text);
    font-size: .9rem;
    font-weight: 750;
    margin-bottom: .35rem;
}

/* ── Login screen ───────────────────────────────────── */
.login-card {
    margin-top: 1rem;
    text-align: center;
}
.login-card .login-logo { font-size: 2.1rem; margin-bottom: .35rem; filter: grayscale(.15); }
.login-card h2 { margin: .2rem 0 .35rem 0; color: var(--text); font-size: 1.18rem; }
.login-card p { color: var(--text-muted); font-size: .82rem; line-height: 1.5; margin: 0; }

/* ── Responsive ─────────────────────────────────────── */
@media (max-width: 1100px) {
    .top-header { grid-template-columns: 1fr; }
    .dataset-head { border-left: 0; padding-left: 0; border-top: 1px solid var(--border); padding-top: .75rem; }
    .top-chips { justify-content: flex-start; }
    .metrics-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""


def inject_css() -> None:
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)
