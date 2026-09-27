# ════════════════════════════════════════════════════════
# DataBridge AI — Scheduled Monitoring Reports
# Stage 17 UI
# ════════════════════════════════════════════════════════
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.security import safe_html
from core.session import append_audit_event
from modules.model_governance import list_governance_families
from modules.monitoring_scheduler import (
    MonitoringSchedulerError,
    create_monitoring_job,
    delete_monitoring_job,
    install_windows_scheduled_task,
    list_monitoring_jobs,
    list_scheduled_reports,
    load_job_source,
    load_monitoring_job,
    load_scheduled_report,
    remove_windows_scheduled_task,
    run_scheduled_job,
    set_job_enabled,
    windows_task_installed,
)
from ui.cards import section_header


WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


def _schedule_text(job: Dict[str, Any]) -> str:
    schedule = dict(job.get("schedule") or {})
    cadence = str(schedule.get("cadence", "")).title()
    at = str(schedule.get("time_of_day", ""))
    if schedule.get("cadence") == "weekly":
        return f"{cadence} · {schedule.get('weekday', '')} · {at}"
    if schedule.get("cadence") == "monthly":
        return f"{cadence} · day {schedule.get('month_day', '')} · {at}"
    return f"{cadence} · {at}"


def _render_jobs() -> None:
    st.markdown("### Existing monitoring jobs")
    try:
        jobs = list_monitoring_jobs()
    except Exception as exc:
        st.error(f"Could not read scheduled jobs safely: {exc}")
        return
    if not jobs:
        st.info("No scheduled monitoring jobs yet.")
        return

    for job in jobs:
        job_id = str(job.get("job_id", ""))
        invalid = bool(job.get("invalid"))
        with st.expander(f"{'⚠️' if invalid else '🕒'} {job.get('name', job_id)}", expanded=False):
            if invalid:
                st.error(str(job.get("error", "Invalid authenticated job.")))
                continue
            family = dict(job.get("family") or {})
            source = dict(job.get("source") or {})
            last = dict(job.get("last_run") or {})
            task_state = "Installed" if os.name == "nt" and windows_task_installed(job_id) else "Not installed"
            st.markdown(
                f"<div class='info-box'><b>{safe_html(str(family.get('family_name','')))}</b> · "
                f"{safe_html(_schedule_text(job))} · Windows task: <b>{safe_html(task_state)}</b><br>"
                f"Source mode: <b>{safe_html(str(source.get('mode','')))}</b> · "
                f"Enabled: <b>{'Yes' if job.get('enabled', True) else 'No'}</b></div>",
                unsafe_allow_html=True,
            )
            if last:
                tone = "success" if last.get("status") == "Completed" else "warning"
                getattr(st, tone)(
                    f"Last run: {last.get('status','')} · {last.get('finished_at','')} · "
                    f"monitor={last.get('overall_status') or '—'} · drift={last.get('drift_score') if last.get('drift_score') is not None else '—'}"
                )
                if last.get("retraining_recommended"):
                    st.error("Retraining is recommended by the latest scheduled monitoring evidence. Promotion remains manual.")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("Run now", key=f"sched_run_{job_id}", width="stretch"):
                    try:
                        with st.spinner("Running signed monitoring job..."):
                            result = run_scheduled_job(job_id)
                        append_audit_event({
                            "event": "scheduled_monitoring_manual_run",
                            "action": "Scheduled monitoring job run manually",
                            "job_id": job_id,
                            "status": result.status,
                            "package_id": result.package_id,
                        })
                        st.success(f"Run result: {result.status}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Run blocked safely: {exc}")
            with c2:
                if os.name == "nt":
                    label = "Update Windows task" if task_state == "Installed" else "Install Windows task"
                    if st.button(label, key=f"sched_install_{job_id}", width="stretch"):
                        try:
                            install_windows_scheduled_task(job_id)
                            st.success("Windows Task Scheduler entry installed/updated.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Task installation failed: {exc}")
                else:
                    st.caption("Windows Task Scheduler is configured on Windows only.")
            with c3:
                enabled = bool(job.get("enabled", True))
                if st.button("Disable" if enabled else "Enable", key=f"sched_toggle_{job_id}", width="stretch"):
                    try:
                        set_job_enabled(job_id, not enabled)
                        if os.name == "nt" and not enabled:
                            install_windows_scheduled_task(job_id)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not update job: {exc}")
            with c4:
                confirm = st.checkbox("Confirm delete", key=f"sched_del_confirm_{job_id}")
                if st.button("Delete job", key=f"sched_delete_{job_id}", disabled=not confirm, width="stretch"):
                    try:
                        delete_monitoring_job(job_id, remove_task=True)
                        st.success("Scheduled job removed. Existing reports were retained.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed safely: {exc}")

            reports = list_scheduled_reports(job_id, limit=12)
            if reports:
                st.markdown("#### Recent reports")
                safe_dataframe(
                    pd.DataFrame([
                        {
                            "Created": r.get("created_at"), "Status": r.get("overall_status"),
                            "Drift": r.get("drift_score"), "Rows": r.get("rows"),
                            "Champion": r.get("package_id"), "Source": r.get("source_name"),
                            "Retrain": r.get("retraining_recommended"),
                        }
                        for r in reports
                    ]),
                    width="stretch", hide_index=True,
                )
                latest = reports[0]
                left, right = st.columns(2)
                with left:
                    if st.button("Load latest report into Monitoring", key=f"sched_open_monitor_{job_id}", width="stretch"):
                        try:
                            report = load_scheduled_report(latest["path"])
                            st.session_state.model_monitoring_report = report
                            st.session_state.model_monitoring_feature_table = pd.DataFrame(report.get("feature_drift", []) or [])
                            st.session_state.current_page = "model_monitoring"
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not open report: {exc}")
                with right:
                    if st.button("Prepare same snapshot for Safe Retraining", key=f"sched_open_retrain_{job_id}", width="stretch"):
                        try:
                            report = load_scheduled_report(latest["path"])
                            source = load_job_source(load_monitoring_job(job_id))
                            expected = str((report.get("scheduled_run") or {}).get("source_sha256", ""))
                            if not expected or source.sha256 != expected:
                                raise MonitoringSchedulerError(
                                    "The source file changed after this report. Run monitoring again before using it for retraining."
                                )
                            if str(report.get("package_id", "")) != str((load_monitoring_job(job_id).get("last_run") or {}).get("package_id", "")):
                                raise MonitoringSchedulerError("The latest report does not match the current job state.")
                            st.session_state.model_monitoring_input_df = source.frame.copy(deep=True)
                            st.session_state.model_monitoring_input_name = source.path.name
                            st.session_state.model_monitoring_report = report
                            st.session_state.model_monitoring_feature_table = pd.DataFrame(report.get("feature_drift", []) or [])
                            st.session_state.current_page = "retraining_workflow"
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Retraining handoff blocked safely: {exc}")


def _render_create() -> None:
    st.markdown("### Create / update a scheduled job")
    families = [f for f in list_governance_families() if f.get("champion_id")]
    if not families:
        st.warning("Promote a governed Champion first. Scheduled monitoring always follows the current Champion for a model family.")
        return
    labels = [f"{f['family_name']} · {f['task']} · {f['target']}" for f in families]
    family_label = st.selectbox("Governed model family", labels, key="sched_family")
    family = families[labels.index(family_label)]

    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Job name", value=f"{family['family_name']} monitoring", key="sched_name")
        source_mode_label = st.radio("Source", ["Exact local file", "Latest matching file in local folder"], key="sched_source_mode")
        source_mode = "file" if source_mode_label.startswith("Exact") else "latest_in_folder"
        source_path = st.text_input(
            "Local file path" if source_mode == "file" else "Local folder path",
            placeholder=r"D:\data\monitoring.csv" if source_mode == "file" else r"D:\data\monitoring",
            key="sched_source_path",
        )
        pattern = ""
        if source_mode == "latest_in_folder":
            pattern = st.text_input("File pattern", value="*.csv", help="Direct folder only; recursive patterns are blocked.", key="sched_pattern")
        actual_target = st.text_input("Actual outcome column (optional)", value=str(family.get("target", "")), key="sched_actual")
        require_actual = st.checkbox("Block run if the actual-outcome column is missing", value=False, key="sched_require_actual")
    with c2:
        cadence_label = st.selectbox("Cadence", ["Daily", "Weekly", "Monthly"], key="sched_cadence")
        cadence = cadence_label.lower()
        at = st.text_input("Local run time (HH:MM)", value="08:00", key="sched_time")
        weekday = st.selectbox("Weekday", WEEKDAYS, key="sched_weekday") if cadence == "weekly" else "MON"
        month_day = st.number_input("Day of month", min_value=1, max_value=28, value=1, step=1, key="sched_monthday") if cadence == "monthly" else 1
        retention = st.number_input("Reports to retain", min_value=1, max_value=365, value=60, step=1, key="sched_retention")
        skip_unchanged = st.checkbox("Skip unchanged source files", value=True, key="sched_skip_same")
        max_age = st.number_input("Maximum source age in hours (0 = disabled)", min_value=0, max_value=8760, value=0, step=1, key="sched_max_age")

    st.caption(
        "The Windows task stores only the job ID. Model family, source path, schedule policy, and run state live in an HMAC-authenticated per-user job file. "
        "The runner resolves the current Champion at execution time, so approved promotions are followed automatically."
    )
    if st.button("Save signed monitoring job", type="primary", width="stretch", key="sched_save"):
        try:
            job = create_monitoring_job(
                name=name, family_id=str(family["family_id"]), source_mode=source_mode,
                source_path=source_path, file_pattern=pattern, cadence=cadence, time_of_day=at,
                weekday=weekday, month_day=int(month_day), actual_target_column=actual_target,
                require_actual_target=require_actual, retention_reports=int(retention),
                skip_unchanged=skip_unchanged, max_source_age_hours=int(max_age), enabled=True,
            )
            append_audit_event({
                "event": "scheduled_monitoring_job_saved",
                "action": "Signed scheduled monitoring job saved",
                "job_id": job["job_id"],
                "family_id": family["family_id"],
            })
            st.success("Signed job saved. Install the Windows task below to run it while DataBridge AI is closed.")
            if os.name == "nt":
                install_windows_scheduled_task(job["job_id"])
                st.success("Windows Task Scheduler entry installed.")
            st.rerun()
        except Exception as exc:
            st.error(f"Scheduled job was blocked safely: {exc}")


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header("🕒", "Scheduled Monitoring Reports", "Stage 17 · unattended governed model health"),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box"><b>Runs even when the desktop UI is closed:</b> on Windows, each signed job can be installed in Task Scheduler. '
        'Jobs follow the current <b>Champion</b> of the selected model family, write aggregate drift/performance reports only, skip unchanged data if requested, '
        'and can recommend retraining — but they never retrain or promote a model automatically.</div>',
        unsafe_allow_html=True,
    )
    _render_jobs()
    st.markdown("---")
    _render_create()
