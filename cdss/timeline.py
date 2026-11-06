"""Block 4 - Historical timeline & visualisation engine (Plotly figures for the dashboard)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SEV_COLOURS = {"High": "#d62728", "Moderate": "#ff7f0e", "Low": "#bcbd22"}


def medication_gantt(patient: dict, meds: pd.DataFrame, today: date | None = None, alerts: list | None = None) -> go.Figure:
    pm = meds[meds["patient_id"] == patient["patient_id"]].copy()
    if pm.empty:
        return go.Figure()
    last_visit = date.fromisoformat(patient["encounters"][-1]["date"])
    horizon = max(today or last_visit, last_visit) + timedelta(days=30)
    pm["finish"] = pm["end"].apply(lambda d: horizon if d is None or pd.isna(d) else d) + timedelta(days=1)
    pm["status"] = pm.apply(lambda r: "Ongoing (no end date)" if r["end"] is None or pd.isna(r["end"])
                            else "Single dose" if r["stat"] else "Course", axis=1)
    pm["label"] = pm["display"].str.replace(r"\s*\(.*\)", "", regex=True)
    pm["hover"] = pm.apply(lambda r: f"{r['raw']}<br>{r['department']} - {r['facility']}<br>"
                                     f"{r['start']} -> {r['end'] or 'ongoing'}<br>source: {r['source_image']}", axis=1)
    fig = px.timeline(pm, x_start="start", x_end="finish", y="label", color="department",
                      hover_name="label", hover_data={"hover": True, "label": False, "department": False,
                                                      "start": False, "finish": False},
                      pattern_shape="status", pattern_shape_map={"Ongoing (no end date)": "/", "Course": "", "Single dose": "."})
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(height=max(320, 26 * pm["label"].nunique() + 140), legend_title_text="Department / pattern",
                      margin=dict(l=10, r=10, t=30, b=10))
    for e in patient["encounters"]:
        fig.add_vline(x=pd.Timestamp(e["date"]).value / 1e6, line_dash="dot", line_color="grey", opacity=0.5)
    if alerts:
        for a in alerts:
            if a.severity != "High" or a.window_start is None:
                continue
            end = (a.window_end or horizon) + timedelta(days=1)
            fig.add_vrect(x0=a.window_start, x1=end, fillcolor=SEV_COLOURS["High"], opacity=0.08, line_width=0)
    return fig


def encounter_timeline(patient: dict) -> go.Figure:
    rows = []
    for e in patient["encounters"]:
        rows.append({"date": pd.Timestamp(e["date"]), "department": e.get("department", ""),
                     "facility": e.get("facility", ""), "n_meds": len(e.get("medications", [])),
                     "what": ", ".join(e.get("diagnoses", []) or e.get("complaints", [])) or e.get("type", "")})
    df = pd.DataFrame(rows)
    fig = px.scatter(df, x="date", y="facility", size=df["n_meds"].clip(lower=1) + 4, color="department",
                     hover_data={"what": True, "n_meds": True})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=None, xaxis_title=None,
                      legend=dict(orientation="h", yanchor="top", y=-0.25, x=0, title=None))
    fig.update_yaxes(tickmode="array", tickvals=df["facility"].unique(),
                     ticktext=[f[:28] + ("..." if len(f) > 28 else "") for f in df["facility"].unique()])
    return fig


def vitals_figure(vitals: pd.DataFrame) -> go.Figure | None:
    if vitals.empty:
        return None
    cols = [c for c in ["bp_systolic", "bp_diastolic", "pulse", "spo2", "random_glucose_mg_dl", "weight_kg"] if c in vitals]
    long = vitals.melt(id_vars=["date"], value_vars=cols, var_name="measure", value_name="value").dropna()
    fig = px.line(long, x="date", y="value", color="measure", markers=True, facet_col="measure", facet_col_wrap=3)
    fig.update_yaxes(matches=None, title=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def polypharmacy_figure(counts: pd.DataFrame) -> go.Figure | None:
    if counts.empty:
        return None
    fig = px.area(counts, x="date", y="active_meds")
    fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="polypharmacy (5+)")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="systemic meds active")
    return fig
