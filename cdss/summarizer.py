"""Block 2b - Medical summarisation of the patient's longitudinal prescription history.

`summarise()` returns a deterministic, fully explainable summary built from the structured
record. If an LLM is configured, `llm_summary()` produces a narrative version grounded in
the same facts (with citations).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import pandas as pd

from . import llm
from .features import medication_changes, vitals_table
from .ingestion import age_on
from .preprocessing import active_on


def _parse_dates(text: str) -> list[date]:
    out = []
    for d, m, y in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", text):
        y = int(y) + (2000 if len(y) == 2 else 0)
        try:
            out.append(date(y, int(m), int(d)))
        except ValueError:
            pass
    return out


def follow_up_audit(patient: dict) -> list[dict]:
    """Compare explicitly scheduled review dates with recorded visits."""
    visits = [date.fromisoformat(e["date"]) for e in patient["encounters"]]
    rows = []
    for e in patient["encounters"]:
        for plan in e.get("plan", []):
            for due in _parse_dates(plan):
                seen = [v for v in visits if abs((v - due).days) <= 3]
                rows.append({"scheduled_in": e["encounter_id"], "due": due, "plan": plan,
                             "status": "Visit recorded" if seen else "No matching visit in records"})
    return rows


def summarise(patient: dict, meds: pd.DataFrame, alerts: list, on: date | None = None) -> dict:
    encs = patient["encounters"]
    on = on or date.fromisoformat(encs[-1]["date"])
    pm = meds[meds["patient_id"] == patient["patient_id"]]
    active = active_on(pm, on)
    changes = medication_changes(patient, meds)
    vit = vitals_table(patient)
    age = age_on(patient, on)

    lines = [f"{patient['name']} ({patient.get('sex')}, {age} y) - {len(encs)} documented encounters "
             f"from {encs[0]['date']} to {encs[-1]['date']} across {len({e['facility'] for e in encs})} facility(ies)."]
    if patient.get("conditions"):
        lines.append("Problem list: " + "; ".join(c["name"] for c in patient["conditions"]) + ".")
    lines.append("Allergies: " + (", ".join(patient.get("allergies", [])) or patient.get("allergy_status", "not documented")) + ".")

    episodes = []
    for e in encs:
        dx = ", ".join(e.get("diagnoses", [])) or ", ".join(e.get("complaints", [])) or "medication review"
        episodes.append(f"{e['date']} [{e['encounter_id']}] {e.get('department', '')}: {dx} "
                        f"- {len(e.get('medications', []))} items prescribed.")

    if not active.empty:
        current = [f"{r['display']} {r['strength']} {r['frequency']}".strip() for _, r in active.iterrows()]
    else:
        current = []

    key_changes = [f"{r['date']}: {r['change']} - {r['medication']} ({r['detail']})"
                   for _, r in changes.iterrows() if r["change"] in ("Switched", "Dose / schedule changed", "Stopped / not continued")]

    high = [a for a in alerts if a.severity == "High"]
    safety = [f"{a.severity}: {a.title} ({a.window_start}) [{a.ref}]" for a in alerts[:6]]

    trends = []
    if not vit.empty:
        for col, label in [("bp_systolic", "Systolic BP"), ("weight_kg", "Weight"), ("random_glucose_mg_dl", "Random glucose"), ("spo2", "SpO2")]:
            if col in vit and vit[col].notna().sum() >= 1:
                vals = vit[["date", col]].dropna()
                trends.append(f"{label}: " + " -> ".join(f"{v} ({d})" for d, v in zip(vals["date"], vals[col])))

    recurrences = []
    dx_dates: dict = {}
    for e in encs:
        for d in e.get("diagnoses", []):
            dx_dates.setdefault(d.lower(), []).append(e["date"])
    for d, ds in dx_dates.items():
        if len(ds) > 1:
            recurrences.append(f"'{d}' documented at {len(ds)} visits ({', '.join(ds)}) - persisting / not resolved between visits.")

    abx = pm[pm["classes"].apply(lambda c: "antibiotic" in c and "topical_antibiotic" not in c)]
    abx_note = None
    if not abx.empty:
        abx_note = f"{abx['encounter_id'].nunique()} systemic antibiotic prescription episode(s): " + \
                   "; ".join(sorted({f"{r['display']} ({r['start']})" for _, r in abx.iterrows()}))

    return {
        "headline": " ".join(lines),
        "episodes": episodes,
        "current_regimen_on": on,
        "current_regimen": current,
        "key_changes": key_changes,
        "vital_trends": trends,
        "recurrences": recurrences,
        "antibiotics": abx_note,
        "safety_highlights": safety,
        "n_high_alerts": len(high),
        "follow_up": follow_up_audit(patient),
    }


def summary_markdown(s: dict) -> str:
    md = [s["headline"], ""]
    md += ["**Care episodes**"] + [f"- {x}" for x in s["episodes"]]
    if s["current_regimen"]:
        md += ["", f"**Regimen active on {s['current_regimen_on']}**"] + [f"- {x}" for x in s["current_regimen"]]
    if s["key_changes"]:
        md += ["", "**Key medication changes**"] + [f"- {x}" for x in s["key_changes"]]
    if s["recurrences"]:
        md += ["", "**Disease progression / recurrence**"] + [f"- {x}" for x in s["recurrences"]]
    if s["antibiotics"]:
        md += ["", f"**Antibiotic exposure:** {s['antibiotics']}"]
    if s["vital_trends"]:
        md += ["", "**Vital-sign trends**"] + [f"- {x}" for x in s["vital_trends"]]
    if s["follow_up"]:
        md += ["", "**Follow-up adherence (scheduled vs recorded)**"] + \
              [f"- Due {f['due']} ({f['plan']}): {f['status']}" for f in s["follow_up"]]
    if s["safety_highlights"]:
        md += ["", "**Safety highlights**"] + [f"- {x}" for x in s["safety_highlights"]]
    return "\n".join(md)


def llm_summary(patient: dict, structured: dict, rag_context: str = "") -> str | None:
    if not llm.available():
        return None
    prompt = (
        "Write a concise clinical summary (max 180 words) of this patient's longitudinal prescription "
        "history for the treating doctor: main problems, how treatment changed over time, follow-up "
        "adherence, and the most important safety concerns. Cite encounter ids and guideline ids.\n\n"
        f"STRUCTURED FACTS:\n{summary_markdown(structured)}\n\nGUIDELINE EXCERPTS:\n{rag_context}"
    )
    try:
        return llm.complete(prompt)
    except Exception as exc:  # network / quota errors should never break the dashboard
        return f"(LLM unavailable: {exc})"
