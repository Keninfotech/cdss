"""Block 1c - Sequential feature engineering.

Builds timeline features that capture disease progression, medication changes and
long-term treatment patterns for a single patient.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

import pandas as pd

from .ingestion import age_on
from .preprocessing import active_on

SYSTEMIC_ROUTES = {"PO", "IV", "IM", "SC"}


def _systemic(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["route"].isin(SYSTEMIC_ROUTES)] if not df.empty else df


def medication_changes(patient: dict, meds: pd.DataFrame) -> pd.DataFrame:
    """Compare each encounter's prescriptions with what was active just before it."""
    pm = meds[meds["patient_id"] == patient["patient_id"]]
    rows = []
    for e in patient["encounters"]:
        d = date.fromisoformat(e["date"])
        before = active_on(pm[pm["encounter_id"] != e["encounter_id"]], d - timedelta(days=1))
        now = pm[pm["encounter_id"] == e["encounter_id"]]
        before_by_comp = _by_component(before)
        now_by_comp = _by_component(now)

        added = set(now_by_comp) - set(before_by_comp)
        stopped = (set(before_by_comp) - set(now_by_comp)) if e.get("reconciles_chronic_meds") else set()
        stopped = {c for c in stopped if any(r["chronic"] for r in before_by_comp[c])}

        switched = set()
        for c_new in sorted(added):
            new_cls = set(now_by_comp[c_new][0]["classes"]) - {c_new}
            for c_old in sorted(stopped):
                old_cls = set(before_by_comp[c_old][0]["classes"]) - {c_old}
                key = new_cls & old_cls & SWITCH_CLASSES
                if key:
                    rows.append(_chg(e, "Switched", f"{_name(before_by_comp[c_old])} -> {_name(now_by_comp[c_new])}",
                                     f"{_regimen(before_by_comp[c_old])} -> {_regimen(now_by_comp[c_new])} (class: {', '.join(sorted(key))})"))
                    switched.update({c_new, c_old})
        for c in sorted(added - switched):
            rows.append(_chg(e, "Started", _name(now_by_comp[c]), _regimen(now_by_comp[c])))
        for c in sorted(stopped - switched):
            rows.append(_chg(e, "Stopped / not continued", _name(before_by_comp[c]), "omitted at medication reconciliation"))
        for c in sorted(set(now_by_comp) & set(before_by_comp)):
            a, b = _regimen(before_by_comp[c]), _regimen(now_by_comp[c])
            kind = "Dose / schedule changed" if a != b else "Continued / renewed"
            rows.append(_chg(e, kind, _name(now_by_comp[c]), f"{a} -> {b}" if a != b else b))
    out = pd.DataFrame(rows, columns=["date", "encounter_id", "change", "medication", "detail"])
    return out.drop_duplicates(subset=["encounter_id", "change", "medication"]).reset_index(drop=True)


SWITCH_CLASSES = {"insulin", "ccb", "ppi", "sulfonylurea", "statin", "arb", "beta_blocker", "nsaid", "ssri"}


def _by_component(df: pd.DataFrame) -> dict:
    out: dict = {}
    for _, r in df.iterrows():
        for c in r["components"]:
            out.setdefault(c, []).append(r)
    return out


def _name(items) -> str:
    return " / ".join(dict.fromkeys(r["display"] for r in items))


def _regimen(items) -> str:
    parts = []
    for r in items:
        txt = f"{r['strength']} {r['frequency']}".strip()
        if r["instructions"] and ("only" in r["instructions"] or "Mon" in r["instructions"]):
            txt += f" ({r['instructions'].split(';')[-1].strip()})"
        parts.append(txt)
    return " + ".join(dict.fromkeys(parts))



def _chg(e, kind, med, detail):
    return {"date": e["date"], "encounter_id": e["encounter_id"], "change": kind, "medication": med, "detail": detail}


def daily_med_counts(patient: dict, meds: pd.DataFrame, systemic_only: bool = True) -> pd.DataFrame:
    """Number of medications active on each day (interval sweep - O(items + days))."""
    pm = meds[meds["patient_id"] == patient["patient_id"]]
    if systemic_only:
        pm = _systemic(pm)
    if pm.empty:
        return pd.DataFrame(columns=["date", "active_meds"])
    start = min(pm["start"])
    ends = [d for d in pm["end"] if d is not None and not pd.isna(d)]
    last = max(ends + [max(pm["start"])])
    n = (last - start).days + 2
    diff = [0] * n
    for s_, e_ in zip(pm["start"], pm["end"]):
        diff[(s_ - start).days] += 1
        e_ = last if e_ is None or pd.isna(e_) else e_
        diff[(e_ - start).days + 1] -= 1
    out, run = [], 0
    for i in range(n - 1):
        run += diff[i]
        out.append({"date": start + timedelta(days=i), "active_meds": run})
    return pd.DataFrame(out)


def vitals_table(patient: dict) -> pd.DataFrame:
    rows = []
    for e in patient["encounters"]:
        v = dict(e.get("vitals") or {})
        if not v:
            continue
        if "bp_systolic" in v and "bp_diastolic" in v:
            v["map"] = round((v["bp_systolic"] + 2 * v["bp_diastolic"]) / 3, 1)
        h = v.get("height_cm") or patient.get("height_cm")
        if "bmi" not in v and h and v.get("weight_kg"):
            v["bmi"] = round(v["weight_kg"] / (h / 100) ** 2, 2)
        rows.append({"date": e["date"], "encounter_id": e["encounter_id"], **v})
    return pd.DataFrame(rows)


def patient_features(patient: dict, meds: pd.DataFrame, today: date | None = None) -> dict:
    today = today or date.today()
    pm = meds[meds["patient_id"] == patient["patient_id"]]
    encs = patient["encounters"]
    dates = [date.fromisoformat(e["date"]) for e in encs]
    dx = Counter(d.lower() for e in encs for d in e.get("diagnoses", []))
    counts = daily_med_counts(patient, meds)
    abx = pm[pm["classes"].apply(lambda c: "antibiotic" in c)] if not pm.empty else pm
    vit = vitals_table(patient)
    latest = vit.ffill().iloc[-1].to_dict() if not vit.empty else {}
    return {
        "patient_id": patient["patient_id"],
        "age": age_on(patient, today),
        "sex": patient.get("sex"),
        "n_encounters": len(encs),
        "first_visit": min(dates).isoformat() if dates else None,
        "last_visit": max(dates).isoformat() if dates else None,
        "follow_up_span_days": (max(dates) - min(dates)).days if dates else 0,
        "days_since_last_visit": (today - max(dates)).days if dates else None,
        "n_facilities": len({e.get("facility") for e in encs}),
        "n_prescribers": len({e.get("doctor") for e in encs if e.get("doctor")}),
        "n_medication_entries": len(pm),
        "n_unique_drugs": len({c for cs in pm["components"] for c in cs}) if not pm.empty else 0,
        "max_concurrent_systemic_meds": int(counts["active_meds"].max()) if not counts.empty else 0,
        "polypharmacy (>=5 concurrent)": bool(not counts.empty and counts["active_meds"].max() >= 5),
        "antibiotic_courses": int(abx["encounter_id"].nunique()) if not abx.empty else 0,
        "recurring_diagnoses": [d for d, n in dx.items() if n > 1],
        "n_chronic_conditions": len(patient.get("conditions", [])),
        "latest_bmi": latest.get("bmi"),
        "latest_map": latest.get("map"),
        "latest_glucose": latest.get("random_glucose_mg_dl"),
        "latest_spo2": latest.get("spo2"),
    }


def all_patient_features(patients: list[dict], meds: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    return pd.DataFrame([patient_features(p, meds, today) for p in patients])
