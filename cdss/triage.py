"""Emergency patient triage: EMERGENCY (transfer) / URGENT (same-day doctor) / ROUTINE (manage locally).

Explainable, rule-based scoring combining
  * NEWS2-style vital-sign points (RCP 2017, without respiratory rate if not available)
  * red-flag symptoms (from the St. John's discharge template + WHO ETAT emergency signs)
  * patient context (age, diabetes, liver disease, recent surgery, recent antibiotics,
    anticoagulant use, documented recurrence)
Every point added is reported so the clinician can see why a level was chosen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from .ingestion import age_on, condition_codes
from .preprocessing import active_on

EMERGENCY_SIGNS = {
    "convulsion": ["convulsion", "seizure", "fits", "fitting"],
    "unresponsive / altered consciousness": ["unresponsive", "unconscious", "drowsy", "confus", "altered sensorium", "not waking"],
    "severe breathlessness": ["severe breathless", "cannot breathe", "can't breathe", "gasping", "respiratory distress"],
    "chest pain": ["chest pain", "chest tightness"],
    "active bleeding": ["bleeding", "vomiting blood", "blood in vomit", "black stool", "malena", "melena", "haematemesis", "hematemesis"],
    "loss of vision": ["loss of vision", "sudden blindness", "cannot see"],
    "no urine output": ["no urine", "absence of urine", "not passed urine", "anuria"],
    "stroke signs": ["slurred speech", "facial droop", "weakness of one side", "hemiparesis"],
    "cauda equina signs": ["bladder control", "bowel control", "saddle numbness", "urinary retention"],
}
URGENT_SIGNS = {
    "fever": ["fever", "febrile", "chills", "rigors"],
    "vomiting / unable to drink": ["vomit", "unable to drink", "inability to drink", "cannot drink"],
    "diarrhoea": ["diarrh", "loose stool"],
    "breathlessness": ["breathless", "shortness of breath", "dyspnoea", "dyspnea"],
    "palpitations": ["palpitation"],
    "jaundice": ["jaundice", "yellow eyes", "yellow skin"],
    "severe pain": ["severe pain", "unbearable pain", "excruciating"],
    "spreading redness / swelling": ["spreading", "redness increasing", "swelling increasing", "red streak"],
    "wound discharge": ["discharge from wound", "pus", "wound discharge", "discharge from surgical site", "wound oozing"],
    "rash": ["rash", "hives", "urticaria"],
    "joint swelling": ["joint swelling", "swollen joint"],
    "behavioural change / self-harm": ["self-harm", "self harm", "suicid", "agitation", "violent"],
    "hypoglycaemia symptoms": ["sweating and shaking", "hypoglyc", "low sugar", "shaky", "tremor"],
}
ROUTINE_HINTS = ["mild", "follow up", "follow-up", "refill", "review", "itch", "acne", "dry skin"]


@dataclass
class TriageResult:
    level: str
    score: int
    reasons: list[tuple[str, int]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    transfer_recommended: bool = False


def _find(text: str, lexicon: dict) -> list[str]:
    t = text.lower()
    return [label for label, words in lexicon.items() if any(w in t for w in words)]


def news2_points(v: dict) -> list[tuple[str, int]]:
    pts = []
    rr = v.get("resp_rate")
    if rr is not None:
        pts.append((f"Respiratory rate {rr}/min", 3 if rr <= 8 or rr >= 25 else 2 if rr >= 21 else 1 if rr <= 11 else 0))
    s = v.get("spo2")
    if s is not None:
        pts.append((f"SpO2 {s}%", 3 if s <= 91 else 2 if s <= 93 else 1 if s <= 95 else 0))
    sbp = v.get("bp_systolic")
    if sbp is not None:
        pts.append((f"Systolic BP {sbp} mmHg", 3 if sbp <= 90 or sbp >= 220 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0))
    hr = v.get("pulse")
    if hr is not None:
        pts.append((f"Pulse {hr}/min", 3 if hr <= 40 or hr >= 131 else 2 if hr >= 111 else 1 if hr <= 50 or hr >= 91 else 0))
    t = v.get("temperature_c")
    if t is None and v.get("temperature_f") is not None:
        t = round((v["temperature_f"] - 32) * 5 / 9, 1)
    if t is not None:
        pts.append((f"Temperature {t} C", 3 if t <= 35.0 else 2 if t >= 39.1 else 1 if t <= 36.0 or t >= 38.1 else 0))
    if v.get("new_confusion"):
        pts.append(("New confusion (ACVPU)", 3))
    return [p for p in pts if p[1] > 0]


def triage(symptoms: str, vitals: dict | None, patient: dict | None = None, meds: pd.DataFrame | None = None,
           on: date | None = None) -> TriageResult:
    vitals = vitals or {}
    on = on or date.today()
    reasons: list[tuple[str, int]] = []
    level = "ROUTINE"

    vital_pts = news2_points(vitals)
    news = sum(p for _, p in vital_pts)
    reasons += [(f"NEWS2: {label}", p) for label, p in vital_pts]
    if news >= 7:
        level = "EMERGENCY"
    elif news >= 5 or any(p == 3 for _, p in vital_pts):
        level = "URGENT"

    g = vitals.get("glucose_mg_dl")
    if g is not None:
        if g < 54:
            reasons.append((f"Severe hypoglycaemia (glucose {g} mg/dL)", 7)); level = "EMERGENCY"
        elif g < 70:
            reasons.append((f"Hypoglycaemia (glucose {g} mg/dL)", 3)); level = _max(level, "URGENT")
        elif g > 400:
            reasons.append((f"Severe hyperglycaemia (glucose {g} mg/dL)", 5)); level = _max(level, "URGENT")
        elif g > 300:
            reasons.append((f"Hyperglycaemia (glucose {g} mg/dL)", 2)); level = _max(level, "URGENT")

    em = _find(symptoms, EMERGENCY_SIGNS)
    for s in em:
        reasons.append((f"Emergency sign: {s}", 7))
    if em:
        level = "EMERGENCY"
    ur = _find(symptoms, URGENT_SIGNS)
    for s in ur:
        reasons.append((f"Red-flag symptom: {s}", 3))
    if ur:
        level = _max(level, "URGENT")

    # ---------------------------------------------------------------- patient context modifiers
    if patient:
        age = age_on(patient, on)
        conds = condition_codes(patient)
        mod = []
        if age is not None and age >= 65:
            mod.append((f"Age {age} (>= 65)", 1))
        if "diabetes" in conds:
            mod.append(("Diabetes - infections and hypo/hyperglycaemia progress faster", 1))
        if "chronic_liver_disease" in conds:
            mod.append(("Chronic liver disease", 1))
        if "cellulitis" in conds and ("fever" in ur or "spreading redness / swelling" in ur):
            mod.append(("Known cellulitis with fever / spreading - possible treatment failure [G-CELL]", 2))
        recent_surgery = [e for e in patient["encounters"] if e.get("type") == "discharge_summary"
                          and 0 <= (on - date.fromisoformat(e["date"])).days <= 30]
        if recent_surgery and (ur or em):
            mod.append(("Within 30 days of hospital discharge / surgery", 2))
        if meds is not None:
            act = active_on(meds[meds["patient_id"] == patient["patient_id"]], on)
            if not act.empty:
                if act["classes"].apply(lambda c: "anticoagulant" in c or "antiplatelet" in c).any() and "active bleeding" in em:
                    mod.append(("On anticoagulant/antiplatelet with bleeding", 3))
                if act["classes"].apply(lambda c: "hypoglycaemic_agent" in c).any() and (
                        "hypoglycaemia symptoms" in ur or "unresponsive / altered consciousness" in em):
                    mod.append(("On insulin / sulfonylurea - consider hypoglycaemia; check glucose now", 3))
        reasons += mod
        if level == "ROUTINE" and sum(p for _, p in mod) >= 3 and (ur or news >= 3):
            level = "URGENT"

    if level == "ROUTINE" and news >= 3:
        level = "URGENT" if news >= 5 else "ROUTINE"

    score = sum(p for _, p in reasons)
    actions = {
        "EMERGENCY": ["Stabilise (ABC), start oxygen / IV access as indicated.",
                      "Arrange immediate transfer to a higher centre with a referral note and medication list.",
                      "Check capillary glucose immediately."],
        "URGENT": ["Same-day doctor review (in person or teleconsultation).",
                   "Repeat vitals within 1 hour; escalate if any parameter worsens.",
                   "Review current medications for causes (see Safety alerts)."],
        "ROUTINE": ["Manage locally / scheduled follow-up - transfer not needed.",
                    "Give safety-net advice: return immediately for any red-flag symptom."],
    }[level]
    if not reasons and level == "ROUTINE":
        reasons.append(("No red-flag symptoms or abnormal vitals entered", 0))
    return TriageResult(level, score, reasons, actions, transfer_recommended=(level == "EMERGENCY"))


def _max(a: str, b: str) -> str:
    order = ["ROUTINE", "URGENT", "EMERGENCY"]
    return a if order.index(a) >= order.index(b) else b
