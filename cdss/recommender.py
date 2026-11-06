"""Block 2c - Context-aware, explainable treatment recommendation engine (LLM + RAG).

Pipeline for a clinician query (current symptoms):
  1. map symptoms -> clinical problem categories
  2. for each category take candidate options from the knowledge base
  3. run every candidate through the Multi-Timeline Safety Engine against the patient's
     active + recently stopped medications, conditions, age and allergies
  4. rank: Preferred -> Use with caution -> Avoid, each with the reasons and evidence ids
  5. add patient-specific considerations (follow-up gaps, monitoring, recurrence)
  6. retrieve guideline + patient-record evidence (RAG) and, if an LLM is configured,
     generate a grounded narrative with citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import llm
from .ingestion import condition_codes
from .rag import ClinicalRAG
from .safety import SEVERITY_RANK, SafetyEngine
from .summarizer import follow_up_audit

CATEGORIES = {
    "pain": ["pain", "ache", "aching", "knee", "back", "joint", "sore", "arthritis"],
    "fever": ["fever", "febrile", "temperature", "chills"],
    "nausea_vomiting": ["nausea", "vomit", "retching"],
    "dyspepsia": ["acidity", "heartburn", "dyspepsia", "gastritis", "epigastric", "reflux"],
    "skin_infection": ["cellulitis", "redness", "swelling of leg", "leg swelling", "skin infection", "warm red", "abscess"],
    "anxiety_sleep": ["anxiety", "anxious", "sleep", "insomnia", "low mood", "worry"],
    "constipation": ["constipation", "hard stool", "not passing stool"],
}

OPTIONS = {
    "pain": [
        {"drug": "paracetamol", "strength": "650 mg", "frequency": "1-1-1", "days": 5,
         "label": "Paracetamol 650 mg up to three times daily", "ref": "G-PAIN-OPTIONS"},
        {"drug": "topical diclofenac", "strength": "", "frequency": "1-1-1", "days": 14,
         "label": "Topical diclofenac gel to the painful joint", "ref": "G-OA"},
        {"drug": "ibuprofen", "strength": "400 mg", "frequency": "1-1-1", "days": 5,
         "label": "Ibuprofen 400 mg three times daily after food (short course)", "ref": "G-NSAID"},
        {"drug": "tramadol", "strength": "50 mg", "frequency": "1-0-1", "days": 3,
         "label": "Tramadol 50 mg (short course, if pain uncontrolled)", "ref": "G-SERO"},
    ],
    "fever": [
        {"drug": "paracetamol", "strength": "650 mg", "frequency": "1-1-1", "days": 3,
         "label": "Paracetamol 650 mg up to three times daily", "ref": "G-FEVER-OPTIONS"},
        {"drug": "ibuprofen", "strength": "400 mg", "frequency": "1-1-1", "days": 3,
         "label": "Ibuprofen 400 mg three times daily after food", "ref": "G-NSAID"},
    ],
    "nausea_vomiting": [
        {"drug": "ondansetron", "strength": "4 mg", "frequency": "1-0-1", "days": 3,
         "label": "Ondansetron 4 mg twice daily", "ref": "G-GI-OPTIONS"},
        {"drug": "domperidone", "strength": "10 mg", "frequency": "1-1-1", "days": 3,
         "label": "Domperidone 10 mg up to three times daily (max 1 week)", "ref": "G-QT"},
    ],
    "dyspepsia": [
        {"drug": "pantoprazole", "strength": "40 mg", "frequency": "1-0-0", "days": 14,
         "label": "Pantoprazole 40 mg once daily before breakfast", "ref": "G-GI-OPTIONS"},
        {"drug": "domperidone", "strength": "10 mg", "frequency": "1-1-1", "days": 5,
         "label": "Domperidone 10 mg (prokinetic) short course", "ref": "G-QT"},
    ],
    "skin_infection": [
        {"drug": "flucloxacillin", "strength": "500 mg", "frequency": "1-1-1-1", "days": 7,
         "label": "Flucloxacillin 500 mg four times daily (NICE first line)", "ref": "G-CELL"},
        {"drug": "cephalexin", "strength": "500 mg", "frequency": "1-1-1", "days": 7,
         "label": "Cephalexin 500 mg three times daily", "ref": "G-CELL"},
        {"drug": "clarithromycin", "strength": "500 mg", "frequency": "1-0-1", "days": 7,
         "label": "Clarithromycin 500 mg twice daily (if penicillin allergy)", "ref": "G-CELL"},
        {"drug": "doxycycline", "strength": "100 mg", "frequency": "1-0-1", "days": 7,
         "label": "Doxycycline 100 mg twice daily (if penicillin allergy)", "ref": "G-CELL"},
        {"drug": "ciprofloxacin", "strength": "500 mg", "frequency": "1-0-1", "days": 7,
         "label": "Ciprofloxacin 500 mg twice daily (poor streptococcal cover - not first line)", "ref": "G-FQ"},
    ],
    "anxiety_sleep": [
        {"drug": "sertraline", "strength": "50 mg", "frequency": "1-0-0", "days": 28,
         "label": "Sertraline 50 mg once daily (SSRI first line)", "ref": "G-ANX"},
        {"drug": "clonazepam", "strength": "0.25 mg", "frequency": "0-0-1", "days": 7,
         "label": "Clonazepam 0.25 mg at night (crisis only, max 2-4 weeks)", "ref": "G-ANX"},
    ],
    "constipation": [
        {"drug": "lactulose", "strength": "15 ml", "frequency": "0-0-1", "days": 7,
         "label": "Lactulose 15 ml at night", "ref": "G-GI-OPTIONS"},
    ],
}

STATUS_ORDER = {"Preferred": 0, "Use with caution": 1, "Avoid": 2}


@dataclass
class OptionAssessment:
    category: str
    label: str
    status: str
    reasons: list = field(default_factory=list)
    ref: str = ""


@dataclass
class Recommendation:
    categories: list[str]
    options: list[OptionAssessment]
    considerations: list[tuple[str, str]]
    evidence: dict
    narrative: str | None = None


def detect_categories(text: str) -> list[str]:
    t = text.lower()
    return [c for c, words in CATEGORIES.items() if any(w in t for w in words)]


class RecommendationEngine:
    def __init__(self, safety: SafetyEngine | None = None, rag: ClinicalRAG | None = None):
        self.safety = safety or SafetyEngine()
        self.rag = rag or ClinicalRAG()

    def assess_option(self, patient: dict, meds: pd.DataFrame, opt: dict, on: date, allergies: list[str]) -> OptionAssessment:
        item = self.safety.make_item(opt["drug"], on, opt["strength"], opt["frequency"], opt["days"])
        res = self.safety.check_proposed(patient, meds, [item], on, allergies)
        alerts = res["alerts"]
        worst = max((SEVERITY_RANK[a.severity] for a in alerts), default=0)
        status = "Avoid" if worst == 3 else "Use with caution" if worst == 2 else "Preferred"
        reasons = [(a.severity, f"{a.title}: {a.effect} -> {a.management}", a.ref) for a in alerts]
        return OptionAssessment("", opt["label"], status, reasons, opt["ref"])

    def considerations(self, patient: dict, meds: pd.DataFrame, on: date, categories: list[str], active_alerts: list) -> list[tuple[str, str]]:
        conds = condition_codes(patient)
        out = []
        for a in active_alerts:
            if a.severity == "High":
                out.append((f"Resolve active HIGH alert first: {a.title} - {a.management}", a.ref))
        from .preprocessing import active_on
        act = active_on(meds[meds["patient_id"] == patient["patient_id"]], on)
        abx = act[act["classes"].apply(lambda c: "antibiotic" in c)] if not act.empty else act
        if not abx.empty:
            names = ", ".join(sorted({f"{r['display']} (since {r['start']})" for _, r in abx.iterrows()}))
            out.append((f"Already on systemic antibiotics: {names}. Assess response and adherence before adding or switching; "
                        "avoid stacking a second course.", "G-CELL"))
        if patient.get("allergy_status") == "not documented":
            out.append(("Allergy status is not documented in any source record - ask and record before prescribing.", "G-DUP"))
        if "diabetes" in conds and ("skin_infection" in categories or "fever" in categories or "cellulitis" in conds):
            out.append(("Check capillary glucose and HbA1c; infection and steroids worsen control, and fluoroquinolones cause dysglycaemia.", "G-DM-FOOT"))
        if "cellulitis" in conds:
            out.append(("Cellulitis documented at more than one visit - mark the margin, reassess response at 48-72 h, and look for venous disease / tinea as recurrence drivers.", "G-CELL"))
        if "varicose_veins" in conds:
            out.append(("Plan compression stockings only after infection resolves and arterial supply is confirmed (as already advised at the last visit).", "G-VEIN"))
        if "chronic_liver_disease" in conds:
            out.append(("Chronic liver disease: keep paracetamol at or below 2 g/day, avoid NSAIDs and sedatives.", "G-LIVER"))
        if "hypothyroidism" in conds:
            out.append(("Levothyroxine dose was changed recently - recheck TSH 6-8 weeks after the change; keep calcium/PPI away from the dose.", "G-THYROID"))
        if "post_op_spine" in conds and "pain" in categories:
            out.append(("After spine surgery: new leg weakness, numbness or bladder/bowel change needs urgent neurosurgical review.", "G-POSTOP-SPINE"))
        for f in follow_up_audit(patient):
            if f["status"] != "Visit recorded":
                out.append((f"Scheduled review on {f['due']} ({f['plan']}) has no matching visit in the records - confirm adherence.", "G-DUP"))
        return out

    def recommend(self, patient: dict, meds: pd.DataFrame, symptoms: str, on: date,
                  allergies: list[str] | None = None, active_alerts: list | None = None, use_llm: bool = True) -> Recommendation:
        allergies = allergies if allergies is not None else patient.get("allergies", [])
        cats = detect_categories(symptoms)
        options = []
        for c in cats:
            for opt in OPTIONS[c]:
                a = self.assess_option(patient, meds, opt, on, allergies)
                a.category = c
                options.append(a)
        options.sort(key=lambda o: (o.category, STATUS_ORDER[o.status]))
        cons = self.considerations(patient, meds, on, cats, active_alerts or [])

        conds = " ".join(c["name"] for c in patient.get("conditions", []))
        evidence = self.rag.retrieve(f"{symptoms} {conds}", patient)
        refs = {o.ref for o in options} | {r for _, r in cons}
        evidence["cited"] = [self.rag.guideline(r) for r in sorted(refs) if self.rag.guideline(r)]

        narrative = None
        if use_llm and llm.available():
            ctx = "\n".join(f"[{c.id}] {c.title}: {c.text}" for c in evidence["cited"][:8])
            pctx = "\n".join(f"[{c.id}] {c.text}" for c, _ in evidence["patient"])
            opts = "\n".join(f"- ({o.status}) {o.label}: " + "; ".join(r[1] for r in o.reasons[:3]) for o in options)
            prompt = (f"Current symptoms reported: {symptoms}\n\nPATIENT RECORD EXCERPTS:\n{pctx}\n\n"
                      f"RULE-ENGINE ASSESSMENT OF OPTIONS:\n{opts or 'none'}\n\nCONSIDERATIONS:\n" +
                      "\n".join(f"- {c} [{r}]" for c, r in cons) +
                      f"\n\nGUIDELINES:\n{ctx}\n\nWrite a short, explainable recommendation for the doctor "
                      "(max 200 words): the safest options and why, what to avoid and why, and what to monitor. Cite ids.")
            try:
                narrative = llm.complete(prompt)
            except Exception as exc:
                narrative = f"(LLM unavailable: {exc})"
        return Recommendation(cats, options, cons, evidence, narrative)
