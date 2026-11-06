"""Block 3 - Dynamic, multi-timeline safety & risk checks.

Checks every pair of medication instances whose active periods overlap - across visits,
departments and hospitals - for:
  * drug-drug interactions (class / ingredient rules + 3-drug 'triple whammy' rules)
  * therapeutic duplication (same ingredient in two products, two drugs of one class)
  * patient-specific contraindications (conditions, age / Beers criteria)
  * allergy and cross-reactivity conflicts
  * cumulative daily-dose ceilings (paracetamol)
and scores each alert Low / Moderate / High.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from itertools import combinations

import pandas as pd

from .ingestion import age_on, condition_codes, load_rules
from .preprocessing import DrugNormaliser, active_on, parse_frequency, parse_strength

SEVERITY_RANK = {"High": 3, "Moderate": 2, "Low": 1}
COMBINATION_MG = {"ultracet": {"tramadol": 37.5, "paracetamol": 325}}
PARACETAMOL_MAX_MG = 4000
PARACETAMOL_LIVER_MAX_MG = 2000
RECENT_WINDOW_DAYS = 14


@dataclass
class Alert:
    rule_id: str
    severity: str
    kind: str
    title: str
    drugs: list[str]
    effect: str
    management: str
    ref: str
    window_start: date | None = None
    window_end: date | None = None
    encounters: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    involves_proposed: bool = False

    def key(self):
        return (self.rule_id, tuple(sorted(self.drugs)))

    def as_dict(self):
        return asdict(self)


def _overlap(a, b):
    s = max(a["start"], b["start"])
    ends = [x for x in (a["end"], b["end"]) if x is not None and not pd.isna(x)]
    e = min(ends) if ends else None
    if e is not None and e < s:
        return None
    return s, e


class SafetyEngine:
    def __init__(self, normaliser: DrugNormaliser | None = None):
        self.rules = load_rules()
        self.norm = normaliser or DrugNormaliser()

    # ------------------------------------------------------------------ pair rules
    def _pair_alerts(self, a: dict, b: dict) -> list[Alert]:
        ta, tb = set(a["classes"]), set(b["classes"])
        found: list[Alert] = []
        generic: list[Alert] = []
        for r in self.rules["interactions"]:
            hit = (r["a"] in ta and r["b"] in tb) or (r["a"] in tb and r["b"] in ta)
            if not hit:
                continue
            al = Alert(r["id"], r["severity"], "Drug-drug interaction", f"{a['display']} + {b['display']}",
                       [a["display"], b["display"]], r["effect"], r["management"], r["ref"])
            (generic if r.get("generic") else found).append(al)
        alerts = found or generic
        # duplication
        same_product_split = a["encounter_id"] == b["encounter_id"] and a["drug_key"] == b["drug_key"]
        if not same_product_split:
            shared = set(a["components"]) & set(b["components"])
            shared -= {"emollient", "skincare", "sunscreen", "nutraceutical"}
            if shared:
                comp = ", ".join(sorted(shared))
                sev = "High" if shared & {"metformin", "glimepiride", "gliclazide", "cholecalciferol"} else "Moderate"
                alerts.append(Alert("DUP-ING", sev, "Therapeutic duplication", f"Same ingredient ({comp}) in two products",
                                    [a["display"], b["display"]],
                                    f"{comp.capitalize()} is present in both products - risk of unintended double dosing.",
                                    "Reconcile: keep one product and stop the other, or confirm the total dose is intended.",
                                    self.rules["duplication_ref"]))
            else:
                for cls, label in self.rules["duplication_classes"].items():
                    if cls in ta and cls in tb:
                        alerts.append(Alert("DUP-CLS", "Moderate", "Therapeutic duplication", label,
                                            [a["display"], b["display"]],
                                            f"{label} prescribed concurrently ({a['display']}, {b['display']}).",
                                            "Confirm both are intended; otherwise stop one.", self.rules["duplication_ref"]))
        return alerts

    def _triple_alerts(self, items: list[dict]) -> list[Alert]:
        out = []
        for r in self.rules["triple_rules"]:
            picks = []
            for cls in r["classes"]:
                cand = [i for i in items if cls in i["classes"] and i not in picks]
                if not cand:
                    break
                picks.append(cand[0])
            else:
                out.append(Alert(r["id"], r["severity"], "Drug-drug interaction (3-drug)",
                                 " + ".join(i["display"] for i in picks), [i["display"] for i in picks],
                                 r["effect"], r["management"], r["ref"]))
        return out

    # ------------------------------------------------------------------ patient rules
    def _patient_alerts(self, item: dict, patient: dict, on: date) -> list[Alert]:
        out = []
        tags = set(item["classes"])
        conds = condition_codes(patient)
        age = age_on(patient, on)
        for r in self.rules["patient_rules"]:
            if r["class"] not in tags:
                continue
            if "condition" in r and r["condition"] in conds:
                cname = next(c["name"] for c in patient["conditions"] if c["code"] == r["condition"])
                out.append(Alert(r["id"], r["severity"], "Condition contraindication / caution",
                                 f"{item['display']} with {cname}", [item["display"]], r["effect"], r["management"], r["ref"]))
            if "min_age" in r and age is not None and age >= r["min_age"]:
                out.append(Alert(r["id"], r["severity"], "Age-related caution (Beers)",
                                 f"{item['display']} at age {age}", [item["display"]], r["effect"], r["management"], r["ref"]))
        return out

    def _allergy_alerts(self, item: dict, allergies: list[str]) -> list[Alert]:
        out = []
        tags = set(item["classes"])
        xr = self.rules["allergy_cross_reactivity"]
        for allergy in allergies:
            a = allergy.strip().lower()
            if not a:
                continue
            _, entry = self.norm.lookup(a)
            direct = set(entry["components"]) if entry else {a}
            if direct & set(item["components"]) or a in tags:
                out.append(Alert("ALG", "High", "Allergy conflict", f"{item['display']} - documented allergy: {allergy}",
                                 [item["display"]], f"Patient has a documented allergy to {allergy}.",
                                 "Do not administer; choose an alternative class.", "G-DUP"))
                continue
            classes = set(xr.get(a, []))
            if entry:
                for c in entry["components"]:
                    classes |= set(self.rules["components"].get(c, [])) & set(xr.keys())
            if a in ("penicillin", "amoxicillin", "amoxycillin", "flucloxacillin") and "cephalosporin" in tags:
                out.append(Alert("ALG-X", "Moderate", "Allergy cross-reactivity", f"{item['display']} - penicillin allergy",
                                 [item["display"]], self.rules["allergy_cross_notes"]["penicillin->cephalosporin"],
                                 "Clarify the nature of the penicillin reaction before giving a cephalosporin.", "G-CELL"))
            elif classes & tags:
                out.append(Alert("ALG", "High", "Allergy conflict", f"{item['display']} - allergy class: {allergy}",
                                 [item["display"]], f"{item['display']} belongs to a class the patient is allergic to ({allergy}).",
                                 "Do not administer; choose an alternative class.", "G-DUP"))
        return out

    # ------------------------------------------------------------------ dose ceilings
    @staticmethod
    def _component_mg_per_day(item: dict, component: str) -> float:
        if component not in item["components"]:
            return 0.0
        mg = COMBINATION_MG.get(item.get("drug_key") or "", {}).get(component)
        if mg is None:
            mg = item.get("strength_value") if item.get("strength_unit") == "mg" else None
        if mg is None:
            return 0.0
        per_day = 3 if item.get("prn") else (item.get("amount_per_day") or 0)
        return float(mg) * float(per_day)

    def _dose_alerts(self, items: list[dict], patient: dict) -> list[Alert]:
        total = sum(self._component_mg_per_day(i, "paracetamol") for i in items)
        if total <= 0:
            return []
        liver = "chronic_liver_disease" in condition_codes(patient)
        ceiling = PARACETAMOL_LIVER_MAX_MG if liver else PARACETAMOL_MAX_MG
        drugs = [i["display"] for i in items if "paracetamol" in i["components"]]
        if total > ceiling:
            return [Alert("DOSE-PCM", "High", "Cumulative dose", f"Paracetamol {total:.0f} mg/day exceeds {ceiling} mg/day",
                          drugs, f"Total paracetamol from all concurrent products is about {total:.0f} mg/day.",
                          "Reduce the dose / remove the duplicate paracetamol source.", "G-LIVER" if liver else "G-PAIN-OPTIONS")]
        if len(drugs) > 1:
            return [Alert("DOSE-PCM", "Low", "Cumulative dose", f"Paracetamol from {len(drugs)} products = {total:.0f} mg/day",
                          drugs, f"Within the {ceiling} mg/day ceiling, but split across several products.",
                          "Tell the patient not to add over-the-counter paracetamol.", "G-PAIN-OPTIONS")]
        return []

    # ------------------------------------------------------------------ public API
    def scan_history(self, patient: dict, meds: pd.DataFrame, allergies: list[str] | None = None) -> list[Alert]:
        """Every alert that existed at any point of the patient's timeline."""
        pm = meds[meds["patient_id"] == patient["patient_id"]].to_dict("records")
        alerts: dict = {}
        allergies = allergies if allergies is not None else patient.get("allergies", [])

        def add(al: Alert, items: list[dict], window):
            al.window_start, al.window_end = window
            al.encounters = sorted({i["encounter_id"] for i in items})
            al.sources = sorted({i["source_image"] for i in items if i["source_image"]})
            k = al.key()
            if k not in alerts or SEVERITY_RANK[al.severity] > SEVERITY_RANK[alerts[k].severity]:
                alerts[k] = al

        for a, b in combinations(pm, 2):
            if a["item"] == b["item"]:
                continue
            win = _overlap(a, b)
            if not win:
                continue
            for al in self._pair_alerts(a, b):
                add(al, [a, b], win)
        # triple rules + dose ceilings evaluated on each distinct start date
        for d in sorted({i["start"] for i in pm}):
            active = [i for i in pm if i["start"] <= d and (i["end"] is None or pd.isna(i["end"]) or i["end"] >= d)]
            for al in self._triple_alerts(active) + self._dose_alerts(active, patient):
                involved = [i for i in active if i["display"] in al.drugs]
                ends = [i["end"] for i in involved if i["end"] is not None and not pd.isna(i["end"])]
                add(al, involved, (d, min(ends) if ends else None))
        for i in pm:
            for al in self._patient_alerts(i, patient, i["start"]) + self._allergy_alerts(i, allergies):
                add(al, [i], (i["start"], i["end"]))
        return sort_alerts(list(alerts.values()))

    def make_item(self, name: str, on: date, strength: str = "", frequency: str = "1-0-0", duration_days: int = 7) -> dict:
        key, entry = self.norm.lookup(name)
        comps = entry["components"] if entry else []
        f = parse_frequency(frequency)
        s_val, s_unit = parse_strength(strength)
        return {
            "item": f"PROPOSED-{name}", "encounter_id": "PROPOSED", "drug_key": key,
            "display": (entry["display"] if entry else name) + " [proposed]",
            "components": comps, "classes": sorted(self.norm.classes_for(comps)),
            "strength": strength, "strength_value": s_val, "strength_unit": s_unit,
            "frequency": frequency, "amount_per_day": f["per_day"], "prn": f["prn"],
            "start": on, "end": on + timedelta(days=max(duration_days, 1) - 1),
            "source_image": "", "recognised": entry is not None,
        }

    def check_proposed(self, patient: dict, meds: pd.DataFrame, proposed: list[dict], on: date,
                       allergies: list[str] | None = None) -> dict:
        """Check a new prescription against everything active (or stopped < 14 days ago) on `on`."""
        pm = meds[meds["patient_id"] == patient["patient_id"]]
        current = active_on(pm, on).to_dict("records")
        recent = [r for r in pm.to_dict("records")
                  if r["end"] is not None and not pd.isna(r["end"]) and on - timedelta(days=RECENT_WINDOW_DAYS) <= r["end"] < on]
        allergies = allergies if allergies is not None else patient.get("allergies", [])
        context = current + recent
        alerts: list[Alert] = []
        for p in proposed:
            for c in context:
                for al in self._pair_alerts(p, c):
                    al.involves_proposed = True
                    al.encounters = [c["encounter_id"]]
                    al.sources = [c["source_image"]] if c["source_image"] else []
                    if c in recent:
                        al.management = f"(Recently stopped on {c['end']}) " + al.management
                    alerts.append(al)
            for al in self._patient_alerts(p, patient, on) + self._allergy_alerts(p, allergies):
                al.involves_proposed = True
                alerts.append(al)
        for a, b in combinations(proposed, 2):
            for al in self._pair_alerts(a, b):
                al.involves_proposed = True
                alerts.append(al)
        for al in self._triple_alerts(current + proposed) + self._dose_alerts(current + proposed, patient):
            if any("[proposed]" in d for d in al.drugs):
                al.involves_proposed = True
                alerts.append(al)
        dedup = {}
        for al in alerts:
            dedup.setdefault(al.key(), al)
        return {"current": current, "recent": recent, "alerts": sort_alerts(list(dedup.values())),
                "unrecognised": [p["display"] for p in proposed if not p.get("recognised")]}

    def active_alerts(self, patient: dict, meds: pd.DataFrame, on: date, allergies: list[str] | None = None) -> list[Alert]:
        hist = self.scan_history(patient, meds, allergies)
        return [a for a in hist if a.window_start <= on and (a.window_end is None or a.window_end >= on)]


def sort_alerts(alerts: list[Alert]) -> list[Alert]:
    return sorted(alerts, key=lambda a: (-SEVERITY_RANK[a.severity], a.window_start or date.min, a.title))


def risk_level(alerts: list[Alert]) -> tuple[str, int]:
    score = sum(SEVERITY_RANK[a.severity] for a in alerts)
    if any(a.severity == "High" for a in alerts):
        return "High", score
    if any(a.severity == "Moderate" for a in alerts):
        return "Moderate", score
    return ("Low", score) if alerts else ("None", 0)


def alerts_frame(alerts: list[Alert]) -> pd.DataFrame:
    return pd.DataFrame([{
        "severity": a.severity, "type": a.kind, "alert": a.title, "effect": a.effect,
        "management": a.management, "evidence": a.ref,
        "from": a.window_start, "to": a.window_end, "encounters": ", ".join(a.encounters),
    } for a in alerts])
