"""Block 1b - Data preprocessing & cleaning.

* normalises brand / free-text drug names to generic components and drug classes
* parses Indian-style dosing schedules ("1-0-1", "1/2-0-0", "18U-0-18U", HS, SOS, stat)
* parses durations ("x 10 days", "2 weeks", "6 months", "continue")
* derives start / end dates for every medication instance
* reports data-quality issues (illegible entries, uncertain dates, inconsistent ages)
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from difflib import get_close_matches

import pandas as pd

from .ingestion import age_on, load_drug_dictionary, load_rules

_FRACTIONS = {"½": 0.5, "1/2": 0.5, "¼": 0.25, "1/4": 0.25, "3/4": 0.75}
_FREQ_WORDS = {"od": 1, "once daily": 1, "bd": 2, "bid": 2, "tds": 3, "tid": 3, "qid": 4,
               "hs": 1, "night": 1, "every day": 1, "twice daily": 2}
DEFAULT_COURSE_DAYS = 30


# --------------------------------------------------------------------------- drug names
class DrugNormaliser:
    def __init__(self):
        self.brands = load_drug_dictionary()
        rules = load_rules()
        self.component_classes = rules["components"]

    def lookup(self, name: str) -> tuple[str | None, dict | None]:
        key = (name or "").strip().lower()
        if not key or key == "unknown":
            return None, None
        if key in self.brands:
            return key, self.brands[key]
        # strip dosage form words and strengths: "tab ciplox 500" -> "ciplox"
        stripped = re.sub(r"\b(tab|tablet|cap|capsule|inj|injection|syp|syrup|oint|gel|cream|lotion|t|mg|mcg|ml)\b\.?", " ", key)
        stripped = re.sub(r"[\d.]+", " ", stripped)
        stripped = " ".join(stripped.split())
        if stripped in self.brands:
            return stripped, self.brands[stripped]
        close = get_close_matches(stripped or key, self.brands.keys(), n=1, cutoff=0.82)
        if close:
            return close[0], self.brands[close[0]]
        return None, None

    def classes_for(self, components: list[str]) -> set[str]:
        tags: set[str] = set()
        for c in components:
            tags.add(c)  # a component name is itself a matchable tag (e.g. "tramadol")
            tags.update(self.component_classes.get(c, []))
        return tags


# --------------------------------------------------------------------------- schedules
def _num(token: str) -> float:
    token = token.strip().lower().rstrip("u").strip()
    if token in _FRACTIONS:
        return _FRACTIONS[token]
    try:
        return float(token)
    except ValueError:
        return 0.0


def parse_frequency(freq: str) -> dict:
    """Return {'per_day': amount units/day, 'times_per_day': n, 'slots': 'M/A/N', 'prn': bool, 'stat': bool}."""
    f = (freq or "").strip().lower()
    out = {"per_day": None, "times_per_day": None, "slots": "", "prn": False, "stat": False, "weekly": False}
    if not f:
        return out
    if f in {"stat", "once"}:
        out.update(per_day=1, times_per_day=1, stat=True)
        return out
    if f in {"sos", "prn"}:
        out.update(per_day=0, times_per_day=0, prn=True)
        return out
    if "week" in f:
        out.update(per_day=1 / 7, times_per_day=1 / 7, weekly=True)
        return out
    parts = re.split(r"\s*-\s*", f)
    if len(parts) in (3, 4) and all(re.fullmatch(r"[\d./½¼]+u?", p.strip()) for p in parts):
        vals = [_num(p) for p in parts]
        labels = ["M", "A", "N", "HS"][: len(vals)]
        out["per_day"] = sum(vals)
        out["times_per_day"] = sum(1 for v in vals if v > 0)
        out["slots"] = "/".join(l for l, v in zip(labels, vals) if v > 0)
        return out
    for word, n in _FREQ_WORDS.items():
        if word in f:
            out.update(per_day=n, times_per_day=n, slots="N" if word in {"hs", "night"} else "")
            return out
    return out


def parse_duration(text: str) -> tuple[int | None, bool]:
    """Return (days, chronic_flag)."""
    t = (text or "").strip().lower()
    if not t:
        return None, False
    if t in {"continue", "to continue", "ongoing", "long term", "lifelong"}:
        return None, True
    if t == "stat":
        return 1, False
    m = re.search(r"([\d.]+)\s*(day|d\b|week|wk|month|mth|mo|year)", t)
    if not m:
        return None, False
    n = float(m.group(1))
    unit = m.group(2)
    mult = 1 if unit.startswith("d") else 7 if unit.startswith("w") else 365 if unit.startswith("y") else 30
    return int(round(n * mult)), False


def parse_strength(text: str) -> tuple[float | None, str]:
    m = re.search(r"([\d.]+)\s*(mg|mcg|g|iu|units?|ml|%)", (text or "").lower())
    if not m:
        return None, ""
    return float(m.group(1)), m.group(2)


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# --------------------------------------------------------------------------- main transform
def build_medication_table(patients: list[dict], normaliser: DrugNormaliser | None = None) -> pd.DataFrame:
    norm = normaliser or DrugNormaliser()
    rows = []
    for p in patients:
        encs = sorted(p["encounters"], key=lambda e: e["date"])
        reconciling_dates = [_d(e["date"]) for e in encs if e.get("reconciles_chronic_meds")]
        for e in encs:
            enc_date = _d(e["date"])
            for i, m in enumerate(e.get("medications", [])):
                key, entry = norm.lookup(m.get("drug", ""))
                components = entry["components"] if entry else []
                freq = parse_frequency(m.get("frequency", ""))
                days, chronic = parse_duration(m.get("duration", ""))
                chronic = chronic or bool(m.get("chronic"))
                start = _d(m["start_date"]) if m.get("start_date") else enc_date
                duration_assumed = False
                if freq["stat"]:
                    end = start
                elif days:
                    end = start + timedelta(days=days - 1)
                elif chronic:
                    later = [d for d in reconciling_dates if d > enc_date]
                    end = (later[0] - timedelta(days=1)) if later else None
                else:
                    end = start + timedelta(days=DEFAULT_COURSE_DAYS - 1)
                    duration_assumed = True
                # chronic meds with an explicit review period stay open until superseded
                if chronic and days:
                    later = [d for d in reconciling_dates if d > enc_date]
                    end = (later[0] - timedelta(days=1)) if later else None
                s_val, s_unit = parse_strength(m.get("strength", ""))
                rows.append({
                    "patient_id": p["patient_id"],
                    "encounter_id": e["encounter_id"],
                    "item": f"{e['encounter_id']}-M{i + 1}",
                    "encounter_date": enc_date,
                    "facility": e.get("facility", ""),
                    "department": e.get("department", ""),
                    "doctor": e.get("doctor", ""),
                    "source_image": e.get("source_image", ""),
                    "raw": m.get("raw", ""),
                    "drug_key": key,
                    "display": entry["display"] if entry else f"Unidentified: {m.get('raw', '')}",
                    "components": components,
                    "classes": sorted(norm.classes_for(components)),
                    "strength": m.get("strength", ""),
                    "strength_value": s_val,
                    "strength_unit": s_unit,
                    "route": m.get("route", ""),
                    "frequency": m.get("frequency", ""),
                    "slots": freq["slots"],
                    "amount_per_day": freq["per_day"],
                    "prn": freq["prn"],
                    "stat": freq["stat"],
                    "duration": m.get("duration", ""),
                    "duration_days": days,
                    "duration_assumed": duration_assumed,
                    "chronic": chronic,
                    "start": start,
                    "end": end,
                    "instructions": m.get("instructions", ""),
                    "legibility": m.get("legibility", "clear"),
                    "note": m.get("note", ""),
                    "administered_in_clinic": bool(m.get("administered_in_clinic")),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = _apply_renewals(df)
        df["daily_dose_mg"] = df.apply(_daily_mg, axis=1)
    return df


def _apply_renewals(df: pd.DataFrame) -> pd.DataFrame:
    """A later prescription of the same product supersedes the earlier one (renewal, not duplication)."""
    df = df.copy()
    df["superseded_by"] = None
    for (pid, key), grp in df[df["drug_key"].notna()].groupby(["patient_id", "drug_key"]):
        grp = grp.sort_values("start")
        enc_starts = grp.groupby("encounter_id")["start"].min().sort_values()
        for idx, r in grp.iterrows():
            later = enc_starts[enc_starts > r["start"]]
            later = later[later.index != r["encounter_id"]]
            if later.empty:
                continue
            new_start = later.iloc[0]
            if r["end"] is None or pd.isna(r["end"]) or r["end"] >= new_start:
                df.at[idx, "end"] = new_start - timedelta(days=1)
                df.at[idx, "superseded_by"] = later.index[0]
    return df


def _daily_mg(row) -> float | None:
    if row["strength_unit"] != "mg" or row["strength_value"] is None:
        return None
    if row["prn"]:  # assume up to 3 doses/day for PRN when checking ceilings
        return row["strength_value"] * 3
    if row["amount_per_day"] is None:
        return None
    return row["strength_value"] * row["amount_per_day"]


def active_on(meds: pd.DataFrame, on: date) -> pd.DataFrame:
    if meds.empty:
        return meds
    mask = meds.apply(lambda r: r["start"] <= on and (r["end"] is None or pd.isna(r["end"]) or r["end"] >= on), axis=1)
    return meds[mask.astype(bool)]


# --------------------------------------------------------------------------- data quality
def data_quality_report(patients: list[dict], meds: pd.DataFrame) -> pd.DataFrame:
    issues = []
    for p in patients:
        if p.get("dob_estimated"):
            issues.append((p["patient_id"], "", "Low", "Date of birth estimated", p.get("dob_note", "")))
        if p.get("identity_linkage_note"):
            issues.append((p["patient_id"], "", "Moderate", "Records merged across facilities", p["identity_linkage_note"]))
        if p.get("allergy_status") == "not documented":
            issues.append((p["patient_id"], "", "Moderate", "Allergy status not documented", "Ask and record allergies before prescribing."))
        for e in p["encounters"]:
            if not e.get("date_certain", True):
                issues.append((p["patient_id"], e["encounter_id"], "Low", "Encounter date uncertain", e.get("date_note", "")))
            if e.get("recorded_age") is not None:
                calc = age_on(p, _d(e["date"]))
                if calc is not None and abs(calc - e["recorded_age"]) > 1:
                    issues.append((p["patient_id"], e["encounter_id"], "Moderate", "Age inconsistency",
                                   f"Recorded age {e['recorded_age']} vs {calc} from DOB - verify identity / DOB."))
    if not meds.empty:
        for _, r in meds.iterrows():
            if r["legibility"] == "illegible" or r["drug_key"] is None:
                issues.append((r["patient_id"], r["encounter_id"], "Moderate", "Unidentified medication",
                               f"'{r['raw']}' could not be mapped to a drug - verify with source image {r['source_image']}."))
            elif r["legibility"] == "partial":
                issues.append((r["patient_id"], r["encounter_id"], "Low", "Partially legible entry",
                               f"'{r['raw']}' - {r['note'] or 'verify against source image.'}"))
            if r["duration_assumed"]:
                issues.append((r["patient_id"], r["encounter_id"], "Low", "Duration not stated",
                               f"'{r['raw']}' - assumed {DEFAULT_COURSE_DAYS} days."))
    return pd.DataFrame(issues, columns=["patient_id", "encounter_id", "severity", "issue", "detail"])
