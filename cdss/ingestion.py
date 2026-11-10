"""Block 1a - Data ingestion and patient record consolidation (unified patient ID mapping)."""

from __future__ import annotations

import json
import re
import shutil
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from . import IMAGE_DIR, KB_DIR, PATIENTS_FILE


# --------------------------------------------------------------------------- loading
def load_store(path: Path = PATIENTS_FILE) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_store(store: dict, path: Path = PATIENTS_FILE) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_patients(path: Path = PATIENTS_FILE) -> list[dict]:
    patients = load_store(path)["patients"]
    for p in patients:
        p["encounters"] = sorted(p["encounters"], key=lambda e: e["date"])
    return patients


def get_patient(patients: list[dict], patient_id: str) -> dict:
    for p in patients:
        if p["patient_id"] == patient_id:
            return p
    raise KeyError(patient_id)


def load_drug_dictionary() -> dict:
    with open(KB_DIR / "drug_dictionary.json", encoding="utf-8") as fh:
        return json.load(fh)["brands"]


def load_rules() -> dict:
    with open(KB_DIR / "clinical_rules.json", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- demographics
def age_on(patient: dict, on: date | None = None) -> int | None:
    if not patient.get("dob"):
        return None
    dob = datetime.strptime(patient["dob"], "%Y-%m-%d").date()
    on = on or date.today()
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


def condition_codes(patient: dict) -> set[str]:
    return {c["code"] for c in patient.get("conditions", [])}


# --------------------------------------------------------------------------- identity resolution
def _norm_name(name: str) -> str:
    name = re.sub(r"\b(mr|mrs|ms|dr|miss)\b\.?", " ", name.lower())
    return " ".join(re.sub(r"[^a-z ]", " ", name).split())


def name_similarity(a: str, b: str) -> float:
    a, b = _norm_name(a), _norm_name(b)
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    # token containment: "leela" inside "sagay josephine leela"
    ta, tb = set(a.split()), set(b.split())
    contain = len(ta & tb) / min(len(ta), len(tb))
    return max(ratio, contain * 0.9)


def match_patient(patients: list[dict], name: str = "", identifier: str = "") -> list[tuple[dict, float, str]]:
    """Return candidate patients for an incoming record, best first.

    An exact identifier (UHID / hospital number) match is decisive; otherwise names are
    compared against the patient's name and every alias.
    """
    results = []
    for p in patients:
        ids = {i["value"].strip().upper() for i in p.get("identifiers", [])}
        if identifier and identifier.strip().upper() in ids:
            results.append((p, 1.0, f"identifier {identifier} matches"))
            continue
        if name:
            best = max(name_similarity(name, n) for n in [p["name"], *p.get("aliases", [])])
            if best >= 0.6:
                results.append((p, round(best, 2), "name similarity"))
    return sorted(results, key=lambda r: -r[1])


# --------------------------------------------------------------------------- adding new records
def next_encounter_id(patient: dict) -> str:
    n = len(patient["encounters"]) + 1
    existing = {e["encounter_id"] for e in patient["encounters"]}
    while f"{patient['patient_id']}-E{n}" in existing:
        n += 1
    return f"{patient['patient_id']}-E{n}"


def store_source_image(uploaded_path: Path | str, patient: dict, enc_date: str, label: str) -> str:
    """Copy an uploaded prescription image into the prescriptions folder and return its file name."""
    uploaded_path = Path(uploaded_path)
    existing = sorted(IMAGE_DIR.glob("[0-9][0-9]_*.*"))
    seq = (max(int(p.name[:2]) for p in existing) + 1) if existing else 1
    first = _norm_name(patient["name"]).split()[0] if patient.get("name") else "patient"
    safe_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "record"
    name = f"{seq:02d}_{first}_{safe_label}_{enc_date}{uploaded_path.suffix.lower() or '.jpg'}"
    shutil.copyfile(uploaded_path, IMAGE_DIR / name)
    return name


def add_encounter(patient_id: str, encounter: dict[str, Any], path: Path = PATIENTS_FILE) -> str:
    store = load_store(path)
    for p in store["patients"]:
        if p["patient_id"] == patient_id:
            encounter.setdefault("encounter_id", next_encounter_id(p))
            encounter.setdefault("medications", [])
            encounter.setdefault("diagnoses", [])
            encounter.setdefault("complaints", [])
            encounter.setdefault("vitals", {})
            encounter.setdefault("investigations_advised", [])
            encounter.setdefault("plan", [])
            p["encounters"].append(encounter)
            p["encounters"].sort(key=lambda e: e["date"])
            save_store(store, path)
            return encounter["encounter_id"]
    raise KeyError(patient_id)


def add_patient(name: str, sex: str, dob: str, identifiers: list[dict], path: Path = PATIENTS_FILE) -> str:
    store = load_store(path)
    nums = [int(p["patient_id"][1:]) for p in store["patients"]]
    pid = f"P{(max(nums) + 1) if nums else 1:03d}"
    store["patients"].append({
        "patient_id": pid, "name": name, "aliases": [], "sex": sex, "dob": dob,
        "dob_estimated": False, "identifiers": identifiers, "allergies": [],
        "allergy_status": "not documented", "conditions": [], "encounters": [],
    })
    save_store(store, path)
    return pid
