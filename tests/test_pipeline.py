"""Run with:  python -m pytest -q"""
from datetime import date

import pytest

from cdss.features import medication_changes
from cdss.ingestion import get_patient, load_patients, match_patient
from cdss.preprocessing import build_medication_table, parse_duration, parse_frequency
from cdss.recommender import RecommendationEngine
from cdss.safety import SafetyEngine
from cdss.triage import triage


@pytest.fixture(scope="module")
def ctx():
    patients = load_patients()
    meds = build_medication_table(patients)
    return patients, meds, SafetyEngine()


def test_frequency_parsing():
    assert parse_frequency("1-0-1")["per_day"] == 2
    assert parse_frequency("1/2-0-0")["per_day"] == 0.5
    assert parse_frequency("18U-0-18U")["per_day"] == 36
    assert parse_frequency("SOS")["prn"]
    assert parse_duration("x 2 weeks") == (14, False)
    assert parse_duration("continue") == (None, True)


def test_every_encounter_has_existing_image(ctx):
    from cdss import IMAGE_DIR
    patients, _, _ = ctx
    for p in patients:
        for e in p["encounters"]:
            assert (IMAGE_DIR / e["source_image"]).exists(), e["source_image"]


def _rules(alerts):
    return {a.rule_id for a in alerts}


def test_philomena_cross_facility_alerts(ctx):
    patients, meds, s = ctx
    ids = _rules(s.scan_history(get_patient(patients, "P002"), meds))
    assert {"I01", "T01", "DUP-ING", "I05"} <= ids  # cipro+glimepiride, triple whammy, metformin dup, NSAID+heparin


def test_leela_clarithromycin_domperidone(ctx):
    patients, meds, s = ctx
    assert "I15" in _rules(s.scan_history(get_patient(patients, "P006"), meds))


def test_asuntha_ccb_duplication_and_insulin_switch(ctx):
    patients, meds, s = ctx
    p = get_patient(patients, "P001")
    assert "DUP-CLS" in _rules(s.scan_history(p, meds))
    ch = medication_changes(p, meds)
    assert (ch["change"] == "Switched").any()


def test_split_dose_not_flagged_as_duplicate(ctx):
    patients, meds, s = ctx
    alerts = s.scan_history(get_patient(patients, "P001"), meds)
    assert not any(a.rule_id == "DUP-ING" and "Levothyroxine" in a.title for a in alerts)


def test_proposed_nsaid_avoided_for_philomena(ctx):
    patients, meds, s = ctx
    rec = RecommendationEngine(s).recommend(get_patient(patients, "P002"), meds, "knee pain", date(2026, 5, 18), use_llm=False)
    ibu = [o for o in rec.options if o.label.startswith("Ibuprofen")][0]
    assert ibu.status == "Avoid"


def test_allergy_cross_reactivity(ctx):
    patients, meds, s = ctx
    p = get_patient(patients, "P001")
    item = s.make_item("cephalexin", date(2026, 3, 20))
    res = s.check_proposed(p, meds, [item], date(2026, 3, 20), allergies=["penicillin"])
    assert any(a.rule_id == "ALG-X" for a in res["alerts"])


def test_triage_levels(ctx):
    patients, meds, _ = ctx
    p = get_patient(patients, "P001")
    assert triage("convulsions", {}, p, meds).level == "EMERGENCY"
    assert triage("fever", {"pulse": 100}, p, meds).level == "URGENT"
    assert triage("routine refill", {"spo2": 98, "pulse": 76}, p, meds).level == "ROUTINE"


def test_identity_matching(ctx):
    patients, _, _ = ctx
    assert match_patient(patients, identifier="38191")[0][0]["patient_id"] == "P001"
    assert match_patient(patients, name="Mrs Leela")[0][0]["patient_id"] == "P006"
