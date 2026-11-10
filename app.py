"""Block 5 - Interactive Clinical Decision Support Dashboard (Streamlit).

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from cdss import DISCLAIMER, IMAGE_DIR, PATIENTS_FILE, llm, risk_model
from cdss.features import (all_patient_features, daily_med_counts, medication_changes,
                           patient_features, vitals_table)
from cdss.ingestion import (add_encounter, add_patient, age_on, get_patient, load_patients,
                            match_patient, store_source_image)
from cdss.ocr import extract, to_encounter
from cdss.preprocessing import DrugNormaliser, active_on, build_medication_table, data_quality_report
from cdss.rag import ClinicalRAG
from cdss.recommender import RecommendationEngine
from cdss.safety import SafetyEngine, alerts_frame, risk_level
from cdss.summarizer import llm_summary, summarise, summary_markdown
from cdss.timeline import encounter_timeline, medication_gantt, polypharmacy_figure, vitals_figure
from cdss.triage import triage

st.set_page_config(page_title="Clinical Decision Support", layout="wide")

LEVEL_STYLE = {"EMERGENCY": st.error, "URGENT": st.warning, "ROUTINE": st.success}
STATUS_STYLE = {"Preferred": "green", "Use with caution": "orange", "Avoid": "red"}


# ----------------------------------------------------------------------------- cached loaders
@st.cache_resource
def engines():
    norm = DrugNormaliser()
    safety = SafetyEngine(norm)
    rag = ClinicalRAG()
    return norm, safety, rag, RecommendationEngine(safety, rag)


@st.cache_data
def load_all(mtime: float):
    patients = load_patients()
    meds = build_medication_table(patients, engines()[0])
    return patients, meds


def data():
    return load_all(PATIENTS_FILE.stat().st_mtime)


norm, safety, rag, recommender = engines()
patients, meds = data()

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("CDSS")
st.sidebar.caption("AI-based clinical decision support from historical prescriptions")
labels = {p["patient_id"]: f"{p['name']} ({p['patient_id']})" for p in patients}
pid = st.sidebar.selectbox("Patient", list(labels), format_func=labels.get)
patient = get_patient(patients, pid)
last_visit = date.fromisoformat(patient["encounters"][-1]["date"]) if patient["encounters"] else date.today()
ref_date = st.sidebar.date_input("Reference date ('today' for the review)", value=last_visit,
                                 help="Active medications and alerts are computed for this date. Defaults to the last visit.")
allergy_txt = st.sidebar.text_input("Allergies (comma separated)", value=", ".join(patient.get("allergies", [])),
                                    help="Recorded status: " + patient.get("allergy_status", "unknown"))
allergies = [a.strip() for a in allergy_txt.split(",") if a.strip()]
st.sidebar.markdown("---")
st.sidebar.write("LLM:", f"connected - {llm.provider_name()}" if llm.available() else "offline (rule/template mode)")
st.sidebar.caption(DISCLAIMER)

# ----------------------------------------------------------------------------- computations
pm = meds[meds["patient_id"] == pid]
history_alerts = safety.scan_history(patient, meds, allergies)
active_alerts = [a for a in history_alerts if a.window_start <= ref_date and (a.window_end is None or a.window_end >= ref_date)]
level, score = risk_level(active_alerts)
feats = patient_features(patient, meds, ref_date)

# ----------------------------------------------------------------------------- header
st.title(patient["name"])
ids = " | ".join(f"{i['system']}: {i['value']}" for i in patient.get("identifiers", []))
age = age_on(patient, ref_date)
st.caption(f"{patient.get('sex')} | {age} y{' (DOB estimated)' if patient.get('dob_estimated') else ''} | {ids}")
if patient.get("conditions"):
    st.markdown(" ".join(f"`{c['name']}`" for c in patient["conditions"]))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Encounters", feats["n_encounters"])
c2.metric("Facilities / prescribers", f"{feats['n_facilities']} / {feats['n_prescribers']}")
c3.metric("Max concurrent systemic meds", feats["max_concurrent_systemic_meds"])
c4.metric(f"Active alerts on {ref_date}", len(active_alerts))
c5.metric("Current safety risk", level)

for a in [a for a in active_alerts if a.severity == "High"]:
    st.error(f"**HIGH - {a.kind}: {a.title}**  \n{a.effect}  \n*Action:* {a.management}  `[{a.ref}]`")
mod = [a for a in active_alerts if a.severity != "High"]
if mod:
    with st.expander(f"{len(mod)} moderate / low alerts active on {ref_date}"):
        for a in mod:
            (st.warning if a.severity == "Moderate" else st.info)(f"**{a.severity} - {a.title}**: {a.effect} *Action:* {a.management} `[{a.ref}]`")

tabs = st.tabs(["Overview & summary", "Prescription timeline", "Source documents", "Safety check",
                "AI assistant & triage", "Risk & analytics", "Add record"])

# ============================================================================ 1 overview
with tabs[0]:
    s = summarise(patient, meds, history_alerts, ref_date)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Longitudinal clinical summary")
        st.markdown(summary_markdown(s))
        if llm.available() and st.button("Generate AI narrative summary"):
            hits = rag.retrieve(" ".join(c["name"] for c in patient.get("conditions", [])), patient)
            ctx = "\n".join(f"[{c.id}] {c.text}" for c, _ in hits["guidelines"])
            with st.spinner("Summarising..."):
                st.info(llm_summary(patient, s, ctx))
    with right:
        st.subheader("Visits")
        st.plotly_chart(encounter_timeline(patient), width="stretch")
        st.subheader("Data quality")
        dq = data_quality_report([patient], pm)
        st.dataframe(dq.drop(columns=["patient_id"]), hide_index=True, width="stretch")

# ============================================================================ 2 timeline
with tabs[1]:
    st.subheader("Medication timeline")
    st.caption("Bars = prescribed course; hatched = no end date (chronic, continued); red shading = HIGH-severity alert window; dotted lines = visits.")
    st.plotly_chart(medication_gantt(patient, meds, ref_date, history_alerts), width="stretch")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Medication changes between visits")
        st.dataframe(medication_changes(patient, meds), hide_index=True, width="stretch")
    with col_b:
        st.subheader("Polypharmacy over time")
        fig = polypharmacy_figure(daily_med_counts(patient, meds))
        if fig:
            st.plotly_chart(fig, width="stretch")
        st.subheader(f"Active on {ref_date}")
        act = active_on(pm, ref_date)
        st.dataframe(act[["display", "strength", "frequency", "start", "end", "department"]] if not act.empty else act,
                     hide_index=True, width="stretch")
    with st.expander("All parsed medication entries"):
        cols = ["encounter_id", "raw", "display", "strength", "route", "frequency", "slots", "amount_per_day",
                "duration", "start", "end", "chronic", "legibility", "source_image"]
        st.dataframe(pm[cols], hide_index=True, width="stretch")

# ============================================================================ 3 source docs
with tabs[2]:
    st.subheader("Source prescriptions (primary data)")
    st.caption("Each record was transcribed from the image shown beside it. Verify partially legible entries against the original.")
    for e in reversed(patient["encounters"]):
        with st.expander(f"{e['date']} - {e.get('department', '')} - {e.get('facility', '')}", expanded=False):
            img_col, data_col = st.columns([1, 1])
            img = IMAGE_DIR / e.get("source_image", "")
            if e.get("source_image") and img.exists():
                img_col.image(str(img), caption=e["source_image"], width="stretch")
            else:
                img_col.info("No image stored for this encounter.")
            data_col.markdown(f"**Doctor:** {e.get('doctor', '-')}  \n**Type:** {e.get('type')}  \n"
                              f"**Date certain:** {e.get('date_certain', True)} {e.get('date_note', '')}")
            if e.get("diagnoses"):
                data_col.markdown("**Diagnoses:** " + "; ".join(e["diagnoses"]))
            if e.get("complaints"):
                data_col.markdown("**Complaints:** " + "; ".join(e["complaints"]))
            if e.get("vitals"):
                data_col.markdown("**Vitals:** " + ", ".join(f"{k}={v}" for k, v in e["vitals"].items()))
            rows = [{"medication": m.get("raw"), "legibility": m.get("legibility", "clear"), "note": m.get("note", "")}
                    for m in e.get("medications", [])]
            if rows:
                data_col.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            for k in ("investigations_advised", "plan"):
                if e.get(k):
                    data_col.markdown(f"**{k.replace('_', ' ').title()}:** " + "; ".join(e[k]))

# ============================================================================ 4 safety
with tabs[3]:
    st.subheader("Multi-timeline safety scan (entire history)")
    st.caption("Interactions are checked only between medicines whose active periods actually overlapped - across visits and hospitals.")
    df = alerts_frame(history_alerts)
    if df.empty:
        st.success("No safety alerts across the documented history.")
    else:
        sev = st.multiselect("Severity", ["High", "Moderate", "Low"], default=["High", "Moderate", "Low"])
        st.dataframe(df[df["severity"].isin(sev)], hide_index=True, width="stretch")

    st.markdown("---")
    st.subheader("Check a new prescription before finalising")
    brand_list = sorted(norm.brands.keys())
    picks = st.multiselect("Drugs to prescribe (from dictionary)", brand_list)
    free = st.text_input("...or type drug names (comma separated)")
    cA, cB, cC = st.columns(3)
    strength = cA.text_input("Strength (applies to all, optional)", "")
    freq = cB.text_input("Frequency", "1-0-1")
    days = cC.number_input("Days", 1, 365, 5)
    proposed_names = picks + [x.strip() for x in free.split(",") if x.strip()]
    if st.button("Run safety check", disabled=not proposed_names):
        items = [safety.make_item(n, ref_date, strength, freq, int(days)) for n in proposed_names]
        res = safety.check_proposed(patient, meds, items, ref_date, allergies)
        if res["unrecognised"]:
            st.warning("Not in drug dictionary (checked by name only): " + ", ".join(res["unrecognised"]))
        st.write(f"Checked against {len(res['current'])} active and {len(res['recent'])} recently-stopped medications on {ref_date}.")
        if not res["alerts"]:
            st.success("No interactions, duplications, contraindications or allergy conflicts found.")
        for a in res["alerts"]:
            fn = st.error if a.severity == "High" else st.warning if a.severity == "Moderate" else st.info
            fn(f"**{a.severity} - {a.kind}: {a.title}**  \n{a.effect}  \n*Action:* {a.management}")
            g = rag.guideline(a.ref)
            if g:
                with st.expander(f"Evidence [{g.id}] {g.title}"):
                    st.write(g.text)

# ============================================================================ 5 assistant
with tabs[4]:
    st.subheader("Current presentation")
    symptoms = st.text_area("Symptoms / reason for visit", placeholder="e.g. fever since 2 days, redness spreading up the left leg, knee pain")
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    def num(col, label, lo, hi):
        val = col.number_input(label, lo, hi, value=None, step=1.0)
        return None if val is None else float(val)
    vit = {"bp_systolic": num(v1, "Systolic BP", 40.0, 260.0), "pulse": num(v2, "Pulse", 20.0, 220.0),
           "spo2": num(v3, "SpO2 %", 50.0, 100.0), "temperature_f": num(v4, "Temp F", 90.0, 110.0),
           "resp_rate": num(v5, "Resp. rate", 4.0, 60.0), "glucose_mg_dl": num(v6, "Glucose mg/dL", 20.0, 700.0)}
    vit = {k: v for k, v in vit.items() if v is not None}
    vit["new_confusion"] = st.checkbox("New confusion / reduced consciousness")
    if st.button("Assess", type="primary", disabled=not symptoms.strip()):
        t = triage(symptoms, vit, patient, meds, ref_date)
        LEVEL_STYLE[t.level](f"### Triage: {t.level}  (score {t.score})\n" + "\n".join(f"- {a}" for a in t.actions))
        with st.expander("Why this triage level?", expanded=True):
            st.table(pd.DataFrame(t.reasons, columns=["factor", "points"]))

        rec = recommender.recommend(patient, meds, symptoms, ref_date, allergies, active_alerts)
        st.subheader("Personalised treatment options")
        if not rec.categories:
            st.info("No mapped problem category - see considerations and retrieved evidence below.")
        for cat in rec.categories:
            st.markdown(f"**{cat.replace('_', ' ').title()}**")
            for o in [o for o in rec.options if o.category == cat]:
                st.markdown(f":{STATUS_STYLE[o.status]}[**{o.status}**] - {o.label}  `[{o.ref}]`")
                for sev, why, ref in o.reasons:
                    st.caption(f"  {sev}: {why} [{ref}]")
        if rec.considerations:
            st.subheader("Patient-specific considerations")
            for text, ref in rec.considerations:
                st.markdown(f"- {text} `[{ref}]`")
        if rec.narrative:
            st.subheader("AI explanation (LLM + RAG)")
            st.info(rec.narrative)
        st.subheader("Retrieved evidence")
        for c in rec.evidence["cited"]:
            with st.expander(f"[{c.id}] {c.title}"):
                st.write(c.text)
        with st.expander("Most relevant parts of this patient's record"):
            for c, sc in rec.evidence["patient"]:
                st.markdown(f"**[{c.id}]** (relevance {sc:.2f}) {c.text}")
        st.caption(DISCLAIMER)

# ============================================================================ 6 analytics
with tabs[5]:
    vt = vitals_table(patient)
    st.subheader("Vital-sign trends")
    fig = vitals_figure(vt)
    if fig:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No vitals recorded for this patient.")

    st.subheader("Disease-progression risk (ML)")
    models, reports = risk_model.train()
    base = risk_model.features_from_patient(patient, vt, age)
    r1, r2, r3, r4 = st.columns(4)
    inp = {}
    for col, f in zip((r1, r2, r3, r4), risk_model.FEATURES):
        v = base.get(f)
        inp[f] = col.number_input(risk_model.LABELS[f], value=None if v is None else float(v), step=1.0, key=f"risk_{f}_{pid}")
    pred = risk_model.predict({k: (None if v is None else float(v)) for k, v in inp.items()})
    st.metric("Probability of high one-year progression", f"{pred['probability']:.0%}", pred["band"])
    if pred["missing_imputed"]:
        st.caption("Imputed (median of training data): " + ", ".join(pred["missing_imputed"]))
    if pred["outside_training_range"]:
        st.warning("Outside the training range - interpret with caution: " + ", ".join(pred["outside_training_range"]))
    st.dataframe(pd.DataFrame([{"model": r.name, "CV ROC-AUC": r.cv_auc, "CV accuracy": r.cv_accuracy, "n": r.n_train,
                                **{f"importance:{k}": v for k, v in r.importances.items()}} for r in reports]),
                 hide_index=True, width="stretch")
    st.caption(risk_model.__doc__)

    st.subheader("Cohort features (sequential feature engineering)")
    st.dataframe(all_patient_features(patients, meds, ref_date), hide_index=True, width="stretch")

# ============================================================================ 7 add record
with tabs[6]:
    st.subheader("Add a new prescription image")
    st.caption("The image is stored in the prescriptions folder and linked to the encounter. Extraction uses Claude vision when an API key is set, otherwise Tesseract or manual entry.")
    up = st.file_uploader("Prescription / discharge summary image", type=["jpg", "jpeg", "png"])
    if up:
        st.image(up, width=360)
        tmp = Path(tempfile.gettempdir()) / up.name
        tmp.write_bytes(up.getvalue())
        if st.button("Extract"):
            with st.spinner("Reading prescription..."):
                st.session_state["extracted"] = extract(tmp)
        ex = st.session_state.get("extracted")
        if ex:
            st.caption(f"Engine: {ex.get('engine')} {ex.get('warning', '')}")
            cands = match_patient(patients, ex.get("patient_name", ""), ex.get("patient_identifier", ""))
            opts = {p["patient_id"]: f"{p['name']} ({p['patient_id']}) - match {s:.2f}, {why}" for p, s, why in cands}
            opts["__new__"] = "Create new patient"
            target = st.selectbox("Link to patient (unified patient ID)", list(opts), format_func=opts.get)
            enc_date = st.text_input("Encounter date (YYYY-MM-DD)", ex.get("date") or date.today().isoformat())
            enc = to_encounter({**ex, "date": enc_date}, "")
            edited = st.text_area("Verify / edit the extracted record (JSON)", json.dumps(enc, indent=2, ensure_ascii=False), height=380)
            if st.button("Save record", type="primary"):
                record = json.loads(edited)
                if target == "__new__":
                    target = add_patient(ex.get("patient_name") or "Unknown", ex.get("sex", ""), "",
                                         [{"system": "Hospital ID", "value": ex.get("patient_identifier", "")}] if ex.get("patient_identifier") else [])
                tgt = get_patient(load_patients(), target)
                record["source_image"] = store_source_image(tmp, tgt, record.get("date", enc_date), record.get("department") or "record")
                eid = add_encounter(target, record)
                st.success(f"Saved {eid}; image stored as {record['source_image']} in the prescriptions folder.")
                st.session_state.pop("extracted", None)
                st.cache_data.clear()
