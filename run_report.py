"""Command-line pipeline: ingest -> preprocess -> features -> safety -> summary for every patient.

Writes one Markdown report per patient into ./reports and prints a cohort overview.
    python run_report.py
"""

from datetime import date
from pathlib import Path

from cdss import DISCLAIMER, PROJECT_ROOT
from cdss.features import all_patient_features
from cdss.ingestion import load_patients
from cdss.preprocessing import build_medication_table, data_quality_report
from cdss.safety import SafetyEngine, alerts_frame, risk_level
from cdss.summarizer import summarise, summary_markdown


def main():
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    patients = load_patients()
    meds = build_medication_table(patients)
    safety = SafetyEngine()
    print(f"Loaded {len(patients)} patients, {sum(len(p['encounters']) for p in patients)} encounters, "
          f"{len(meds)} medication entries from {meds['source_image'].nunique()} source images.\n")
    for p in patients:
        alerts = safety.scan_history(p, meds)
        lvl, score = risk_level(alerts)
        s = summarise(p, meds, alerts)
        md = [f"# {p['name']} ({p['patient_id']})", "", summary_markdown(s), "", "## All safety alerts", ""]
        df = alerts_frame(alerts)
        md.append(df.to_markdown(index=False) if not df.empty else "None")
        dq = data_quality_report([p], meds[meds["patient_id"] == p["patient_id"]])
        md += ["", "## Data quality", "", dq.drop(columns=["patient_id"]).to_markdown(index=False) if not dq.empty else "None",
               "", f"_{DISCLAIMER}_"]
        (out / f"{p['patient_id']}_{p['name'].split()[0].lower()}.md").write_text("\n".join(md), encoding="utf-8")
        print(f"{p['patient_id']} {p['name']:<24} alerts={len(alerts):>2}  highest={lvl:<8} score={score}")
    print("\nCohort features:")
    print(all_patient_features(patients, meds, date.today()).to_string(index=False))
    print(f"\nReports written to {out}")


if __name__ == "__main__":
    main()
