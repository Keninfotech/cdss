# AI-Based Clinical Decision Support System Using Historical Medical Prescriptions and Symptom Analysis

Project Work Phase 1 (CSE 784 / CSEIOT 784): Sunith K (2360471, 7BTCS A). Guide: Dr. Jyothi Thomas.
Department of Computer Science and Engineering.

The system turns a patient's paper prescriptions and discharge summaries into a structured
longitudinal record. It checks every medicine against every other medicine that was active
at the same time, even when they came from different hospitals. It also triages the current
presentation and suggests personalised, explainable treatment options in a Streamlit dashboard.

> **Clinical decision support only.** All alerts and suggestions must be verified by the treating
> clinician against the source documents and current guidelines.

---

## 1. Primary data: the prescription images (project root)

The 16 source documents are kept in the **project root** and every encounter in
`data/patients.json` points to its image (`source_image`). The dashboard shows each image next
to its transcription.

| # | Image (root folder) | Patient | Document |
|---|---|---|---|
| 01 | `01_asuntha_discharge_summary_2025-06.jpg` | Asuntha Anandraj (P001) | SJMCH Surgery discharge summary (CT s/o CLD, CBD stent) |
| 02 | `02_asuntha_pharmacy_indent_2025-08-28.jpg` | Asuntha Anandraj (P001) | Inpatient pharmacy indent (cefuroxime, dalteparin) |
| 03 | `03_asuntha_opd_medicine_2026-03-14.jpg` | Asuntha Anandraj (P001) | SJMCH Medicine OPD (insulin, thyroxine, BP, DM) |
| 04 | `04_philomena_clinic_2026-04-26.jpg` | Philomena R (P002) | Mahesh Health Care: DM, HTN, knee arthritis |
| 05 | `05_philomena_surgery_first_visit_2026-05-02.jpg` | Philomena R (P002) | St. Philomena's Surgery: cellulitis, first visit |
| 06 | `06_philomena_surgery_followup_2026-05-18.jpg` | Philomena R (P002) | St. Philomena's Surgery: follow-up, vitals, varicose veins |
| 07 | `07_nishith_dermatology_2025-10-06.jpg` | Nishith D (P003) | SJMCH Dermatology |
| 08 | `08_nishith_dermatology_2026-01-03.jpg` | Nishith D (P003) | SJMCH Dermatology review |
| 09 | `09_anthony_neurosurgery_discharge_2021-06.jpg` | Anthony Reddy (P004) | Neurosurgery discharge (LS spine) |
| 10 | `10_anthony_prescription_2021-06-27.jpg` | Anthony Reddy (P004) | Post-op prescription (Ultracet, dressing, LS belt) |
| 11 | `11_deva_prasad_orthopaedics_2025-08.jpg` | Deva Prasad (P005) | SJMCH Orthopaedics |
| 12 | `12_leela_surgeon_2018-04-13.jpg` | Sagay Josephine Leela (P006) | H. pylori triple therapy |
| 13 | `13_leela_dermatology_2021-04-09.jpg` | Sagay Josephine Leela (P006) | Dermatology (?PIH) |
| 14 | `14_leela_psychiatry_2021-11-11.jpg` | Sagay Josephine Leela (P006) | Psychiatry (GAD + grief) |
| 15 | `15_leela_gastroenterology_2022-09-29.jpg` | Sagay Josephine Leela (P006) | Gastroenterology (mebeverine, amitriptyline) |
| 16 | `16_yellamma_rheumatology_2024-09.jpg` | Yellamma (P007) | Clinical Immunology & Rheumatology |

Handwritten entries that could not be read with confidence keep their raw text and are marked
`legibility: partial | illegible`. The dashboard lists them under **Data quality** so they can be
checked against the image. New images uploaded through the dashboard (**Add record** tab) are
also saved into the project root and linked to the new encounter.

Phone numbers and street addresses visible in the images are **not** copied into the structured
data. These are real patient documents: do not publish this repository.

---

## 2. Architecture (maps to the methodology slide, `docs/methodology.png`)

```
images (root) ──► 1. Ingestion & processing ──► 2. Recommendation engine (LLM + RAG) ──► 3. Safety & risk checks
                     ingestion.py                   rag.py, llm.py, summarizer.py            safety.py, triage.py,
                     preprocessing.py               recommender.py                           risk_model.py
                     features.py                                   │                                │
                         └──────────────► 4. Timeline & visualisation (timeline.py) ◄──────────────┘
                                                        │
                                       5. Streamlit dashboard (app.py)
```

| Methodology block | Module | What it does |
|---|---|---|
| Data ingestion from multiple sources | `cdss/ingestion.py`, `cdss/ocr.py` | Loads the record store. New images are read with Claude vision, falling back to Tesseract and then manual entry. Images are saved to the root folder. |
| Data fusion / unified patient ID | `ingestion.match_patient` | Matches records on UHID or hospital number first, then on fuzzy name and alias matching. Leela's three name variants across three facilities are merged this way, and an age inconsistency is flagged. |
| Preprocessing & cleaning | `cdss/preprocessing.py` | Maps brand names to generic components and drug classes (100+ entries). Parses `1-0-1`, `½-0-0`, `18U-0-18U`, HS, SOS, stat and weekly doses, plus durations. Derives start and end dates, handles renewals and supersession, and produces a data-quality report. |
| Feature engineering | `cdss/features.py` | Builds the medication-change history (started, stopped, switched within class, dose changed), daily polypharmacy counts, antibiotic courses, recurring diagnoses, vitals trends and cohort features. |
| LLM-based summarisation | `cdss/summarizer.py` | Writes a deterministic structured summary and checks follow-up adherence (scheduled vs. recorded visits). An optional LLM narrative cites encounter and guideline ids. |
| RAG pipeline | `cdss/rag.py` + `data/knowledge_base/guidelines.md` | Uses TF-IDF retrieval over 29 guideline chunks (NICE, Beers 2023, FDA, EMA, ADA, NEWS2 and others) and over the patient's own encounters. |
| Personalised recommendations + explainability | `cdss/recommender.py` | Puts every candidate drug through the safety engine for this patient and ranks it Preferred, Use with caution or Avoid. Each rating comes with its reasons and evidence ids. |
| Multi-timeline safety check | `cdss/safety.py` + `clinical_rules.json` | Covers drug-drug interactions (pairwise and the 3-drug "triple whammy"), same-ingredient and same-class duplication, condition contraindications, Beers age rules, allergy cross-reactivity and cumulative paracetamol dose. Only drugs whose active periods overlapped are compared. |
| Risk scoring & alerts | `safety.risk_level`, `triage.py` | Rates alerts Low, Moderate or High. Triage uses NEWS2 vitals, red-flag symptoms (from the discharge template) and patient context, and returns EMERGENCY, URGENT or ROUTINE with the reason for each point. |
| Disease-risk prediction (ML) | `cdss/risk_model.py` | Logistic regression vs. gradient boosting with 5-fold cross-validated ROC-AUC (about 0.80-0.82) and permutation importance. Features: age, BMI, mean arterial pressure and glucose. |
| Timeline & visualisation | `cdss/timeline.py` | Medication Gantt chart with HIGH-alert windows shaded, a visit map, vitals trends and a polypharmacy curve. |
| Dashboard | `app.py` | Seven tabs, with real-time banners for alerts that are active on the chosen date. |

### Examples the system finds in the supplied prescriptions

* **Philomena R** (two clinics within 1 week):
  * glimepiride (Glycomet GP) from the physician + **ciprofloxacin** from the surgeon: dysglycaemia risk in a 72-year-old [High]
  * diclofenac + telmisartan + torsemide: **triple whammy** AKI risk [High]
  * metformin in **two** combination tablets [High]
  * diclofenac + heparin [High]
  * depot steroid in a diabetic patient [Moderate]
* **Sagay Josephine Leela (2018):** clarithromycin + domperidone (Nexpro RD), a QT-prolongation combination that is contraindicated [High].
* **Anthony Reddy:** pregabalin + tramadol (Ultracet), a CNS and respiratory depression warning [High], plus paracetamol in two products.
* **Asuntha Anandraj:**
  * amlodipine + cilnidipine, two calcium-channel blockers
  * insulin + gliclazide + metoprolol, a hypoglycaemia risk that metoprolol can mask
  * chronic liver disease limits paracetamol and sulfonylureas
  * an automatically detected switch from NovoMix to Huminsulin
* **Deva Prasad:** oral vitamin D3 60K + Arachitol 6 lakh IU injection, a duplicate vitamin D loading.

---

## 3. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                     # dashboard at http://localhost:8501
python run_report.py                                     # CLI pipeline -> reports/*.md
python -m pytest -q                                      # 10 unit tests
```

### Optional: enable the LLM (summaries, explanations, handwriting OCR)

```bash
export ANTHROPIC_API_KEY=sk-ant-...          # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
export CDSS_LLM_MODEL=claude-sonnet-4-5      # any available Claude model id
```

Without a key, every feature still works using the rule engine, templates and extractive retrieval.
The LLM is only ever given the retrieved patient record and guideline excerpts. It is told
to cite them and not to invent facts.

---

## 4. Project structure

```
cdss_project/
├── 01_...jpg … 16_...jpg          # source prescription images (primary data, root folder)
├── app.py                         # Streamlit dashboard (block 5)
├── run_report.py                  # batch pipeline / reports
├── requirements.txt
├── cdss/
│   ├── ingestion.py               # load/save, unified patient ID, store new images in root
│   ├── ocr.py                     # image -> structured encounter (Claude vision / Tesseract)
│   ├── preprocessing.py           # drug normalisation, dose/duration parsing, data quality
│   ├── features.py                # sequential features, medication change detection
│   ├── rag.py                     # retrieval over guidelines + patient record
│   ├── llm.py                     # optional Anthropic client
│   ├── summarizer.py              # longitudinal summary, follow-up audit
│   ├── recommender.py             # context-aware, explainable options
│   ├── safety.py                  # multi-timeline safety engine
│   ├── triage.py                  # EMERGENCY / URGENT / ROUTINE
│   ├── risk_model.py              # ML disease-progression model
│   └── timeline.py                # Plotly visualisations
├── data/
│   ├── patients.json              # 7 patients, 16 encounters, 76 medication entries
│   └── knowledge_base/
│       ├── drug_dictionary.json   # brand -> generic components
│       ├── clinical_rules.json    # classes, interactions, contraindications, allergy map
│       └── guidelines.md          # RAG corpus (29 cited chunks)
├── docs/
│   ├── methodology.png            # methodology diagram from the Review-0 slides
│   └── review0_presentation.pptx
├── reports/                       # generated by run_report.py
└── tests/test_pipeline.py
```

## 5. Limitations and next steps (Phase 2)

* The rule base is educational and hand-curated. For deployment it would be replaced or extended
  with a licensed interaction database (e.g. DrugBank or the Indian NLEM-linked formulary).
* The risk model is trained on the public scikit-learn diabetes-progression dataset (442 patients),
  not on local data. It needs retraining and validation on the clinic's own records. Glucose values
  outside the training range are flagged.
* Several handwritten entries are only partly legible. A clinician has to verify them before any
  real use.
* Planned next steps:
  * replace TF-IDF with clinical sentence embeddings
  * add a lab-report parser
  * build a prospective triage validation set
  * add role-based access and audit logging
#   c d s s  
 