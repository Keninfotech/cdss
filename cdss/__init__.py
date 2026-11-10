"""AI-Based Clinical Decision Support System (CDSS).

Modules follow the five blocks of the project methodology diagram (docs/methodology.png):

1. Clinical data ingestion & processing  -> ingestion, preprocessing, features
2. Clinical recommendation engine (LLM+RAG) -> rag, llm, summarizer, recommender
3. Dynamic safety & risk checks           -> safety, triage, risk_model
4. Historical timeline & visualisation    -> timeline
5. Clinical decision support dashboard    -> ../app.py (Streamlit)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
PATIENTS_FILE = DATA_DIR / "patients.json"
IMAGE_DIR = PROJECT_ROOT / "prescriptions"  # source prescription images are kept in the prescriptions folder

DISCLAIMER = (
    "Clinical decision SUPPORT only. Outputs are generated from transcribed records and an "
    "educational rule base; every alert and suggestion must be verified by the treating "
    "clinician against the source documents and current guidelines."
)
