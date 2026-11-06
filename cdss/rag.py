"""Block 2a - Retrieval layer of the RAG pipeline.

Two indexes are searched with TF-IDF cosine similarity (works fully offline):
  * the clinical knowledge base (data/knowledge_base/guidelines.md, one chunk per section)
  * the patient's own record, chunked per encounter, so answers are anchored to the
    specific health record ("patient context").
Swap `TfidfRetriever` for a sentence-embedding retriever if a GPU / internet is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import KB_DIR


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    source: str  # "guideline" | "patient"

    def cite(self) -> str:
        return f"[{self.id}]"


def load_guideline_chunks(path=KB_DIR / "guidelines.md") -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    chunks = []
    for m in re.finditer(r"^## \[(?P<id>[^\]]+)\] (?P<title>.+?)\n(?P<body>.*?)(?=^## \[|\Z)", text, re.S | re.M):
        chunks.append(Chunk(m["id"], m["title"].strip(), " ".join(m["body"].split()), "guideline"))
    return chunks


def patient_chunks(patient: dict) -> list[Chunk]:
    out = []
    for e in patient["encounters"]:
        meds = "; ".join(m.get("raw", "") for m in e.get("medications", []))
        parts = [
            f"{e['date']} {e.get('department', '')} at {e.get('facility', '')} by {e.get('doctor', '')}.",
            f"Diagnoses: {', '.join(e.get('diagnoses', [])) or 'not stated'}.",
            f"Complaints: {', '.join(e.get('complaints', [])) or 'not stated'}.",
            f"Vitals: {e.get('vitals') or 'not recorded'}.",
            f"Medications: {meds or 'none'}.",
            f"Plan: {'; '.join(e.get('plan', [])) or 'none'}.",
            f"Investigations advised: {', '.join(e.get('investigations_advised', [])) or 'none'}.",
        ]
        out.append(Chunk(e["encounter_id"], f"{e['date']} - {e.get('department', '')}", " ".join(parts), "patient"))
    conds = "; ".join(f"{c['name']} ({c['source']})" for c in patient.get("conditions", []))
    out.append(Chunk(f"{patient['patient_id']}-PROFILE", "Patient profile",
                     f"{patient['name']}, sex {patient.get('sex')}, DOB {patient.get('dob')}. Conditions: {conds}. "
                     f"Allergies: {', '.join(patient.get('allergies', [])) or patient.get('allergy_status', 'unknown')}.",
                     "patient"))
    return out


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([f"{c.title} {c.text}" for c in chunks]) if chunks else None

    def search(self, query: str, k: int = 4, min_score: float = 0.03) -> list[tuple[Chunk, float]]:
        if not self.chunks or not query.strip():
            return []
        sims = cosine_similarity(self.vec.transform([query]), self.matrix).ravel()
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order if sims[i] >= min_score]


class ClinicalRAG:
    """Retrieves guideline evidence + patient-record context for a query."""

    def __init__(self):
        self.guidelines = load_guideline_chunks()
        self.by_id = {c.id: c for c in self.guidelines}
        self.kb = TfidfRetriever(self.guidelines)

    def guideline(self, ref: str) -> Chunk | None:
        return self.by_id.get(ref)

    def retrieve(self, query: str, patient: dict | None = None, k_kb: int = 4, k_patient: int = 3) -> dict:
        kb_hits = self.kb.search(query, k_kb)
        pt_hits = TfidfRetriever(patient_chunks(patient)).search(query, k_patient, 0.0) if patient else []
        return {"guidelines": kb_hits, "patient": pt_hits}
