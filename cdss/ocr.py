"""Prescription image -> structured encounter (ingestion of NEW records).

Strategy
  1. Claude vision (if ANTHROPIC_API_KEY is set) - handles handwriting well.
  2. Tesseract OCR (if `pytesseract` + tesseract binary are installed) - printed forms only;
     returns raw text, medications must then be filled in manually.
  3. Manual entry in the dashboard.

Every extracted record is shown to the clinician for verification before it is saved.

CLI:
    python -m cdss.ocr path/to/image.jpg            # prints extracted JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import llm

EXTRACTION_INSTRUCTION = """Transcribe this medical prescription / discharge summary into JSON with keys:
{
 "patient_name": str, "patient_identifier": str (UHID / hospital no / IP no, or ""), "age": str, "sex": "M"|"F"|"",
 "date": "YYYY-MM-DD" or "", "date_certain": bool,
 "facility": str, "department": str, "doctor": str,
 "type": "opd_prescription"|"discharge_summary"|"clinic_prescription"|"pharmacy_indent"|"prescription",
 "diagnoses": [str], "complaints": [str],
 "vitals": {"bp_systolic": int, "bp_diastolic": int, "pulse": int, "spo2": int, "weight_kg": float,
            "temperature_f": float, "random_glucose_mg_dl": int}  (only those present),
 "medications": [{"raw": exact text, "drug": brand or generic name in lower case, "strength": str,
                  "route": "PO"|"IV"|"IM"|"SC"|"topical", "frequency": e.g. "1-0-1", "HS", "SOS", "stat",
                  "duration": e.g. "5 days", "2 weeks", "1 month", "continue", "instructions": str,
                  "legibility": "clear"|"partial"|"illegible"}],
 "investigations_advised": [str], "plan": [str]
}
Rules: copy what is written; never guess an illegible drug name - set legibility "illegible" and keep raw text.
Indian notation 1-0-1 means morning-afternoon-night. Output JSON only."""


def extract_with_llm(image_path: str | Path) -> dict:
    return llm.vision_json(image_path, EXTRACTION_INSTRUCTION)


def extract_with_tesseract(image_path: str | Path) -> dict:
    import importlib
    pytesseract = importlib.import_module("pytesseract")
    from PIL import Image, ImageOps

    img = ImageOps.grayscale(Image.open(image_path))
    text = pytesseract.image_to_string(img)
    return {"raw_text": text, "medications": [], "note": "Tesseract output - fill medications manually."}


def extract(image_path: str | Path) -> dict:
    if llm.available():
        try:
            return {"engine": llm.provider_name(), **extract_with_llm(image_path)}
        except Exception as exc:  # fall through to tesseract
            err = str(exc)
        else:
            err = ""
    else:
        err = "No LLM API key set in .env (GEMINI_API_KEY or ANTHROPIC_API_KEY)"
    try:
        return {"engine": "tesseract", "warning": err, **extract_with_tesseract(image_path)}
    except Exception as exc:
        return {"engine": "none", "warning": f"{err}; tesseract unavailable: {exc}", "medications": []}


def to_encounter(extracted: dict, source_image: str) -> dict:
    keys = ["date", "date_certain", "facility", "department", "doctor", "type", "diagnoses", "complaints",
            "vitals", "medications", "investigations_advised", "plan"]
    enc = {k: extracted.get(k) for k in keys if extracted.get(k) not in (None, "")}
    enc.setdefault("type", "prescription")
    enc.setdefault("date_certain", bool(extracted.get("date")))
    enc["source_image"] = source_image
    enc["reconciles_chronic_meds"] = False
    return enc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(json.dumps(extract(sys.argv[1]), indent=2, ensure_ascii=False))
