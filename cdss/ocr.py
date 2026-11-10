"""Prescription image -> structured encounter (ingestion of NEW records).

Extraction strategy (tried in priority order):
  1. Cloud LLM vision  (GEMINI_API_KEY or ANTHROPIC_API_KEY in .env)
     Handles handwriting well; uses the structured prompt below.
  2. Local Ollama vision model  (OLLAMA_MODEL set in .env, Ollama running locally)
     Fully offline, near-cloud accuracy on modern hardware (Qwen2.5-VL, MiniCPM-V, etc.)
  3. PaddleOCR  (pip install paddlepaddle paddleocr)
     Deep-learning OCR, runs on CPU, much better than Tesseract for printed forms.
     Followed by regex-based slot extraction and medical lexicon fuzzy correction.
  4. Tesseract OCR  (pip install pytesseract + Windows Tesseract binary)
     Classical OCR fallback; works for clean printed forms.
     Followed by the same regex / lexicon post-processing as PaddleOCR.
  5. Manual entry in the dashboard.

All OCR paths (3 & 4) run the same OpenCV image pre-processing pipeline first
(dewarping → CLAHE contrast → Sauvola binarisation) to maximise text recognition.

Every extracted record is shown to the clinician for verification before it is saved.

CLI:
    python -m cdss.ocr path/to/image.jpg            # prints extracted JSON
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from . import llm

# ---------------------------------------------------------------------------
# Structured extraction prompt (identical across cloud & Ollama paths)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# STAGE 0: OpenCV image pre-processing
# ---------------------------------------------------------------------------
def _preprocess_image(image_path: str | Path):
    """Apply a 4-step OpenCV pipeline to improve OCR accuracy.

    Steps:
      1. Grayscale conversion
      2. Denoising (fastNlMeansDenoising)
      3. CLAHE contrast enhancement (equalises uneven clinic lighting)
      4. Sauvola-style adaptive binarisation (isolates ink from paper texture)

    Returns a uint8 numpy array (H x W) or None if cv2 is not installed.
    """
    try:
        import importlib
        cv2 = importlib.import_module("cv2")
        np = importlib.import_module("numpy")

        img = cv2.imread(str(image_path))
        if img is None:
            # Fallback: load with Pillow, convert to numpy
            from PIL import Image
            pil = Image.open(image_path).convert("RGB")
            img = np.array(pil)[:, :, ::-1]  # RGB -> BGR

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mild denoising without blurring fine pen strokes
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # CLAHE: locally equalise contrast (helps with uneven clinic lighting and shadows)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Adaptive Gaussian threshold (Sauvola-style): binarise ink vs paper
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=12,
        )
        return binary
    except Exception:
        return None


def _save_preprocessed(image_path: str | Path) -> str:
    """Pre-process and save to a temp file; return its path as a string."""
    import tempfile
    processed = _preprocess_image(image_path)
    if processed is None:
        return str(image_path)
    try:
        import importlib
        cv2 = importlib.import_module("cv2")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, processed)
        return tmp.name
    except Exception:
        return str(image_path)


# ---------------------------------------------------------------------------
# STAGE 1: Cloud LLM vision (Gemini / Claude)
# ---------------------------------------------------------------------------
def extract_with_llm(image_path: str | Path) -> dict:
    return llm.vision_json(image_path, EXTRACTION_INSTRUCTION)


# ---------------------------------------------------------------------------
# STAGE 2: Ollama local vision model (fully offline, near-cloud accuracy)
# ---------------------------------------------------------------------------
def _ollama_available() -> bool:
    llm.load_env()
    return bool(
        (import_or_none("requests") is not None)
        and (
            os.environ.get("OLLAMA_MODEL")
            or _ollama_running()
        )
    )


def _ollama_running() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def extract_with_ollama(image_path: str | Path) -> dict:
    """Send image to a locally running Ollama vision model and parse JSON output.

    Set OLLAMA_MODEL in .env, e.g.:
        OLLAMA_MODEL=qwen2.5-vl:7b
        OLLAMA_MODEL=minicpm-v:8b
        OLLAMA_MODEL=moondream:latest
    """
    import base64
    import os
    import requests

    llm.load_env()
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5-vl:7b").strip()
    path = Path(image_path)
    img_data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": EXTRACTION_INSTRUCTION,
        "images": [img_data],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2500},
        "system": "You transcribe Indian medical prescriptions into structured JSON. Output JSON only.",
        "format": "json",
    }
    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error ({resp.status_code}): {resp.text[:200]}")

    text = resp.json().get("response", "")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}


# ---------------------------------------------------------------------------
# STAGE 3: PaddleOCR + post-processing (offline deep learning OCR)
# ---------------------------------------------------------------------------
_MEDICAL_LEXICON = [
    # Common Indian brand names and generics (extend as needed)
    "paracetamol", "dolo", "calpol", "ibuprofen", "aspirin", "diclofenac", "aceclofenac",
    "amoxicillin", "augmentin", "cefuroxime", "cephalexin", "azithromycin", "clarithromycin",
    "ciprofloxacin", "levofloxacin", "metronidazole", "clindamycin", "doxycycline",
    "pantoprazole", "omeprazole", "esomeprazole", "rabeprazole", "razo",
    "metformin", "glimepiride", "gliclazide", "insulin", "linagliptin", "sitagliptin",
    "glycomet", "glycomet gp", "reclide", "januvia", "tradjenta",
    "amlodipine", "amlong", "cilnidipine", "cilacar", "telmisartan", "losartan",
    "metoprolol", "atenolol", "carvedilol", "ramipril", "enalapril",
    "torsemide", "furosemide", "hydrochlorothiazide",
    "levothyroxine", "thyronorm", "eltroxin",
    "prednisolone", "methylprednisolone", "dexamethasone", "betamethasone",
    "pregabalin", "gabapentin", "tramadol", "ultracet", "tapentadol",
    "cetirizine", "loratadine", "fexofenadine", "levocetrizine",
    "domperidone", "ondansetron", "metoclopramide",
    "dalteparin", "heparin", "warfarin", "clopidogrel",
    "atorvastatin", "rosuvastatin", "shelcal", "calcium carbonate", "vitamin d",
    "mebeverine", "amitriptyline", "sertraline", "alprazolam", "clonazepam",
    "arachitol", "cholecalciferol",
]


def _fuzzy_correct(word: str, threshold: float = 0.75) -> str:
    """Return the closest medical lexicon match if similarity >= threshold."""
    from difflib import SequenceMatcher
    word_lc = word.lower().strip()
    best_match, best_score = word_lc, 0.0
    for term in _MEDICAL_LEXICON:
        score = SequenceMatcher(None, word_lc, term).ratio()
        if score > best_score:
            best_score, best_match = score, term
    return best_match if best_score >= threshold else word_lc


_FREQ_RE = re.compile(
    r"\b(\d+[\-\/]\d+[\-\/]\d+)\b"       # 1-0-1, ½-0-0, 18U-0-18U
    r"|\b(SOS|PRN|stat|HS|BD|TDS|OD|QID)\b",
    re.IGNORECASE,
)
_STRENGTH_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|µg|g|ml|units?|IU|%)\b",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"\b(?:x|for)?\s*(\d+)\s*(days?|weeks?|months?)\b"
    r"|\b(continue|chronic|long.?term)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b"
)


def _extract_slots_from_text(raw_text: str) -> dict:
    """Parse Indian prescription slots from raw OCR text using regex."""
    meds = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or len(line) < 4:
            continue
        # Heuristic: a medication line usually starts with a drug token and has a frequency
        freq_m = _FREQ_RE.search(line)
        str_m = _STRENGTH_RE.search(line)
        if not (freq_m or str_m):
            continue  # Skip lines with no dosing cues

        # Extract drug name: first alpha token before the first number/strength/freq
        drug_token = re.split(r"\d", line)[0].strip().rstrip(".,:")
        drug = _fuzzy_correct(drug_token) if len(drug_token) > 2 else drug_token.lower()

        strength = str_m.group(0) if str_m else ""
        frequency = (freq_m.group(1) or freq_m.group(2) or "").upper() if freq_m else ""
        dur_m = _DURATION_RE.search(line)
        duration = ""
        if dur_m:
            if dur_m.group(3):
                duration = "continue"
            elif dur_m.group(1):
                duration = f"{dur_m.group(1)} {dur_m.group(2)}"

        meds.append({
            "raw": line,
            "drug": drug,
            "strength": strength,
            "frequency": frequency,
            "duration": duration,
            "legibility": "partial",
        })

    # Try to find a date in the full text
    date_str = ""
    date_m = _DATE_RE.search(raw_text)
    if date_m:
        d, m_, y = date_m.groups()
        y_int = int(y) + (2000 if len(y) == 2 else 0)
        try:
            date_str = f"{y_int:04d}-{int(m_):02d}-{int(d):02d}"
        except Exception:
            pass

    return {
        "date": date_str,
        "medications": meds,
        "note": "Extracted by offline OCR + regex. Verify all entries against the original image.",
    }


def extract_with_paddleocr(image_path: str | Path) -> dict:
    """Deep-learning OCR via PaddleOCR followed by regex/lexicon post-processing.

    Install: pip install paddlepaddle paddleocr
    """
    import importlib
    paddle_mod = importlib.import_module("paddleocr")
    PaddleOCR = getattr(paddle_mod, "PaddleOCR")

    proc_path = _save_preprocessed(image_path)
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr.ocr(proc_path, cls=True)
    lines = []
    if result and result[0]:
        for block in result[0]:
            # block: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)
            text, conf = block[1]
            if conf >= 0.4:
                lines.append(text)
    raw_text = "\n".join(lines)
    slots = _extract_slots_from_text(raw_text)
    slots["raw_text"] = raw_text
    slots["engine"] = "paddleocr"
    return slots


# ---------------------------------------------------------------------------
# STAGE 4: Tesseract OCR + post-processing (classical fallback)
# ---------------------------------------------------------------------------
def extract_with_tesseract(image_path: str | Path) -> dict:
    import importlib
    pytesseract = importlib.import_module("pytesseract")
    from PIL import Image

    proc_path = _save_preprocessed(image_path)
    # Use the pre-processed numpy array directly if available, otherwise fallback
    processed = _preprocess_image(image_path)
    if processed is not None:
        from PIL import Image as PILImage
        import numpy as np
        pil_img = PILImage.fromarray(processed)
    else:
        pil_img = Image.open(image_path).convert("L")

    # PSM 6 = Assume a single uniform block of text (good for prescriptions)
    custom_cfg = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(pil_img, config=custom_cfg)
    slots = _extract_slots_from_text(text)
    slots["raw_text"] = text
    slots["engine"] = "tesseract"
    return slots


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def import_or_none(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None


import os


# ---------------------------------------------------------------------------
# Main entry-point: try each engine in priority order
# ---------------------------------------------------------------------------
def extract(image_path: str | Path) -> dict:
    """Try Cloud LLM (Gemini primary) -> Ollama local fallback -> PaddleOCR -> Tesseract -> manual fallback."""
    errors = []

    # 1. Cloud LLM (Gemini / Claude) primary
    if llm.available_cloud():
        try:
            return {"engine": llm.provider_name(), **extract_with_llm(image_path)}
        except Exception as exc:
            errors.append(f"Cloud LLM ({llm.provider_name()}): {exc}")

    # 2. Local Ollama fallback if cloud key missing or failed
    if _ollama_running():
        try:
            return {"engine": f"ollama/{os.environ.get('OLLAMA_MODEL', 'qwen2.5:3b')}", **extract_with_ollama(image_path)}
        except Exception as exc:
            errors.append(f"Ollama: {exc}")

    # 3. PaddleOCR (deep-learning offline OCR)
    if import_or_none("paddleocr") is not None:
        try:
            return extract_with_paddleocr(image_path)
        except Exception as exc:
            errors.append(f"PaddleOCR: {exc}")

    # 4. Tesseract (classical OCR)
    if import_or_none("pytesseract") is not None:
        try:
            result = extract_with_tesseract(image_path)
            result["warning"] = "; ".join(errors) if errors else ""
            return result
        except Exception as exc:
            errors.append(f"Tesseract: {exc}")

    # 5. No engine available
    return {
        "engine": "none",
        "warning": " | ".join(errors) or "No OCR engine available. Install paddleocr or pytesseract.",
        "medications": [],
    }


# ---------------------------------------------------------------------------
# Convert raw extraction dict -> encounter record for patients.json
# ---------------------------------------------------------------------------
def to_encounter(extracted: dict, source_image: str) -> dict:
    keys = ["date", "date_certain", "facility", "department", "doctor", "type", "diagnoses", "complaints",
            "vitals", "medications", "investigations_advised", "plan"]
    enc = {k: extracted.get(k) for k in keys if extracted.get(k) not in (None, "")}
    enc.setdefault("type", "prescription")
    enc.setdefault("date_certain", bool(extracted.get("date")))
    enc["source_image"] = source_image
    enc["reconciles_chronic_meds"] = False
    return enc


# ---------------------------------------------------------------------------
# CLI usage: python -m cdss.ocr path/to/image.jpg
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(json.dumps(extract(sys.argv[1]), indent=2, ensure_ascii=False))
