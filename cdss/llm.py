"""Optional LLM backend (Anthropic Claude) for summarisation, RAG answers and prescription OCR.

Everything in the CDSS works WITHOUT an API key - template / extractive fallbacks are used.
Set ANTHROPIC_API_KEY (and optionally CDSS_LLM_MODEL) to enable generative features.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path

DEFAULT_MODEL = os.environ.get("CDSS_LLM_MODEL", "claude-sonnet-4-5")

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant helping a licensed doctor in a rural Indian clinic. "
    "Use ONLY the patient record and guideline excerpts provided. Cite evidence with the bracketed ids "
    "given, e.g. [G-FQ] or [P002-E1]. Never invent drugs, doses, results or history. If information is "
    "missing or illegible, say so. Present options for the clinician to consider, not orders. Be concise."
)


def available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _client():
    import anthropic
    return anthropic.Anthropic()


def complete(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 900, model: str | None = None) -> str:
    msg = _client().messages.create(
        model=model or DEFAULT_MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def vision_json(image_path: str | Path, instruction: str, max_tokens: int = 2500, model: str | None = None) -> dict:
    """Send an image + instruction, expect a JSON object back."""
    path = Path(image_path)
    media = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.standard_b64encode(path.read_bytes()).decode()
    msg = _client().messages.create(
        model=model or DEFAULT_MODEL, max_tokens=max_tokens,
        system="You transcribe Indian medical prescriptions into structured JSON. Output JSON only.",
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
            {"type": "text", "text": instruction},
        ]}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}
