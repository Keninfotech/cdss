"""LLM backend for summarisation, RAG answers and prescription OCR.

Supported providers (priority order):
  1. Google Gemini  (GEMINI_API_KEY in .env  — free tier at aistudio.google.com)
  2. Anthropic Claude  (ANTHROPIC_API_KEY in .env)
  3. Ollama  (local model — OLLAMA_MODEL in .env, e.g. qwen2.5-vl:7b)
  
Everything in the CDSS works WITHOUT any API key — template / extractive fallbacks are used.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant helping a licensed doctor in a rural Indian clinic. "
    "Use ONLY the patient record and guideline excerpts provided. Cite evidence with the bracketed ids "
    "given, e.g. [G-FQ] or [P002-E1]. Never invent drugs, doses, results or history. If information is "
    "missing or illegible, say so. Present options for the clinician to consider, not orders. Be concise."
)


def load_env() -> None:
    """Read key-value pairs from .env in the project root if present."""
    root = Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and not os.environ.get(k):
                        os.environ[k] = v
        except Exception:
            pass


load_env()


def available() -> bool:
    load_env()
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or _ollama_running())


def available_cloud() -> bool:
    load_env()
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def _ollama_running() -> bool:
    """Check if a local Ollama instance is reachable."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def provider_name() -> str:
    load_env()
    if os.environ.get("GEMINI_API_KEY"):
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        return f"Gemini ({model})"
    if os.environ.get("ANTHROPIC_API_KEY"):
        model = os.environ.get("CDSS_LLM_MODEL", "claude-sonnet-4-5")
        return f"Claude ({model})"
    if _ollama_running():
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
        return f"Ollama ({model}) [offline]"
    return "offline (rule/template mode)"


def _gemini_generate(parts: list, system: str | None = None, response_json: bool = False,
                     max_tokens: int = 2500, model: str | None = None) -> str:
    import requests

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload: dict = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.1 if response_json else 0.2,
        },
    }
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}
    if response_json:
        payload["generationConfig"]["response_mime_type"] = "application/json"

    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates: {data}")
    return candidates[0]["content"]["parts"][0].get("text", "").strip()


def _anthropic_client():
    import importlib
    anthropic = importlib.import_module("anthropic")
    return anthropic.Anthropic()


def _ollama_complete(prompt: str, system: str = "", max_tokens: int = 900, model: str | None = None) -> str:
    """Send a text prompt to the locally running Ollama instance."""
    import requests
    model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5-vl:7b")
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": max_tokens},
    }
    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error ({resp.status_code}): {resp.text[:200]}")
    return resp.json().get("response", "").strip()


def complete(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 900, model: str | None = None) -> str:
    load_env()
    if os.environ.get("GEMINI_API_KEY"):
        return _gemini_generate([{"text": prompt}], system=system, max_tokens=max_tokens, model=model)
    elif os.environ.get("ANTHROPIC_API_KEY"):
        model = model or os.environ.get("CDSS_LLM_MODEL", "claude-sonnet-4-5")
        msg = _anthropic_client().messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    elif _ollama_running():
        return _ollama_complete(prompt, system=system, max_tokens=max_tokens, model=model)
    raise RuntimeError("No LLM available. Set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env, or start Ollama.")


def vision_json(image_path: str | Path, instruction: str, max_tokens: int = 2500, model: str | None = None) -> dict:
    """Send an image + instruction, expect a JSON object back."""
    load_env()
    path = Path(image_path)
    media = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

    if os.environ.get("GEMINI_API_KEY"):
        parts = [
            {"inline_data": {"mime_type": media, "data": data}},
            {"text": instruction},
        ]
        text = _gemini_generate(
            parts,
            system="You transcribe Indian medical prescriptions into structured JSON. Output JSON only.",
            response_json=True,
            max_tokens=max_tokens,
            model=model,
        )
    elif os.environ.get("ANTHROPIC_API_KEY"):
        model = model or os.environ.get("CDSS_LLM_MODEL", "claude-sonnet-4-5")
        msg = _anthropic_client().messages.create(
            model=model, max_tokens=max_tokens,
            system="You transcribe Indian medical prescriptions into structured JSON. Output JSON only.",
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
                {"type": "text", "text": instruction},
            ]}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    elif _ollama_running():
        # Ollama vision: handled in ocr.py's extract_with_ollama (uses /api/generate with images)
        # For the llm.py path, fall back to text-only with the instruction
        text = _ollama_complete(instruction, system="Output JSON only.", max_tokens=max_tokens, model=model)
    else:
        raise RuntimeError("No LLM available. Set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env, or start Ollama.")

    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}
