from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

try:
    # optional, but convenient for local dev
    from dotenv import load_dotenv  # type: ignore

    # Load .env from both repo root and backend folder (whichever exists),
    # so it works whether server starts from root or backend directory.
    _repo = Path(__file__).resolve().parents[2]
    load_dotenv(_repo / ".env")
    load_dotenv(_repo / "backend" / ".env")
except Exception:
    pass

try:
    from openai import OpenAI  # type: ignore
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore
    _OPENAI_IMPORT_ERROR = e
else:
    _OPENAI_IMPORT_ERROR = None


def _repo_root() -> Path:
    # backend/app/ai_assistant.py -> repo root is 2 levels up: app -> backend -> repo
    return Path(__file__).resolve().parents[2]


def _class_names() -> list[str]:
    class_names_path = (
        Path(os.getenv("CLASS_NAMES_PATH", str(_repo_root() / "saved_models" / "class_names.json")))
        .expanduser()
        .resolve()
    )
    try:
        with class_names_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        pass
    return ["Acne", "Eczema", "Psoriasis", "SkinCancer", "Vitiligo", "Warts"]


def _system_prompt(class_names: list[str]) -> str:
    return (
        "You are a dermatology information assistant embedded in a student project UI. "
        "Use a friendly, casual tone.\n\n"
        "Rules:\n"
        "- Do NOT claim to diagnose. This is informational only.\n"
        "- Start replies with a short casual greeting (example: 'Hi!' or 'Hey there!').\n"
        "- Keep replies short, natural, and conversational.\n"
        "- Do not give long disease descriptions unless the user explicitly asks for detailed explanation.\n"
        "- Include a short disclaimer only when giving medical guidance.\n"
        "- If the user describes alarming symptoms (rapid growth, bleeding, irregular border, "
        "severe pain, fever, spreading rash, immunocompromised), advise urgent medical care.\n"
        "- Prefer plain sentences over heavy formatting.\n\n"
        f"The model's dataset classes are: {', '.join(class_names)}.\n"
        "When asked about 'skin cancer', discuss general warning signs and next steps.\n"
    )


def ask_ai(
    *,
    question: str,
    predicted_class: Optional[str] = None,
    top_predictions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    api_key = groq_key or openai_key
    if not api_key:
        return {
            "error": (
                "Missing API key. Set GROQ_API_KEY (preferred) or OPENAI_API_KEY "
                "in your environment (do not hardcode it)."
            ),
        }

    if OpenAI is None:
        return {
            "error": f"openai package not available: {_OPENAI_IMPORT_ERROR}",
        }

    class_names = _class_names()
    is_groq = bool(groq_key)
    model_name = (
        os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if is_groq
        else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    context_bits: list[str] = []
    if predicted_class:
        context_bits.append(f"Predicted class: {predicted_class}")
    if top_predictions:
        # Keep only small summary to avoid huge payloads.
        trimmed = top_predictions[:5]
        context_bits.append(f"Top predictions: {trimmed}")
    context = "\n".join(context_bits).strip()

    user_content = question.strip()
    if context:
        user_content = f"{user_content}\n\nContext from classifier:\n{context}"

    # Groq supports an OpenAI-compatible API surface.
    client = (
        OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        if is_groq
        else OpenAI(api_key=api_key)
    )
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": _system_prompt(class_names)},
            {"role": "user", "content": user_content},
        ],
    )
    text = resp.choices[0].message.content or ""
    return {"answer": text}

