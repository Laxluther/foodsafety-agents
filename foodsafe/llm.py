"""Model access.

Two providers are supported. Gemini is the default because it answers in seconds
where a 26B model offloaded to CPU takes minutes, and because only one of four
locally installed models could emit a tool call at all.

The Ollama path is kept, not deleted: it is what makes this runnable with no API
key and no data leaving the machine, which matters for anyone evaluating a food
safety tool on unpublished results. Set FOODSAFE_PROVIDER=ollama to use it.
"""

import os
from pathlib import Path

import requests

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = _ENV_FILE) -> None:
    """Read KEY=value lines from .env without adding a dependency.

    Existing environment variables win, so a shell export overrides the file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env()

_CONFIGURED_PROVIDER = os.environ.get("FOODSAFE_PROVIDER", "gemini").lower()

GEMINI_MODEL = os.environ.get("FOODSAFE_GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- Ollama (previous default, still supported) ---------------------------
OLLAMA_MODEL = os.environ.get("FOODSAFE_MODEL", "gemma4:26b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

DEFAULT_BASE_URL = OLLAMA_BASE_URL


class LLMUnavailable(RuntimeError):
    """The configured provider is unreachable or misconfigured."""


def gemini_usable() -> tuple[bool, str]:
    """Can Gemini actually generate, not merely authenticate?

    A valid key still fails at generation time when the account has no credit,
    and listing models succeeds regardless - so the check has to be a real
    generateContent call.
    """
    if not GEMINI_API_KEY:
        return False, "GEMINI_API_KEY is not set"
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "ok"}]}],
                  "generationConfig": {"maxOutputTokens": 1}},
            headers={"x-goog-api-key": GEMINI_API_KEY},
            timeout=30,
        )
    except requests.RequestException as exc:
        return False, f"unreachable: {exc}"

    if response.status_code == 200:
        return True, ""
    if response.status_code == 429:
        return False, "quota or prepayment credits exhausted"
    return False, f"HTTP {response.status_code}"


def _resolve_provider() -> tuple[str, str]:
    """Pick the provider that can actually answer, and say why if it changed."""
    if _CONFIGURED_PROVIDER != "gemini":
        return _CONFIGURED_PROVIDER, ""
    if os.environ.get("FOODSAFE_SKIP_PROBE"):
        return "gemini", ""
    usable, reason = gemini_usable()
    if usable:
        return "gemini", ""
    if installed_models():
        return "ollama", f"Gemini unavailable ({reason}); using local Ollama instead"
    return "gemini", f"Gemini unavailable ({reason}) and no local model installed"


PROVIDER, PROVIDER_NOTE = "gemini", ""


def active_model() -> str:
    return GEMINI_MODEL if PROVIDER == "gemini" else OLLAMA_MODEL


def describe_provider() -> dict:
    """What the app is actually talking to, for the /api/models endpoint."""
    return {
        "configured": _CONFIGURED_PROVIDER,
        "provider": PROVIDER,
        "model": active_model(),
        "local": PROVIDER == "ollama",
        "note": PROVIDER_NOTE,
    }


def installed_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Locally pulled Ollama models. Empty when Ollama is not running."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return []
    return sorted(m.get("name", "") for m in response.json().get("models", []))


def supports_tools(model: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    """Whether Ollama will accept a tools payload for this model.

    Some models are refused outright with "does not support tools" because their
    chat template has none, which is worth distinguishing from a model that
    accepts tools and simply declines to call one. Both were observed here:
    gemma3n:e4b is refused, phi4-mini is accepted and never calls.
    """
    probe = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "tools": [{
            "type": "function",
            "function": {"name": "probe", "description": "probe",
                         "parameters": {"type": "object", "properties": {}}},
        }],
    }
    try:
        response = requests.post(f"{base_url}/api/chat", json=probe, timeout=120)
    except requests.RequestException:
        return False
    return "does not support tools" not in response.text


def chat(system: str, user: str, model: str | None = None, timeout: int = 600) -> str:
    """Single-turn completion from whichever provider is configured."""
    if PROVIDER == "gemini":
        return _chat_gemini(system, user, model or GEMINI_MODEL, timeout)
    return _chat_ollama(system, user, model or OLLAMA_MODEL, timeout)


def _chat_gemini(system: str, user: str, model: str, timeout: int) -> str:
    if not GEMINI_API_KEY:
        raise LLMUnavailable("GEMINI_API_KEY is not set; put it in .env")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.1},  # explanation, not invention
    }
    try:
        response = requests.post(
            url, json=payload, timeout=timeout,
            headers={"x-goog-api-key": GEMINI_API_KEY},
        )
        response.raise_for_status()
        parts = response.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except requests.RequestException as exc:
        raise LLMUnavailable(f"Gemini request failed ({model}): {exc}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMUnavailable(f"unexpected Gemini response: {exc}") from exc


def _chat_ollama(system: str, user: str, model: str, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except requests.RequestException as exc:
        raise LLMUnavailable(f"Ollama request failed ({OLLAMA_BASE_URL}, {model}): {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise LLMUnavailable(f"unexpected Ollama response: {exc}") from exc


# Resolved once at import: the probe is a network call and the answer does not
# change within a run.
PROVIDER, PROVIDER_NOTE = _resolve_provider()
DEFAULT_MODEL = active_model()
