"""Local LLM access via Ollama.

Kept deliberately small. The language model's only job in this system is to
explain evidence that has already been gathered and verified -- it is never a
source of facts, so it needs no tool-calling loop of its own.
"""

import os

import requests

DEFAULT_MODEL = os.environ.get("FOODSAFE_MODEL", "gemma4:26b")
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


class LLMUnavailable(RuntimeError):
    """Ollama is not reachable or the requested model is not pulled."""


def installed_models(base_url: str = DEFAULT_BASE_URL) -> list[str]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return []
    return sorted(m.get("name", "") for m in response.json().get("models", []))


def is_available(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> bool:
    return model in installed_models(base_url)


def supports_tools(model: str, base_url: str = DEFAULT_BASE_URL) -> bool:
    """Whether Ollama will accept a tools payload for this model.

    Some models are refused outright with "does not support tools" because their
    chat template has no tool-calling support, which is worth distinguishing from
    a model that accepts tools and simply declines to call one.
    """
    probe = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "probe",
                    "description": "probe",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    try:
        response = requests.post(f"{base_url}/api/chat", json=probe, timeout=120)
    except requests.RequestException:
        return False
    return "does not support tools" not in response.text


def chat(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.1,  # explanation, not invention
    timeout: int = 600,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except requests.RequestException as exc:
        raise LLMUnavailable(f"Ollama request failed ({base_url}, {model}): {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise LLMUnavailable(f"unexpected Ollama response: {exc}") from exc
