"""Local LLM access via Ollama.

Kept deliberately small. The language model's only job in this system is to
explain evidence that has already been gathered and verified — it is never a
source of facts, so it needs no tool-calling loop of its own.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import requests

DEFAULT_MODEL = os.environ.get("FOODSAFE_MODEL", "gemma4:26b")
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


class LLMUnavailable(RuntimeError):
    """Ollama is not reachable or the requested model is not pulled."""


@dataclass
class Ollama:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.1  # explanation, not invention
    timeout: int = 600

    def available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException:
            return False
        names = {m.get("name", "") for m in response.json().get("models", [])}
        return self.model in names

    def installed_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException:
            return []
        return sorted(m.get("name", "") for m in response.json().get("models", []))

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMUnavailable(f"Ollama request failed ({self.base_url}, {self.model}): {exc}") from exc

        try:
            return response.json()["message"]["content"].strip()
        except (KeyError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"unexpected Ollama response: {exc}") from exc
