"""Connect a model served by Ollama (for example your fine-tuned SLM) to the RAG answer layer.

Ollama runs as a separate program on your laptop and listens on http://localhost:11434.
This class only sends it an HTTP request, so the rest of the code does not care which model is behind it.

    ollama create pension-slm -f Modelfile      # after you converted/merged your trained model
    OLLAMA_MODEL=pension-slm  python -m src.api.main
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from src.rag.answer_generation import LLMProvider


class OllamaError(RuntimeError):
    pass


class OllamaLLM(LLMProvider):
    def __init__(self, model: str | None = None, base_url: str | None = None, temperature: float = 0.0, timeout: float = 120.0):
        self.model = model or os.environ.get("OLLAMA_MODEL")
        if not self.model:
            raise OllamaError("OLLAMA_MODEL is not set (the name you gave the model in Ollama).")
        self.base_url = (base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.temperature, self.timeout = temperature, timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        body = json.dumps({"model": self.model, "stream": False, "options": {"temperature": self.temperature},
                           "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Could not get an answer from Ollama at {self.base_url} ({exc}). Is `ollama serve` running and the model pulled?") from exc
        text = (data.get("message") or {}).get("content", "").strip()
        if not text:
            raise OllamaError("Ollama returned an empty answer")
        return text
