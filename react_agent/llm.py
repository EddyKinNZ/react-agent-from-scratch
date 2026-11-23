"""Thin LLM clients over plain HTTP, with retry, backoff and model fallback.

No SDKs on purpose: the point of this project is to see every request.
"""
import os
import time
from dataclasses import dataclass, field

import requests

# USD per million tokens (input, output). Placeholder values: check the providers' price pages.
PRICES = {
    "claude-haiku-5-5": (1.0, 5.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "gemini-3.6-flash": (0.0, 0.0),  # free tier
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: list = field(default_factory=list)

    def add(self, model, tin, tout):
        pin, pout = PRICES.get(model, (0.0, 0.0))
        cost = (tin * pin + tout * pout) / 1_000_000
        self.input_tokens += tin
        self.output_tokens += tout
        self.cost_usd += cost
        self.calls.append({"model": model, "in": tin, "out": tout, "usd": round(cost, 6)})


class LLMError(Exception):
    pass


class AnthropicLLM:
    def __init__(self, model="claude-haiku-5-5", api_key=None, timeout=60):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.timeout = timeout

    def complete(self, system, messages, stop):
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": self.model, "max_tokens": 1024, "system": system,
                  "messages": messages, "stop_sequences": stop},
            timeout=self.timeout)
        if r.status_code != 200:
            raise LLMError(f"{r.status_code} {r.text[:200]}")
        d = r.json()
        text = "".join(b.get("text", "") for b in d["content"])
        return text, d["usage"]["input_tokens"], d["usage"]["output_tokens"]


class GeminiLLM:
    def __init__(self, model="gemini-3.6-flash", api_key=None, timeout=60):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.timeout = timeout

    def complete(self, system, messages, stop):
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]} for m in messages]
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "content-type": "application/json"},
            json={"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                  "generationConfig": {"stopSequences": stop, "temperature": 0}},
            timeout=self.timeout)
        if r.status_code != 200:
            raise LLMError(f"{r.status_code} {r.text[:200]}")
        d = r.json()
        text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        u = d.get("usageMetadata", {})
        return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


class ResilientLLM:
    """Try each model in order; retry each with exponential backoff before falling back."""

    def __init__(self, models, retries=2, base_delay=2.0, sleep=time.sleep):
        self.models = models
        self.retries = retries
        self.base_delay = base_delay
        self.sleep = sleep
        self.usage = Usage()

    def complete(self, system, messages, stop):
        errors = []
        for llm in self.models:
            for attempt in range(self.retries + 1):
                try:
                    text, tin, tout = llm.complete(system, messages, stop)
                    self.usage.add(llm.model, tin, tout)
                    return text
                except (LLMError, requests.RequestException, KeyError, IndexError) as e:
                    errors.append(f"{llm.model} attempt {attempt + 1}: {e}")
                    if attempt < self.retries:
                        self.sleep(self.base_delay * 2 ** attempt)
        raise LLMError("all models failed: " + " | ".join(errors))
