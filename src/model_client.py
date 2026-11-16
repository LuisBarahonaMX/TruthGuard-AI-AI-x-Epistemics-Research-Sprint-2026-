"""Adaption multimodel client for TruthGuard AI. Python >=3.10."""
from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any

MODELS = (
    "glm-53-flash", "glm-5.2", "qwen3.5-397b",
    "gemma-4-26b-a4b-it", "qwen3.8-27b",
    "deepseek-v4-flash", "gemma-4-31b-it",
)
# Adaption published credits per 1 million tokens; rates may change.
# https://docs.adaptionlabs.ai/inference-stacks/models
RATES = {
    "glm-53-flash": (0.15, 0.50),
    "glm-5.2": (1.40, 4.40),
    "qwen3.5-397b": (0.55, 3.50),
    "gemma-4-26b-a4b-it": (0.077, 0.26),
    "qwen3.8-27b": (0.039, 1.25),
    "deepseek-v4-flash": (0.045, 0.14),
    "gemma-4-31b-it": (0.09, 0.34),
}

class ResponseFailure(RuntimeError):
    def __init__(self, message: str, *, retriable: bool = False, response_data: dict | None = None):
        super().__init__(message)
        self.retriable = retriable
        self.response_data = response_data


def _as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _error_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status"):
        status = getattr(exc, attr, None)
        if isinstance(status, int):
            return status
    return None


def is_transient(exc: Exception) -> bool:
    code = _error_code(exc)
    if code is not None:
        return code in (408, 409, 425, 429) or code >= 500
    name = type(exc).__name__.lower()
    return any(part in name for part in ("timeout", "connection", "ratelimit", "internalserver"))


def estimate_credits(model: str, usage: dict) -> float | None:
    """Estimate using observed input/output tokens; NOT the invoiced amount.

    Ignores cached-token discount, which can only lower the estimated amount.
    """
    if model not in RATES or "input_tokens" not in usage or "output_tokens" not in usage:
        return None
    inp, out = RATES[model]
    return (int(usage["input_tokens"]) * inp + int(usage["output_tokens"]) * out) / 1_000_000


@dataclass
class ModelClient:
    """One reusable async Adaption session, externally rate-limited by caller."""
    timeout: float = 120.0
    max_output_tokens: int = 512
    retries: int = 2
    instructions: str = "Answer the user's question accurately."

    def __post_init__(self):
        self._client = None

    async def __aenter__(self):
        if not os.getenv("ADAPTION_API_KEY"):
            raise RuntimeError("Set ADAPTION_API_KEY in PowerShell before running live requests")
        from adaption import AsyncAdaption
        self._client = AsyncAdaption(timeout=self.timeout, max_retries=0)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
            self._client = None

    async def generate(self, model: str, messages: list[dict]) -> dict:
        if model not in MODELS:
            raise ValueError(f"Model not in approved 7-model list: {model}")
        if self._client is None:
            raise RuntimeError("Use 'async with ModelClient() as client:'")
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                response = await self._client.responses.create(
                    model=model,
                    input=messages,
                    instructions=self.instructions,
                    max_output_tokens=self.max_output_tokens,
                    tool_choice="none",
                    store=False,
                    agentic_memory=False,
                )
                elapsed = time.perf_counter() - start
                payload = _as_dict(response)
                status = getattr(response, "status", None)
                usage = _as_dict(getattr(response, "usage", None))
                text = getattr(response, "output_text", None) or ""
                data = {
                    "model": model, "response_id": getattr(response, "id", None),
                    "status": status, "text": text,
                    "latency_seconds": round(elapsed, 4), "usage": usage,
                    "estimated_credits_from_usage": estimate_credits(model, usage),
                    "pricing": "published_list_rate_estimate_not_invoiced",
                    "raw_response": payload,
                }
                if status != "completed" or not text.strip():
                    # No retry after a response object: it may have been billed.
                    raise ResponseFailure(f"Response status={status}, empty_text={not bool(text.strip())}", response_data=data)
                return data
            except ResponseFailure:
                raise
            except Exception as exc:
                if attempt >= self.retries or not is_transient(exc):
                    raise
                await asyncio.sleep(min(20, 2 ** attempt + random.random()))
        raise AssertionError("unreachable")
