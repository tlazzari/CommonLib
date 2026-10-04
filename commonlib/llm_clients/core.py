from __future__ import annotations

import os
import time
from typing import Any, Dict
import json
from pathlib import Path
import logging

from openai import OpenAI
import requests

__all__ = [
    "QuotaExceededError",
    "OpenAILLMClient",
    "GeminiLLMClient",
    "OpenRouterLLMClient",
    "create_llm_client",
]


class QuotaExceededError(RuntimeError):
    """Raised when an upstream LLM rejects a request due to quota limits."""


_TRANSIENT_STATUS = {500, 502, 503, 504}
_TRANSIENT_BACKOFF = (15, 45, 90)  # seconds between retries; ~2.5 min in total


def _resolve_timeout() -> float:
    value = os.getenv("LLM_TIMEOUT_SECONDS")
    if not value:
        return 120.0
    try:
        parsed = float(value)
    except ValueError:
        return 120.0
    return max(5.0, parsed)


class OpenAILLMClient:
    """OpenAI Chat Completions wrapper matching the LLMClient protocol."""

    def __init__(self, *, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_API_KEY environment variable must be set for OpenAI client")
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.timeout = _resolve_timeout()
        self.label = f"openai:{model}"

    def complete(self, prompt: str, **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an SEO content strategist."},
                {"role": "user", "content": prompt},
            ],
            timeout=self.timeout,
            **kwargs,
        )
        return response.choices[0].message.content or ""


class GeminiLLMClient:
    """Google Gemini wrapper matching the LLMClient protocol."""

    def __init__(self, *, model: str = "gemini-2.5-flash") -> None:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_API_KEY environment variable must be set for Gemini client")
        self.model_name = model
        self.api_key = api_key
        self.timeout = _resolve_timeout()
        self.label = f"gemini:{model}"
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        )

    def _openrouter_fallback(self, prompt: str, temperature: float | None) -> str:
        payload: Dict[str, Any] = {
            "model": f"google/{self.model_name}",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        response = requests.post(
            OpenRouterLLMClient.BASE_URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            raise requests.HTTPError(
                f"Gemini 402 and OpenRouter fallback failed ({response.status_code}): {response.text[:300]}",
                response=response,
            )
        choices = response.json().get("choices") or []
        return (choices[0].get("message") or {}).get("content") or "" if choices else ""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature")
        attempt_limits = [None, 6000]
        last_exc: Exception | None = None

        for limit in attempt_limits:
            if limit is None or len(prompt) <= limit:
                working_prompt = prompt
            else:
                working_prompt = prompt[:limit]

            payload: Dict[str, Any] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": working_prompt}],
                    }
                ],
            }
            if temperature is not None:
                payload["generationConfig"] = {"temperature": temperature}

            params = {"key": self.api_key}
            print(
                f"[GeminiLLMClient] POST {self.endpoint} prompt_chars={len(working_prompt)} temp={temperature}"
            )
            debug_path_value = os.getenv("GEMINI_DEBUG_PAYLOAD")
            if debug_path_value:
                try:
                    debug_path = Path(debug_path_value)
                    debug_payload = {
                        "prompt": working_prompt,
                        "payload": payload,
                    }
                    debug_path.write_text(
                        json.dumps(debug_payload, ensure_ascii=False, indent=2)
                    )
                    print(f"[GeminiLLMClient] wrote debug payload to {debug_path}")
                except Exception as exc:
                    print(f"[GeminiLLMClient] failed to write debug payload: {exc}")
            # Gemini answers 503 "high demand" for minutes at a time. One failed call used to kill
            # the whole run: the Seta National Day post (2026-09-30) was never generated. Retry
            # transient failures with backoff before giving up.
            response = None
            conn_exc: Exception | None = None
            for wait in (*_TRANSIENT_BACKOFF, None):
                conn_exc = None
                try:
                    response = requests.post(
                        self.endpoint,
                        params=params,
                        json=payload,
                        timeout=self.timeout,
                    )
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                    print(f"[GeminiLLMClient] connection error with limit={limit}: {exc}")
                    conn_exc = exc
                except requests.RequestException as exc:  # pragma: no cover - other network failures
                    print(f"[GeminiLLMClient] request failed to send: {exc}")
                    raise exc
                transient = conn_exc is not None or response.status_code in _TRANSIENT_STATUS
                if not transient or wait is None:
                    break
                print(f"[GeminiLLMClient] transient failure, retrying in {wait}s")
                time.sleep(wait)
            if conn_exc is not None:
                last_exc = conn_exc
                continue

            if response.status_code == 429:
                raise QuotaExceededError("Gemini quota exceeded")

            # 402 = Gemini prepaid credits depleted (LinkedIn posts stopped 2026-10-02). Same model,
            # billed to OpenRouter, so a missed top-up no longer costs a day's post.
            if response.status_code == 402 and os.getenv("OPENROUTER_API_KEY"):
                print("[GeminiLLMClient] credits depleted (402) - falling back to OpenRouter")
                return self._openrouter_fallback(working_prompt, temperature)

            if not response.ok:
                try:
                    detail = response.json()
                except Exception:  # pragma: no cover - diagnostics only
                    detail = response.text
                print(
                    f"[GeminiLLMClient] non-OK response {response.status_code}: {detail}"
                )
                raise requests.HTTPError(
                    f"Gemini request failed ({response.status_code}): {detail}", response=response
                )

            print(
                f"[GeminiLLMClient] response {response.status_code} {response.headers.get('content-type')}"
            )

            data = response.json()
            candidates = data.get("candidates") or []
            contents: list[str] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content") or {}
                parts = []
                if isinstance(content, dict):
                    parts = content.get("parts") or []
                if not parts:
                    parts = candidate.get("parts") or []
                for part in parts:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if text:
                            contents.append(text)
            if contents:
                return "\n".join(contents)
            return data.get("text", "")

        if last_exc:
            raise last_exc
        raise RuntimeError("Gemini request failed for all prompt lengths")


class OpenRouterLLMClient:
    """OpenRouter API wrapper supporting Anthropic Claude Sonnet via /chat/completions."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, *, model: str = "anthropic/claude-3.5-sonnet") -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable must be set for OpenRouter client"
            )
        self.model = model or "anthropic/claude-3.5-sonnet"
        self.api_key = api_key
        self.timeout = _resolve_timeout()
        self.label = f"openrouter:{self.model}"

    def complete(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1600)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://tntbearings.com"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "TNT Motion Scheduler"),
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an SEO content strategist."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            self.BASE_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        debug_path_value = os.getenv("OPENROUTER_DEBUG_PATH")
        if debug_path_value:
            try:
                debug_path = Path(debug_path_value)
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(response.text, encoding="utf-8")
            except Exception as exc:  # pragma: no cover - debug only
                print(f"[OpenRouterLLMClient] failed to write debug payload: {exc}")
        if response.status_code == 429:
            raise QuotaExceededError("OpenRouter rate limit hit")
        if not response.ok:
            detail = None
            try:
                detail = response.json()
            except Exception:  # pragma: no cover - diagnostic path
                detail = response.text
            raise requests.HTTPError(
                f"OpenRouter request failed ({response.status_code}): {detail}",
                response=response,
            )
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason")
        if finish_reason and finish_reason.lower() not in {"stop", "completed"}:
            raise ValueError(f"OpenRouter response incomplete (finish_reason={finish_reason})")
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""


def create_llm_client(*, provider: str, model: str) -> Any:
    normalized = (provider or "openai").strip().lower()
    if normalized in {"openai", "gpt"}:
        return OpenAILLMClient(model=model)
    if normalized in {"gemini", "google", "google-gemini"}:
        return GeminiLLMClient(model=model)
    if normalized in {"openrouter", "sonnet", "anthropic", "claude"}:
        return OpenRouterLLMClient(model=model)
    raise ValueError(f"Unsupported LLM provider: {provider}")
