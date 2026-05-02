"""Shared LLM client library — used by GenerativeSEOProject and LinkedinGeneration.

Originally lived in seo_generation/seo_audit/llm_clients.py — extracted to
CommonLib on 2026-05-02 so LinkedinGeneration no longer depends on the SEO
project's internal package layout.
"""
from commonlib.llm_clients.core import (
    QuotaExceededError,
    OpenAILLMClient,
    GeminiLLMClient,
    OpenRouterLLMClient,
    create_llm_client,
)

__all__ = [
    'QuotaExceededError',
    'OpenAILLMClient',
    'GeminiLLMClient',
    'OpenRouterLLMClient',
    'create_llm_client',
]
