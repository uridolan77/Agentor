"""
LLM module for the Agentor framework.

This module provides various LLM implementations for agents, including:
- OpenAI LLM provider
- Semantic caching for LLMs
- Adapters for using LLM components with the standardized interfaces
"""

from agentor.llm_gateway.llm.base import BaseLLM, LLMRequest, LLMResponse
from agentor.llm_gateway.llm.providers.openai import OpenAILLM
from agentor.llm_gateway.llm.semantic_cache import SemanticCache, SemanticCachedLLM
from agentor.llm_gateway.llm.adapters import (
    LLMAdapter,
    StreamingLLMAdapter,
    CachedLLMAdapter,
    SemanticCachedLLMAdapter,
    OpenAILLMAdapter,
    SemanticCachedOpenAILLMAdapter,
    OpenAILLMPlugin,
    SemanticCachedOpenAILLMPlugin
)

__all__ = [
    # Base LLM classes
    'BaseLLM',
    'LLMRequest',
    'LLMResponse',
    'OpenAILLM',
    'SemanticCache',
    'SemanticCachedLLM',

    # LLM adapters
    'LLMAdapter',
    'StreamingLLMAdapter',
    'CachedLLMAdapter',
    'SemanticCachedLLMAdapter',
    'OpenAILLMAdapter',
    'SemanticCachedOpenAILLMAdapter',

    # LLM plugins
    'OpenAILLMPlugin',
    'SemanticCachedOpenAILLMPlugin',
]