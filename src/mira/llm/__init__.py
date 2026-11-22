"""LLM provider package — factory entry point."""

from __future__ import annotations

import os

from mira.config import LLMConfig
from mira.llm.base import LLMProviderProtocol


def create_llm(config: LLMConfig) -> LLMProviderProtocol:
    """Create the appropriate LLM provider based on config.provider.

    A model slug such as ``zai/glm`` first resolves to its harness and model
    (see ``mira.llm.slugs``). Returns an instance satisfying LLMProviderProtocol.
    """
    from mira.llm import slugs

    config = slugs.resolve_config(config, os.environ)

    if config.provider == "bedrock":
        from mira.llm.bedrock import BedrockProvider

        return BedrockProvider(config)

    if config.provider in {"codex-cli", "codex_cli", "codex"}:
        from mira.llm.codex_cli import CodexCLIProvider

        return CodexCLIProvider(config)

    from mira.llm.agent_cli import CLAUDE_CODE_PROVIDER_NAMES, OPENCODE_PROVIDER_NAMES

    if config.provider in CLAUDE_CODE_PROVIDER_NAMES:
        from mira.llm.agent_cli import ClaudeCodeProvider

        return ClaudeCodeProvider(config)

    if config.provider in OPENCODE_PROVIDER_NAMES:
        from mira.llm.agent_cli import OpenCodeProvider

        return OpenCodeProvider(config)

    if config.api_style == "responses":
        from mira.llm.responses import ResponsesProvider

        return ResponsesProvider(config)

    # Default: OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, etc.)
    from mira.llm.provider import LLMProvider

    return LLMProvider(config)
