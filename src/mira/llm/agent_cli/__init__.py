"""Providers that drive a full agent CLI (Claude Code, OpenCode) as Mira's model.

Unlike ``mira.llm.codex_cli``, which uses its CLI only to complete text, these
providers also offer a workspace mode: while ``workspace`` is bound to a
read-only checkout of the pull request, every call runs the CLI's own agent
loop inside it. See ``AgentCLIProvider``.
"""

from mira.llm.agent_cli.base import AgentCLIProvider, CLIRunResult
from mira.llm.agent_cli.claude import ClaudeCodeProvider
from mira.llm.agent_cli.opencode import OpenCodeProvider

CLAUDE_CODE_PROVIDER_NAMES = frozenset({"claude-code", "claude_code", "claude-cli"})
OPENCODE_PROVIDER_NAMES = frozenset({"opencode"})

__all__ = [
    "CLAUDE_CODE_PROVIDER_NAMES",
    "OPENCODE_PROVIDER_NAMES",
    "AgentCLIProvider",
    "CLIRunResult",
    "ClaudeCodeProvider",
    "OpenCodeProvider",
]
