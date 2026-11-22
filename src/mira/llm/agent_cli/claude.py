"""Claude Code CLI provider (``claude -p``).

This provider runs the real Claude Code CLI. With ``CLAUDE_CODE_OAUTH_TOKEN``
(from ``claude setup-token``) it bills a Claude Pro or Max subscription. With
``ANTHROPIC_API_KEY`` it bills the API.

With ``ANTHROPIC_BASE_URL`` set, the CLI talks to an Anthropic-compatible
gateway instead, such as Z.ai's GLM Coding Plan. The gateway's key comes from
``ANTHROPIC_AUTH_TOKEN`` (or ``ANTHROPIC_API_KEY``), and the
``ANTHROPIC_DEFAULT_*_MODEL`` variables map ``--model sonnet`` and the other
aliases to the gateway's models. A subscription token is never sent to a
gateway.

Security posture of every invocation (checked against Claude Code 2.1.330):

* ``--setting-sources ""``: no user, project or local settings are loaded.
  A pull request cannot add hooks, permissions or MCP servers through a
  ``.claude/settings.json`` it commits, and a ``CLAUDE.md`` in the checkout
  is not loaded automatically. Mira injects repository conventions itself.
* ``--strict-mcp-config`` with no ``--mcp-config``: no MCP servers.
* ``--permission-mode dontAsk``: any tool call that is not pre-approved is
  denied, and the denial is reported in ``permission_denials``.
* Completion mode passes ``--tools ""``, so no tools exist. Workspace mode
  passes ``--tools Read,Grep,Glob`` and pre-approves ``Read(./**)``. That
  rule also bounds Grep and Glob, so reads outside the checkout are denied.
* ``--bare`` is deliberately not used: in bare mode the CLI never reads
  OAuth credentials, which would break subscription auth.
* The child process gets a scrubbed environment and a throwaway ``HOME``
  (see ``AgentCLIProvider._child_env``).

Output is ``--output-format stream-json`` with partial messages, so the stall
watchdog sees each token arrive. ``system`` events (startup, API retries) do
not count as progress. The final ``result`` event carries the same fields
``--output-format json`` prints.
"""

from __future__ import annotations

import json
import os
from typing import Any

from mira.exceptions import LLMError, NonRetriableLLMError
from mira.llm.agent_cli.base import AgentCLIProvider, CLIRunResult

# Credentials Claude Code accepts, in the order they are preferred.
_AUTH_ENV_KEYS = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
# With ANTHROPIC_BASE_URL set, only a key meant for that gateway is sent.
_GATEWAY_AUTH_ENV_KEYS = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")
# Gateway settings the CLI reads. None of them is a secret.
_GATEWAY_ENV_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "API_TIMEOUT_MS",
)

# Values accepted by ``claude --effort``.
_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})

# Model ids that mean "let Claude Code choose".
_DEFAULT_MODEL_IDS = frozenset({"", "default", "claude-code-default"})

_WORKSPACE_TOOLS = "Read,Grep,Glob"
_WORKSPACE_ALLOW_RULE = "Read(./**)"

_RATE_LIMIT_MARKERS = ("usage limit", "rate limit", "limit reached", "rate_limit")
_AUTH_MARKERS = ("not logged in", "invalid api key", "oauth token", "authentication")


class ClaudeCodeProvider(AgentCLIProvider):
    harness = "claude-code"
    error_prefix = "claude"

    @property
    def _executable(self) -> str:
        # No fallback: an empty setting must fail validation, not silently
        # resolve to whatever `claude` is first on PATH.
        return self.config.claude_command

    @property
    def _timeout_seconds(self) -> int:
        return self.config.claude_timeout_seconds

    def _auth_env(self) -> dict[str, str]:
        keys = _GATEWAY_AUTH_ENV_KEYS if os.environ.get("ANTHROPIC_BASE_URL") else _AUTH_ENV_KEYS
        for key in keys:
            value = os.environ.get(key)
            if value:
                return {key: value}
        raise NonRetriableLLMError("claude_auth_missing", keys=", ".join(keys))

    def _extra_env(self) -> dict[str, str]:
        return {
            # A long-running server must not self-update its pinned CLI.
            "DISABLE_AUTOUPDATER": "1",
            # Drop telemetry and other traffic a headless review does not need.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            **{key: os.environ[key] for key in _GATEWAY_ENV_KEYS if os.environ.get(key)},
        }

    def _command(
        self,
        *,
        workspace: bool,
        schema: dict | None,
        system_prompt_file: str | None,
    ) -> list[str]:
        cmd = [
            self._executable,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--permission-mode",
            "dontAsk",
        ]
        if workspace:
            cmd += ["--tools", _WORKSPACE_TOOLS, "--allowedTools", _WORKSPACE_ALLOW_RULE]
        else:
            cmd += ["--tools", ""]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
        if system_prompt_file:
            cmd += ["--append-system-prompt-file", system_prompt_file]
        if self.config.model not in _DEFAULT_MODEL_IDS:
            cmd += ["--model", self.config.model]
        if self.config.fallback_model and self.config.fallback_model != self.config.model:
            cmd += ["--fallback-model", self.config.fallback_model]
        effort = (self.config.reasoning_effort or "").lower()
        if effort in _EFFORT_LEVELS:
            cmd += ["--effort", effort]
        return cmd

    def _is_progress(self, line: bytes) -> bool:
        return bool(line.strip()) and _event_type(line) != "system"

    def _keeps_line(self, line: bytes) -> bool:
        # Token deltas only feed the watchdog; the result event repeats what matters.
        return _event_type(line) != "stream_event"

    def _parse_output(self, stdout: str, stderr: str, returncode: int) -> CLIRunResult:
        payload = _result_event(stdout)
        if payload is None:
            detail = (stderr or stdout).strip()[-2000:]
            raise LLMError("claude_exit_failed", exit_code=returncode, detail=detail)

        if payload.get("is_error") or returncode != 0:
            message = str(payload.get("result") or payload.get("subtype") or "").strip()
            status = payload.get("api_error_status")
            lowered = message.lower()
            if status in (401, 403) or any(m in lowered for m in _AUTH_MARKERS):
                raise NonRetriableLLMError("claude_auth_failed", detail=message[:500])
            if status == 429 or any(m in lowered for m in _RATE_LIMIT_MARKERS):
                raise NonRetriableLLMError("claude_rate_limited", detail=message[:500])
            raise LLMError("claude_reported_error", exit_code=returncode, detail=message[:2000])

        structured = payload.get("structured_output")
        result = CLIRunResult(
            text=str(payload.get("result") or ""),
            structured=structured if isinstance(structured, dict) else None,
            cost_usd=_as_float(payload.get("total_cost_usd")),
            num_turns=_as_int(payload.get("num_turns")),
            duration_ms=_as_int(payload.get("duration_ms")),
            permission_denials=list(payload.get("permission_denials") or []),
        )
        _apply_model_usage(result, payload)
        return result


def _event_type(line: bytes) -> str | None:
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event.get("type") if isinstance(event, dict) else None


def _result_event(stdout: str) -> dict[str, Any] | None:
    """The last ``{"type": "result"}`` event in a stream-json transcript."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return event
    return None


def _apply_model_usage(result: CLIRunResult, payload: dict[str, Any]) -> None:
    """Fill tokens and model from ``modelUsage``, which totals every turn.

    The top-level ``usage`` block covers only part of the run, so it is
    used only as a fallback.
    """
    model_usage = payload.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        main_model, main_output = None, -1
        for model_id, stats in model_usage.items():
            if not isinstance(stats, dict):
                continue
            result.input_tokens += _as_int(stats.get("inputTokens")) or 0
            result.output_tokens += _as_int(stats.get("outputTokens")) or 0
            result.cache_read_tokens += _as_int(stats.get("cacheReadInputTokens")) or 0
            result.cache_write_tokens += _as_int(stats.get("cacheCreationInputTokens")) or 0
            output = _as_int(stats.get("outputTokens")) or 0
            if output > main_output:
                main_model, main_output = model_id, output
        result.model = main_model
        return
    usage = payload.get("usage")
    if isinstance(usage, dict):
        result.input_tokens = _as_int(usage.get("input_tokens")) or 0
        result.output_tokens = _as_int(usage.get("output_tokens")) or 0
        result.cache_read_tokens = _as_int(usage.get("cache_read_input_tokens")) or 0
        result.cache_write_tokens = _as_int(usage.get("cache_creation_input_tokens")) or 0


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
