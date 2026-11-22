"""OpenCode CLI provider (``opencode run``).

OpenCode runs any provider it knows (OpenAI, Google, Z.ai, DeepSeek,
OpenRouter, OpenCode Zen and others) and any OpenAI-compatible endpoint. The
model is ``<provider>/<model>``, such as ``zai/glm-5.3``.

Credentials, checked in this order:

* ``OPENCODE_AUTH_CONTENT``, the JSON of an OpenCode ``auth.json``, for a
  subscription such as a ChatGPT plan. Only the model's provider's entry is
  passed on, and never to a re-pointed gateway.
* The provider's API key: the ``api_key_env`` of the providers.json profile
  with the provider's name (``ZAI_API_KEY`` for ``zai``), else
  ``<PROVIDER>_API_KEY``. The config names the variable through OpenCode's
  ``{env:...}`` syntax, so the key never appears in argv or the config.
* ``openai-compatible/<model>`` reads ``OPENAI_COMPATIBLE_BASE_URL``,
  ``OPENAI_COMPATIBLE_API_KEY``, ``OPENAI_COMPATIBLE_CONTEXT`` and
  ``OPENAI_COMPATIBLE_MAX_OUTPUT``. OpenCode has no catalog entry for such an
  endpoint, so the token limits are required.

``<PREFIX>_BASE_URL`` (``OPENAI_BASE_URL``, ``ANTHROPIC_BASE_URL`` and the
others in ``_GATEWAY_URL_ENV``) re-points a native provider at a gateway.

Security posture of every invocation, checked against opencode-ai 1.18.29:

* OpenCode imports ``{plugin,plugins}/*.{ts,js}`` and custom tools from every
  ``.opencode`` directory between its working directory and the git root,
  even with ``OPENCODE_DISABLE_PROJECT_CONFIG=1`` and ``--pure``. So the CLI
  always runs in an empty directory, never in the checkout, and workspace
  mode reads the checkout by absolute path. ``--dir <checkout>`` would load
  the checkout's plugins too.
* ``OPENCODE_DISABLE_PROJECT_CONFIG=1`` and ``--pure`` as well, so no project
  ``opencode.json``, local MCP server or external plugin loads either.
* The config in ``OPENCODE_CONFIG_CONTENT`` denies every tool except read,
  grep, glob and list, which are allowed only in workspace mode and only
  inside the checkout (``external_directory``). Denied tools are not offered
  to the model at all. No MCP servers, no snapshots, no sharing, no autoupdate.
* The global config directory is read-only, so OpenCode skips its background
  ``npm install`` of ``@opencode-ai/plugin``. With
  ``OPENCODE_DISABLE_MODELS_FETCH=1`` it uses the models.dev catalog compiled
  into the pinned binary, so a run fetches nothing but the model API.
  ``OPENCODE_MODELS_PATH`` in Mira's environment points it at a catalog file
  instead.
* A custom ``mira`` agent carries Mira's system prompt in place of OpenCode's
  coding-agent prompt, and ``--title`` skips the extra title-generation call.
* The child gets a scrubbed environment and a throwaway ``HOME``. Mira's
  scrub also drops ``OPENCODE_EXPERIMENTAL*``, which would enable a script
  interpreter tool.

Adapted from Pullfrog's OpenCode harness (``agents/opencode.ts``,
``agents/opencodeShared.ts``, ``models.ts``, ``mcp/askModel.ts``), MIT
License, Copyright (c) 2026 Pullfrog, Inc.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mira.exceptions import LLMError, NonRetriableLLMError
from mira.llm import provider_profiles as profiles
from mira.llm.agent_cli.base import (
    UNTRUSTED_REPOSITORY_NOTE,
    AgentCLIProvider,
    CLICall,
    CLIRunResult,
)

OPENAI_COMPATIBLE = "openai-compatible"
_OPENAI_COMPATIBLE_ENV = {
    "base_url": "OPENAI_COMPATIBLE_BASE_URL",
    "api_key": "OPENAI_COMPATIBLE_API_KEY",
    "context": "OPENAI_COMPATIBLE_CONTEXT",
    "max_output": "OPENAI_COMPATIBLE_MAX_OUTPUT",
}

# Providers Mira can re-point at a gateway, and the variable that does it.
# OpenCode reads no base-URL variable itself; Mira sets ``options.baseURL``.
_GATEWAY_URL_ENV = {
    "anthropic": "ANTHROPIC_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "google": "GEMINI_BASE_URL",
    "xai": "XAI_BASE_URL",
    "deepseek": "DEEPSEEK_BASE_URL",
    "moonshotai": "MOONSHOT_BASE_URL",
    "kimi-for-coding": "KIMI_BASE_URL",
}
# ``@ai-sdk/anthropic`` posts to ``{baseURL}/messages``, but ANTHROPIC_BASE_URL
# is a bare host by convention.
_GATEWAY_PATH = {"anthropic": "/v1"}

AUTH_CONTENT_ENV = "OPENCODE_AUTH_CONTENT"
# The runner workflow's one model key, for a provider whose own key isn't set.
RUNNER_KEY_ENV = "MIRA_LLM_API_KEY"

_READ_TOOLS = ("read", "grep", "glob", "list")
_DENIED_TOOLS = (
    "bash",
    "edit",
    "write",
    "webfetch",
    "websearch",
    "task",
    "skill",
    "todowrite",
    "question",
    "lsp",
)
_AGENT = "mira"
_DEFAULT_SYSTEM_PROMPT = "You are Mira, a careful code reviewer."

_ENV = {
    "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
    "OPENCODE_DISABLE_MODELS_FETCH": "1",
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_SHARE": "1",
    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
    "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
}

# What OpenCode reports when a tool call is refused by policy.
_DENIAL_MARKERS = ("prevents you from using this specific tool call", "unavailable tool")


@dataclass(frozen=True)
class _Credential:
    env: dict[str, str]
    #: The API-key variable the config names, or None for a subscription.
    key_env: str | None


class OpenCodeProvider(AgentCLIProvider):
    harness = "opencode"
    error_prefix = "opencode"
    runs_in_workspace = False
    supports_json_schema = False

    @property
    def _executable(self) -> str:
        return self.config.opencode_command

    @property
    def _timeout_seconds(self) -> int:
        return self.config.opencode_timeout_seconds

    # ── Model and credentials ──────────────────────────────────────────

    def _split_model(self) -> tuple[str, str]:
        provider, _, model = self.config.model.partition("/")
        if not provider or not model:
            raise NonRetriableLLMError("opencode_model_invalid", model=self.config.model)
        return provider, model

    def _key_env(self, provider: str) -> str:
        if provider == OPENAI_COMPATIBLE:
            return _OPENAI_COMPATIBLE_ENV["api_key"]
        profile = profiles.get(provider) or {}
        return profile.get("api_key_env") or f"{provider.upper().replace('-', '_')}_API_KEY"

    def _gateway_url(self, provider: str) -> str | None:
        url = os.environ.get(_GATEWAY_URL_ENV.get(provider, ""), "").rstrip("/")
        return url + _GATEWAY_PATH.get(provider, "") if url else None

    def _subscription(self, provider: str) -> dict | None:
        raw = os.environ.get(AUTH_CONTENT_ENV)
        if not raw:
            return None
        try:
            entries = json.loads(raw)
        except ValueError:
            entries = None
        if not isinstance(entries, dict):
            raise NonRetriableLLMError("opencode_auth_content_invalid")
        entry = entries.get(provider)
        return entry if isinstance(entry, dict) else None

    def _credential(self) -> _Credential:
        provider, _ = self._split_model()
        # A subscription never goes to a gateway.
        takes_subscription = provider != OPENAI_COMPATIBLE and self._gateway_url(provider) is None
        if takes_subscription and (entry := self._subscription(provider)):
            return _Credential({AUTH_CONTENT_ENV: json.dumps({provider: entry})}, None)
        key_env = self._key_env(provider)
        for name in (key_env, RUNNER_KEY_ENV):
            if key := os.environ.get(name):
                return _Credential({name: key}, name)
        keys = f"{key_env} or {RUNNER_KEY_ENV}"
        if takes_subscription:
            keys += f" or a {provider!r} entry in {AUTH_CONTENT_ENV}"
        raise NonRetriableLLMError("opencode_auth_missing", provider=provider, keys=keys)

    def _auth_env(self) -> dict[str, str]:
        return self._credential().env

    def _secret_values(self) -> list[str]:
        values = super()._secret_values()
        # A subscription's tokens sit inside the JSON blob, so redact each one.
        try:
            entry = self._subscription(self._split_model()[0]) or {}
        except LLMError:
            entry = {}
        return values + [v for v in entry.values() if isinstance(v, str) and len(v) >= 8]

    # ── Config ─────────────────────────────────────────────────────────

    def _provider_config(self, key_env: str | None) -> dict[str, Any]:
        """The ``provider`` block: an OpenAI-compatible endpoint, a gateway, or a key."""
        provider, model = self._split_model()
        options: dict[str, Any] = {"apiKey": f"{{env:{key_env}}}"} if key_env else {}
        if provider == OPENAI_COMPATIBLE:
            base_url = os.environ.get(_OPENAI_COMPATIBLE_ENV["base_url"], "").rstrip("/")
            if not base_url:
                raise NonRetriableLLMError(
                    "opencode_endpoint_invalid", problem=_OPENAI_COMPATIBLE_ENV["base_url"]
                )
            limit = {
                "context": _positive_int_env(_OPENAI_COMPATIBLE_ENV["context"]),
                "output": _positive_int_env(_OPENAI_COMPATIBLE_ENV["max_output"]),
            }
            return {
                provider: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "OpenAI-compatible",
                    "options": {"baseURL": base_url, **options},
                    "models": {model: {"name": model, "limit": limit}},
                }
            }
        if gateway := self._gateway_url(provider):
            options["baseURL"] = gateway
        return {provider: {"options": options}} if options else {}

    def _config(self, call: CLICall, key_env: str | None) -> dict[str, Any]:
        provider, _ = self._split_model()
        read = "allow" if call.workspace is not None else "deny"
        outside: dict[str, str] = {"*": "deny"}
        if call.workspace is not None:
            outside |= {str(call.workspace): "allow", f"{call.workspace}/*": "allow"}
        prompt = f"{{file:{call.system_prompt_file}}}" if call.system_prompt_file else None
        return {
            "model": self.config.model,
            "enabled_providers": [provider],
            "provider": self._provider_config(key_env),
            # Last match wins, so the catch-all deny comes first.
            "permission": {
                "*": "deny",
                **dict.fromkeys(_READ_TOOLS, read),
                **dict.fromkeys(_DENIED_TOOLS, "deny"),
                "external_directory": outside,
            },
            "agent": {
                _AGENT: {
                    "mode": "primary",
                    "description": "Mira code review",
                    "prompt": prompt or _DEFAULT_SYSTEM_PROMPT,
                }
            },
            "mcp": {},
            "plugin": [],
            "snapshot": False,
            "share": "disabled",
            "autoupdate": False,
        }

    def _extra_env(self, call: CLICall) -> dict[str, str]:
        # Read-only, so OpenCode skips its background npm install into it.
        config_dir = call.home / ".config" / "opencode"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_dir.chmod(0o500)
        env = dict(_ENV)
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(self._config(call, self._credential().key_env))
        if models_path := os.environ.get("OPENCODE_MODELS_PATH"):
            env["OPENCODE_MODELS_PATH"] = models_path
        return env

    def preflight(self) -> None:
        super().preflight()
        self._provider_config(self._credential().key_env)

    # ── argv and prompt ────────────────────────────────────────────────

    def _command(
        self,
        *,
        workspace: bool,
        schema: dict | None,
        system_prompt_file: str | None,
    ) -> list[str]:
        cmd = [
            self._executable,
            "run",
            "--format",
            "json",
            "--pure",
            "--agent",
            _AGENT,
            "--model",
            self.config.model,
            "--title",
            "mira",
        ]
        if self.config.reasoning_effort:
            cmd += ["--variant", self.config.reasoning_effort]
        return cmd

    def _workspace_instructions(self, workspace: Path) -> str:
        return (
            "\n\n## Repository access\n"
            f"The pull request's head commit is checked out at `{workspace}`. Use the read, "
            "grep and glob tools to check any code your task depends on, including "
            "definitions, callers, tests and configuration in files not shown above. Give "
            "them absolute paths under that directory; nothing outside it can be read. When "
            "you cite a file, use its path relative to that directory (for example "
            "`src/app.py`), never the absolute path. You cannot run commands or modify "
            "files. " + UNTRUSTED_REPOSITORY_NOTE
        )

    # ── Output ─────────────────────────────────────────────────────────

    def _parse_output(self, stdout: str, stderr: str, returncode: int) -> CLIRunResult:
        """Read ``--format json`` events: the final step's text, and tokens and cost summed."""
        result = CLIRunResult(text="", model=self.config.model, num_turns=0)
        steps: list[list[str]] = [[]]
        error: dict | None = None
        for line in stdout.splitlines():
            event = _json_object(line)
            if event is None:
                continue
            kind, part = event.get("type"), event.get("part") or {}
            if kind == "step_start":
                steps.append([])
            elif kind == "text":
                steps[-1].append(str(part.get("text") or ""))
                # Text after an error means the session recovered from it.
                error = None
            elif kind == "step_finish":
                _add_usage(result, part)
            elif kind == "tool_use":
                _record_denial(result, part)
            elif kind == "error":
                error = event.get("error") if isinstance(event.get("error"), dict) else {}
        if error is not None:
            raise _classify(error, returncode)
        answer = next(("\n\n".join(texts) for texts in reversed(steps) if texts), "")
        if not answer.strip() and returncode != 0:
            detail = (stderr or stdout).strip()[-2000:]
            raise LLMError("opencode_exit_failed", exit_code=returncode, detail=detail)
        result.text = answer
        return result


def _positive_int_env(name: str) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        value = 0
    if value <= 0:
        raise NonRetriableLLMError("opencode_endpoint_invalid", problem=f"{name} (a token count)")
    return value


def _json_object(line: str) -> dict | None:
    if not line.startswith("{"):
        return None
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _add_usage(result: CLIRunResult, part: dict) -> None:
    tokens = part.get("tokens") or {}
    cache = tokens.get("cache") or {}
    result.input_tokens += _int(tokens.get("input"))
    result.output_tokens += _int(tokens.get("output")) + _int(tokens.get("reasoning"))
    result.cache_read_tokens += _int(cache.get("read"))
    result.cache_write_tokens += _int(cache.get("write"))
    result.cost_usd = (result.cost_usd or 0.0) + _float(part.get("cost"))
    result.num_turns = (result.num_turns or 0) + 1


def _record_denial(result: CLIRunResult, part: dict) -> None:
    state = part.get("state") or {}
    message = str(state.get("error") or "")
    if state.get("status") == "error" and any(m in message for m in _DENIAL_MARKERS):
        result.permission_denials.append({"tool": part.get("tool"), "input": state.get("input")})


def _classify(error: dict, returncode: int) -> LLMError:
    data = error.get("data")
    if not isinstance(data, dict):
        data = {}
    message = str(data.get("message") or error.get("name") or "unknown error")
    status = data.get("statusCode")
    if status in (401, 403):
        return NonRetriableLLMError("opencode_auth_failed", detail=message[:500])
    if status == 429:
        # OpenCode has already retried it before giving up.
        return NonRetriableLLMError("opencode_rate_limited", detail=message[:500])
    return LLMError("opencode_reported_error", exit_code=returncode, detail=message[:2000])


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
