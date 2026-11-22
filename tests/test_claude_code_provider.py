"""Tests for the Claude Code CLI provider, driven by a fake ``claude`` executable."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from mira.config import LLMConfig, _strip_deployment_only_settings
from mira.dashboard.model_catalog import active_backend, build_options
from mira.exceptions import LLMError, NonRetriableLLMError
from mira.llm import create_llm
from mira.llm.agent_cli import ClaudeCodeProvider
from mira.llm.agent_cli.base import _REDACTED
from mira.llm.tool_schemas import SUBMIT_CRITIQUE_TOOL

TOKEN = "sk-ant-oat01-" + "x" * 40

SUCCESS_PAYLOAD = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": '{"comments": []}',
    "structured_output": {"comments": [], "summary": "Looks fine."},
    "total_cost_usd": 0.0123,
    "num_turns": 3,
    "duration_ms": 4200,
    "permission_denials": [],
    "usage": {"input_tokens": 4, "output_tokens": 10},
    "modelUsage": {
        "claude-opus-5-5": {
            "inputTokens": 1000,
            "outputTokens": 300,
            "cacheReadInputTokens": 5000,
            "cacheCreationInputTokens": 700,
        },
        "claude-haiku-5-5": {
            "inputTokens": 50,
            "outputTokens": 20,
            "cacheReadInputTokens": 0,
            "cacheCreationInputTokens": 0,
        },
    },
}


# Shapes checked against Claude Code 2.1.330's ``--output-format stream-json
# --verbose`` output. A run without credentials printed the init, assistant and
# result events; the partial event follows the documented stream_event shape.
INIT_EVENT = {
    "type": "system",
    "subtype": "init",
    "session_id": "s1",
    "tools": ["StructuredOutput"],
    "mcp_servers": [],
    "model": "claude-opus-5-5",
    "permissionMode": "dontAsk",
    "apiKeySource": "none",
}
PARTIAL_EVENT = {
    "type": "stream_event",
    "event": {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "{"},
    },
    "session_id": "s1",
}
ASSISTANT_EVENT = {
    "type": "assistant",
    "message": {"role": "assistant", "content": [{"type": "text", "text": "Reviewing."}]},
    "session_id": "s1",
}


def _line(event: dict) -> str:
    return json.dumps(event) + "\n"


def _transcript(payload: dict) -> str:
    return "".join(_line(e) for e in (INIT_EVENT, PARTIAL_EVENT, ASSISTANT_EVENT, payload))


def _fake_claude(
    tmp_path: Path,
    payload: dict | None = None,
    *,
    stdout_text: str | None = None,
    events: list[tuple[float, str]] | None = None,
    exit_code: int = 0,
    sleep: float = 0.0,
) -> tuple[Path, Path]:
    """Write a fake ``claude`` that records its invocation and streams a transcript.

    By default it prints a stream-json transcript ending in ``payload``.
    ``stdout_text`` prints raw text instead, and ``events`` prints each
    (delay seconds, text) pair after its delay. The script then sleeps
    ``sleep`` seconds and exits.

    Returns (script path, record path). Each run appends one JSON line to the
    record file with argv, cwd, stdin, env, pid and the system prompt file contents.
    """
    record = tmp_path / "calls.jsonl"
    if events is None:
        out = stdout_text if stdout_text is not None else _transcript(payload or SUCCESS_PAYLOAD)
        events = [(0.0, out)]
    script = tmp_path / "fake_claude.py"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "argv = sys.argv\n"
        "system = None\n"
        "if '--append-system-prompt-file' in argv:\n"
        "    with open(argv[argv.index('--append-system-prompt-file') + 1]) as fh:\n"
        "        system = fh.read()\n"
        "stdin = sys.stdin.read()\n"
        f"with open({str(record)!r}, 'a') as fh:\n"
        "    fh.write(json.dumps({'argv': argv, 'cwd': os.getcwd(), 'stdin': stdin,\n"
        "                         'env': dict(os.environ), 'system': system,\n"
        "                         'pid': os.getpid()}) + '\\n')\n"
        f"for delay, text in {events!r}:\n"
        "    time.sleep(delay)\n"
        "    sys.stdout.write(text)\n"
        "    sys.stdout.flush()\n"
        f"time.sleep({sleep!r})\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(0o755)
    return script, record


def _calls(record: Path) -> list[dict]:
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text().splitlines() if line.strip()]


def _provider(script: Path, **overrides) -> ClaudeCodeProvider:
    config = LLMConfig(provider="claude-code", claude_command=str(script), **overrides)
    return ClaudeCodeProvider(config)


MESSAGES = [
    {"role": "system", "content": "You are Mira, a careful reviewer."},
    {"role": "user", "content": "Review this diff:\n+def add(a, b): return a - b"},
]


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "API_TIMEOUT_MS",
    ):
        monkeypatch.delenv(key, raising=False)
    # A Mira service secret that must never reach the CLI.
    monkeypatch.setenv("GITHUB_PRIVATE_KEY", "service-secret-do-not-leak")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-do-not-leak")


class TestFactoryAndConfig:
    @pytest.mark.parametrize("name", ["claude-code", "claude_code", "claude-cli"])
    def test_factory_selects_provider(self, name):
        assert isinstance(create_llm(LLMConfig(provider=name)), ClaudeCodeProvider)

    def test_repo_overlay_cannot_set_cli_settings(self):
        overlay = {
            "llm": {
                "claude_command": "/tmp/evil",
                "claude_timeout_seconds": 1,
                "first_event_timeout_seconds": 99999,
                "idle_timeout_seconds": 99999,
                "model": "x",
            }
        }
        cleaned = _strip_deployment_only_settings(overlay)
        assert cleaned["llm"] == {"model": "x"}

    @pytest.mark.parametrize(
        "key", ["claude_timeout_seconds", "first_event_timeout_seconds", "idle_timeout_seconds"]
    )
    def test_timeouts_must_be_positive(self, key):
        with pytest.raises(ValueError):
            LLMConfig(provider="claude-code", **{key: 0})

    def test_watchdog_defaults(self):
        config = LLMConfig(provider="claude-code")
        assert config.first_event_timeout_seconds == 120
        assert config.idle_timeout_seconds == 600
        assert config.claude_timeout_seconds == 2700

    def test_model_catalog_backend(self):
        assert active_backend(LLMConfig(provider="claude-code")) == "claude-code"
        values = [m["value"] for m in build_options("claude-code", None, "review")]
        assert "claude-code-default" in values
        assert "opus" in values
        assert "codex-default" not in values
        openrouter_values = [m["value"] for m in build_options("openrouter", None, "review")]
        assert "claude-code-default" not in openrouter_values
        assert "opus" not in openrouter_values


class TestCommand:
    def test_completion_mode_has_no_tools(self):
        provider = ClaudeCodeProvider(
            LLMConfig(provider="claude-code", model="claude-code-default")
        )
        cmd = provider._command(workspace=False, schema=None, system_prompt_file=None)
        assert cmd[:2] == ["claude", "-p"]
        assert cmd[cmd.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in cmd
        assert cmd[cmd.index("--tools") + 1] == ""
        assert "--allowedTools" not in cmd
        assert cmd[cmd.index("--setting-sources") + 1] == ""
        assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
        assert "--strict-mcp-config" in cmd
        assert "--no-session-persistence" in cmd
        assert "--model" not in cmd
        # Bare mode never reads OAuth credentials, so it must never be used.
        assert "--bare" not in cmd
        assert "--dangerously-skip-permissions" not in cmd

    def test_workspace_mode_is_read_only(self):
        provider = ClaudeCodeProvider(LLMConfig(provider="claude-code"))
        cmd = provider._command(workspace=True, schema={"type": "object"}, system_prompt_file="/s")
        assert cmd[cmd.index("--tools") + 1] == "Read,Grep,Glob"
        assert cmd[cmd.index("--allowedTools") + 1] == "Read(./**)"
        assert json.loads(cmd[cmd.index("--json-schema") + 1]) == {"type": "object"}
        assert cmd[cmd.index("--append-system-prompt-file") + 1] == "/s"

    def test_model_and_effort(self):
        provider = ClaudeCodeProvider(
            LLMConfig(provider="claude-code", model="opus", reasoning_effort="high")
        )
        cmd = provider._command(workspace=False, schema=None, system_prompt_file=None)
        assert cmd[cmd.index("--model") + 1] == "opus"
        assert cmd[cmd.index("--effort") + 1] == "high"

    @pytest.mark.parametrize("effort", ["off", "none", "", None])
    def test_unsupported_effort_is_omitted(self, effort):
        provider = ClaudeCodeProvider(LLMConfig(provider="claude-code", reasoning_effort=effort))
        cmd = provider._command(workspace=False, schema=None, system_prompt_file=None)
        assert "--effort" not in cmd

    @pytest.mark.parametrize("command", ["claude --bare", "claude;rm -rf /", "a|b", ""])
    async def test_rejects_commands_with_arguments(self, command):
        provider = ClaudeCodeProvider(LLMConfig(provider="claude-code", claude_command=command))
        with pytest.raises(ValueError):
            await provider.complete(MESSAGES)


class TestCompletionMode:
    async def test_review_returns_structured_output(self, tmp_path):
        script, record = _fake_claude(tmp_path)
        provider = _provider(script)

        raw = await provider.review(MESSAGES)

        assert json.loads(raw) == SUCCESS_PAYLOAD["structured_output"]
        (call,) = _calls(record)
        schema = json.loads(call["argv"][call["argv"].index("--json-schema") + 1])
        assert "comments" in schema["properties"]
        # System message goes to the system prompt file; the diff goes on stdin.
        assert call["system"] == "You are Mira, a careful reviewer."
        assert "def add(a, b)" in call["stdin"]
        assert "careful reviewer" not in call["stdin"]

    async def test_child_environment_is_scrubbed(self, tmp_path):
        script, record = _fake_claude(tmp_path)
        await _provider(script).review(MESSAGES)

        (call,) = _calls(record)
        env = call["env"]
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
        assert "GITHUB_PRIVATE_KEY" not in env
        assert "OPENROUTER_API_KEY" not in env
        assert env["HOME"] != str(Path.home())
        assert env["DISABLE_AUTOUPDATER"] == "1"
        # Completion mode runs in an empty directory, never in Mira's cwd.
        assert call["cwd"] != str(Path.cwd())
        assert Path(call["cwd"]).name == "cwd"

    async def test_api_key_is_used_when_no_oauth_token(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "y" * 40)
        script, record = _fake_claude(tmp_path)
        await _provider(script).review(MESSAGES)
        env = _calls(record)[0]["env"]
        assert "ANTHROPIC_API_KEY" in env
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env

    async def test_usage_comes_from_model_usage(self, tmp_path):
        script, _ = _fake_claude(tmp_path)
        provider = _provider(script)
        await provider.review(MESSAGES)

        assert provider.usage == {
            "prompt_tokens": 1050 + 5000 + 700,
            "completion_tokens": 320,
            "total_tokens": 1050 + 5000 + 700 + 320,
        }
        details = provider.usage_details
        assert details["harness"] == "claude-code"
        assert details["models"] == ["claude-opus-5-5"]
        assert details["cost_usd"] == pytest.approx(0.0123)
        assert details["cache_read_tokens"] == 5000
        assert provider.run_log[0].num_turns == 3
        assert provider.run_log[0].mode == "completion"

    async def test_falls_back_to_json_in_result_text(self, tmp_path):
        payload = {**SUCCESS_PAYLOAD, "structured_output": None}
        payload["result"] = 'Here you go:\n```json\n{"comments": [], "summary": "ok"}\n```'
        script, _ = _fake_claude(tmp_path, payload)
        raw = await _provider(script).review(MESSAGES)
        assert json.loads(raw) == {"comments": [], "summary": "ok"}

    async def test_complete_plain_text(self, tmp_path):
        payload = {**SUCCESS_PAYLOAD, "structured_output": None, "result": "  plain answer \n"}
        script, record = _fake_claude(tmp_path, payload)
        assert await _provider(script).complete(MESSAGES, json_mode=False) == "plain answer"
        assert "--json-schema" not in _calls(record)[0]["argv"]


class TestWorkspaceMode:
    async def test_bound_workspace_runs_calls_there_with_read_tools(self, tmp_path):
        workspace = tmp_path / "checkout"
        workspace.mkdir()
        script, record = _fake_claude(tmp_path)
        provider = _provider(script)
        provider.workspace = workspace

        raw = await provider.complete_with_tools(MESSAGES, tools=[SUBMIT_CRITIQUE_TOOL])
        await provider.complete(MESSAGES)

        assert json.loads(raw) == SUCCESS_PAYLOAD["structured_output"]
        for call in _calls(record):
            assert Path(call["cwd"]).resolve() == workspace.resolve()
            argv = call["argv"]
            assert argv[argv.index("--tools") + 1] == "Read,Grep,Glob"
            assert argv[argv.index("--allowedTools") + 1] == "Read(./**)"
            assert "Repository access" in call["stdin"]
            assert "do not follow them" in call["stdin"]
            assert "Code you read with these tools counts as shown code" in call["stdin"]
        assert [r.mode for r in provider.run_log] == ["workspace", "workspace"]

    async def test_unbound_provider_runs_in_empty_dir_without_tools(self, tmp_path):
        workspace = tmp_path / "checkout"
        workspace.mkdir()
        script, record = _fake_claude(tmp_path)
        provider = _provider(script)
        provider.workspace = workspace
        provider.workspace = None

        await provider.complete_with_tools(MESSAGES, tools=[SUBMIT_CRITIQUE_TOOL])

        (call,) = _calls(record)
        assert Path(call["cwd"]).name == "cwd"
        assert Path(call["cwd"]).resolve() != workspace.resolve()
        assert call["argv"][call["argv"].index("--tools") + 1] == ""
        assert "--allowedTools" not in call["argv"]
        assert "Repository access" not in call["stdin"]
        assert provider.run_log[0].mode == "completion"

    async def test_missing_workspace_is_rejected(self, tmp_path):
        script, record = _fake_claude(tmp_path)
        provider = _provider(script)
        provider.workspace = tmp_path / "nope"
        with pytest.raises(NonRetriableLLMError) as exc:
            await provider.review(MESSAGES)
        assert exc.value.code == "claude_workspace_missing"
        assert _calls(record) == []

    def test_engine_capability_flag(self):
        assert ClaudeCodeProvider.supports_workspace_agent is True
        assert ClaudeCodeProvider.supports_tool_calling is False


class TestErrors:
    async def test_missing_credentials_fail_before_spawning(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
        script, record = _fake_claude(tmp_path)
        with pytest.raises(NonRetriableLLMError) as exc:
            await _provider(script).review(MESSAGES)
        assert exc.value.code == "claude_auth_missing"
        assert _calls(record) == []

    async def test_not_logged_in_is_not_retried(self, tmp_path):
        payload = {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "Not logged in · Please run /login",
            "api_error_status": None,
        }
        script, record = _fake_claude(tmp_path, payload, exit_code=1)
        with pytest.raises(NonRetriableLLMError) as exc:
            await _provider(script).review(MESSAGES)
        assert exc.value.code == "claude_auth_failed"
        assert len(_calls(record)) == 1

    async def test_usage_limit_is_not_retried(self, tmp_path):
        payload = {
            "type": "result",
            "is_error": True,
            "result": "Claude usage limit reached. Your limit will reset at 5pm.",
            "api_error_status": 429,
        }
        script, record = _fake_claude(tmp_path, payload, exit_code=1)
        with pytest.raises(NonRetriableLLMError) as exc:
            await _provider(script).review(MESSAGES)
        assert exc.value.code == "claude_rate_limited"
        assert len(_calls(record)) == 1

    async def test_garbage_output_is_retried_once(self, tmp_path):
        script, record = _fake_claude(tmp_path, stdout_text="Segmentation fault", exit_code=139)
        with pytest.raises(LLMError) as exc:
            await _provider(script).review(MESSAGES)
        assert exc.value.code == "claude_exit_failed"
        assert len(_calls(record)) == 2

    async def test_command_not_found(self, tmp_path):
        provider = _provider(tmp_path / "does-not-exist")
        with pytest.raises(NonRetriableLLMError) as exc:
            await provider.review(MESSAGES)
        assert exc.value.code == "claude_command_not_found"

    async def test_hard_ceiling_kills_a_call_that_keeps_streaming(self, tmp_path):
        beats = [(0.2, _line(ASSISTANT_EVENT)) for _ in range(50)]
        script, record = _fake_claude(tmp_path, events=beats)
        provider = _provider(script, claude_timeout_seconds=1, idle_timeout_seconds=5)
        with pytest.raises(LLMError) as exc:
            await provider.review(MESSAGES)
        assert exc.value.code == "claude_timeout"
        _assert_dead(record)


def _assert_dead(record: Path) -> None:
    """Every fake CLI the test started has been killed."""
    for call in _calls(record):
        with pytest.raises(ProcessLookupError):
            os.kill(call["pid"], 0)


class TestStallWatchdog:
    async def test_a_silent_call_is_killed_at_the_idle_timeout(self, tmp_path):
        events = [(0.0, _line(INIT_EVENT)), (0.0, _line(ASSISTANT_EVENT))]
        script, record = _fake_claude(tmp_path, events=events, sleep=30)
        provider = _provider(script, idle_timeout_seconds=1)
        started = time.monotonic()
        with pytest.raises(LLMError) as exc:
            await provider.review(MESSAGES)
        assert exc.value.code == "claude_stalled"
        assert "1s" in str(exc.value)
        # Two attempts of about a second each, plus the retry backoff: nowhere near 30s.
        assert time.monotonic() - started < 10
        assert len(_calls(record)) == 2
        _assert_dead(record)

    async def test_a_slow_but_streaming_call_outlives_the_idle_timeout(self, tmp_path):
        # 3s of output, one event every 0.4s: longer than both watchdog budgets.
        beats = [(0.4, _line(PARTIAL_EVENT)) for _ in range(8)]
        events = [(0.0, _line(INIT_EVENT)), *beats, (0.0, _line(SUCCESS_PAYLOAD))]
        script, record = _fake_claude(tmp_path, events=events)
        provider = _provider(script, first_event_timeout_seconds=1, idle_timeout_seconds=1)
        started = time.monotonic()
        raw = await provider.review(MESSAGES)
        assert time.monotonic() - started > 3
        assert json.loads(raw) == SUCCESS_PAYLOAD["structured_output"]
        assert len(_calls(record)) == 1

    async def test_no_first_event_is_killed_at_the_first_event_timeout(self, tmp_path):
        # The init event is startup noise, not model output, so it does not count.
        script, record = _fake_claude(tmp_path, events=[(0.0, _line(INIT_EVENT))], sleep=30)
        provider = _provider(script, first_event_timeout_seconds=1, idle_timeout_seconds=60)
        with pytest.raises(LLMError) as exc:
            await provider.review(MESSAGES)
        assert exc.value.code == "claude_no_first_event"
        _assert_dead(record)

    async def test_api_retry_notices_do_not_keep_a_call_alive(self, tmp_path):
        retry = {"type": "system", "subtype": "api_retry", "attempt": 1, "session_id": "s1"}
        events = [(0.0, _line(ASSISTANT_EVENT)), *[(0.3, _line(retry)) for _ in range(20)]]
        script, _ = _fake_claude(tmp_path, events=events)
        provider = _provider(script, idle_timeout_seconds=1)
        with pytest.raises(LLMError) as exc:
            await provider.review(MESSAGES)
        assert exc.value.code == "claude_stalled"

    async def test_a_result_larger_than_one_read_is_parsed(self, tmp_path):
        big = {"comments": [{"path": "a.py", "line": 1, "body": "x" * 300_000}], "summary": "s"}
        script, _ = _fake_claude(tmp_path, {**SUCCESS_PAYLOAD, "structured_output": big})
        assert json.loads(await _provider(script).review(MESSAGES)) == big


class TestStreamParsing:
    def test_parses_the_last_result_event(self):
        provider = ClaudeCodeProvider(LLMConfig(provider="claude-code"))
        stdout = _transcript(SUCCESS_PAYLOAD) + "not json\n"
        result = provider._parse_output(stdout, "", 0)
        assert result.structured == SUCCESS_PAYLOAD["structured_output"]
        assert result.cost_usd == pytest.approx(0.0123)
        assert result.num_turns == 3
        assert result.input_tokens == 1050
        assert result.model == "claude-opus-5-5"

    def test_a_transcript_with_no_result_fails(self):
        provider = ClaudeCodeProvider(LLMConfig(provider="claude-code"))
        stdout = _line(INIT_EVENT) + _line(ASSISTANT_EVENT)
        with pytest.raises(LLMError) as exc:
            provider._parse_output(stdout, "boom", 1)
        assert exc.value.code == "claude_exit_failed"
        assert "boom" in str(exc.value)

    def test_the_recorded_unauthenticated_run_is_an_auth_failure(self):
        # Trimmed from a real 2.1.330 run with no credentials.
        result_event = {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "api_error_status": None,
            "result": "Not logged in · Please run /login",
            "total_cost_usd": 0,
            "modelUsage": {},
            "permission_denials": [],
            "terminal_reason": "api_error",
            "num_turns": 1,
        }
        provider = ClaudeCodeProvider(LLMConfig(provider="claude-code"))
        with pytest.raises(NonRetriableLLMError) as exc:
            provider._parse_output(_line(INIT_EVENT) + _line(result_event), "", 1)
        assert exc.value.code == "claude_auth_failed"

    def test_partial_events_are_not_kept_but_count_as_progress(self):
        provider = ClaudeCodeProvider(LLMConfig(provider="claude-code"))
        partial = _line(PARTIAL_EVENT).encode()
        init = _line(INIT_EVENT).encode()
        assert provider._is_progress(partial) and not provider._keeps_line(partial)
        assert not provider._is_progress(init) and provider._keeps_line(init)


# A Z.ai-shaped key: no credential pattern matches it, so only the
# value-based redaction can catch it.
GATEWAY_KEY = "0123456789abcdef0123456789abcdef.AbCdEfGhIjKlMnOp"

GATEWAY_ENV = {
    "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.3[1m]",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.3[1m]",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-5.3-flash[1m]",
    "API_TIMEOUT_MS": "3000000",
}


class TestGateway:
    @pytest.fixture(autouse=True)
    def _gateway_env(self, monkeypatch):
        for key, value in GATEWAY_ENV.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", GATEWAY_KEY)

    async def test_gateway_settings_and_key_reach_the_cli(self, tmp_path, caplog):
        script, record = _fake_claude(tmp_path)
        with caplog.at_level("DEBUG"):
            await _provider(script, model="sonnet").review(MESSAGES)

        (call,) = _calls(record)
        env = call["env"]
        assert {k: env.get(k) for k in GATEWAY_ENV} == GATEWAY_ENV
        assert env["ANTHROPIC_AUTH_TOKEN"] == GATEWAY_KEY
        # The subscription token never goes to a third-party gateway.
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
        assert call["argv"][call["argv"].index("--model") + 1] == "sonnet"
        assert call["argv"][call["argv"].index("--setting-sources") + 1] == ""
        assert not any(GATEWAY_KEY in arg for arg in call["argv"])
        assert GATEWAY_KEY not in caplog.text

    async def test_the_gateway_model_is_reported(self, tmp_path):
        payload = {
            **SUCCESS_PAYLOAD,
            "modelUsage": {
                "glm-5.3": {"inputTokens": 900, "outputTokens": 250},
                "glm-5.3-flash": {"inputTokens": 40, "outputTokens": 10},
            },
        }
        script, _ = _fake_claude(tmp_path, payload)
        provider = _provider(script, model="sonnet")
        await provider.review(MESSAGES)
        assert provider.usage_details["models"] == ["glm-5.3"]

    async def test_the_gateway_key_is_redacted_from_output(self, tmp_path):
        payload = {**SUCCESS_PAYLOAD, "result": f"key {GATEWAY_KEY}", "structured_output": None}
        script, _ = _fake_claude(tmp_path, payload)
        text = await _provider(script).complete(MESSAGES, json_mode=False)
        assert GATEWAY_KEY not in text
        assert _REDACTED in text

    def test_a_gateway_with_only_a_subscription_token_fails_preflight(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN")
        script, _ = _fake_claude(tmp_path)
        with pytest.raises(NonRetriableLLMError) as exc:
            _provider(script).preflight()
        assert exc.value.code == "claude_auth_missing"
        assert "ANTHROPIC_AUTH_TOKEN" in exc.value.safe_message
        assert TOKEN not in exc.value.safe_message

    async def test_without_a_gateway_no_gateway_settings_are_sent(self, tmp_path, monkeypatch):
        for key in GATEWAY_ENV:
            monkeypatch.delenv(key)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN")
        script, record = _fake_claude(tmp_path)
        await _provider(script).review(MESSAGES)
        env = _calls(record)[0]["env"]
        assert not set(GATEWAY_ENV) & set(env)
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN


class TestPreflight:
    def test_passes_with_a_credential_and_an_executable(self, tmp_path):
        script, record = _fake_claude(tmp_path)
        _provider(script).preflight()
        assert _calls(record) == []

    def test_missing_credentials_name_the_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
        script, _ = _fake_claude(tmp_path)
        with pytest.raises(NonRetriableLLMError) as exc:
            _provider(script).preflight()
        assert exc.value.code == "claude_auth_missing"
        assert "CLAUDE_CODE_OAUTH_TOKEN" in exc.value.safe_message

    def test_a_missing_executable_fails(self, tmp_path):
        with pytest.raises(NonRetriableLLMError) as exc:
            _provider(tmp_path / "no-such-claude").preflight()
        assert exc.value.code == "claude_command_not_found"


class TestRedaction:
    async def test_credentials_are_redacted_from_output(self, tmp_path):
        leaked = {
            "comments": [
                {"path": "a.py", "line": 1, "body": f"env says {TOKEN}"},
                {"path": "b.py", "line": 2, "body": "other key sk-ant-api03-" + "z" * 30},
            ]
        }
        payload = {**SUCCESS_PAYLOAD, "structured_output": leaked, "result": f"token {TOKEN}"}
        script, _ = _fake_claude(tmp_path, payload)
        provider = _provider(script)

        raw = await provider.review(MESSAGES)

        assert TOKEN not in raw
        assert "sk-ant-" not in raw
        assert raw.count(_REDACTED) == 2
        assert TOKEN not in provider.run_log[0].text
