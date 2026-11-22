"""Model slugs: resolution, harness choice, effort ladders, cheap siblings and fallback."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mira.config import LLMConfig, MiraConfig
from mira.dashboard.models_config import llm_config_for
from mira.exceptions import ConfigError, LLMError, NonRetriableLLMError
from mira.llm import create_llm, provider_profiles, slugs
from mira.llm.agent_cli import ClaudeCodeProvider
from mira.llm.provider import LLMProvider
from mira.llm.slugs import Harness

_CLAUDE_ENV = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
)


@pytest.fixture
def no_claude_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _CLAUDE_ENV:
        monkeypatch.delenv(key, raising=False)


# ── The bundled data ─────────────────────────────────────────────────


@pytest.mark.parametrize("slug", sorted(slugs.all_slugs()))
def test_every_slug_points_at_known_data(slug: str) -> None:
    model = slugs.all_slugs()[slug]
    assert provider_profiles.get(slugs.provider_of(model).profile) is not None
    assert model.resolve.startswith(f"{model.provider}/")
    # A fallback or a cheap sibling runs on the same harness and endpoint.
    for sibling in (model.fallback, model.cheap):
        if sibling is not None:
            assert slugs.get(sibling) is not None
            assert slugs.all_slugs()[sibling].provider == model.provider


def test_a_slug_resolves_to_a_concrete_model() -> None:
    model = slugs.get("anthropic/claude-sonnet")
    assert model is not None
    assert model.resolve == "anthropic/claude-sonnet-4-6"
    assert model.cheap == "anthropic/claude-haiku"
    assert slugs.get("zai/glm") == slugs.ModelSlug(
        slug="zai/glm",
        provider="zai",
        resolve="zai/glm-5.3",
        effort=("low", "high", "max"),
        fallback="zai/glm-flash",
        cheap="zai/glm-flash",
    )


def test_a_plain_model_id_is_not_a_slug() -> None:
    assert slugs.get("anthropic/claude-sonnet-4-6") is None
    assert slugs.cheap_for("anthropic/claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"


def test_credentials_per_provider() -> None:
    assert slugs.credentials("anthropic/claude-opus") == (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    )
    assert slugs.credentials("zai/glm") == ("ZAI_API_KEY",)
    assert slugs.credentials("deepseek/deepseek-flash") == ("DEEPSEEK_API_KEY",)


def test_cheap_sibling() -> None:
    assert slugs.cheap_for("anthropic/claude-opus") == "anthropic/claude-sonnet"
    assert slugs.cheap_for("zai/glm") == "zai/glm-flash"
    # The cheapest tier is its own cheap sibling.
    assert slugs.cheap_for("anthropic/claude-haiku") == "anthropic/claude-haiku"


# ── Effort ladders ───────────────────────────────────────────────────

FIVE = ("low", "medium", "high", "xhigh", "max")
THREE = ("low", "high", "max")


@pytest.mark.parametrize(
    ("ladder", "level", "accepted", "expected"),
    [
        # A five-rung ladder lands each level on itself.
        *((FIVE, level, None, level) for level in FIVE),
        # A shorter ladder rounds down: medium (0.25) and xhigh (0.75) fold onto lower rungs.
        (THREE, "low", None, "low"),
        (THREE, "medium", None, "low"),
        (THREE, "high", None, "high"),
        (THREE, "xhigh", None, "high"),
        (THREE, "max", None, "max"),
        # Rungs that turn reasoning off are dropped before the lookup.
        (("none", "low", "medium", "high", "xhigh"), "low", None, "low"),
        (("none", "low", "medium", "high", "xhigh"), "max", None, "xhigh"),
        (("minimal", "low", "medium", "high"), "medium", None, "low"),
        # So are rungs the harness's CLI does not take.
        (("low", "high", "ultra"), "max", frozenset({"low", "high"}), "high"),
        # Reasoning off, or a model with no effort control: no flag.
        (FIVE, None, None, None),
        ((), "high", None, None),
        (("none", "minimal"), "high", None, None),
        # A level Mira has no position for passes through.
        (THREE, "ultra", None, "ultra"),
    ],
)
def test_effort_level(
    ladder: tuple[str, ...], level: str | None, accepted: frozenset[str] | None, expected: str
) -> None:
    assert slugs.effort_level(ladder, level, accepted) == expected


# ── Harness choice ───────────────────────────────────────────────────

GATEWAY = {"ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic", "ANTHROPIC_AUTH_TOKEN": "k"}


@pytest.mark.parametrize(
    ("slug", "env", "override", "harness", "model", "profile"),
    [
        # 1. An explicit override wins over every credential.
        ("anthropic/claude-sonnet", {"CLAUDE_CODE_OAUTH_TOKEN": "t"}, "openai", Harness.API,
         "anthropic/claude-sonnet-4-6", "openrouter"),
        ("zai/glm", {"ZAI_API_KEY": "k"}, "claude-code", Harness.CLAUDE_CODE, "glm-5.3", None),
        ("zai/glm", {}, "claude_code", Harness.CLAUDE_CODE, "glm-5.3", None),
        ("zai/glm", {}, "together", Harness.API, "zai/glm-5.3", "together"),
        ("zai/glm", {}, "opencode", Harness.OPENCODE, "zai/glm-5.3", None),
        # 2. The vendor's harness when its credential is set.
        ("anthropic/claude-sonnet", {"CLAUDE_CODE_OAUTH_TOKEN": "t"}, None, Harness.CLAUDE_CODE,
         "claude-sonnet-4-6", None),
        ("anthropic/claude-opus", {"ANTHROPIC_API_KEY": "k"}, None, Harness.CLAUDE_CODE,
         "claude-opus-4-8", None),
        # An unset Actions variable arrives as an empty string.
        ("anthropic/claude-sonnet", {"CLAUDE_CODE_OAUTH_TOKEN": ""}, None, Harness.API,
         "anthropic/claude-sonnet-4-6", "openrouter"),
        # 3. A gateway forces Claude Code, also for another vendor's slug.
        ("zai/glm", GATEWAY, None, Harness.CLAUDE_CODE, "glm-5.3", None),
        ("zai/glm", {**GATEWAY, "ZAI_API_KEY": "k"}, None, Harness.CLAUDE_CODE, "glm-5.3", None),
        ("zai/glm", {"ANTHROPIC_BASE_URL": GATEWAY["ANTHROPIC_BASE_URL"]}, None, Harness.API,
         "zai/glm-5.3", "zai"),
        # 4. Otherwise the slug's API profile, with or without its key.
        ("zai/glm", {"ZAI_API_KEY": "k"}, None, Harness.API, "zai/glm-5.3", "zai"),
        ("deepseek/deepseek-flash", {}, None, Harness.API, "deepseek/deepseek-v4-flash",
         "deepseek"),
        ("anthropic/claude-sonnet", {}, None, Harness.API, "anthropic/claude-sonnet-4-6",
         "openrouter"),
    ],
)  # fmt: skip
def test_choose_harness(
    slug: str,
    env: dict[str, str],
    override: str | None,
    harness: Harness,
    model: str,
    profile: str | None,
) -> None:
    choice = slugs.choose_harness(slug, env, override)
    assert (choice.harness, choice.model, choice.profile) == (harness, model, profile)


@pytest.mark.parametrize("override", ["bedrock", "codex-cli", "zia"])
def test_an_override_that_cannot_run_a_slug_is_refused(override: str) -> None:
    with pytest.raises(ConfigError, match=override):
        slugs.choose_harness("zai/glm", {}, override)


# ── create_llm ───────────────────────────────────────────────────────


def _review_llm(llm: dict, monkeypatch: pytest.MonkeyPatch, **env: str):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = MiraConfig.model_validate({"llm": llm}).llm
    return create_llm(llm_config_for("review", config, use_dashboard=False))


def test_create_llm_runs_a_slug_on_its_api_profile(
    no_claude_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _review_llm({"model": "zai/glm"}, monkeypatch, ZAI_API_KEY="k" * 20)

    assert isinstance(llm, LLMProvider)
    assert llm.config.provider == "openai"
    assert llm.config.model == "zai/glm-5.3"
    assert llm.config.base_url == "https://api.z.ai/api/paas/v4"
    assert llm.config.api_key_env == "ZAI_API_KEY"
    assert llm.config.fallback_model == "zai/glm-5.3-flash"
    # Mira's default "medium" rounds down to GLM's floor rung.
    assert llm.config.reasoning_effort == "low"


def test_create_llm_runs_an_anthropic_slug_on_claude_code_with_a_token(
    no_claude_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _review_llm(
        {"model": "anthropic/claude-fable", "review_reasoning_effort": "xhigh"},
        monkeypatch,
        CLAUDE_CODE_OAUTH_TOKEN="t" * 20,
    )

    assert isinstance(llm, ClaudeCodeProvider)
    cmd = llm._command(workspace=False, schema=None, system_prompt_file=None)
    assert cmd[cmd.index("--model") + 1] == "claude-fable-5"
    assert cmd[cmd.index("--fallback-model") + 1] == "claude-opus-4-8"
    assert cmd[cmd.index("--effort") + 1] == "xhigh"


def test_an_explicit_endpoint_keeps_its_base_url(
    no_claude_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _review_llm(
        {"model": "deepseek/deepseek-pro", "base_url": "https://litellm.internal/v1"},
        monkeypatch,
    )

    assert llm.config.base_url == "https://litellm.internal/v1"
    assert llm.config.model == "deepseek/deepseek-v4-pro"
    assert llm.config.fallback_model == "deepseek/deepseek-v4-flash"


def test_an_explicit_fallback_model_wins(
    no_claude_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _review_llm({"model": "zai/glm", "fallback_model": "zai/glm-4.7"}, monkeypatch)
    assert llm.config.fallback_model == "zai/glm-4.7"


def test_the_opencode_slot_is_not_implemented(no_claude_env: None) -> None:
    with pytest.raises(ConfigError, match="opencode"):
        create_llm(LLMConfig(provider="opencode", model="zai/glm"))


@pytest.mark.parametrize(
    "llm",
    [
        {"model": "anthropic/claude-sonnet-4-6", "review_reasoning_effort": "max"},
        {"model": "glm-5.3", "provider": "openai", "base_url": "https://api.z.ai/api/paas/v4"},
        {"model": "sonnet", "provider": "claude-code"},
    ],
)
def test_a_plain_model_id_is_unchanged(
    llm: dict, monkeypatch: pytest.MonkeyPatch, no_claude_env: None
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "t" * 20)
    config = llm_config_for(
        "review", MiraConfig.model_validate({"llm": llm}).llm, use_dashboard=False
    )
    assert slugs.resolve_config(config, {"CLAUDE_CODE_OAUTH_TOKEN": "t"}) is config
    assert create_llm(config).config == config


# ── Cheap siblings ───────────────────────────────────────────────────


def test_indexing_and_security_take_the_cheap_sibling() -> None:
    config = MiraConfig.model_validate({"llm": {"model": "anthropic/claude-opus"}}).llm

    assert llm_config_for("review", config, use_dashboard=False).model == "anthropic/claude-opus"
    assert (
        llm_config_for("indexing", config, use_dashboard=False).model == "anthropic/claude-sonnet"
    )
    assert (
        llm_config_for("security", config, use_dashboard=False).model == "anthropic/claude-sonnet"
    )


def test_a_configured_pass_model_beats_the_cheap_sibling() -> None:
    config = MiraConfig.model_validate(
        {"llm": {"model": "zai/glm", "indexing_model": "zai/glm", "security_model": "zai/glm"}}
    ).llm

    assert llm_config_for("indexing", config, use_dashboard=False).model == "zai/glm"
    assert llm_config_for("security", config, use_dashboard=False).model == "zai/glm"


# ── Fallback on a missing model ──────────────────────────────────────


def _response(status: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


_OK = {"choices": [{"message": {"content": '{"ok": true}'}}]}
_MISSING = {"error": {"message": "The model `glm-5.3` does not exist", "code": "model_not_found"}}


async def _complete(config: LLMConfig, answer) -> tuple[str, list[str]]:
    models: list[str] = []

    async def post(*args, **kwargs):
        models.append(kwargs["json"]["model"])
        return answer(kwargs["json"]["model"])

    with patch("mira.llm.provider.httpx.AsyncClient") as client_cls:
        client = AsyncMock()
        client.post = AsyncMock(side_effect=post)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client_cls.return_value = client
        result = await LLMProvider(config).complete([{"role": "user", "content": "hi"}])
    return result, models


@pytest.mark.parametrize("status", [400, 404])
async def test_a_missing_model_falls_back_once(
    status: int, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "k" * 20)
    config = slugs.resolve_config(
        LLMConfig(model="zai/glm", reasoning_effort=None, max_retries=3), {}
    )

    result, models = await _complete(
        config,
        lambda model: _response(status, _MISSING) if model == "glm-5.3" else _response(200, _OK),
    )

    assert result == '{"ok": true}'
    # Not retried: one try of the model, one of its fallback.
    assert models == ["glm-5.3", "glm-5.3-flash"]
    assert "trying fallback zai/glm-5.3-flash" in caplog.text


async def test_a_missing_model_with_no_fallback_stays_a_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "k" * 20)
    config = slugs.resolve_config(LLMConfig(model="zai/glm-flash"), {})

    with pytest.raises(NonRetriableLLMError, match="model missing or unavailable"):
        await _complete(config, lambda model: _response(404, _MISSING))


async def test_another_client_error_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "k" * 20)
    config = slugs.resolve_config(LLMConfig(model="zai/glm"), {})

    with pytest.raises(NonRetriableLLMError) as exc:
        await _complete(config, lambda model: _response(400, {"error": "bad request"}))
    assert exc.value.code == "api_error"


async def test_both_models_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "k" * 20)
    config = slugs.resolve_config(LLMConfig(model="zai/glm"), {})

    with pytest.raises(LLMError, match="Both primary"):
        await _complete(config, lambda model: _response(404, _MISSING))
