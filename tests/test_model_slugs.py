"""Model slugs: resolution, harness choice, effort ladders, cheap siblings and fallback."""

from __future__ import annotations

import pytest

from mira.exceptions import ConfigError
from mira.llm import provider_profiles, slugs
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
