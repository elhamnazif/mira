"""Model slugs: one stable ``provider/alias`` name per model, such as ``zai/glm``.

A slug resolves to a concrete model id and says how to run it: which harness
and credentials can serve it, which slug to try when the model is missing or
unavailable, which cheaper sibling runs the indexing and security passes, and
which effort levels the model accepts. The data lives in ``slugs.json``. Wire
quirks stay in ``providers.json``, which a slug's provider names by profile.

A plain model id is not a slug, and nothing here touches it.

Adapted from Pullfrog's model registry (``models.ts``, ``effort.ts`` and
``utils/agent.ts``), MIT License, Copyright (c) 2026 Pullfrog, Inc.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from mira.config import LLMConfig
from mira.exceptions import ConfigError
from mira.llm import provider_profiles as profiles
from mira.llm.agent_cli import CLAUDE_CODE_PROVIDER_NAMES
from mira.llm.agent_cli.claude import _EFFORT_LEVELS as CLAUDE_EFFORT_LEVELS

_PATH = Path(__file__).parent / "slugs.json"

# Mira's effort names as points on [0, 1]: 0 is a model's cheapest level and
# 1 its most expensive. A model's own ladder decides what each point runs as.
EFFORT_POSITIONS: Mapping[str, float] = {
    "low": 0.0,
    "medium": 0.25,
    "high": 0.5,
    "xhigh": 0.75,
    "max": 1.0,
}
# Levels that turn reasoning off, which no review wants.
_DISABLING_LEVELS = frozenset({"none", "minimal"})

# A gateway such as Z.ai's Anthropic endpoint that only Claude Code can use.
_GATEWAY_ENV = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")


class Harness(StrEnum):
    """What runs a slug. Each value is the ``llm.provider`` that selects it."""

    CLAUDE_CODE = "claude-code"
    # Chosen only by an explicit override until the harness exists.
    OPENCODE = "opencode"
    # The OpenAI-compatible client, pointed at a providers.json profile.
    API = "openai"


@dataclass(frozen=True)
class ModelSlug:
    slug: str
    provider: str
    resolve: str
    effort: tuple[str, ...] = ()
    fallback: str | None = None
    cheap: str | None = None


@dataclass(frozen=True)
class SlugProvider:
    profile: str
    harness: Harness | None = None
    harness_env: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarnessChoice:
    harness: Harness
    # The model id as this harness takes it.
    model: str
    # The providers.json profile, for the API harness only.
    profile: str | None
    # Which rule chose the harness, for the log.
    reason: str


@dataclass(frozen=True)
class _Registry:
    providers: dict[str, SlugProvider]
    slugs: dict[str, ModelSlug]


@lru_cache(maxsize=1)
def _load() -> _Registry:
    raw = json.loads(_PATH.read_text())
    providers = {
        name: SlugProvider(
            profile=entry["profile"],
            harness=Harness(entry["harness"]) if entry.get("harness") else None,
            harness_env=tuple(entry.get("harness_env", ())),
        )
        for name, entry in raw["providers"].items()
    }
    slugs = {
        slug: ModelSlug(
            slug=slug,
            provider=slug.split("/", 1)[0],
            resolve=entry["resolve"],
            effort=tuple(entry.get("effort", ())),
            fallback=entry.get("fallback"),
            cheap=entry.get("cheap"),
        )
        for slug, entry in raw["slugs"].items()
    }
    return _Registry(providers, slugs)


def all_slugs() -> dict[str, ModelSlug]:
    return _load().slugs


def get(model: str) -> ModelSlug | None:
    """The slug named ``model``, or None for a plain model id."""
    return _load().slugs.get(model)


def provider_of(slug: ModelSlug) -> SlugProvider:
    return _load().providers[slug.provider]


def credentials(model: str) -> tuple[str, ...]:
    """The env vars that can authenticate ``model``: its vendor harness's,
    then its API profile's key."""
    slug = _require(model)
    provider = provider_of(slug)
    api_key_env = (profiles.get(provider.profile) or {}).get("api_key_env")
    return provider.harness_env + ((api_key_env,) if api_key_env else ())


def cheap_for(model: str) -> str:
    """``model``'s cheaper sibling for indexing and security, else ``model``."""
    slug = get(model)
    return slug.cheap if slug and slug.cheap else model


def effort_level(
    ladder: Sequence[str], level: str | None, accepted: Collection[str] | None = None
) -> str | None:
    """Land Mira's effort ``level`` on a model's ``ladder``.

    A rung with the level's own name wins. Otherwise the level's position picks
    the nearest rung, ties going up, so ``medium`` on a ``low, high, max``
    ladder runs at ``high`` as the provider profile's effort map does, not at
    the cheapest rung. Rungs that turn reasoning off, and rungs outside
    ``accepted`` (what the harness's pinned CLI takes), are dropped first. None means no
    effort flag: reasoning is off, or the model has no effort control. A level
    Mira has no position for passes through unchanged.
    """
    if level is None:
        return None
    position = EFFORT_POSITIONS.get(level)
    if position is None:
        return level
    rungs = [
        rung
        for rung in ladder
        if rung not in _DISABLING_LEVELS and (accepted is None or rung in accepted)
    ]
    if not rungs:
        return None
    if level in rungs:
        return level
    return rungs[int(position * (len(rungs) - 1) + 0.5)]


def choose_harness(
    model: str, env: Mapping[str, str], override: str | None = None
) -> HarnessChoice:
    """Pick the harness for slug ``model``. Pure: reads only its arguments.

    1. An explicit ``override`` (``llm.provider``, ``MIRA_RUNNER_LLM_PROVIDER``)
       wins: an agent CLI, ``openai`` for the slug's own API profile, or a
       providers.json profile name.
    2. The vendor's own harness, when one of its credentials is set.
    3. A Claude Code gateway (``ANTHROPIC_BASE_URL`` plus
       ``ANTHROPIC_AUTH_TOKEN``) forces Claude Code.
    4. Otherwise the API harness with the slug's profile.
    """
    slug = _require(model)
    provider = provider_of(slug)
    if override:
        harness, profile = _parse_override(override, provider.profile)
        return _choice(slug, harness, profile, f"override {override!r}")
    if provider.harness and any(env.get(key) for key in provider.harness_env):
        return _choice(slug, provider.harness, None, f"{slug.provider} credential set")
    if all(env.get(key) for key in _GATEWAY_ENV):
        return _choice(slug, Harness.CLAUDE_CODE, None, "Claude Code gateway set")
    return _choice(slug, Harness.API, provider.profile, f"API profile {provider.profile!r}")


def resolve_config(config: LLMConfig, env: Mapping[str, str]) -> LLMConfig:
    """``config`` with its slug resolved, or ``config`` itself for a plain id.

    Sets the harness, the concrete model, the fallback model (unless one is
    set) and the effort level the model accepts. The API harness also takes
    the profile's endpoint, unless ``llm.base_url`` was set explicitly.
    """
    slug = get(config.model)
    if slug is None:
        return config
    explicit = config.model_fields_set
    choice = choose_harness(
        config.model, env, override=config.provider if "provider" in explicit else None
    )
    accepted = CLAUDE_EFFORT_LEVELS if choice.harness is Harness.CLAUDE_CODE else None
    update: dict[str, object] = {
        "provider": choice.harness.value,
        "model": choice.model,
        "reasoning_effort": effort_level(slug.effort, config.reasoning_effort, accepted),
    }
    if slug.fallback and not config.fallback_model:
        update["fallback_model"] = _wire_model(_require(slug.fallback).resolve, choice.harness)
    if choice.profile and "base_url" not in explicit:
        profile = profiles.get(choice.profile) or {}
        update["base_url"] = profile["base_url"]
        update["api_key_env"] = profile.get("api_key_env") or ""
    return config.model_copy(update=update)


def _require(model: str) -> ModelSlug:
    slug = get(model)
    if slug is None:
        raise ConfigError(f"{model!r} is not a model slug")
    return slug


def _parse_override(override: str, slug_profile: str) -> tuple[Harness, str | None]:
    if override in CLAUDE_CODE_PROVIDER_NAMES:
        return Harness.CLAUDE_CODE, None
    if override == Harness.OPENCODE:
        return Harness.OPENCODE, None
    if override == Harness.API:
        return Harness.API, slug_profile
    if profiles.get(override) is not None:
        return Harness.API, override
    raise ConfigError(
        f"LLM provider {override!r} cannot run a model slug: use claude-code, opencode, "
        "openai or a providers.json profile, or a plain model id"
    )


def _choice(slug: ModelSlug, harness: Harness, profile: str | None, reason: str) -> HarnessChoice:
    return HarnessChoice(harness, _wire_model(slug.resolve, harness), profile, reason)


def _wire_model(resolved: str, harness: Harness) -> str:
    """Claude Code takes the bare id (``claude-sonnet-4-6``). The API harness's
    profile strips or keeps the prefix itself, and OpenCode wants it."""
    return resolved.split("/", 1)[1] if harness is Harness.CLAUDE_CODE else resolved
