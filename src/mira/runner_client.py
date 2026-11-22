"""``mira runner``: review one dispatched job inside its GitHub Actions run.

The run proves who it is with its Actions OIDC token, claims the job from
the server, reviews the PR with the real engine against the claimed state
snapshot, and posts the writes back in one result bundle. Exit codes:

- 0: the review finished and the server has its result, or the job was
  no longer this run's to claim (409).
- 1: the server refused or could not be reached, or the review failed.
  A failed review still posts ``outcome=failed`` first.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr, ValidationError

from mira.config import MiraConfig, _is_local_host, set_global_defaults_data
from mira.core.engine import ReviewEngine
from mira.dashboard.models_config import llm_config_for
from mira.exceptions import ConfigError, MiraError
from mira.llm import create_llm
from mira.llm import provider_profiles as profiles
from mira.llm.agent_cli import CLAUDE_CODE_PROVIDER_NAMES
from mira.llm.provider_errors import classify_provider_error
from mira.providers.github import _GITHUB_API_URL as GITHUB_API_URL
from mira.providers.github import GitHubProvider
from mira.runner_protocol import JobSpec, RemoteStateStore, ResultBundle, TokenGrant, UsageReport

logger = logging.getLogger(__name__)

# Waits between attempts after a network error, 502, 503 or 504. Tests zero them.
_BACKOFF_S: tuple[float, ...] = (2, 4, 8, 16, 30, 60, 60)
# A claim refused with 403 can be a run that started before the server
# stored its run id, so it is retried a few times.
_FORBIDDEN_RETRIES_S: tuple[float, ...] = (5, 10, 15)
_RETRY_STATUSES = frozenset({502, 503, 504})
# Refresh the GitHub token this long before it expires.
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)
# The key of an endpoint the runner is pointed at by URL.
API_KEY_ENV = "MIRA_LLM_API_KEY"
# Providers that are not an OpenAI-compatible endpoint.
_NON_ENDPOINT_PROVIDERS = CLAUDE_CODE_PROVIDER_NAMES | {
    "codex-cli",
    "codex_cli",
    "codex",
    "bedrock",
}


class RunnerError(Exception):
    """The job cannot go on. The message is safe to log."""


class _OIDCUnavailable(Exception):
    """GitHub's OIDC endpoint answered 5xx. Retried like a network error."""


def _in_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _mask(secret: str) -> None:
    """Have GitHub Actions hide ``secret`` in this run's logs from now on."""
    if secret and _in_actions():
        # A workflow command: Actions reads it from stdout and does not log it.
        sys.stdout.write(f"::add-mask::{secret}\n")
        sys.stdout.flush()


class RunnerAPI:
    """The server's /runner API, as one job's Actions run."""

    def __init__(self, http: httpx.AsyncClient, server_url: str, job_id: str, audience: str):
        self._http = http
        self._base = f"{server_url}/runner/jobs/{job_id}"
        self._audience = audience
        self.job_id = job_id

    async def _oidc_token(self) -> str:
        """A fresh OIDC token for this run. Each lives a few minutes, so every
        request asks for its own."""
        url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "")
        bearer = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "")
        if not url or not bearer:
            raise RunnerError(
                "ACTIONS_ID_TOKEN_REQUEST_URL is not set: the job needs permissions: id-token: write"
            )
        # The URL carries its own query (api-version), which params= would drop.
        resp = await self._http.request(
            "GET",
            httpx.URL(url).copy_merge_params({"audience": self._audience}),
            headers={"Authorization": f"bearer {bearer}"},
        )
        if resp.status_code >= 500:
            raise _OIDCUnavailable(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise RunnerError(f"GitHub refused an OIDC token: HTTP {resp.status_code}")
        token = str(resp.json()["value"])
        _mask(token)
        return token

    async def post(
        self, path: str, content: bytes = b"", *, forbidden_retries: Iterable[float] = ()
    ) -> httpx.Response:
        """POST with a fresh OIDC token, retrying network errors, 502, 503
        and 504 with backoff, and a 403 once per ``forbidden_retries`` delay.
        Returns the last response, whatever its status."""
        backoff = iter(_BACKOFF_S)
        forbidden = iter(forbidden_retries)
        while True:
            try:
                resp = await self._http.request(
                    "POST",
                    f"{self._base}/{path}",
                    content=content,
                    headers={
                        "Authorization": f"Bearer {await self._oidc_token()}",
                        "Content-Type": "application/json",
                    },
                )
            except (httpx.TransportError, _OIDCUnavailable) as exc:
                delay = next(backoff, None)
                if delay is None:
                    raise RunnerError(f"POST {path} failed: {type(exc).__name__}") from exc
                logger.warning("POST %s failed (%s), retrying in %ss", path, exc, delay)
                await asyncio.sleep(delay)
                continue
            if resp.status_code in _RETRY_STATUSES:
                delay = next(backoff, None)
            elif resp.status_code == 403:
                delay = next(forbidden, None)
            else:
                delay = None
            if delay is None:
                return resp
            logger.warning("POST %s answered %s, retrying in %ss", path, resp.status_code, delay)
            await asyncio.sleep(delay)

    async def claim(self) -> JobSpec | None:
        """The job's spec, or None when the job is no longer claimable (409)."""
        resp = await self.post("claim", forbidden_retries=_FORBIDDEN_RETRIES_S)
        if resp.status_code == 409:
            logger.info("Job %s is %s, so this run has nothing to do", self.job_id, _status(resp))
            return None
        if resp.status_code != 200:
            raise RunnerError(f"claim refused: HTTP {resp.status_code}")
        spec = JobSpec.model_validate_json(resp.content)
        _mask(spec.github_token.get_secret_value())
        return spec

    async def refresh_token(self) -> TokenGrant:
        resp = await self.post("token")
        if resp.status_code != 200:
            raise RunnerError(f"token refresh refused: HTTP {resp.status_code}")
        grant = TokenGrant.model_validate_json(resp.content)
        _mask(grant.github_token.get_secret_value())
        return grant

    async def post_result(self, bundle: ResultBundle) -> None:
        resp = await self.post("result", bundle.model_dump_json().encode())
        if resp.status_code == 409:
            logger.warning("The server no longer takes a result for job %s", self.job_id)
            return
        if resp.status_code != 200:
            raise RunnerError(f"result refused: HTTP {resp.status_code}")
        body = resp.json()
        logger.info("Result posted: job %s, applied=%s", body.get("status"), body.get("applied"))


def _status(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("status", "finished"))
    except ValueError:
        return "finished"


class TokenSupplier:
    """The job's GitHub token, refreshed through the server shortly before it
    expires or once GitHub refuses it. GitHubProvider awaits it before every
    call. It keeps every token it received, so the run can revoke them all
    when it ends."""

    def __init__(self, api: RunnerAPI, spec: JobSpec) -> None:
        self._api = api
        self._token = spec.github_token
        self._expires_at = spec.token_expires_at
        self._lock = asyncio.Lock()
        self.received = [spec.github_token]

    async def __call__(self) -> str:
        async with self._lock:
            if datetime.now(UTC) >= self._expires_at - TOKEN_REFRESH_MARGIN:
                await self._refresh()
            return self._token.get_secret_value()

    async def invalidate(self, stale: str) -> None:
        """GitHub refused ``stale``. Concurrent callers that saw the same
        token fail share one refresh: the lock serializes them, and a token
        already replaced is not refreshed again."""
        async with self._lock:
            if self._token.get_secret_value() == stale:
                logger.warning("GitHub refused the job's token before it expired; refreshing")
                await self._refresh(refused=True)

    async def _refresh(self, *, refused: bool = False) -> None:
        try:
            grant = await self._api.refresh_token()
        except (RunnerError, ValidationError) as exc:
            if refused or datetime.now(UTC) >= self._expires_at:
                raise
            logger.warning("GitHub token refresh failed, using the current one: %s", exc)
            return
        self._token, self._expires_at = grant.github_token, grant.token_expires_at
        self.received.append(grant.github_token)
        logger.info("Refreshed the GitHub token, now valid until %s", self._expires_at)

    async def revoke(self, http: httpx.AsyncClient) -> None:
        """Revoke every token received, best effort, so none outlives the run.
        Adapted from Pullfrog's utils/token.ts (MIT, Copyright (c) 2026
        Pullfrog, Inc.)."""
        await asyncio.gather(*(_revoke(http, token) for token in self.received))


async def _revoke(http: httpx.AsyncClient, token: SecretStr) -> None:
    try:
        resp = await http.request(
            "DELETE",
            f"{GITHUB_API_URL}/installation/token",
            headers={
                "Authorization": f"token {token.get_secret_value()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=10,
        )
    except httpx.HTTPError as exc:
        logger.warning("Could not revoke a GitHub token: %s", type(exc).__name__)
        return
    # 401: the token had already expired or been revoked.
    if resp.status_code not in (204, 401):
        logger.warning("Revoking a GitHub token answered HTTP %s", resp.status_code)


@dataclass(frozen=True)
class RunnerLLM:
    """How this run reaches its model, from the runner's flags.

    ``provider`` is an agent CLI, ``bedrock``, or the name of a
    ``providers.json`` profile such as ``zai``, which brings its own URL and
    key env. ``base_url`` points at any OpenAI-compatible endpoint instead,
    with its key in ``MIRA_LLM_API_KEY``, and then ``provider`` is ignored.
    """

    provider: str = "claude-code"
    model: str | None = None
    base_url: str | None = None
    api_style: str | None = None

    def settings(self) -> dict[str, Any]:
        """The llm settings that choose the endpoint. They replace the spec's
        in full: those describe the server's machine, and a repo's
        ``.mira.yaml`` can set its URL and key env."""
        if self.base_url:
            key_env = API_KEY_ENV
            # A local endpoint such as Ollama may need no key.
            host = urlparse(self.base_url).hostname or ""
            if not os.environ.get(key_env) and _is_local_host(host):
                key_env = ""
            return self._endpoint(self.base_url, key_env)
        if self.provider in _NON_ENDPOINT_PROVIDERS:
            return {"provider": self.provider}
        profile = profiles.get(self.provider)
        if profile is None:
            raise ConfigError(
                f"Unknown LLM provider {self.provider!r}: name an agent CLI, bedrock or "
                "a providers.json profile, or set MIRA_RUNNER_LLM_BASE_URL"
            )
        return self._endpoint(profile["base_url"], profile.get("api_key_env") or API_KEY_ENV)

    def _endpoint(self, base_url: str, api_key_env: str) -> dict[str, Any]:
        return {
            "provider": "openai",
            "base_url": base_url,
            "api_key_env": api_key_env,
            "api_style": self.api_style or "chat",
        }


def job_config(spec: JobSpec, llm: RunnerLLM) -> MiraConfig:
    """The spec's review config plus the runner's own LLM execution settings,
    which the server strips. ``llm.model`` replaces every per-purpose model."""
    data: dict[str, Any] = {**spec.config, "llm": dict(spec.config.get("llm", {}))}
    data["llm"].update(llm.settings())
    if llm.model:
        data["llm"].update(
            model=llm.model, review_model=None, indexing_model=None, security_model=None
        )
    # Code that reads load_config() instead of the engine's config sees the same.
    set_global_defaults_data(data)
    return MiraConfig.model_validate(data)


def usage_report(llms: Iterable[Any]) -> UsageReport:
    """Usage summed over the distinct LLM objects. A model counts once it
    was used: the agent CLIs report each call's model, other providers only
    their configured one."""
    harnesses: set[str] = set()
    models: set[str] = set()
    input_tokens = output_tokens = 0
    cost: float | None = None
    for llm in {id(llm): llm for llm in llms}.values():
        details = getattr(llm, "usage_details", None)
        config = getattr(llm, "config", None)
        if isinstance(details, dict):
            harnesses.add(str(details["harness"]))
            models.update(details["models"])
            input_tokens += int(details["input_tokens"])
            output_tokens += int(details["output_tokens"])
            cost = (cost or 0.0) + float(details["cost_usd"])
            continue
        usage = getattr(llm, "usage", None) or {}
        tokens = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        input_tokens += tokens[0]
        output_tokens += tokens[1]
        if config is not None:
            harnesses.add(config.provider)
            if any(tokens):
                models.add(config.model)
    return UsageReport(
        harness=", ".join(sorted(harnesses)) or None,
        model=", ".join(sorted(models)) or None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        # The CLIs report 0 when they report no cost.
        cost_usd=round(cost, 6) if cost else None,
    )


def _safe_error(exc: BaseException) -> str:
    """A short error for the job record and the PR's check, without the
    exception's own text unless Mira wrote it to be shown. A model
    provider's refusal leads with its cause."""
    message = exc.safe_message if isinstance(exc, MiraError) else ""
    error = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    kind = classify_provider_error(exc)
    return f"Review failed: {kind} ({error})" if kind else error


async def run_job(
    http: httpx.AsyncClient,
    *,
    job_id: str,
    server_url: str,
    audience: str,
    llm: RunnerLLM,
) -> int:
    """Claim, review and report one job. Returns the process exit code."""
    api = RunnerAPI(http, server_url, job_id, audience)
    try:
        spec = await api.claim()
    except (RunnerError, ValidationError) as exc:
        logger.error("Could not claim job %s: %s", job_id, exc)
        return 1
    if spec is None:
        return 0
    logger.info("Claimed job %s: %s at %s", job_id, spec.pr_url, spec.head_sha[:12])

    tokens = TokenSupplier(api, spec)
    try:
        return await _review_and_report(api, spec, llm, tokens)
    finally:
        await tokens.revoke(http)


async def _review_and_report(
    api: RunnerAPI, spec: JobSpec, llm: RunnerLLM, tokens: TokenSupplier
) -> int:
    job_id = spec.job_id
    store = RemoteStateStore(spec.snapshot)
    llms: list[Any] = []
    try:
        config = job_config(spec, llm)
        llms = [
            create_llm(llm_config_for(purpose, config.llm, use_dashboard=False))
            for purpose in ("review", "indexing", "security")
        ]
        # A missing secret fails the job here, not after a review of failed calls.
        for each in {id(each): each for each in llms}.values():
            each.preflight()
        engine = ReviewEngine(
            config=config,
            llm=llms[0],
            provider=GitHubProvider(tokens, on_unauthorized=tokens.invalidate),
            bot_name=spec.bot_name,
            indexing_llm=llms[1],
            security_llm=llms[2],
            state=store,
            check_run_id=spec.check_run_id,
        )
        result = await engine.review_pr(spec.pr_url)
    except Exception as exc:
        logger.exception("Review of %s failed", spec.pr_url)
        bundle = store.bundle(job_id, "failed", error=_safe_error(exc), usage=usage_report(llms))
        try:
            await api.post_result(bundle)
        except RunnerError as post_exc:
            logger.error("Could not report the failure: %s", post_exc)
        return 1

    error = ""
    if result.all_chunks_failed:
        # Nothing was reviewed. A partial failure still succeeds: the engine
        # records the failed paths as retryable. The reasons are Mira's own text.
        reasons = "; ".join(sorted({f.reason for f in result.failed_chunks if f.reason}))
        error = f"Review failed: all {result.total_chunks} review chunk(s) failed ({reasons})"
        logger.error("%s", error)
    else:
        logger.info("Reviewed %s: %d comments", spec.pr_url, len(result.comments))
    outcome: Literal["succeeded", "failed"] = "failed" if error else "succeeded"
    try:
        await api.post_result(store.bundle(job_id, outcome, error=error, usage=usage_report(llms)))
    except RunnerError as exc:
        logger.error("Could not post the result of job %s: %s", job_id, exc)
        return 1
    return 1 if error else 0
