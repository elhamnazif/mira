"""``mira runner``: run one dispatched job inside its GitHub Actions run.

The run proves who it is with its Actions OIDC token, claims the job from
the server, and runs the job's task: a review with the real engine against
the claimed state snapshot, an answer to an @mention through the same
handler the server uses in local mode, an index of the repo through the
server's own indexer, or the model pass of learning from a merged PR with
the server's own synthesis. It posts the writes back in one result bundle,
after an index's writes in parts. Exit codes:

- 0: the task finished and the server has its result, or the job was
  no longer this run's to claim (409).
- 1: the server refused or could not be reached, or the task failed.
  A failed task still posts ``outcome=failed`` first.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr, ValidationError

from mira.analysis.feedback import synthesize_from_human_reviews
from mira.config import MiraConfig, _is_local_host, set_global_defaults_data
from mira.core.engine import ReviewEngine
from mira.dashboard.models_config import llm_config_for
from mira.exceptions import ConfigError, MiraError
from mira.index.indexer import IndexingRecoveryPending, index_repo
from mira.llm import create_llm, slugs
from mira.llm import provider_profiles as profiles
from mira.llm.agent_cli import CLAUDE_CODE_PROVIDER_NAMES
from mira.llm.provider_errors import classify_provider_error
from mira.models import ReviewResult
from mira.platforms.fetch import EmptyRepoError, GitHubRepoFetcher
from mira.platforms.handlers import answer_question, reply_in_thread
from mira.platforms.index_handlers import index_push
from mira.providers.github import _GITHUB_API_URL as GITHUB_API_URL
from mira.providers.github import GitHubProvider
from mira.runner import (
    AskTask,
    IndexDiffTask,
    IndexRepoTask,
    IndexTask,
    LearnTask,
    ReviewRestTask,
    ReviewTask,
    ThreadReplyTask,
)
from mira.runner_protocol import (
    MAX_INDEX_PART_BYTES,
    IndexEnd,
    IndexReport,
    IndexSnapshot,
    JobSpec,
    LearningSnapshot,
    RemoteIndexStore,
    RemoteLearningStore,
    RemoteStateStore,
    ResultBundle,
    TokenGrant,
    UsageReport,
    index_parts,
)

logger = logging.getLogger(__name__)

# Waits between attempts after a network error, 502, 503 or 504. Tests zero them.
_BACKOFF_S: tuple[float, ...] = (2, 4, 8, 16, 30, 60, 60)
# A claim refused with 403 can be a run that started before the server
# stored its run id, so it is retried a few times.
_FORBIDDEN_RETRIES_S: tuple[float, ...] = (5, 10, 15)
_RETRY_STATUSES = frozenset({502, 503, 504})
# The largest index part the runner uploads. Tests shrink it.
_PART_BYTES = MAX_INDEX_PART_BYTES
# Refresh the GitHub token this long before it expires.
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)
# The key of an endpoint the runner is pointed at by URL.
API_KEY_ENV = "MIRA_LLM_API_KEY"
# Providers that are not an OpenAI-compatible endpoint.
_NON_ENDPOINT_PROVIDERS = CLAUDE_CODE_PROVIDER_NAMES | {
    slugs.Harness.OPENCODE,
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

    async def post_index_part(self, seq: int, body: bytes) -> None:
        resp = await self.post(f"index/{seq}", body)
        if resp.status_code != 200:
            raise RunnerError(f"index part {seq} refused: HTTP {resp.status_code}")

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
    Unset, ``provider`` is chosen from a model slug, else ``claude-code``.
    """

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_style: str | None = None

    def chosen(self) -> bool:
        """Whether the runner named a model or endpoint of its own."""
        return bool(self.provider or self.model or self.base_url)

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
        provider = self.provider or "claude-code"
        if self.model and slugs.get(self.model):
            choice = slugs.choose_harness(self.model, os.environ, override=self.provider)
            logger.info("Model %s runs on %s (%s)", self.model, choice.harness, choice.reason)
            provider = choice.profile or choice.harness.value
        if provider in _NON_ENDPOINT_PROVIDERS:
            return {"provider": provider}
        profile = profiles.get(provider)
        if profile is None:
            raise ConfigError(
                f"Unknown LLM provider {provider!r}: name an agent CLI, bedrock or "
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


def _safe_error(exc: BaseException, label: str) -> str:
    """A short error for the job record and the PR's check, without the
    exception's own text unless Mira wrote it to be shown. A model
    provider's refusal leads with its cause and the job's kind, ``label``."""
    message = exc.safe_message if isinstance(exc, MiraError) else ""
    error = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    kind = classify_provider_error(exc)
    return f"{label.capitalize()} failed: {kind} ({error})" if kind else error


# Actions stops a cancelled run with SIGINT, then SIGTERM 7.5 s later, and
# kills it 2.5 s after that. Reporting and revoking must fit in between.
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_REPORT_AFTER_STOP_S = 6.0
_REVOKE_AFTER_STOP_S = 2.0

# What report-failure tells the server, by why the runner step ended. That
# step never learned the job's kind, so these name none.
REPORTED_FAILURES = {
    "cancelled": "Job stopped: the Actions run was cancelled before the job reported",
    "failed": "Job failed: the runner step ended before it reported a result",
}


async def run_job(
    http: httpx.AsyncClient,
    *,
    job_id: str,
    server_url: str,
    audience: str,
    llm: RunnerLLM,
    timeout: timedelta | None = None,
) -> int:
    """Claim, run and report one job. Returns the process exit code.

    The first SIGINT or SIGTERM, which is how Actions cancels a run, or the
    end of ``timeout`` stops the task, and the job is reported failed with
    the reason. The run revokes its GitHub tokens before it returns."""
    api = RunnerAPI(http, server_url, job_id, audience)
    try:
        spec = await api.claim()
    except (RunnerError, ValidationError) as exc:
        logger.error("Could not claim job %s: %s", job_id, exc)
        return 1
    if spec is None:
        return 0
    logger.info(
        "Claimed %s job %s: %s at %s", spec.task.kind, job_id, spec.pr_url, spec.head_sha[:12]
    )
    _adopt_credentials(spec, llm)
    if not llm.chosen():
        llm = RunnerLLM(
            provider=spec.llm_provider or None,
            model=spec.llm_model or None,
            api_style=llm.api_style,
        )

    tokens = TokenSupplier(api, spec)
    store = _store(spec)
    # Filled as the LLMs are made, so a failed review still reports their usage.
    llms: list[Any] = []
    work = asyncio.create_task(_run_task(spec, llm, api, tokens, store, llms))
    with _cancel_on_signal(work) as signals:
        try:
            error = await _task_error(work, signals, timeout, spec)
            outcome: Literal["succeeded", "failed"] = "failed" if error else "succeeded"
            bundle = store.bundle(job_id, outcome, error=error, usage=usage_report(llms))
            try:
                async with asyncio.timeout(_REPORT_AFTER_STOP_S if signals else None):
                    await api.post_result(bundle)
            except (RunnerError, TimeoutError) as exc:
                logger.error("Could not post the result of job %s: %r", job_id, exc)
                return 1
            return 1 if error else 0
        finally:
            with suppress(TimeoutError):
                async with asyncio.timeout(_REVOKE_AFTER_STOP_S if signals else None):
                    await tokens.revoke(http)


def _repo_endpoints(llm: RunnerLLM) -> list[str]:
    """The model endpoints the repo chose for this run, by setting name.
    Any ``*_BASE_URL`` variable can point a provider at another host."""
    chosen = [name for name, value in os.environ.items() if value and name.endswith("_BASE_URL")]
    return ["MIRA_RUNNER_LLM_BASE_URL"] * bool(llm.base_url) + sorted(chosen)


def _adopt_credentials(spec: JobSpec, llm: RunnerLLM) -> None:
    """Put the server's model credentials in this process's environment,
    where the LLM providers read them. One the runner already has, from a
    repo secret, wins. When the repo points the model at an endpoint of its
    own, the server's keys are not used at all: they would go to that host."""
    if spec.credentials and (endpoints := _repo_endpoints(llm)):
        logger.warning(
            "Not using the server's model keys: this repo sets its own endpoint (%s). "
            "Give the endpoint's key as a repo secret.",
            ", ".join(endpoints),
        )
        return
    for name, secret in spec.credentials.items():
        value = secret.get_secret_value()
        _mask(value)
        if not os.environ.get(name):
            os.environ[name] = value


Store = RemoteStateStore | RemoteIndexStore | RemoteLearningStore


def _store(spec: JobSpec) -> Store:
    """Where the job's task reads and writes state. JobSpec checks that the
    snapshot fits the task."""
    if isinstance(spec.snapshot, IndexSnapshot):
        task = spec.task
        reconcile = isinstance(task, IndexRepoTask) and task.reconcile
        return RemoteIndexStore(spec.snapshot, reconcile=reconcile)
    if isinstance(spec.snapshot, LearningSnapshot):
        return RemoteLearningStore(spec.snapshot)
    return RemoteStateStore(spec.snapshot)


async def _run_task(
    spec: JobSpec,
    llm: RunnerLLM,
    api: RunnerAPI,
    tokens: TokenSupplier,
    store: Store,
    llms: list[Any],
) -> str:
    """Run the job's task. Returns why it failed, or "" if it did not."""
    config = job_config(spec, llm)
    llms += [
        create_llm(llm_config_for(purpose, config.llm, use_dashboard=False))
        for purpose in ("review", "indexing", "security")
    ]
    # A missing secret fails the job here, not after a review of failed calls.
    for each in {id(each): each for each in llms}.values():
        each.preflight()
    task = spec.task
    if isinstance(task, IndexRepoTask | IndexDiffTask):
        assert isinstance(store, RemoteIndexStore)
        return await _run_index(api, task, config, llms[1], tokens, store)
    if isinstance(task, LearnTask):
        assert isinstance(store, RemoteLearningStore)
        # The server's own merge-time synthesis, on the indexing model.
        rules = await synthesize_from_human_reviews(store, llms[1])
        logger.info("Learned %d rules from the human reviews of %s", rules, spec.pr_url)
        return ""
    assert isinstance(store, RemoteStateStore)
    provider = GitHubProvider(tokens, on_unauthorized=tokens.invalidate)
    match task:
        case AskTask():
            await answer_question(provider, llms[0], spec.pr_url, task)
            return ""
        case ThreadReplyTask(may_resolve=verdict):

            async def may_resolve() -> bool:
                return verdict

            await reply_in_thread(
                provider,
                llms[1],
                await provider.get_pr_info(spec.pr_url),
                task.reply,
                may_resolve=may_resolve,
                state=store,
                platform=spec.platform,
            )
            return ""
        case ReviewTask() | ReviewRestTask():
            engine = ReviewEngine(
                config=config,
                llm=llms[0],
                provider=provider,
                bot_name=spec.bot_name,
                indexing_llm=llms[1],
                security_llm=llms[2],
                state=store,
                check_run_id=spec.check_run_id,
                review_only_paths=set(task.paths) if isinstance(task, ReviewRestTask) else None,
            )
            return _review_failure(await engine.review_pr(spec.pr_url), spec.pr_url)


_INCOMPLETE = (
    "Index incomplete: some files could not be read or summarized, "
    "so the next index run reconciles them"
)


async def _run_index(
    api: RunnerAPI,
    task: IndexTask,
    config: MiraConfig,
    llm: Any,
    tokens: TokenSupplier,
    store: RemoteIndexStore,
) -> str:
    """Index the repo into ``store`` with the server's indexer, then upload
    the writes. Returns why the job failed, or "" if it did not. A run that
    fails outright uploads nothing, so the server's index is untouched."""
    owner, repo = store.repo.owner, store.repo.repo
    fetcher = GitHubRepoFetcher(tokens, GITHUB_API_URL)
    run = store.claim_index_run()
    error = ""
    try:
        match task:
            case IndexRepoTask(full=full, reset=reset):
                await index_repo(
                    owner,
                    repo,
                    config=config,
                    store=store,
                    llm=llm,
                    full=full,
                    reset=reset,
                    fetcher=fetcher,
                    run=run,
                )
            case IndexDiffTask():
                await index_push(
                    owner,
                    repo,
                    fetcher,
                    list(task.changed_paths),
                    list(task.removed_paths),
                    task.branch,
                    config=config,
                    store=store,
                    llm=llm,
                    run=run,
                )
        end = IndexEnd.COMPLETE if store.finished else IndexEnd.INCOMPLETE
    except IndexingRecoveryPending as exc:
        logger.warning("Index of %s/%s is incomplete: %s", owner, repo, exc)
        end = IndexEnd.INCOMPLETE
    except EmptyRepoError:
        end = IndexEnd.EMPTY
    if end is IndexEnd.INCOMPLETE:
        error = _INCOMPLETE
    try:
        parts = index_parts(store.writes, _PART_BYTES)
        for seq, body in enumerate(parts):
            await api.post_index_part(seq, body)
    except (ValueError, RunnerError) as exc:
        logger.error("Could not upload the index of %s/%s: %s", owner, repo, exc)
        return f"Index upload failed: {exc}"
    store.report = IndexReport(end=end, parts=len(parts))
    logger.info(
        "Indexed %s/%s (%s): %d writes in %d parts", owner, repo, end, len(store.writes), len(parts)
    )
    return error


def _review_failure(result: ReviewResult, pr_url: str) -> str:
    if result.all_chunks_failed:
        # Nothing was reviewed. A partial failure still succeeds: the engine
        # records the failed paths as retryable. The reasons are Mira's own text.
        reasons = "; ".join(sorted({f.reason for f in result.failed_chunks if f.reason}))
        error = f"Review failed: all {result.total_chunks} review chunk(s) failed ({reasons})"
        logger.error("%s", error)
        return error
    logger.info("Reviewed %s: %d comments", pr_url, len(result.comments))
    return ""


async def _task_error(
    work: asyncio.Task[str], signals: list[str], timeout: timedelta | None, spec: JobSpec
) -> str:
    """Wait for the task. Returns why the job failed, or "" if it did not."""
    label = spec.task.kind.label
    try:
        return await asyncio.wait_for(work, timeout.total_seconds() if timeout else None)
    except asyncio.CancelledError:
        if not signals:
            raise
        logger.error("The %s of %s stopped on %s", label, spec.pr_url, signals[0])
        return f"{label.capitalize()} stopped: the Actions run was cancelled before it finished"
    except Exception as exc:
        if isinstance(exc, TimeoutError) and work.cancelled() and timeout:
            logger.error("The %s of %s did not finish within %s", label, spec.pr_url, timeout)
            minutes = int(timeout.total_seconds() // 60)
            return f"{label.capitalize()} timed out after {minutes} minutes"
        logger.exception("The %s of %s failed", label, spec.pr_url)
        return _safe_error(exc, label)


@contextmanager
def _cancel_on_signal(task: asyncio.Task[Any]) -> Iterator[list[str]]:
    """Cancel ``task`` on the first SIGINT or SIGTERM. Later ones are
    ignored, so the run can still report. Yields the signals received."""
    loop = asyncio.get_running_loop()
    received: list[str] = []

    def stop(sig: signal.Signals) -> None:
        received.append(sig.name)
        if len(received) == 1:
            logger.warning("Got %s: stopping the job and reporting it", sig.name)
            task.cancel()

    for sig in _STOP_SIGNALS:
        loop.add_signal_handler(sig, stop, sig)
    try:
        yield received
    finally:
        for sig in _STOP_SIGNALS:
            loop.remove_signal_handler(sig)


async def report_failure(
    http: httpx.AsyncClient, *, job_id: str, server_url: str, audience: str, reason: str
) -> int:
    """Post a failed result for a job whose runner step ended without
    reporting, for the workflow's last step. The server applies only a
    job's first result and leaves a superseded job's check alone, so a run
    that did report keeps its result."""
    api = RunnerAPI(http, server_url, job_id, audience)
    bundle = ResultBundle(job_id=job_id, outcome="failed", error=REPORTED_FAILURES[reason])
    try:
        await api.post_result(bundle)
    except RunnerError as exc:
        logger.error("Could not report job %s as failed: %s", job_id, exc)
        return 1
    return 0
