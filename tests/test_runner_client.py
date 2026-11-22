"""``mira runner`` end to end, in one process.

The runner talks to the real runner API (PR3's router on a real SQLite app
DB) through an ASGI transport, gets its OIDC tokens from a local route that
signs them like GitHub, and reviews with the real engine and the engine
tests' mock LLM. GitHub is a fake HTTP layer for check runs plus a mock for
the PR's content.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, Request
from pydantic import SecretStr

from mira import config as mira_config
from mira import runner_client
from mira.config import MiraConfig, ReviewConfig
from mira.dashboard.db import AppDatabase
from mira.dashboard.models_config import llm_config_for
from mira.exceptions import ConfigError, NonRetriableLLMError, ProviderError
from mira.index.store import IndexStore
from mira.llm import create_llm
from mira.llm.provider import LLMProvider
from mira.llm.responses import ResponsesProvider
from mira.platforms.github.auth import InstallationToken
from mira.platforms.github.oidc import GitHubOIDCVerifier
from mira.platforms.github.runner_api import create_runner_router
from mira.providers.github import GitHubProvider
from mira.runner import JobStatus
from mira.runner_client import RunnerLLM, run_job, usage_report
from mira.runner_protocol import JobSpec, ResultBundle, UsageReport
from tests.test_engine import mock_llm  # noqa: F401  (fixture)
from tests.test_runner_api import (
    _KEY,
    PR,
    REF,
    FakeAppAuth,
    _engine_provider,
    db,  # noqa: F401  (fixture)
    oidc_token,
    seed_job,
    status_of,
)
from tests.test_runner_protocol import state_rows

SERVER = "http://mira.test"
OIDC_URL = f"{SERVER}/oidc?api-version=2.0"


class ExpiringAppAuth(FakeAppAuth):
    """Mints tokens that expire after the given delays, in order."""

    def __init__(self, *lifetimes: timedelta) -> None:
        super().__init__()
        self._lifetimes = list(lifetimes)

    async def mint_repo_token(self, installation_id: int, repo: str, permissions: dict[str, str]):
        token = await super().mint_repo_token(installation_id, repo, permissions)
        lifetime = self._lifetimes.pop(0) if self._lifetimes else timedelta(hours=1)
        return InstallationToken(token=token.token, expires_at=datetime.now(UTC) + lifetime)


class GitHubChecks:
    """The check-run REST calls, with the token each one used."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any] | None, str]] = []

    def _record(self, method: str, url: str, json: Any, headers: dict[str, str] | None) -> None:
        self.requests.append((method, url, json, (headers or {}).get("Authorization", "")))

    async def post(self, url: str, *, json: Any = None, headers: Any = None, **_: Any):
        self._record("POST", url, json, headers)
        return httpx.Response(201, json={"id": 999}, request=httpx.Request("POST", url))

    async def patch(self, url: str, *, json: Any = None, headers: Any = None, **_: Any):
        self._record("PATCH", url, json, headers)
        return httpx.Response(200, json={}, request=httpx.Request("PATCH", url))


@pytest.fixture
def github() -> Iterator[GitHubChecks]:
    # The runner's own requests go through AsyncClient.request, so only
    # GitHubProvider's check-run calls land here.
    fake = GitHubChecks()
    with (
        patch.object(httpx.AsyncClient, "post", fake.post),
        patch.object(httpx.AsyncClient, "patch", fake.patch),
    ):
        yield fake


@pytest.fixture
def app_auth() -> ExpiringAppAuth:
    return ExpiringAppAuth()


@pytest.fixture
def oidc_requests() -> list[httpx.QueryParams]:
    return []


@pytest.fixture
def revoked() -> list[str]:
    """The GitHub tokens the runner revoked, in order."""
    return []


@pytest.fixture
def server(
    db: AppDatabase,  # noqa: F811
    app_auth: ExpiringAppAuth,
    oidc_requests: list[httpx.QueryParams],
    revoked: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> FastAPI:
    app = FastAPI()
    verifier = GitHubOIDCVerifier(lambda token: _KEY.public_key())
    app.include_router(create_runner_router(app_auth, "mira-bot", verifier=verifier))  # type: ignore[arg-type]

    @app.get("/oidc")
    async def issue(request: Request) -> dict[str, str]:
        assert request.headers["authorization"] == "bearer request-token"
        oidc_requests.append(request.query_params)
        return {"value": oidc_token(aud=request.query_params["audience"])}

    # The ASGI transport routes by path, so GitHub's revoke lands here too.
    @app.delete("/installation/token", status_code=204)
    async def revoke(request: Request) -> None:
        revoked.append(request.headers["authorization"].removeprefix("token "))

    # The runner sets the process's global config from its job spec, which
    # has no server settings. The server here keeps its own.
    server_config = MiraConfig(review=ReviewConfig(runner_workflow_refs=[REF]))
    monkeypatch.setattr(
        "mira.platforms.github.runner_api.load_config", lambda *a, **k: server_config
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", OIDC_URL)
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "request-token")
    monkeypatch.setattr(runner_client, "_BACKOFF_S", (0,) * 7)
    monkeypatch.setattr(runner_client, "_FORBIDDEN_RETRIES_S", (0,) * 3)
    return app


@pytest.fixture
def pr_content(sample_diff_text: str) -> AsyncMock:
    return _engine_provider(sample_diff_text, "sha-1")


@pytest.fixture
def runner_side(
    monkeypatch: pytest.MonkeyPatch,
    mock_llm,  # noqa: F811
    pr_content: AsyncMock,
) -> MagicMock:
    """Wire the runner's engine to the mock LLM and to a provider whose check
    runs are the real GitHubProvider's. Returns the create_llm mock."""

    def provider(token_supplier: Any, **kwargs: Any) -> AsyncMock:
        real = GitHubProvider(token_supplier, **kwargs)
        pr_content.start_check_run = real.start_check_run
        pr_content.start_queued_check_run = real.start_queued_check_run
        pr_content.finish_check_run = real.finish_check_run
        return pr_content

    create_llm = MagicMock(return_value=mock_llm)
    monkeypatch.setattr(runner_client, "GitHubProvider", provider)
    monkeypatch.setattr(runner_client, "create_llm", create_llm)
    return create_llm


async def _run(
    app: FastAPI, job_id: str = "job-1", transport: Any = None, llm: RunnerLLM | None = None
) -> int:
    async with httpx.AsyncClient(transport=transport or httpx.ASGITransport(app=app)) as http:
        return await run_job(
            http,
            job_id=job_id,
            server_url=SERVER,
            audience="mira",
            llm=llm or RunnerLLM(),
        )


async def test_claim_review_result(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    github: GitHubChecks,
    runner_side: MagicMock,
    mock_llm,  # noqa: F811
    oidc_requests: list[httpx.QueryParams],
) -> None:
    seed_job(db)
    rules = IndexStore.open("test", "repo")
    try:
        rules.upsert_review_context("Money", "Amounts are cents, never dollars.")
    finally:
        rules.close()

    assert await _run(server) == 0

    job = db.get_review_job("job-1")
    assert job is not None and job.status is JobStatus.SUCCEEDED
    assert job.usage == UsageReport(input_tokens=100, output_tokens=50).model_dump(mode="json")
    rows = state_rows(PR)
    assert rows["sha"] == "sha-1" and len(rows["events"]) == 1

    # The job's queued check is the only Mira check: started, then completed.
    assert [m for m, url, *_ in github.requests if url.endswith("/check-runs")] == []
    checks = [(m, body) for m, url, body, _ in github.requests if url.endswith("/check-runs/501")]
    assert [(m, body["status"]) for m, body in checks] == [
        ("PATCH", "in_progress"),
        ("PATCH", "completed"),
    ]
    assert checks[-1][1]["conclusion"] == "success"

    # The dashboard's review-context rule reached the runner's review prompt.
    assert "Amounts are cents, never dollars." in str(mock_llm.review.await_args_list)
    # Every model came from the job's config, run on the runner's provider.
    purposes = [c.args[0] for c in runner_side.call_args_list]
    assert {cfg.provider for cfg in purposes} == {"claude-code"}
    # A fresh OIDC token for each request, for the right audience.
    assert len(oidc_requests) == 2
    assert all(q["audience"] == "mira" and q["api-version"] == "2.0" for q in oidc_requests)


async def test_a_job_that_is_not_claimable_exits_0_without_reviewing(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    runner_side: MagicMock,
    mock_llm,  # noqa: F811
) -> None:
    seed_job(db, status=JobStatus.SUPERSEDED)

    assert await _run(server) == 0

    runner_side.assert_not_called()
    mock_llm.review.assert_not_awaited()
    assert status_of(db) is JobStatus.SUPERSEDED


async def test_an_unknown_job_fails(server: FastAPI, db: AppDatabase, runner_side) -> None:  # noqa: F811
    assert await _run(server, job_id="no-such-job") == 1
    runner_side.assert_not_called()


async def test_a_failed_review_posts_a_failed_result_and_exits_1(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    github: GitHubChecks,
    runner_side: MagicMock,
    pr_content: AsyncMock,
) -> None:
    seed_job(db)
    pr_content.get_pr_diff.side_effect = RuntimeError("diff fetch failed for ghs_secret")

    assert await _run(server) == 1

    job = db.get_review_job("job-1")
    assert job is not None and job.status is JobStatus.FAILED
    # The exception's own text may carry anything, so only its type is reported.
    assert job.error == "RuntimeError"
    final = [body for m, url, body, _ in github.requests if url.endswith("/check-runs/501")][-1]
    assert (final["conclusion"], final["output"]["summary"]) == ("failure", "RuntimeError")


async def test_the_github_token_is_refreshed_before_it_expires(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    github: GitHubChecks,
    runner_side: MagicMock,
    app_auth: ExpiringAppAuth,
) -> None:
    # The claimed token is two minutes from expiry, inside the refresh margin.
    app_auth._lifetimes = [timedelta(minutes=2), timedelta(hours=1)]
    seed_job(db)

    assert await _run(server) == 0

    assert len(app_auth.minted) == 2
    used = {auth for _, url, _, auth in github.requests if "/check-runs/501" in url}
    assert used == {"token ghs_job_token_2"}


async def test_a_run_whose_id_the_server_has_not_stored_yet_retries_its_claim(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    github: GitHubChecks,
    runner_side: MagicMock,
    oidc_requests: list[httpx.QueryParams],
) -> None:
    seed_job(db, run_id=None)

    class StoreRunIdAfterFirstClaim(httpx.ASGITransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            response = await super().handle_async_request(request)
            if request.url.path.endswith("/claim") and response.status_code == 403:
                db.set_review_job_run("job-1", 1001, "https://github.com/test/repo/runs/1001")
            return response

    assert await _run(server, transport=StoreRunIdAfterFirstClaim(app=server)) == 0
    assert status_of(db) is JobStatus.SUCCEEDED


async def test_network_errors_and_502s_are_retried(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    github: GitHubChecks,
    runner_side: MagicMock,
) -> None:
    seed_job(db)
    failures = {"claim": [httpx.ConnectError("refused"), 502], "result": [503]}

    class Flaky(httpx.ASGITransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            pending = failures.get(request.url.path.rsplit("/", 1)[-1], [])
            if pending:
                failure = pending.pop(0)
                if isinstance(failure, Exception):
                    raise failure
                return httpx.Response(failure, request=request)
            return await super().handle_async_request(request)

    assert await _run(server, transport=Flaky(app=server)) == 0
    assert failures == {"claim": [], "result": []}
    assert status_of(db) is JobStatus.SUCCEEDED


async def test_a_missing_oidc_permission_fails_the_claim(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    runner_side: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seed_job(db)
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN")

    assert await _run(server) == 1
    assert "id-token: write" in caplog.text
    assert status_of(db) is JobStatus.DISPATCHED


def test_usage_is_summed_over_distinct_llms() -> None:
    class CLI:
        def __init__(self, model: str, tokens: int, cost: float) -> None:
            self.usage_details = {
                "harness": "claude-code",
                "models": [model],
                "input_tokens": tokens,
                "output_tokens": tokens // 10,
                "cost_usd": cost,
            }

    review, indexing = CLI("claude-opus-4-1", 1000, 0.25), CLI("claude-haiku-4-5", 200, 0.01)
    report = usage_report([review, indexing, review])

    assert report == UsageReport(
        harness="claude-code",
        model="claude-haiku-4-5, claude-opus-4-1",
        input_tokens=1200,
        output_tokens=120,
        cost_usd=0.26,
    )


def test_tokens_are_masked_in_actions_logs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    runner_client._mask("ghs_abc")
    assert capsys.readouterr().out == "::add-mask::ghs_abc\n"
    monkeypatch.delenv("GITHUB_ACTIONS")
    runner_client._mask("ghs_abc")
    assert capsys.readouterr().out == ""


def test_the_cli_rejects_a_plain_http_server() -> None:
    from click.testing import CliRunner

    from mira.cli import main

    result = CliRunner().invoke(main, ["runner", "--job", "j", "--server", "http://mira.example"])
    assert result.exit_code == 2
    assert "https" in result.output


# Credentials a test machine may have that would mask a missing one.
_LLM_ENV = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "MIRA_LLM_API_KEY",
    "ZAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
)
API_KEY = "sk-endpoint-" + "k" * 32


@pytest.fixture
def no_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _LLM_ENV:
        monkeypatch.delenv(key, raising=False)
    # job_config sets the process's global config; put it back afterwards.
    monkeypatch.setattr(mira_config, "_global_defaults", {})


def _bundles(monkeypatch: pytest.MonkeyPatch) -> list[ResultBundle]:
    """Record every result bundle the runner posts."""
    posted: list[ResultBundle] = []
    original = runner_client.RunnerAPI.post_result

    async def post_result(self: Any, bundle: ResultBundle) -> None:
        posted.append(bundle)
        await original(self, bundle)

    monkeypatch.setattr(runner_client.RunnerAPI, "post_result", post_result)
    return posted


async def test_a_review_where_every_chunk_failed_fails_and_keeps_the_sha(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    github: GitHubChecks,
    runner_side: MagicMock,
    no_llm_env: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The real OpenAI-compatible provider against an endpoint that answers
    every call with an error: nothing is reviewed, so the job fails and the
    head is not recorded as reviewed."""
    seed_job(db)
    monkeypatch.setattr(runner_client, "create_llm", create_llm)
    monkeypatch.setenv("MIRA_LLM_API_KEY", API_KEY)
    posted = _bundles(monkeypatch)
    llm = RunnerLLM(base_url="https://llm.example/v1", model="vendor/some-model")

    with caplog.at_level(logging.DEBUG):
        assert await _run(server, llm=llm) == 1

    (bundle,) = posted
    assert bundle.outcome == "failed"
    assert bundle.error.startswith("Review failed: all 1 review chunk(s) failed")
    assert [w for w in bundle.writes if w.op == "set_last_reviewed_sha"] == []
    job = db.get_review_job("job-1")
    assert job is not None and job.status is JobStatus.FAILED
    assert not state_rows(PR)["sha"]
    final = [body for m, url, body, _ in github.requests if url.endswith("/check-runs/501")][-1]
    assert final["conclusion"] == "failure"
    assert final["output"]["summary"] == bundle.error

    # Every model call went to the configured endpoint with its key.
    calls = [(body, auth) for _, url, body, auth in github.requests if "llm.example" in url]
    assert calls
    assert {auth for _, auth in calls} == {f"Bearer {API_KEY}"}
    assert {body["model"] for body, _ in calls} == {"some-model"}
    assert API_KEY not in caplog.text
    assert API_KEY not in bundle.model_dump_json()


async def test_a_missing_secret_fails_the_job_before_any_review(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    github: GitHubChecks,
    runner_side: MagicMock,
    no_llm_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_job(db)
    monkeypatch.setattr(runner_client, "create_llm", create_llm)

    assert await _run(server) == 1

    job = db.get_review_job("job-1")
    assert job is not None and job.status is JobStatus.FAILED
    assert job.error.startswith("NonRetriableLLMError: Claude Code credentials are not set")
    assert "CLAUDE_CODE_OAUTH_TOKEN" in job.error
    # The PR's check says which secret to set.
    final = [body for m, url, body, _ in github.requests if url.endswith("/check-runs/501")][-1]
    assert (final["conclusion"], final["output"]["summary"]) == ("failure", job.error)
    # It failed before the review started its own check run.
    assert [m for m, url, *_ in github.requests if url.endswith("/check-runs/501")] == ["PATCH"]


def _review_llm(llm: RunnerLLM, repo_llm: dict[str, Any] | None = None) -> Any:
    spec = cast(JobSpec, SimpleNamespace(config={"llm": repo_llm or {}}))
    config = runner_client.job_config(spec, llm)
    return create_llm(llm_config_for("review", config.llm, use_dashboard=False))


def test_a_profile_name_builds_that_endpoint(
    no_llm_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A repo's .mira.yaml can set these; the runner must not send its key there.
    repo_llm = {"base_url": "https://evil.example/v1", "api_key_env": "GITHUB_TOKEN"}
    llm = _review_llm(RunnerLLM(provider="zai", model="glm-5.3"), repo_llm)

    assert isinstance(llm, LLMProvider)
    assert llm.config.base_url == "https://api.z.ai/api/paas/v4"
    assert llm.config.api_key_env == "ZAI_API_KEY"
    assert llm.config.model == "glm-5.3"
    with pytest.raises(NonRetriableLLMError) as exc:
        llm.preflight()
    assert "ZAI_API_KEY" in exc.value.safe_message

    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    llm.preflight()


def test_a_base_url_builds_a_generic_endpoint(
    no_llm_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _review_llm(
        RunnerLLM(base_url="https://api.together.xyz/v1", model="glm-5.3", api_style="responses")
    )

    assert isinstance(llm, ResponsesProvider)
    assert llm.config.base_url == "https://api.together.xyz/v1"
    assert llm.config.api_key_env == "MIRA_LLM_API_KEY"
    assert llm.config.model == "glm-5.3"
    with pytest.raises(NonRetriableLLMError) as exc:
        llm.preflight()
    assert "MIRA_LLM_API_KEY" in exc.value.safe_message

    monkeypatch.setenv("MIRA_LLM_API_KEY", API_KEY)
    llm.preflight()


def test_only_a_local_endpoint_may_go_without_a_key(
    no_llm_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = RunnerLLM(base_url="http://localhost:11434/v1", model="qwen3")
    _review_llm(local).preflight()

    monkeypatch.setenv("MIRA_LLM_API_KEY", API_KEY)
    assert _review_llm(local)._build_headers()["Authorization"] == f"Bearer {API_KEY}"


def test_an_unknown_provider_is_refused(no_llm_env: None) -> None:
    with pytest.raises(ConfigError, match="zia"):
        _review_llm(RunnerLLM(provider="zia"))


def test_the_cli_reads_the_endpoint_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from mira.cli import main

    seen: list[RunnerLLM] = []

    async def fake_run_job(http: Any, **kwargs: Any) -> int:
        seen.append(kwargs["llm"])
        return 0

    monkeypatch.setattr(runner_client, "run_job", fake_run_job)
    env = {
        "MIRA_RUNNER_LLM_PROVIDER": "",
        "MIRA_RUNNER_MODEL": "glm-5.3",
        "MIRA_RUNNER_LLM_BASE_URL": "https://api.together.xyz/v1",
        "MIRA_RUNNER_LLM_API_STYLE": "",
    }
    result = CliRunner().invoke(
        main, ["runner", "--job", "j", "--server", "https://mira.example"], env=env
    )

    assert result.exit_code == 0, result.output
    # Unset repo variables reach the job as empty strings and mean "default".
    assert seen == [
        RunnerLLM(provider="claude-code", model="glm-5.3", base_url="https://api.together.xyz/v1")
    ]


async def test_every_token_received_is_revoked_when_the_review_succeeds(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    github: GitHubChecks,
    runner_side: MagicMock,
    app_auth: ExpiringAppAuth,
    revoked: list[str],
) -> None:
    # The claimed token is inside the refresh margin, so the run gets two.
    app_auth._lifetimes = [timedelta(minutes=2), timedelta(hours=1)]
    seed_job(db)

    assert await _run(server) == 0

    assert sorted(revoked) == ["ghs_job_token_1", "ghs_job_token_2"]


async def test_the_token_is_revoked_when_the_review_fails(
    server: FastAPI,
    db: AppDatabase,  # noqa: F811
    github: GitHubChecks,
    runner_side: MagicMock,
    pr_content: AsyncMock,
    revoked: list[str],
) -> None:
    seed_job(db)
    pr_content.get_pr_diff.side_effect = RuntimeError("diff fetch failed")

    assert await _run(server) == 1

    assert status_of(db) is JobStatus.FAILED
    assert revoked == ["ghs_job_token_1"]


class _Grants:
    """The server's token route: a new token per refresh, counted."""

    def __init__(self) -> None:
        self.refreshes = 0

    async def refresh_token(self) -> Any:
        self.refreshes += 1
        await asyncio.sleep(0)
        return SimpleNamespace(
            github_token=SecretStr(f"ghs_fresh_{self.refreshes}"),
            token_expires_at=datetime.now(UTC) + timedelta(hours=1),
        )


def _spec_with_token(token: str) -> Any:
    return SimpleNamespace(
        github_token=SecretStr(token), token_expires_at=datetime.now(UTC) + timedelta(hours=1)
    )


async def test_a_401_refreshes_the_token_once_for_every_caller_and_retries() -> None:
    grants = _Grants()
    tokens = runner_client.TokenSupplier(cast(Any, grants), _spec_with_token("ghs_revoked"))
    provider = GitHubProvider(tokens, on_unauthorized=tokens.invalidate)
    used: list[str] = []

    async def fake_patch(
        self: Any, url: str, *, headers: dict[str, str], **_: Any
    ) -> httpx.Response:
        token = headers["Authorization"].removeprefix("token ")
        used.append(token)
        await asyncio.sleep(0)
        status = 401 if token == "ghs_revoked" else 200
        return httpx.Response(status, json={}, request=httpx.Request("PATCH", url))

    pr = SimpleNamespace(owner="test", repo="repo")
    with patch.object(httpx.AsyncClient, "patch", fake_patch):
        await asyncio.gather(
            provider.start_queued_check_run(cast(Any, pr), 501),
            provider.finish_check_run(cast(Any, pr), 501, "success"),
        )

    assert grants.refreshes == 1
    assert used == ["ghs_revoked", "ghs_revoked", "ghs_fresh_1", "ghs_fresh_1"]
    assert tokens.received == [SecretStr("ghs_revoked"), SecretStr("ghs_fresh_1")]


async def test_without_a_refresher_a_401_is_not_retried() -> None:
    calls: list[str] = []

    async def fake_patch(self: Any, url: str, **_: Any) -> httpx.Response:
        calls.append(url)
        return httpx.Response(401, json={}, request=httpx.Request("PATCH", url))

    pr = SimpleNamespace(owner="test", repo="repo")
    with (
        patch.object(httpx.AsyncClient, "patch", fake_patch),
        pytest.raises(ProviderError) as raised,
    ):
        await GitHubProvider("ghs_static").finish_check_run(cast(Any, pr), 501, "success")

    assert raised.value.status_code == 401
    assert len(calls) == 1
