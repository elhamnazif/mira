"""``mira runner`` end to end, in one process.

The runner talks to the real runner API (PR3's router on a real SQLite app
DB) through an ASGI transport, gets its OIDC tokens from a local route that
signs them like GitHub, and reviews with the real engine and the engine
tests' mock LLM. GitHub is a fake HTTP layer for check runs plus a mock for
the PR's content.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, Request

from mira import runner_client
from mira.config import MiraConfig, ReviewConfig
from mira.dashboard.db import AppDatabase
from mira.index.store import IndexStore
from mira.platforms.github.auth import InstallationToken
from mira.platforms.github.oidc import GitHubOIDCVerifier
from mira.platforms.github.runner_api import create_runner_router
from mira.providers.github import GitHubProvider
from mira.runner import JobStatus
from mira.runner_client import run_job, usage_report
from mira.runner_protocol import UsageReport
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
def server(
    db: AppDatabase,  # noqa: F811
    app_auth: ExpiringAppAuth,
    oidc_requests: list[httpx.QueryParams],
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

    def provider(token_supplier: Any) -> AsyncMock:
        real = GitHubProvider(token_supplier)
        pr_content.start_check_run = real.start_check_run
        pr_content.start_queued_check_run = real.start_queued_check_run
        pr_content.finish_check_run = real.finish_check_run
        return pr_content

    create_llm = MagicMock(return_value=mock_llm)
    monkeypatch.setattr(runner_client, "GitHubProvider", provider)
    monkeypatch.setattr(runner_client, "create_llm", create_llm)
    return create_llm


async def _run(app: FastAPI, job_id: str = "job-1", transport: Any = None) -> int:
    async with httpx.AsyncClient(transport=transport or httpx.ASGITransport(app=app)) as http:
        return await run_job(
            http,
            job_id=job_id,
            server_url=SERVER,
            audience="mira",
            llm_provider="claude-code",
            model=None,
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
