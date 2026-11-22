"""The runner API end to end: OIDC tokens signed by a local RSA key, a real
SQLite app DB, a fake GitHub token mint, and the real engine reviewing
through a RemoteStateStore on the runner side."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mira.config as mira_config
from mira.config import ConfigError, MiraConfig, load_config
from mira.core.engine import ReviewEngine
from mira.core.state_store import LocalStateStore
from mira.dashboard.db import AppDatabase
from mira.models import BotThreadRecord, PRInfo
from mira.platforms.github import runner_api
from mira.platforms.github.auth import GitHubAppAuth, InstallationToken
from mira.platforms.github.oidc import ISSUER, GitHubOIDCVerifier
from mira.platforms.github.runner_api import (
    RUNNER_TOKEN_PERMISSIONS,
    create_runner_router,
    expire_stale_jobs,
)
from mira.runner import JobStatus, ReviewJob
from mira.runner_protocol import (
    JobSpec,
    PRKey,
    RemoteStateStore,
    SetLastReviewedSha,
    TokenGrant,
    UsageReport,
)
from tests.test_engine import mock_llm  # noqa: F401  (fixture)
from tests.test_runner_dispatch import FakeGitHub
from tests.test_runner_protocol import _review, state_rows

REF = "elhamnazif/mira/.github/workflows/review.yml@refs/heads/main"
INSTALLATION = 184691606
PR = PRKey(platform="github", owner="test", repo="repo", pr_number=1)
_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def oidc_token(key: Any = _KEY, **overrides: Any) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": "mira",
        "iat": now,
        "nbf": now,
        "exp": now + 300,
        "repository": "Test/Repo",
        "run_id": "1001",
        "event_name": "workflow_dispatch",
        "job_workflow_ref": REF,
        "workflow_ref": "test/repo/.github/workflows/mira.yml@refs/heads/main",
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test"})


def auth(**overrides: Any) -> dict[str, str]:
    return {"Authorization": f"Bearer {oidc_token(**overrides)}"}


class FakeAppAuth:
    """Mints numbered repo tokens and records each request."""

    def __init__(self) -> None:
        self.minted: list[tuple[int, str, dict[str, str]]] = []

    async def mint_repo_token(
        self, installation_id: int, repo: str, permissions: dict[str, str]
    ) -> InstallationToken:
        self.minted.append((installation_id, repo, dict(permissions)))
        return InstallationToken(
            token=f"ghs_job_token_{len(self.minted)}",
            expires_at=datetime(2026, 11, 23, 12, 0, tzinfo=UTC),
        )

    async def get_installation_token(self, installation_id: int) -> str:
        return "ghs_app_token"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path / "server"))
    app_db = AppDatabase(url="", admin_password="pw")
    monkeypatch.setattr("mira.dashboard.api._app_db", app_db)
    monkeypatch.setattr(
        mira_config, "_global_defaults", {"review": {"runner_workflow_refs": [REF]}}
    )
    return app_db


@pytest.fixture
def app_auth() -> FakeAppAuth:
    return FakeAppAuth()


@pytest.fixture
def client(db: AppDatabase, app_auth: FakeAppAuth) -> TestClient:
    app = FastAPI()
    verifier = GitHubOIDCVerifier(lambda token: _KEY.public_key())
    app.include_router(create_runner_router(app_auth, "mira-bot", verifier=verifier))  # type: ignore[arg-type]
    return TestClient(app)


@pytest.fixture
def github() -> Iterator[FakeGitHub]:
    fake = FakeGitHub()
    with (
        patch.object(httpx.AsyncClient, "post", fake.post),
        patch.object(httpx.AsyncClient, "patch", fake.patch),
    ):
        yield fake


def seed_job(db: AppDatabase, job_id: str = "job-1", **overrides: Any) -> ReviewJob:
    fields: dict[str, Any] = {
        "id": job_id,
        "owner": "test",
        "repo": "repo",
        "pr_number": 1,
        "head_sha": "sha-1",
        "installation_id": INSTALLATION,
        "runs_on": ("ubuntu-latest",),
        "check_run_id": 501,
        "run_id": 1001,
        "created_at": time.time(),
    }
    fields.update(overrides)
    job = ReviewJob(**fields)
    db.create_review_job(job)
    return job


def claim(client: TestClient, job_id: str = "job-1", **claims: Any) -> httpx.Response:
    return client.post(f"/runner/jobs/{job_id}/claim", headers=auth(**claims))


def post_result(client: TestClient, bundle: Any, job_id: str = "job-1", **claims: Any):
    return client.post(
        f"/runner/jobs/{job_id}/result",
        content=bundle.model_dump_json(),
        headers={**auth(**claims), "content-type": "application/json"},
    )


def status_of(db: AppDatabase, job_id: str = "job-1") -> JobStatus:
    job = db.get_review_job(job_id)
    assert job is not None
    return job.status


# ── Round trip with the real engine ──


class _TeeStore(RemoteStateStore):
    """A RemoteStateStore that also logs each write call, to replay it later
    on a LocalStateStore."""

    def __init__(self, snapshot: Any) -> None:
        super().__init__(snapshot)
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if name in {
            "set_last_reviewed_sha",
            "upsert_pr_review_progress",
            "record_review",
            "upsert_pr_fingerprint",
        }:
            calls = super().__getattribute__("calls")

            def logged(*args: Any, **kwargs: Any) -> Any:
                calls.append((name, args, kwargs))
                return attr(*args, **kwargs)

            return logged
        return attr


def _engine_provider(diff: str, head_sha: str) -> AsyncMock:
    provider = AsyncMock()
    provider.get_pr_info.return_value = PRInfo(
        title="Test PR",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://github.com/test/repo/pull/1",
        number=1,
        owner="test",
        repo="repo",
        head_sha=head_sha,
    )
    provider.get_pr_diff.return_value = diff
    provider.get_compare_diff.return_value = diff
    provider.get_all_bot_threads.return_value = []
    provider.get_unresolved_bot_threads.return_value = []
    provider.find_bot_comment.return_value = None
    provider.list_open_prs.return_value = []
    return provider


async def test_claim_token_result_round_trip_matches_local_writes(
    client: TestClient,
    db: AppDatabase,
    app_auth: FakeAppAuth,
    mock_llm,  # noqa: F811
    sample_diff_text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay: list[tuple[str, tuple, dict]] = []
    for job_id, run_id, sha, prior_sha in [
        ("job-1", 1001, "sha-1", ""),
        ("job-2", 1002, "sha-2", "sha-1"),
    ]:
        seed_job(db, job_id, run_id=run_id, head_sha=sha)

        resp = claim(client, job_id, run_id=str(run_id))
        assert resp.status_code == 200
        spec = JobSpec.model_validate_json(resp.content)
        assert status_of(db, job_id) is JobStatus.CLAIMED
        assert (spec.pr_url, spec.head_sha, spec.bot_name) == (PR_URL, sha, "mira-bot")
        assert spec.snapshot.pr == PR and spec.snapshot.last_reviewed_sha == prior_sha
        assert "database" not in spec.config

        # The runner: the real engine, reviewing through the snapshot.
        store = _TeeStore(spec.snapshot)
        provider = _engine_provider(sample_diff_text, sha)
        if prior_sha:
            # Round 2: an open bot thread from round 1 makes it incremental.
            provider.get_all_bot_threads.return_value = [
                BotThreadRecord(
                    thread_id="t1", path="src/utils.py", line=12, body="x", is_resolved=False
                )
            ]
        engine = ReviewEngine(
            config=MiraConfig.model_validate(spec.config),
            llm=mock_llm,
            provider=provider,
            state=store,
        )
        await engine.review_pr(spec.pr_url)
        if prior_sha:
            args = provider.get_compare_diff.call_args.args
            assert (args[1], args[2]) == (prior_sha, sha)

        refreshed = client.post(f"/runner/jobs/{job_id}/token", headers=auth(run_id=str(run_id)))
        assert refreshed.status_code == 200
        grant = TokenGrant.model_validate_json(refreshed.content)
        assert grant.github_token.get_secret_value() != spec.github_token.get_secret_value()

        usage = UsageReport(harness="claude-code", model="claude-opus", input_tokens=1200)
        resp = post_result(
            client, store.bundle(job_id, "succeeded", usage=usage), job_id, run_id=str(run_id)
        )
        assert resp.json() == {"status": "succeeded", "applied": True}
        job = db.get_review_job(job_id)
        assert job is not None and job.status is JobStatus.SUCCEEDED
        assert job.usage == usage.model_dump(mode="json")
        replay += store.calls

    assert {"set_last_reviewed_sha", "record_review", "upsert_pr_fingerprint"} <= {
        name for name, _, _ in replay
    }
    assert [inst for inst, _, _ in app_auth.minted] == [INSTALLATION] * 4
    assert {(repo, tuple(perms.items())) for _, repo, perms in app_auth.minted} == {
        ("repo", tuple(RUNNER_TOKEN_PERMISSIONS.items()))
    }
    applied = state_rows(PR)

    # The same calls straight into a LocalStateStore on a fresh server.
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path / "direct"))
    monkeypatch.setattr("mira.dashboard.api._app_db", AppDatabase(url="", admin_password="pw"))
    local = LocalStateStore()
    for name, args, kwargs in replay:
        getattr(local, name)(*args, **kwargs)
    assert state_rows(PR) == applied
    assert applied["sha"] == "sha-2" and len(applied["events"]) == 2


PR_URL = "https://github.com/test/repo/pull/1"


# ── Authentication and binding ──


@pytest.mark.parametrize(
    ("claims", "expected"),
    [
        pytest.param({"repository": "test/other"}, 403, id="wrong-repo"),
        pytest.param({"run_id": "9999"}, 403, id="wrong-run"),
        pytest.param({"event_name": "push"}, 403, id="wrong-event"),
        pytest.param(
            {"job_workflow_ref": "test/repo/.github/workflows/mira.yml@refs/heads/main"},
            403,
            id="wrong-workflow-ref",
        ),
        pytest.param({"job_workflow_ref": None}, 403, id="no-workflow-ref"),
        pytest.param({"aud": "someone-else"}, 401, id="wrong-aud"),
        pytest.param({"iss": "https://evil.example"}, 401, id="wrong-issuer"),
        pytest.param({"exp": int(time.time()) - 3600}, 401, id="expired"),
        pytest.param({"iat": int(time.time()) + 3600}, 401, id="issued-in-future"),
        pytest.param({"run_id": None}, 401, id="no-run-id"),
        pytest.param({"key": _OTHER_KEY}, 401, id="wrong-signing-key"),
    ],
)
def test_claim_refuses_a_token_not_bound_to_the_job(
    client: TestClient,
    db: AppDatabase,
    caplog: pytest.LogCaptureFixture,
    claims: dict[str, Any],
    expected: int,
) -> None:
    seed_job(db)
    token = oidc_token(**claims)
    with caplog.at_level(logging.INFO):
        resp = client.post("/runner/jobs/job-1/claim", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == expected
    assert resp.json() == {"detail": "unauthorized" if expected == 401 else "forbidden"}
    assert status_of(db) is JobStatus.DISPATCHED
    assert token not in caplog.text


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic abc"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer junk"},
    ],
)
def test_claim_without_a_valid_bearer_token_is_unauthorized(
    client: TestClient, db: AppDatabase, headers: dict[str, str]
) -> None:
    seed_job(db)
    resp = client.post("/runner/jobs/job-1/claim", headers=headers)
    assert resp.status_code == 401
    assert status_of(db) is JobStatus.DISPATCHED


def test_an_empty_workflow_allowlist_refuses_every_runner(
    client: TestClient, db: AppDatabase, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.setattr(mira_config, "_global_defaults", {})
    seed_job(db)
    with caplog.at_level(logging.WARNING):
        resp = claim(client)
    assert resp.status_code == 403
    assert "runner_workflow_refs is empty" in caplog.text


def test_a_job_whose_run_id_is_not_recorded_yet_is_refused(
    client: TestClient, db: AppDatabase
) -> None:
    seed_job(db, run_id=None)
    assert claim(client).status_code == 403


def test_unknown_job_is_not_found(client: TestClient, db: AppDatabase) -> None:
    assert claim(client, "no-such-job").status_code == 404
    # Without a valid token the caller learns nothing about which jobs exist.
    resp = client.post("/runner/jobs/no-such-job/claim", headers=auth(key=_OTHER_KEY))
    assert resp.status_code == 401


# ── Claim and token ──


@pytest.mark.parametrize(
    "status", [JobStatus.SUPERSEDED, JobStatus.EXPIRED, JobStatus.FAILED, JobStatus.SUCCEEDED]
)
def test_claiming_a_finished_job_conflicts(
    client: TestClient, db: AppDatabase, app_auth: FakeAppAuth, status: JobStatus
) -> None:
    seed_job(db, status=status)
    resp = claim(client)
    assert resp.status_code == 409
    assert resp.json()["status"] == status.value
    assert app_auth.minted == []


def test_a_claimed_job_can_be_claimed_again_by_its_run(
    client: TestClient, db: AppDatabase, caplog: pytest.LogCaptureFixture
) -> None:
    seed_job(db)
    with caplog.at_level(logging.DEBUG):
        first = claim(client)
        second = claim(client)
    assert (first.status_code, second.status_code) == (200, 200)
    tokens = [JobSpec.model_validate_json(r.content).github_token for r in (first, second)]
    assert tokens[0].get_secret_value() == "ghs_job_token_1"
    assert tokens[1].get_secret_value() == "ghs_job_token_2"
    assert "ghs_job_token" not in caplog.text
    assert status_of(db) is JobStatus.CLAIMED


def test_token_refresh_needs_a_claimed_job(client: TestClient, db: AppDatabase) -> None:
    seed_job(db)
    resp = client.post("/runner/jobs/job-1/token", headers=auth())
    assert resp.status_code == 409 and resp.json()["status"] == "dispatched"
    assert client.post("/runner/jobs/job-1/token", headers=auth(run_id="7")).status_code == 403


def test_a_failed_token_mint_is_a_bad_gateway(
    client: TestClient, db: AppDatabase, app_auth: FakeAppAuth, monkeypatch
) -> None:
    seed_job(db)
    monkeypatch.setattr(app_auth, "mint_repo_token", AsyncMock(side_effect=RuntimeError("422")))
    assert claim(client).status_code == 502
    # The job stays claimed, so the runner can claim again.
    assert status_of(db) is JobStatus.CLAIMED


# ── Result ──


def _claimed_store(client: TestClient, db: AppDatabase) -> RemoteStateStore:
    seed_job(db)
    spec = JobSpec.model_validate_json(claim(client).content)
    store = RemoteStateStore(spec.snapshot)
    store.record_review("test", "repo", _review(pr_number=1), platform="github")
    store.set_last_reviewed_sha("test", "repo", 1, "sha-1", platform="github")
    return store


def test_a_duplicate_result_is_accepted_and_ignored(client: TestClient, db: AppDatabase) -> None:
    bundle = _claimed_store(client, db).bundle("job-1", "succeeded")
    first = post_result(client, bundle)
    rows = state_rows(PR)
    second = post_result(client, bundle)

    assert first.json() == {"status": "succeeded", "applied": True}
    assert second.status_code == 200
    assert second.json() == {"status": "succeeded", "applied": False}
    assert state_rows(PR) == rows and len(rows["events"]) == 1


def test_a_superseded_job_keeps_its_status_but_gets_its_writes(
    client: TestClient, db: AppDatabase
) -> None:
    store = _claimed_store(client, db)
    assert db.transition_review_job("job-1", JobStatus.CLAIMED, JobStatus.SUPERSEDED)

    resp = post_result(client, store.bundle("job-1", "succeeded"))
    again = post_result(client, store.bundle("job-1", "succeeded"))

    assert resp.json() == {"status": "superseded", "applied": True}
    assert again.json() == {"status": "superseded", "applied": False}
    rows = state_rows(PR)
    assert rows["sha"] == "sha-1" and len(rows["events"]) == 1
    assert status_of(db) is JobStatus.SUPERSEDED


@pytest.mark.parametrize("status", [JobStatus.DISPATCHED, JobStatus.EXPIRED, JobStatus.SUPERSEDED])
def test_a_result_for_a_job_that_never_ran_conflicts(
    client: TestClient, db: AppDatabase, status: JobStatus
) -> None:
    seed_job(db, status=status)
    bundle = RemoteStateStore(_empty_snapshot()).bundle("job-1", "succeeded")
    resp = post_result(client, bundle)
    assert resp.status_code == 409 and resp.json()["status"] == status.value
    assert state_rows(PR)["events"] == []


def test_a_result_after_expiry_of_a_claimed_job_is_applied(
    client: TestClient, db: AppDatabase
) -> None:
    store = _claimed_store(client, db)
    assert db.transition_review_job("job-1", JobStatus.CLAIMED, JobStatus.EXPIRED)
    resp = post_result(client, store.bundle("job-1", "succeeded"))
    assert resp.json() == {"status": "expired", "applied": True}
    assert state_rows(PR)["sha"] == "sha-1"
    assert status_of(db) is JobStatus.EXPIRED
    # A retry of the same result applies nothing twice.
    assert post_result(client, store.bundle("job-1", "succeeded")).json()["applied"] is False
    assert len(state_rows(PR)["events"]) == 1


def test_a_failed_result_fails_the_job_and_its_check(
    client: TestClient, db: AppDatabase, github: FakeGitHub
) -> None:
    store = _claimed_store(client, db)
    resp = post_result(client, store.bundle("job-1", "failed", error="claude exited 1"))

    assert resp.json() == {"status": "failed", "applied": True}
    job = db.get_review_job("job-1")
    assert job is not None and (job.status, job.error) == (JobStatus.FAILED, "claude exited 1")
    [patch_body] = github.calls("PATCH", "/repos/test/repo/check-runs/501")
    assert patch_body is not None
    assert patch_body["conclusion"] == "failure"
    assert patch_body["output"]["summary"] == "claude exited 1"
    # The writes of a failed review describe comments it did post.
    assert state_rows(PR)["sha"] == "sha-1"


def test_a_result_must_describe_its_own_job(client: TestClient, db: AppDatabase) -> None:
    store = _claimed_store(client, db)
    other_job = post_result(client, store.bundle("job-2", "succeeded"))
    store.writes.append(store.writes[0].model_copy(update={"review": _review(pr_number=2)}))
    other_pr = post_result(client, store.bundle("job-1", "succeeded"))

    assert (other_job.status_code, other_pr.status_code) == (422, 422)
    assert status_of(db) is JobStatus.CLAIMED
    assert state_rows(PR)["sha"] == ""


def test_a_result_body_is_bounded(
    client: TestClient, db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _claimed_store(client, db)
    monkeypatch.setattr(runner_api, "MAX_RESULT_BYTES", 100)
    assert post_result(client, store.bundle("job-1", "succeeded")).status_code == 413
    assert status_of(db) is JobStatus.CLAIMED


def test_a_result_needs_the_jobs_run(client: TestClient, db: AppDatabase) -> None:
    store = _claimed_store(client, db)
    resp = post_result(client, store.bundle("job-1", "succeeded"), run_id="1002")
    assert resp.status_code == 403
    assert status_of(db) is JobStatus.CLAIMED


def _empty_snapshot():
    from mira.runner_protocol import StateSnapshot

    return StateSnapshot(
        pr=PR,
        repo_record=None,
        global_rules=[],
        learned_rules=[],
        review_context=[],
        manifest_package_names=[],
        last_reviewed_sha="",
        review_progress=None,
        fingerprints=[],
    )


# ── Expiry sweep ──


async def test_the_sweep_expires_stale_jobs_and_closes_their_checks(
    db: AppDatabase, app_auth: FakeAppAuth, github: FakeGitHub
) -> None:
    now = time.time()
    hour = 3600
    seed_job(db, "old-dispatched", created_at=now - 26 * hour, check_run_id=601)
    seed_job(db, "new-dispatched", created_at=now - 23 * hour, check_run_id=602)
    seed_job(
        db, "old-claimed", status=JobStatus.CLAIMED, claimed_at=now - 3 * hour, check_run_id=603
    )
    seed_job(db, "new-claimed", status=JobStatus.CLAIMED, claimed_at=now - hour, check_run_id=604)
    seed_job(db, "old-done", status=JobStatus.SUCCEEDED, created_at=now - 48 * hour)

    expired = await expire_stale_jobs(db, app_auth, now=now, claim_timeout_minutes=120)  # type: ignore[arg-type]

    assert sorted(job.id for job in expired) == ["old-claimed", "old-dispatched"]
    statuses = {
        j: status_of(db, j)
        for j in ("old-dispatched", "new-dispatched", "old-claimed", "new-claimed", "old-done")
    }
    assert statuses == {
        "old-dispatched": JobStatus.EXPIRED,
        "new-dispatched": JobStatus.DISPATCHED,
        "old-claimed": JobStatus.EXPIRED,
        "new-claimed": JobStatus.CLAIMED,
        "old-done": JobStatus.SUCCEEDED,
    }
    conclusions = {
        url.rsplit("/", 1)[1]: body["conclusion"]
        for method, url, body in github.requests
        if method == "PATCH" and body is not None
    }
    assert conclusions == {"601": "neutral", "603": "failure"}
    job = db.get_review_job("old-claimed")
    assert job is not None and "120 minutes" in job.error

    # A second sweep finds nothing left to do.
    assert await expire_stale_jobs(db, app_auth, now=now, claim_timeout_minutes=120) == []  # type: ignore[arg-type]


# ── Wiring, config and the token mint ──


def test_the_server_mounts_the_runner_api_outside_dashboard_auth(db: AppDatabase) -> None:
    from mira.platforms.server import create_app

    app = create_app(app_auth=FakeAppAuth(), webhook_secret="s")  # type: ignore[arg-type]
    resp = TestClient(app).post("/runner/jobs/job-1/claim")
    # The runner API's own 401, not the dashboard's session error.
    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}


def test_a_repo_cannot_widen_the_runner_allowlist(tmp_path: Path, db: AppDatabase) -> None:
    repo_config = tmp_path / ".mira.yaml"
    repo_config.write_text(
        "review:\n"
        "  runner_workflow_refs: ['evil/x/.github/workflows/a.yml@refs/heads/main']\n"
        "  runner_oidc_audience: evil\n"
        "  max_chunks_per_review: 2\n"
    )
    review = load_config(repo_config).review
    assert (review.runner_workflow_refs, review.runner_oidc_audience) == ([REF], "mira")
    assert review.max_chunks_per_review == 2


@pytest.mark.parametrize("ref", ["review.yml", "o/r/.github/workflows/review.yml@", "o/r@main"])
def test_workflow_refs_must_name_a_workflow_at_a_ref(ref: str) -> None:
    with pytest.raises(ConfigError):
        load_config(overrides={"review.runner_workflow_refs": [ref]}, use_dashboard=False)


async def test_mint_repo_token_asks_github_for_one_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[str, dict]] = []

    async def fake_post(self, url: str, *, json: dict, **_: Any) -> httpx.Response:
        sent.append((url, json))
        expires = (datetime.now(UTC) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return httpx.Response(
            201, json={"token": "ghs_x", "expires_at": expires}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    auth_ = GitHubAppAuth("1", "unused")
    monkeypatch.setattr(auth_, "_generate_jwt", lambda: "app-jwt")

    token = await auth_.mint_repo_token(INSTALLATION, "repo", RUNNER_TOKEN_PERMISSIONS)

    [(url, body)] = sent
    assert url.endswith(f"/app/installations/{INSTALLATION}/access_tokens")
    assert body == {"repositories": ["repo"], "permissions": RUNNER_TOKEN_PERMISSIONS}
    assert token.token == "ghs_x" and token.expires_at.tzinfo is not None
    assert "ghs_x" not in repr(token)


def test_a_bundle_write_is_typed_on_the_wire() -> None:
    from mira.runner_protocol import ResultBundle

    raw = '{"job_id": "j", "outcome": "succeeded", "writes": [{"op": "drop_table"}]}'
    with pytest.raises(ValueError):
        ResultBundle.model_validate_json(raw)
    ok = ResultBundle.model_validate_json(
        '{"job_id": "j", "outcome": "succeeded", "writes": [{"op": "set_last_reviewed_sha", "sha": "a"}]}'
    )
    assert ok.writes == [SetLastReviewedSha(sha="a")]
