"""handle_pull_request with a runner target: local review or Actions dispatch.

Drives the real GitHubProvider against a fake GitHub HTTP layer and a real
SQLite app DB, and asserts what GitHub receives and what the DB records.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from mira.config import MiraConfig, ReviewConfig
from mira.dashboard.db import AppDatabase
from mira.platforms.github.webhook import handle_pull_request
from mira.runner import JobStatus

INSTALLATION = 184691606


class FakeGitHub:
    """Answers the REST calls dispatch makes and records every request."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []
        self.dispatch_status = 200
        self.cancel_status = 202
        self._next_run = 1000
        self._next_check = 500
        self.during_dispatch: Callable[[], Awaitable[None]] | None = None

    def calls(self, method: str, path_suffix: str) -> list[dict[str, Any] | None]:
        return [body for m, url, body in self.requests if m == method and url.endswith(path_suffix)]

    async def post(self, url: str, *, json: dict[str, Any] | None = None, **_: Any):
        self.requests.append(("POST", url, json))
        request = httpx.Request("POST", url)
        if url.endswith("/check-runs"):
            self._next_check += 1
            return httpx.Response(201, json={"id": self._next_check}, request=request)
        if url.endswith("/dispatches"):
            if self.during_dispatch is not None:
                hook, self.during_dispatch = self.during_dispatch, None
                await hook()
            if self.dispatch_status != 200:
                return httpx.Response(self.dispatch_status, text="Not Found", request=request)
            self._next_run += 1
            run = self._next_run
            html_url = f"https://github.com/o/r/actions/runs/{run}"
            return httpx.Response(
                200, json={"workflow_run_id": run, "html_url": html_url}, request=request
            )
        if url.endswith("/cancel"):
            return httpx.Response(self.cancel_status, json={}, request=request)
        raise AssertionError(f"unexpected POST {url}")

    async def patch(self, url: str, *, json: dict[str, Any] | None = None, **_: Any):
        self.requests.append(("PATCH", url, json))
        return httpx.Response(200, json={}, request=httpx.Request("PATCH", url))


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    return AppDatabase(url="", admin_password="admin")


@pytest.fixture
def github() -> Iterator[FakeGitHub]:
    fake = FakeGitHub()
    with (
        patch.object(httpx.AsyncClient, "post", fake.post),
        patch.object(httpx.AsyncClient, "patch", fake.patch),
    ):
        yield fake


@pytest.fixture
def app_auth() -> AsyncMock:
    auth = AsyncMock()
    auth.get_installation_token = AsyncMock(return_value="ghs_test_token")
    return auth


@pytest.fixture
def local_review() -> Iterator[AsyncMock]:
    with patch("mira.platforms.github.webhook.run_pr_review", new_callable=AsyncMock) as m:
        yield m


@pytest.fixture
def review_failed() -> Iterator[AsyncMock]:
    with patch("mira.outbound_webhooks.dispatch_event", new_callable=AsyncMock) as m:
        yield m


def _payload(sha: str = "sha1", action: str = "opened") -> dict[str, Any]:
    return {
        "installation": {"id": INSTALLATION},
        "action": action,
        "pull_request": {
            "number": 7,
            "title": "Fix cents",
            "head": {"sha": sha, "ref": "fix"},
            "base": {"ref": "main"},
        },
        "repository": {
            "owner": {"login": "o"},
            "name": "r",
            "default_branch": "trunk",
            "private": False,
        },
    }


async def _deliver(
    db: AppDatabase, payload: dict[str, Any], app_auth: AsyncMock, **review: Any
) -> None:
    review.setdefault("runner_server_url", "https://mira.example")
    cfg = MiraConfig(review=ReviewConfig(**review))
    with (
        patch("mira.dashboard.api._app_db", db),
        patch("mira.platforms.github.webhook.load_config", return_value=cfg),
    ):
        await handle_pull_request(payload, app_auth, "mira-bot")


def _jobs(db: AppDatabase) -> list[tuple[str, str, str]]:
    rows = db._rows("SELECT id, head_sha, status FROM review_jobs ORDER BY created_at")
    return [(r[0], r[1], r[2]) for r in rows]


async def test_local_mode_reviews_in_process(db, github, app_auth, local_review):
    await _deliver(db, _payload(), app_auth)

    local_review.assert_awaited_once()
    assert local_review.await_args.args[1:4] == ("o", "r", 7)
    assert github.requests == []
    assert _jobs(db) == []


async def test_actions_mode_queues_check_and_dispatches(db, github, app_auth, local_review):
    await _deliver(db, _payload(), app_auth, runner="actions")

    local_review.assert_not_awaited()
    [(job_id, head_sha, status)] = _jobs(db)
    assert (head_sha, status) == ("sha1", "dispatched")
    assert len(job_id) >= 40  # token_urlsafe(32)

    [check] = github.calls("POST", "/repos/o/r/check-runs")
    assert check is not None
    assert (check["head_sha"], check["status"]) == ("sha1", "queued")

    [dispatch] = github.calls("POST", "/repos/o/r/actions/workflows/mira.yml/dispatches")
    assert dispatch == {
        "ref": "trunk",
        "inputs": {
            "job_id": job_id,
            "runs_on": '["ubuntu-latest"]',
            "server_url": "https://mira.example",
        },
        "return_run_details": True,
    }

    job = db.get_review_job(job_id)
    assert job is not None
    assert job.installation_id == INSTALLATION
    assert job.check_run_id == 501
    assert job.run_id == 1001
    assert job.run_url == "https://github.com/o/r/actions/runs/1001"


async def test_repo_override_beats_local_default(db, github, app_auth, local_review):
    db.register_repo("o", "r")
    db.set_repo_runner("o", "r", "actions:self-hosted,sora")

    await _deliver(db, _payload(), app_auth)

    local_review.assert_not_awaited()
    [dispatch] = github.calls("POST", "/dispatches")
    assert dispatch is not None
    assert json.loads(dispatch["inputs"]["runs_on"]) == ["self-hosted", "sora"]


async def test_repo_override_beats_actions_default(db, github, app_auth, local_review):
    db.register_repo("o", "r")
    db.set_repo_runner("o", "r", "local")

    await _deliver(db, _payload(), app_auth, runner="actions")

    local_review.assert_awaited_once()
    assert github.requests == []


async def test_new_head_supersedes_and_cancels(db, github, app_auth, local_review):
    await _deliver(db, _payload("sha1"), app_auth, runner="actions")
    [(old_id, _, _)] = _jobs(db)

    await _deliver(db, _payload("sha2", "synchronize"), app_auth, runner="actions")

    jobs = _jobs(db)
    assert [(sha, status) for _, sha, status in jobs] == [
        ("sha1", "superseded"),
        ("sha2", "dispatched"),
    ]
    new_id = jobs[1][0]
    assert len(github.calls("POST", "/repos/o/r/actions/runs/1001/cancel")) == 1
    [closed] = github.calls("PATCH", "/repos/o/r/check-runs/501")
    assert closed is not None
    assert (closed["status"], closed["conclusion"]) == ("completed", "neutral")
    dispatches = github.calls("POST", "/dispatches")
    assert [d["inputs"]["job_id"] for d in dispatches if d] == [old_id, new_id]


async def test_push_during_dispatch_retires_the_older_run(db, github, app_auth, local_review):
    # sha2 arrives while sha1's dispatch request is still in flight, so sha1
    # has no run to cancel yet when it is superseded.
    async def push_sha2() -> None:
        await _deliver(db, _payload("sha2", "synchronize"), app_auth, runner="actions")

    github.during_dispatch = push_sha2
    await _deliver(db, _payload("sha1"), app_auth, runner="actions")

    assert [(sha, status) for _, sha, status in _jobs(db)] == [
        ("sha1", "superseded"),
        ("sha2", "dispatched"),
    ]
    # sha2's dispatch answered first with run 1001; sha1's run 1002 is cancelled.
    assert len(github.calls("POST", "/actions/runs/1002/cancel")) == 1
    assert github.calls("POST", "/actions/runs/1001/cancel") == []
    # Both paths may close sha1's check; each close sets the same end state.
    closes = github.calls("PATCH", "/repos/o/r/check-runs/501")
    assert closes
    assert {(c["status"], c["conclusion"]) for c in closes if c} == {("completed", "neutral")}


async def test_same_head_is_not_dispatched_twice(db, github, app_auth, local_review):
    await _deliver(db, _payload("sha1"), app_auth, runner="actions")
    await _deliver(db, _payload("sha1", "reopened"), app_auth, runner="actions")

    assert len(_jobs(db)) == 1
    assert len(github.calls("POST", "/dispatches")) == 1


async def test_failed_cancel_still_dispatches_new_head(
    db, github, app_auth, local_review, caplog: pytest.LogCaptureFixture
):
    await _deliver(db, _payload("sha1"), app_auth, runner="actions")
    github.cancel_status = 409

    with caplog.at_level(logging.WARNING):
        await _deliver(db, _payload("sha2", "synchronize"), app_auth, runner="actions")

    assert [s for _, _, s in _jobs(db)] == ["superseded", "dispatched"]
    assert "Could not cancel workflow run 1001" in caplog.text


async def test_missing_workflow_fails_job_and_check(
    db, github, app_auth, local_review, review_failed
):
    github.dispatch_status = 404

    await _deliver(db, _payload(), app_auth, runner="actions")

    local_review.assert_not_awaited()  # never falls back to an in-process review
    [(job_id, _, status)] = _jobs(db)
    assert status == "failed"
    job = db.get_review_job(job_id)
    assert job is not None
    assert "Mira runner workflow .github/workflows/mira.yml not found" in job.error
    assert job.status is JobStatus.FAILED

    [check] = github.calls("PATCH", "/repos/o/r/check-runs/501")
    assert check is not None
    assert (check["status"], check["conclusion"]) == ("completed", "failure")
    assert check["output"]["summary"] == job.error

    [(event, data)] = [c.args for c in review_failed.await_args_list]
    assert event == "review.failed"
    assert data["error"] == job.error


async def test_no_server_url_fails_job_and_check_without_dispatching(
    db, github, app_auth, local_review
):
    await _deliver(db, _payload(), app_auth, runner="actions", runner_server_url="")

    [(job_id, _, status)] = _jobs(db)
    job = db.get_review_job(job_id)
    assert job is not None and status == "failed"
    assert "review.runner_server_url" in job.error
    assert github.calls("POST", "/dispatches") == []
    [check] = github.calls("PATCH", "/repos/o/r/check-runs/501")
    assert check is not None
    assert (check["conclusion"], check["output"]["summary"]) == ("failure", job.error)
    local_review.assert_not_awaited()


async def test_missing_permission_names_actions_write(db, github, app_auth, local_review):
    github.dispatch_status = 403

    await _deliver(db, _payload(), app_auth, runner="actions")

    [(job_id, _, status)] = _jobs(db)
    assert status == "failed"
    job = db.get_review_job(job_id)
    assert job is not None
    assert "Actions: write" in job.error


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://mira.example/", "https://mira.example"),
        ("http://localhost:8000", "http://localhost:8000"),
        ("http://mira.example", None),
        ("mira.example", None),
    ],
)
def test_runner_server_url_is_https_unless_local(url: str, ok: str | None) -> None:
    from mira.config import ConfigError, load_config

    overrides = {"review.runner_server_url": url}
    if ok is None:
        with pytest.raises(ConfigError):
            load_config(overrides=overrides, use_dashboard=False)
    else:
        assert load_config(overrides=overrides, use_dashboard=False).review.runner_server_url == ok


def test_a_repo_cannot_point_its_runner_at_another_server(tmp_path) -> None:
    from mira.config import load_config

    repo_config = tmp_path / ".mira.yaml"
    repo_config.write_text("review:\n  runner_server_url: https://evil.example\n")
    assert load_config(repo_config, use_dashboard=False).review.runner_server_url == ""
