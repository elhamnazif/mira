"""Learning from a merged PR on a repo that reviews on GitHub Actions, end to end.

The server side is the real merged-PR handler, the real dispatch and runner
API on a real SQLite app DB and index store. The runner is ``mira runner``
against that API through an ASGI transport, learning with a fake model. The
server keeps the merge's signals and rule-based rules, and every server-side
way to make a model raises, so a model rule lands only if the runner learned
it.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from mira import config as mira_config
from mira import runner_client
from mira.dashboard.db import AppDatabase
from mira.index.store import IndexStore
from mira.models import HumanReviewComment, PRInfo
from mira.platforms.github import webhook
from mira.platforms.github.runner_api import DISPATCHED_TTL_S, expire_stale_jobs
from mira.providers.github import GitHubProvider
from mira.runner import JobStatus, LearnTask, ReviewJob
from mira.runner_client import RunnerLLM, run_job
from mira.runner_protocol import UsageReport
from tests.test_runner_api import INSTALLATION, REF, db, seed_job  # noqa: F401
from tests.test_runner_client import (  # noqa: F401  (fixtures)
    SERVER,
    app_auth,
    oidc_requests,
    revoked,
    server,
)
from tests.test_runner_dispatch import FakeGitHub

OWNER, REPO = "test", "repo"
RULES = {
    "rules": [
        {
            "rule": "Keep raw SQL in the data layer.",
            "rationale": "Reviewers keep asking for it.",
            "evidence_count": 2,
        }
    ]
}


class LearningModel:
    """Answers the synthesis prompt with ``RULES``, and records each prompt."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(model="test-model", provider="openai")
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.prompts: list[str] = []
        self.error: Exception | None = None
        self.hang = False

    def preflight(self) -> None:
        pass

    async def complete(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.prompts.append(messages[0]["content"])
        self.usage["prompt_tokens"] += 30
        self.usage["completion_tokens"] += 5
        if self.hang:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return json.dumps(RULES)


def _server_model_is_never_built(*_: Any, **__: Any) -> Any:
    raise AssertionError("the server built a model")


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch, db: AppDatabase) -> LearningModel:  # noqa: F811
    """The runner's model, with the server side set to review on Actions and
    unable to build one."""
    fake = LearningModel()
    monkeypatch.setattr(runner_client, "create_llm", lambda config: fake)
    monkeypatch.setattr("mira.platforms.handlers.create_llm", _server_model_is_never_built)
    monkeypatch.setattr(
        mira_config,
        "_global_defaults",
        {"review": {"runner_workflow_refs": [REF], "runner_server_url": "https://mira.test"}},
    )
    db.register_repo(OWNER, REPO, INSTALLATION)
    db.set_repo_runner(OWNER, REPO, "actions:ubuntu-latest")
    return fake


@pytest.fixture
def github() -> Iterator[FakeGitHub]:
    fake = FakeGitHub()
    with (
        patch.object(httpx.AsyncClient, "post", fake.post),
        patch.object(httpx.AsyncClient, "patch", fake.patch),
    ):
        yield fake


@pytest.fixture
def pr(monkeypatch: pytest.MonkeyPatch) -> GitHubProvider:
    """The real provider, so the dispatch reaches the fake GitHub, reading a
    merged PR with two human review comments."""
    provider = GitHubProvider("ghs_test_token")
    reads: dict[str, Any] = {
        "get_pr_info": PRInfo(
            title="Move queries",
            description="",
            base_branch="main",
            head_branch="move",
            url=f"https://github.com/{OWNER}/{REPO}/pull/7",
            number=7,
            owner=OWNER,
            repo=REPO,
            head_sha="sha-7",
        ),
        "get_all_bot_threads": [],
        "get_human_review_comments": [
            HumanReviewComment(
                path="api.py", line=3, body="No raw SQL in handlers.", author="lead"
            ),
            HumanReviewComment(
                path="jobs.py", line=9, body="Move this query to db/.", author="sam"
            ),
        ],
    }
    for name, value in reads.items():
        setattr(provider, name, AsyncMock(return_value=value))
    monkeypatch.setattr(webhook, "create_provider", lambda *a, **k: provider)
    return provider


async def _merge() -> None:
    payload = {
        "action": "closed",
        "installation": {"id": INSTALLATION},
        "pull_request": {
            "number": 7,
            "merged": True,
            "merged_by": {"login": "lead"},
            "user": {"login": "dev"},
            "title": "Move queries",
            "body": "",
            "base": {"ref": "main"},
            "head": {"ref": "move", "sha": "sha-7"},
        },
        "repository": {"owner": {"login": OWNER}, "name": REPO, "default_branch": "main"},
        "sender": {"login": "lead"},
    }
    await webhook.handle_pr_merged(payload, MagicMock(), "mira-bot")


async def _run(app: FastAPI, job_id: str, timeout: timedelta | None = None) -> int:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
        return await run_job(
            http,
            job_id=job_id,
            server_url=SERVER,
            audience="mira",
            llm=RunnerLLM(),
            timeout=timeout,
        )


def _human_rules() -> list[tuple[str, str]]:
    store = IndexStore.open(OWNER, REPO)
    try:
        return [
            (r.rule_text, r.status)
            for r in store.list_learned_rules()
            if r.source_signal == "human_pattern"
        ]
    finally:
        store.close()


def _signals() -> list[str]:
    store = IndexStore.open(OWNER, REPO)
    try:
        return sorted(f.signal for f in store.list_feedback())
    finally:
        store.close()


def _the_job(db: AppDatabase) -> ReviewJob:  # noqa: F811
    [job] = db.list_active_review_jobs(OWNER, REPO, 7)
    return job


async def test_a_merged_pr_learns_from_its_human_reviews_on_the_runner(
    server: FastAPI,  # noqa: F811
    db: AppDatabase,  # noqa: F811
    app_auth,  # noqa: F811
    github: FakeGitHub,
    model: LearningModel,
    pr: GitHubProvider,
) -> None:
    await _merge()

    # The server recorded the merge's signals itself and dispatched the
    # model pass, with no check run.
    assert _signals() == ["human_review", "human_review"]
    job = _the_job(db)
    assert (job.task, job.head_sha, job.check_run_id) == (LearnTask(), "sha-7", None)
    assert job.run_url == "https://github.com/o/r/actions/runs/1001"
    [dispatch] = github.calls("POST", "/actions/workflows/mira.yml/dispatches")
    assert dispatch is not None and dispatch["inputs"]["job_id"] == job.id
    assert github.calls("POST", "/check-runs") == []
    assert _human_rules() == [] and model.prompts == []

    assert await _run(server, job.id) == 0

    done = db.get_review_job(job.id)
    assert done is not None and done.status is JobStatus.SUCCEEDED
    assert done.usage == UsageReport(
        harness="openai", model="test-model", input_tokens=30, output_tokens=5
    ).model_dump(mode="json")
    # The rule waits for an admin, as one learned on the server does.
    assert _human_rules() == [
        ("Keep raw SQL in the data layer. (Reviewers keep asking for it.)", "pending")
    ]
    [prompt] = model.prompts
    assert "No raw SQL in handlers." in prompt and "Move this query to db/." in prompt
    # The job's token can only read: learning posts nothing to GitHub.
    assert [permissions for _, _, permissions in app_auth.minted] == [{"contents": "read"}]


async def test_a_merge_without_human_reviews_dispatches_nothing(
    db: AppDatabase,  # noqa: F811
    github: FakeGitHub,
    model: LearningModel,
    pr: GitHubProvider,
) -> None:
    pr.get_human_review_comments.return_value = []  # type: ignore[attr-defined]

    await _merge()

    assert db.list_active_review_jobs(OWNER, REPO, 7) == []
    assert github.requests == []


async def test_a_local_repo_still_learns_in_process(
    db: AppDatabase,  # noqa: F811
    github: FakeGitHub,
    model: LearningModel,
    pr: GitHubProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.set_repo_runner(OWNER, REPO, "local")
    monkeypatch.setattr("mira.platforms.handlers.create_llm", lambda config: model)

    await _merge()

    assert db.list_active_review_jobs(OWNER, REPO, 7) == []
    assert github.requests == []
    assert [text for text, _ in _human_rules()] == [
        "Keep raw SQL in the data layer. (Reviewers keep asking for it.)"
    ]


async def test_a_learning_job_whose_model_fails_fails_and_learns_nothing(
    server: FastAPI,  # noqa: F811
    db: AppDatabase,  # noqa: F811
    github: FakeGitHub,
    model: LearningModel,
    pr: GitHubProvider,
) -> None:
    await _merge()
    job = _the_job(db)
    model.error = RuntimeError("model down")

    assert await _run(server, job.id) == 1

    done = db.get_review_job(job.id)
    assert done is not None and (done.status, done.error) == (JobStatus.FAILED, "RuntimeError")
    assert _human_rules() == []


async def test_a_learning_job_past_its_timeout_says_learning_timed_out(
    server: FastAPI,  # noqa: F811
    db: AppDatabase,  # noqa: F811
    github: FakeGitHub,
    model: LearningModel,
    pr: GitHubProvider,
) -> None:
    await _merge()
    job = _the_job(db)
    model.hang = True

    assert await _run(server, job.id, timeout=timedelta(seconds=0.2)) == 1

    done = db.get_review_job(job.id)
    assert done is not None and done.status is JobStatus.FAILED
    assert done.error.startswith("Learning timed out after")


async def test_a_learning_job_no_runner_picked_up_expires_as_learning(
    db: AppDatabase,  # noqa: F811
    app_auth,  # noqa: F811
) -> None:
    now = time.time()
    seed_job(db, task=LearnTask(), check_run_id=None, created_at=now - DISPATCHED_TTL_S - 1)

    [expired] = await expire_stale_jobs(db, app_auth, now=now, claim_timeout_minutes=120)

    done = db.get_review_job(expired.id)
    assert done is not None and done.status is JobStatus.EXPIRED
    assert done.error == "No Mira runner picked up this learning job within 25 hours."
