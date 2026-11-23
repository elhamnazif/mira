"""@mention commands and thread replies on a repo that reviews on Actions.

The server runs no model for such a repo: the commands that need one become
jobs for the repo's runner, and the ones that do not (help, pause, reject)
still run on the server. A local repo answers in process, as before.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mira.config import CommandsConfig, MiraConfig, ReviewConfig
from mira.dashboard.db import AppDatabase, PRReviewProgress
from mira.index.store import IndexStore
from mira.models import PRInfo
from mira.platforms.github.webhook import dispatch_github_event
from mira.providers.github import GitHubProvider
from mira.runner import (
    AskTask,
    JobStatus,
    ReviewJob,
    ReviewRestTask,
    ReviewTask,
    ThreadReply,
    ThreadReplyTask,
)
from tests.test_runner_dispatch import FakeGitHub


class _RunNow:
    def __init__(self) -> None:
        self.tasks: list[tuple[Any, tuple, dict]] = []

    def add_task(self, func: Any, *args: Any, **kwargs: Any) -> None:
        self.tasks.append((func, args, kwargs))

    async def run(self) -> None:
        for func, args, kwargs in self.tasks:
            await func(*args, **kwargs)


@pytest.fixture
def provider() -> GitHubProvider:
    """The real provider, so a dispatch's check run and workflow dispatch
    reach the fake GitHub, with mocks for what it reads and posts."""
    real = GitHubProvider("ghs_test_token")
    mocks: dict[str, Any] = {
        "get_pr_info": PRInfo(
            title="Fix cents",
            description="",
            base_branch="main",
            head_branch="fix",
            url="https://github.com/o/r/pull/7",
            number=7,
            owner="o",
            repo="r",
            head_sha="sha-7",
        ),
        "get_pr_diff": "diff --git a/x b/x\n",
        "get_comment_body": "Divide by 100.",
        "get_thread_id_for_comment": "thread-1",
        "resolve_threads": 1,
        "get_repo_permission": "write",
        "post_comment": None,
        "reply_to_review_comment": None,
        "add_label": None,
    }
    for name, value in mocks.items():
        setattr(real, name, AsyncMock(return_value=value))
    return real


@pytest.fixture
def llm() -> AsyncMock:
    model = AsyncMock()
    model.complete.return_value = "Because amounts are cents."
    model.complete_with_tools.return_value = '{"intent": "question", "reply": "It is cents."}'
    return model


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    return AppDatabase(url="", admin_password="pw")


@pytest.fixture
def github() -> Iterator[FakeGitHub]:
    fake = FakeGitHub()
    with (
        patch.object(httpx.AsyncClient, "post", fake.post),
        patch.object(httpx.AsyncClient, "patch", fake.patch),
    ):
        yield fake


@pytest.fixture
def world(
    db: AppDatabase, provider: GitHubProvider, llm: AsyncMock, github: FakeGitHub
) -> Iterator[MagicMock]:
    """Patch the app DB, the provider and the model factory. Yields create_llm."""
    create_llm = MagicMock(return_value=llm)
    with (
        patch("mira.dashboard.api._app_db", db),
        patch("mira.platforms.github.webhook.create_provider", return_value=provider),
        patch("mira.platforms.handlers.create_llm", create_llm),
    ):
        yield create_llm


async def _deliver(
    event: str, payload: dict[str, Any], runner: str, commands: dict[str, Any] | None = None
) -> str:
    cfg = MiraConfig(
        review=ReviewConfig(runner=runner, runner_server_url="https://mira.example"),  # type: ignore[arg-type]
        commands=CommandsConfig.model_validate(commands or {}),
    )
    app_auth = AsyncMock()
    app_auth.get_bot_identity.return_value = "mira-bot"
    tasks = _RunNow()
    with (
        patch("mira.platforms.github.webhook.load_config", return_value=cfg),
        patch("mira.platforms.handlers.load_config", return_value=cfg),
    ):
        status = await dispatch_github_event(event, payload, app_auth, "mira-bot", tasks)
        await tasks.run()
    return status


_REPO = {"owner": {"login": "o"}, "name": "r", "default_branch": "trunk"}
_HUMAN = {"login": "dev", "id": 42, "type": "User"}


def _mention(body: str) -> dict[str, Any]:
    return {
        "action": "created",
        "installation": {"id": 1},
        "issue": {"number": 7, "title": "Fix cents", "pull_request": {}},
        "comment": {"id": 9, "body": body, "user": _HUMAN},
        "repository": _REPO,
    }


def _thread_reply(body: str) -> dict[str, Any]:
    return {
        "action": "created",
        "installation": {"id": 1},
        "pull_request": {"number": 7},
        "comment": {
            "id": 10,
            "node_id": "C_10",
            "in_reply_to_id": 3,
            "body": body,
            "path": "billing.py",
            "line": 4,
            "user": _HUMAN,
        },
        "repository": _REPO,
    }


def _the_job(db: AppDatabase, github: FakeGitHub) -> ReviewJob:
    """The one job dispatched, after checking GitHub was asked to run it."""
    [job] = db.list_active_review_jobs("o", "r", 7)
    [dispatch] = github.calls("POST", "/actions/workflows/mira.yml/dispatches")
    assert dispatch is not None
    assert dispatch["ref"] == "trunk" and dispatch["inputs"]["job_id"] == job.id
    assert (job.head_sha, job.installation_id) == ("sha-7", 1)
    return job


# ── Actions repo: model work becomes a job ──


async def test_a_question_becomes_an_ask_job(
    world: MagicMock, db: AppDatabase, github: FakeGitHub, provider: GitHubProvider
) -> None:
    await _deliver("issue_comment", _mention("@mira-bot why cents?"), runner="actions")

    job = _the_job(db, github)
    assert job.task == AskTask(question="why cents?", actor="dev")
    # An answer shows no check run, and the server ran no model.
    assert job.check_run_id is None
    assert github.calls("POST", "/check-runs") == []
    world.assert_not_called()
    provider.post_comment.assert_not_awaited()  # type: ignore[attr-defined]


async def test_a_review_command_dispatches_a_review_like_a_push(
    world: MagicMock, db: AppDatabase, github: FakeGitHub
) -> None:
    await _deliver("issue_comment", _mention("@mira-bot review"), runner="actions")

    job = _the_job(db, github)
    assert job.task == ReviewTask()
    [check] = github.calls("POST", "/check-runs")
    assert check is not None and check["status"] == "queued" and check["head_sha"] == "sha-7"
    assert job.check_run_id == 501
    world.assert_not_called()


async def test_review_rest_dispatches_the_skipped_files(
    world: MagicMock, db: AppDatabase, github: FakeGitHub
) -> None:
    db.upsert_pr_review_progress(
        PRReviewProgress(
            owner="o",
            repo="r",
            pr_number=7,
            total_paths=["a.py", "b.py", "c.py"],
            reviewed_paths=["a.py"],
            skipped_paths=["b.py", "c.py"],
            chunk_index=1,
        ),
        platform="github",
    )

    await _deliver("issue_comment", _mention("@mira-bot rest"), runner="actions")

    assert _the_job(db, github).task == ReviewRestTask(paths=("b.py", "c.py"))
    world.assert_not_called()


async def test_review_rest_with_nothing_skipped_answers_on_the_server(
    world: MagicMock, db: AppDatabase, github: FakeGitHub, provider: GitHubProvider
) -> None:
    await _deliver("issue_comment", _mention("@mira-bot rest"), runner="actions")

    assert db.list_active_review_jobs("o", "r", 7) == []
    [posted] = provider.post_comment.await_args_list  # type: ignore[attr-defined]
    assert "nothing left to review" in posted.args[1]


async def test_a_thread_reply_becomes_a_thread_reply_job(
    world: MagicMock, db: AppDatabase, github: FakeGitHub, provider: GitHubProvider
) -> None:
    await _deliver("pull_request_review_comment", _thread_reply("@mira-bot why?"), "actions")

    reply = ThreadReply(
        comment_id=10,
        body="why?",
        original_suggestion="Divide by 100.",
        comment_node_id="C_10",
        path="billing.py",
        line=4,
        actor="dev",
    )
    job = _the_job(db, github)
    assert job.task == ThreadReplyTask(reply=reply, may_resolve=True)
    assert job.check_run_id is None
    world.assert_not_called()
    provider.reply_to_review_comment.assert_not_awaited()  # type: ignore[attr-defined]
    # The human's reply is still recorded for the activity timeline.
    store = IndexStore.open("o", "r")
    try:
        assert [r.body for r in store.list_replies(7)] == ["@mira-bot why?"]
    finally:
        store.close()


async def test_a_thread_reply_job_carries_a_denied_reject(
    world: MagicMock, db: AppDatabase, github: FakeGitHub
) -> None:
    await _deliver(
        "pull_request_review_comment",
        _thread_reply("@mira-bot you are wrong"),
        "actions",
        commands={"reject": {"blocked_users": ["dev"]}},
    )

    task = _the_job(db, github).task
    assert isinstance(task, ThreadReplyTask) and task.may_resolve is False


@pytest.mark.parametrize(
    ("event", "payload"),
    [
        ("issue_comment", _mention("@mira-bot why cents?")),
        ("issue_comment", _mention("@mira-bot review")),
        ("pull_request_review_comment", _thread_reply("@mira-bot why?")),
    ],
)
async def test_a_user_who_may_not_run_the_command_starts_no_job(
    world: MagicMock,
    db: AppDatabase,
    github: FakeGitHub,
    provider: GitHubProvider,
    event: str,
    payload: dict[str, Any],
) -> None:
    provider.get_repo_permission.return_value = "read"  # type: ignore[attr-defined]

    await _deliver(event, payload, runner="actions")

    assert db.list_active_review_jobs("o", "r", 7) == []
    assert github.requests == []


async def test_a_question_does_not_supersede_the_review_in_flight(
    world: MagicMock, db: AppDatabase, github: FakeGitHub
) -> None:
    await _deliver("issue_comment", _mention("@mira-bot review"), runner="actions")
    await _deliver("issue_comment", _mention("@mira-bot why cents?"), runner="actions")
    # A second review of the same head is not dispatched while one is active.
    await _deliver("issue_comment", _mention("@mira-bot review"), runner="actions")

    jobs = db.list_active_review_jobs("o", "r", 7)
    assert [(j.task, j.status) for j in jobs] == [
        (ReviewTask(), JobStatus.DISPATCHED),
        (AskTask(question="why cents?", actor="dev"), JobStatus.DISPATCHED),
    ]
    assert github.calls("POST", "/cancel") == []


# ── Commands that run no model, and local repos ──


async def test_a_question_is_answered_in_process_on_a_local_repo(
    world: MagicMock, db: AppDatabase, provider: GitHubProvider, github: FakeGitHub
) -> None:
    await _deliver("issue_comment", _mention("@mira-bot why cents?"), runner="local")

    world.assert_called()
    [posted] = provider.post_comment.await_args_list  # type: ignore[attr-defined]
    assert posted.args[1] == "> @dev asked: why cents?\n\nBecause amounts are cents."
    assert db.list_active_review_jobs("o", "r", 7) == []
    assert github.requests == []


async def test_a_thread_reply_is_answered_on_a_local_repo(
    world: MagicMock, provider: GitHubProvider, github: FakeGitHub
) -> None:
    await _deliver("pull_request_review_comment", _thread_reply("@mira-bot why?"), "local")

    provider.reply_to_review_comment.assert_awaited_once()  # type: ignore[attr-defined]
    assert provider.reply_to_review_comment.await_args.args[2] == "It is cents."  # type: ignore[attr-defined]
    assert github.requests == []


async def test_help_still_works_on_an_actions_repo(
    world: MagicMock, db: AppDatabase, provider: GitHubProvider
) -> None:
    await _deliver("issue_comment", _mention("@mira-bot help"), runner="actions")

    [posted] = provider.post_comment.await_args_list  # type: ignore[attr-defined]
    assert "Mira commands" in posted.args[1]
    assert db.list_active_review_jobs("o", "r", 7) == []


async def test_pause_still_works_on_an_actions_repo(
    world: MagicMock, provider: GitHubProvider
) -> None:
    await _deliver("issue_comment", _mention("@mira-bot pause"), runner="actions")

    provider.add_label.assert_awaited_once()  # type: ignore[attr-defined]
    world.assert_not_called()


async def test_reject_still_resolves_and_records_feedback_on_an_actions_repo(
    world: MagicMock, db: AppDatabase, provider: GitHubProvider
) -> None:
    await _deliver("pull_request_review_comment", _thread_reply("@mira-bot reject"), "actions")

    world.assert_not_called()
    assert db.list_active_review_jobs("o", "r", 7) == []
    provider.resolve_threads.assert_awaited_once()  # type: ignore[attr-defined]
    assert provider.resolve_threads.await_args.args[1] == ["thread-1"]  # type: ignore[attr-defined]
    store = IndexStore.open("o", "r")
    try:
        assert [f.signal for f in store.list_feedback()] == ["rejected"]
    finally:
        store.close()
