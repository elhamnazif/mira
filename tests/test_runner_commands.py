"""@mention commands and thread replies on a repo that reviews on Actions.

The server runs no model for such a repo: the commands that need one are
skipped, and the ones that do not (help, pause, reject) still work.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mira.config import MiraConfig, ReviewConfig
from mira.dashboard.db import AppDatabase
from mira.index.store import IndexStore
from mira.models import PRInfo
from mira.platforms.github.webhook import dispatch_github_event


class _RunNow:
    def __init__(self) -> None:
        self.tasks: list[tuple[Any, tuple, dict]] = []

    def add_task(self, func: Any, *args: Any, **kwargs: Any) -> None:
        self.tasks.append((func, args, kwargs))

    async def run(self) -> None:
        for func, args, kwargs in self.tasks:
            await func(*args, **kwargs)


@pytest.fixture
def provider() -> AsyncMock:
    fake = AsyncMock()
    fake.get_pr_info.return_value = PRInfo(
        title="Fix cents",
        description="",
        base_branch="main",
        head_branch="fix",
        url="https://github.com/o/r/pull/7",
        number=7,
        owner="o",
        repo="r",
    )
    fake.get_pr_diff.return_value = "diff --git a/x b/x\n"
    fake.get_comment_body.return_value = "Divide by 100."
    fake.get_thread_id_for_comment.return_value = "thread-1"
    fake.resolve_threads.return_value = 1
    fake.get_repo_permission.return_value = "write"
    return fake


@pytest.fixture
def llm() -> AsyncMock:
    model = AsyncMock()
    model.complete.return_value = "Because amounts are cents."
    model.complete_with_tools.return_value = '{"intent": "question", "reply": "It is cents."}'
    return model


@pytest.fixture
def world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: AsyncMock, llm: AsyncMock
) -> Iterator[MagicMock]:
    """Patch the app DB, the provider and the model factory. Yields create_llm."""
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    db = AppDatabase(url="", admin_password="pw")
    create_llm = MagicMock(return_value=llm)
    with (
        patch("mira.dashboard.api._app_db", db),
        patch("mira.platforms.github.webhook.create_provider", return_value=provider),
        patch("mira.platforms.handlers.create_llm", create_llm),
    ):
        yield create_llm


async def _deliver(event: str, payload: dict[str, Any], runner: str) -> str:
    cfg = MiraConfig(
        review=ReviewConfig(runner=runner, runner_server_url="https://mira.example")  # type: ignore[arg-type]
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


_REPO = {"owner": {"login": "o"}, "name": "r", "default_branch": "main"}
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


@pytest.mark.parametrize("body", ["@mira-bot why cents?", "@mira-bot review", "@mira-bot rest"])
async def test_model_commands_are_skipped_on_an_actions_repo(
    world: MagicMock, provider: AsyncMock, body: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="mira.platforms.github.webhook"):
        await _deliver("issue_comment", _mention(body), runner="actions")

    world.assert_not_called()
    provider.post_comment.assert_not_awaited()
    assert "reviews on GitHub Actions" in caplog.text


async def test_a_question_is_answered_in_process_on_a_local_repo(
    world: MagicMock, provider: AsyncMock
) -> None:
    await _deliver("issue_comment", _mention("@mira-bot why cents?"), runner="local")

    world.assert_called()
    [posted] = provider.post_comment.await_args_list
    assert "Because amounts are cents." in posted.args[1]


async def test_help_still_works_on_an_actions_repo(world: MagicMock, provider: AsyncMock) -> None:
    await _deliver("issue_comment", _mention("@mira-bot help"), runner="actions")

    [posted] = provider.post_comment.await_args_list
    assert "Mira commands" in posted.args[1]


async def test_pause_still_works_on_an_actions_repo(world: MagicMock, provider: AsyncMock) -> None:
    await _deliver("issue_comment", _mention("@mira-bot pause"), runner="actions")

    provider.add_label.assert_awaited_once()
    world.assert_not_called()


async def test_a_thread_reply_is_skipped_on_an_actions_repo(
    world: MagicMock, provider: AsyncMock
) -> None:
    await _deliver("pull_request_review_comment", _thread_reply("@mira-bot why?"), "actions")

    world.assert_not_called()
    provider.reply_to_review_comment.assert_not_awaited()
    # The human's reply is still recorded for the activity timeline.
    store = IndexStore.open("o", "r")
    try:
        assert [r.body for r in store.list_replies(7)] == ["@mira-bot why?"]
    finally:
        store.close()


async def test_a_thread_reply_is_answered_on_a_local_repo(
    world: MagicMock, provider: AsyncMock
) -> None:
    await _deliver("pull_request_review_comment", _thread_reply("@mira-bot why?"), "local")

    provider.reply_to_review_comment.assert_awaited_once()
    assert provider.reply_to_review_comment.await_args.args[2] == "It is cents."


async def test_reject_still_resolves_and_records_feedback_on_an_actions_repo(
    world: MagicMock, provider: AsyncMock
) -> None:
    await _deliver("pull_request_review_comment", _thread_reply("@mira-bot reject"), "actions")

    world.assert_not_called()
    provider.resolve_threads.assert_awaited_once()
    assert provider.resolve_threads.await_args.args[1] == ["thread-1"]
    store = IndexStore.open("o", "r")
    try:
        assert [f.signal for f in store.list_feedback()] == ["rejected"]
    finally:
        store.close()
