"""Retiring the bot's own threads: verified fixes get a reply, then resolve."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.core.threads import resolve_verified_threads
from mira.llm.prompts.verify_fixes import build_verify_fixes_prompt
from mira.models import PRInfo, UnresolvedThread

_PR = PRInfo(
    title="t",
    description="",
    base_branch="main",
    head_branch="f",
    url="https://github.com/o/r/pull/1",
    number=1,
    owner="o",
    repo="r",
    head_sha="7f017290aa05",
)


def _thread(thread_id: str, comment_id: int | None, outdated: bool = False) -> UnresolvedThread:
    return UnresolvedThread(
        thread_id=thread_id,
        path="a.py",
        line=3,
        body="Token compared with ==",
        is_outdated=outdated,
        comment_id=comment_id,
    )


def _provider(threads: list[UnresolvedThread]) -> MagicMock:
    provider = MagicMock()
    provider.get_unresolved_bot_threads = AsyncMock(return_value=threads)
    provider.get_file_content = AsyncMock(return_value="a\nb\nif token == expected:\n")
    provider.reply_to_review_comment = AsyncMock()
    provider.resolve_threads = AsyncMock(side_effect=lambda _pr, ids: len(ids))
    return provider


def _llm(fixed: dict[str, bool]) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=json.dumps({"results": [{"id": k, "fixed": v} for k, v in fixed.items()]})
    )
    return llm


@pytest.mark.asyncio
async def test_replies_addressed_in_head_sha_before_resolving():
    provider = _provider([_thread("T1", 101)])
    order = MagicMock()
    order.attach_mock(provider.reply_to_review_comment, "reply")
    order.attach_mock(provider.resolve_threads, "resolve")

    checked, resolved, remaining, _ = await resolve_verified_threads(
        provider, _llm({"T1": True}), _PR, "mira", dry_run=False
    )

    assert (checked, resolved, remaining) == (1, 1, [])
    assert [c[0] for c in order.mock_calls] == ["reply", "resolve"]
    provider.reply_to_review_comment.assert_awaited_once_with(_PR, 101, "Addressed in `7f01729`.")
    provider.resolve_threads.assert_awaited_once_with(_PR, ["T1"])


@pytest.mark.asyncio
async def test_outdated_thread_that_is_not_fixed_stays_open():
    outdated = _thread("T1", 101, outdated=True)
    provider = _provider([outdated])

    _, resolved, remaining, decisions = await resolve_verified_threads(
        provider, _llm({"T1": False}), _PR, "mira", dry_run=False
    )

    assert resolved == 0
    assert remaining == [outdated]
    assert [d.fixed for d in decisions] == [False]
    provider.reply_to_review_comment.assert_not_awaited()
    provider.resolve_threads.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_reply_leaves_the_thread_open():
    provider = _provider([_thread("T1", 101), _thread("T2", 102)])
    provider.reply_to_review_comment = AsyncMock(side_effect=[RuntimeError("502"), None])

    _, resolved, _, _ = await resolve_verified_threads(
        provider, _llm({"T1": True, "T2": True}), _PR, "mira", dry_run=False
    )

    assert resolved == 1
    provider.resolve_threads.assert_awaited_once_with(_PR, ["T2"])


@pytest.mark.asyncio
async def test_a_thread_without_a_reply_id_is_not_resolved():
    provider = _provider([_thread("T1", None)])

    _, resolved, _, _ = await resolve_verified_threads(
        provider, _llm({"T1": True}), _PR, "mira", dry_run=False
    )

    assert resolved == 0
    provider.reply_to_review_comment.assert_not_awaited()
    provider.resolve_threads.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_the_bots_own_threads_are_fetched():
    provider = _provider([])

    await resolve_verified_threads(provider, _llm({}), _PR, "mira", dry_run=False)

    provider.get_unresolved_bot_threads.assert_awaited_once_with(_PR, "mira")


def test_verify_prompt_says_outdated_is_not_addressed_and_every_concern_counts():
    system = build_verify_fixes_prompt([("a.py", "1| x", [_thread("T1", 1, outdated=True)])])[0][
        "content"
    ]

    assert "That says nothing about whether the issue was addressed" in system
    assert "only when every concern the issue raises is resolved" in system
    assert "when the fix is partial, or when you are unsure" in system
    assert "very likely fixed" not in system
