"""The engine binds a PR checkout to its agent-CLI LLMs for the length of a review."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.config import MiraConfig
from mira.core.engine import ReviewEngine
from mira.core.git_snapshot import GitSnapshot
from mira.core.state_store import LocalStateStore
from mira.core.workspace import CheckoutSource
from mira.dashboard.db import ReviewedVersion
from mira.models import PRInfo


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Mira Tests",
            "-c",
            "user.email=mira@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _review(title: str) -> dict:
    return {
        "comments": [
            {
                "path": "src/app.py",
                "line": 1,
                "severity": "blocker",
                "category": "bug",
                "title": title,
                "body": f"{title} body",
                "confidence": 0.99,
            }
        ],
        "summary": "",
    }


class _FakeLLM:
    supports_temperature = False

    def __init__(self, *, workspace_agent: bool) -> None:
        self.supports_workspace_agent = workspace_agent
        self.workspace: Path | None = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # (call, bound workspace, src/app.py as the call could read it)
        self.calls: list[tuple[str, Path | None, str | None]] = []
        self.agentic_calls = 0

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def _saw(self, call: str) -> None:
        ws = self.workspace
        self.calls.append((call, ws, (ws / "src/app.py").read_text() if ws else None))

    def bound(self, call: str) -> list[Path | None]:
        return [ws for name, ws, _ in self.calls if name == call]

    async def complete_agentic(self, messages, tools):
        self.agentic_calls += 1
        if self.supports_workspace_agent:
            # Like AgentCLIProvider: no step protocol, so the loop forces one submission.
            return {"content": "", "tool_calls": []}
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_submit",
                    "type": "function",
                    "function": {
                        "name": "submit_review",
                        "arguments": json.dumps(_review("from agentic loop")),
                    },
                }
            ],
        }

    async def complete_with_tools(self, messages, tools, temperature=None, force_tool=None):
        name = force_tool or tools[0]["function"]["name"]
        self._saw(name)
        if name == "submit_critique":
            return json.dumps({"verdicts": [{"index": 0, "evidence": "proven", "reason": "ok"}]})
        return json.dumps(_review("from forced submit"))

    async def review(self, messages, temperature=None):
        self._saw("review")
        return json.dumps(
            _review("from workspace agent" if self.workspace else "from plain review")
        )

    async def complete(self, messages, json_mode=True, temperature=None, max_tokens=None):
        return "{}"

    async def walkthrough(self, messages):
        return json.dumps({"summary": "wt", "change_groups": []})


def _engine(llm: _FakeLLM, **llms: _FakeLLM) -> ReviewEngine:
    cfg = MiraConfig()
    cfg.review.agentic_tools = True
    cfg.review.walkthrough = False
    cfg.review.security_pass = False
    cfg.review.dependency_overlap = False
    cfg.review.self_critique = False
    cfg.review.second_look = False
    return ReviewEngine(config=cfg, llm=llm, **llms, state=LocalStateStore())


def _titles(result) -> list[str]:
    return [c.title for c in result.comments]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A repo with a `change` branch, plus a bare clone to fetch from."""
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path / "index"))
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    _git(work, "init", "-q", "--initial-branch=main", "--template=")
    (work / "src/app.py").write_text("VERSION = 1\n")
    _git(work, "add", ".")
    _git(work, "commit", "-qm", "base")
    base = _git(work, "rev-parse", "HEAD")
    _git(work, "checkout", "-qb", "change")
    (work / "src/app.py").write_text("VERSION = 2\n")
    _git(work, "commit", "-qam", "bump")
    head = _git(work, "rev-parse", "HEAD")
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(work), str(bare))
    return {
        "work": work,
        "url": bare.as_uri(),
        "head": head,
        "diff": _git(work, "diff", base, head) + "\n",
    }


# ── Local `mira review --base` ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_local_review_runs_agent_in_checkout_root(repo, monkeypatch):
    monkeypatch.chdir(repo["work"] / "src")
    snapshot = GitSnapshot("main")
    llm = _FakeLLM(workspace_agent=True)

    result = await _engine(llm).review_diff(snapshot.diff(), local_snapshot=snapshot)

    assert llm.bound("review") == [repo["work"]]
    assert llm.agentic_calls == 0
    assert llm.workspace is None
    assert _titles(result) == ["from workspace agent"]
    assert {"stage": "agent_cli", "chunk": 0} in result.audit


@pytest.mark.asyncio
async def test_local_review_without_workspace_agent_uses_agentic_loop(repo, monkeypatch):
    monkeypatch.chdir(repo["work"])
    snapshot = GitSnapshot("main")
    llm = _FakeLLM(workspace_agent=False)

    result = await _engine(llm).review_diff(snapshot.diff(), local_snapshot=snapshot)

    assert {ws for _, ws, _ in llm.calls} <= {None}
    assert llm.agentic_calls >= 1
    assert _titles(result) == ["from agentic loop"]


@pytest.mark.asyncio
async def test_plain_diff_review_has_no_workspace(repo):
    llm = _FakeLLM(workspace_agent=True)

    result = await _engine(llm).review_diff(repo["diff"])

    assert llm.bound("review") == [None]
    assert _titles(result) == ["from plain review"]


# ── Hosted PR review ───────────────────────────────────────────────────


def _provider(repo: dict, source: CheckoutSource | None) -> AsyncMock:
    provider = AsyncMock()
    provider.get_pr_info.return_value = PRInfo(
        title="Bump",
        description="",
        base_branch="main",
        head_branch="change",
        url="https://github.com/test/repo/pull/1",
        number=1,
        owner="test",
        repo="repo",
        head_sha=repo["head"],
    )
    provider.get_pr_diff.return_value = repo["diff"]
    provider.get_all_bot_threads.return_value = []
    provider.get_unresolved_bot_threads.return_value = []
    provider.get_file_history.return_value = {}
    provider.get_repo_tree.return_value = ["src/app.py"]
    provider.get_file_content.return_value = "VERSION = 2\n"
    provider.find_bot_comment = AsyncMock(return_value=None)
    provider.checkout_source = AsyncMock(return_value=source)
    return provider


async def _review_pr(engine: ReviewEngine, provider: AsyncMock, monkeypatch):
    db = MagicMock()
    db.get_last_reviewed.return_value = ReviewedVersion()
    db.get_repo.return_value = None
    db.get_pr_review_progress.return_value = None
    monkeypatch.setattr("mira.dashboard.api._app_db", db)
    engine.provider = provider
    engine.dry_run = True
    return await engine.review_pr("https://github.com/test/repo/pull/1")


@pytest.mark.asyncio
async def test_pr_review_runs_agent_in_fresh_checkout_of_head(repo, monkeypatch):
    llm = _FakeLLM(workspace_agent=True)
    provider = _provider(repo, CheckoutSource(repo["url"], repo["head"]))

    result = await _review_pr(_engine(llm), provider, monkeypatch)

    ((call, workspace, seen),) = llm.calls
    assert call == "review"
    assert seen == "VERSION = 2\n"
    assert workspace is not None
    assert workspace != repo["work"]
    assert not workspace.exists()
    assert llm.workspace is None
    assert llm.agentic_calls == 0
    assert _titles(result) == ["from workspace agent"]


@pytest.mark.asyncio
async def test_pr_review_binds_checkout_to_critic_and_security_llms(repo, monkeypatch):
    llm = _FakeLLM(workspace_agent=True)
    critic = _FakeLLM(workspace_agent=True)
    security = _FakeLLM(workspace_agent=True)
    provider = _provider(repo, CheckoutSource(repo["url"], repo["head"]))
    engine = _engine(llm, indexing_llm=critic, security_llm=security)
    engine.config.review.self_critique = True
    engine.config.review.security_pass = True

    result = await _review_pr(engine, provider, monkeypatch)

    (workspace,) = llm.bound("review")
    assert workspace is not None
    assert critic.bound("submit_critique") == [workspace]
    assert security.bound("submit_review") == [workspace]
    seen = {name: text for fake in (llm, critic, security) for name, _, text in fake.calls}
    assert seen == {
        "review": "VERSION = 2\n",
        "submit_critique": "VERSION = 2\n",
        "submit_review": "VERSION = 2\n",
    }
    assert [fake.workspace for fake in (llm, critic, security)] == [None, None, None]
    assert "from workspace agent" in _titles(result)


@pytest.mark.asyncio
async def test_pr_review_unbinds_llms_when_review_raises(repo, monkeypatch):
    llm = _FakeLLM(workspace_agent=True)
    critic = _FakeLLM(workspace_agent=True)
    provider = _provider(repo, CheckoutSource(repo["url"], repo["head"]))
    bound: list[Path | None] = []

    async def failing_diff(pr_info):
        bound.extend([llm.workspace, critic.workspace])
        raise RuntimeError("diff fetch failed")

    provider.get_pr_diff.side_effect = failing_diff

    with pytest.raises(RuntimeError, match="diff fetch failed"):
        await _review_pr(_engine(llm, indexing_llm=critic), provider, monkeypatch)

    assert bound[0] is not None
    assert bound == [bound[0], bound[0]]
    assert llm.workspace is None
    assert critic.workspace is None
    assert not bound[0].exists()


@pytest.mark.asyncio
async def test_pr_review_falls_back_when_checkout_fails(repo, monkeypatch):
    llm = _FakeLLM(workspace_agent=True)
    provider = _provider(repo, CheckoutSource(repo["url"], "0" * 40))

    result = await _review_pr(_engine(llm), provider, monkeypatch)

    assert llm.bound("review") == []
    assert llm.workspace is None
    assert result.comments
    assert "from workspace agent" not in _titles(result)


@pytest.mark.asyncio
async def test_pr_review_without_workspace_agent_skips_checkout(repo, monkeypatch):
    llm = _FakeLLM(workspace_agent=False)
    provider = _provider(repo, CheckoutSource(repo["url"], repo["head"]))

    result = await _review_pr(_engine(llm), provider, monkeypatch)

    provider.checkout_source.assert_not_awaited()
    assert {ws for _, ws, _ in llm.calls} <= {None}
    assert llm.agentic_calls >= 1
    assert result.comments
