"""Indexing a repo that reviews on GitHub Actions, end to end in one process.

The server side is the real dashboard trigger or push webhook, the real
dispatch and runner API on a real SQLite app DB and index store. The runner
is ``mira runner`` against that API through an ASGI transport, indexing with
a fake model and a fake copy of the repo. Every server-side way to make a
model raises, so a test passes only if the runner did the model work.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mira import config as mira_config
from mira import runner_client
from mira.dashboard.db import AppDatabase, User
from mira.dashboard.routers import repos
from mira.index.status import tracker
from mira.index.store import FileSummary, IndexStore
from mira.platforms.github import index_jobs
from mira.platforms.github.index_jobs import dispatch_index_job
from mira.platforms.github.runner_api import PARTS_KEPT_S, expire_stale_jobs
from mira.platforms.github.webhook import handle_push_index
from mira.providers.github import GitHubProvider
from mira.runner import IndexDiffTask, IndexRepoTask, JobStatus, ReviewJob
from mira.runner_client import RunnerLLM, run_job
from mira.runner_protocol import (
    IndexEnd,
    IndexPart,
    IndexReport,
    ManifestPackage,
    RemoveIndexedPaths,
    ReplaceManifestPackages,
    ResultBundle,
    index_parts,
)
from tests.test_runner_api import INSTALLATION, REF, auth, db, seed_job  # noqa: F401
from tests.test_runner_client import (  # noqa: F401  (fixtures)
    SERVER,
    app_auth,
    oidc_requests,
    revoked,
    server,
)
from tests.test_runner_dispatch import FakeGitHub

OWNER, REPO = "test", "repo"
CODE = "def handler(event):\n    return event\n" * 30  # over the trivial-file size


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class FakeRepo:
    """The repo's default branch, as the runner's fetcher reads it."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.tokens: list[str] = []

    def fetcher(self, token: Any, api_url: str = "") -> FakeRepo:
        self._token = token
        return self

    async def default_branch(self, owner: str, repo: str) -> str:
        return "main"

    async def branch_head(self, owner: str, repo: str, branch: str) -> str:
        return "sha-2"

    async def repo_tree(self, owner: str, repo: str, branch: str) -> list[str]:
        if not isinstance(self._token, str):
            self.tokens.append(await self._token())
        return sorted(self.files)

    async def file_content(self, owner: str, repo: str, path: str, ref: str, semaphore=None):
        return self.files.get(path)

    async def repo_tarball(self, owner, repo, ref, max_file_size=0, indexable_paths=None):
        return {p: c for p, c in self.files.items() if p in (indexable_paths or ())}


class IndexingModel:
    """Summarizes whatever files and directories it is asked about."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(model="test-model", provider="openai", indexing_temperature=0)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.asked: list[str] = []
        self.hang = False

    def preflight(self) -> None:
        pass

    async def complete(self, messages: list[dict[str, str]], **_: Any) -> str:
        system, user = messages[0]["content"], messages[1]["content"]
        self.usage["prompt_tokens"] += 10
        if self.hang:
            await asyncio.Event().wait()
        if user == "Summarize these directories.":
            dirs = [
                {"path": "(root)", "summary": "the root"},
                {"path": "src", "summary": "dir src"},
            ]
            return json.dumps({"directories": dirs})
        if user == "Summarize this directory.":
            return json.dumps({"summary": "src after the push"})
        paths = [p for p in ("src/app.py", "src/util.py", "same.py") if f"{p}" in system]
        self.asked += paths
        return json.dumps(
            {
                "files": [
                    {
                        "path": p,
                        "language": "python",
                        "summary": f"new summary of {p}",
                        "symbols": [
                            {
                                "name": "handler",
                                "kind": "function",
                                "signature": "",
                                "description": "",
                            }
                        ],
                        "imports": [],
                    }
                    for p in paths
                ]
            }
        )


def _server_model_is_never_built(*_: Any, **__: Any) -> Any:
    raise AssertionError("the server built a model")


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch, db: AppDatabase) -> IndexingModel:  # noqa: F811
    """The runner's model, plus the server side configured to index on
    Actions and unable to reach any model or open an index on the runner."""
    model = IndexingModel()
    monkeypatch.setattr(runner_client, "create_llm", lambda config: model)
    for module in ("mira.llm", "mira.index.indexer", "mira.platforms.index_handlers"):
        monkeypatch.setattr(f"{module}.create_llm", _server_model_is_never_built)
    # Indexing on the runner must go through the job's store, never one it opens.
    for module in ("mira.index.indexer", "mira.platforms.index_handlers"):
        monkeypatch.setattr(f"{module}.IndexStore", None)
    monkeypatch.setattr(tracker, "_jobs", {})
    monkeypatch.setattr(
        mira_config,
        "_global_defaults",
        {"review": {"runner_workflow_refs": [REF], "runner_server_url": "https://mira.test"}},
    )
    db.register_repo(OWNER, REPO, INSTALLATION)
    db.set_repo_runner(OWNER, REPO, "actions:ubuntu-latest")
    return model


@pytest.fixture
def github() -> Iterator[FakeGitHub]:
    fake = FakeGitHub()
    with (
        patch.object(httpx.AsyncClient, "post", fake.post),
        patch.object(httpx.AsyncClient, "patch", fake.patch),
    ):
        yield fake


@pytest.fixture
def parts(db: AppDatabase, monkeypatch: pytest.MonkeyPatch) -> list[int]:  # noqa: F811
    """The part numbers the server stored, in order."""
    stored: list[int] = []
    put = db.put_job_part

    def spy(job_id: str, seq: int, body: str) -> None:
        stored.append(seq)
        put(job_id, seq, body)

    monkeypatch.setattr(db, "put_job_part", spy)
    return stored


def _seed_index(files: dict[str, tuple[str, str]]) -> None:
    """Index ``files`` (path -> (content, summary)) on the server."""
    store = IndexStore.open(OWNER, REPO)
    try:
        for path, (content, summary) in files.items():
            store.upsert_summary(
                FileSummary(
                    path=path, language="python", summary=summary, content_hash=_hash(content)
                )
            )
    finally:
        store.close()


def _summaries() -> dict[str, str]:
    store = IndexStore.open(OWNER, REPO)
    try:
        return {p: store.get_summary(p).summary for p in store.all_paths()}  # type: ignore[union-attr]
    finally:
        store.close()


def _pending() -> bool:
    """Whether the next index run must reconcile the server's index."""
    store = IndexStore.open(OWNER, REPO)
    try:
        return store.claim_index_run().needs_reconciliation
    finally:
        store.close()


def _only_job(db: AppDatabase) -> ReviewJob:  # noqa: F811
    (job,) = db.list_active_index_jobs()
    return job


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


_ADMIN = SimpleNamespace(state=SimpleNamespace(user=User(id=1, username="a", is_admin=True)))


async def test_a_full_index_from_the_dashboard_runs_on_the_runner(
    server: FastAPI,  # noqa: F811
    db: AppDatabase,  # noqa: F811
    app_auth,  # noqa: F811
    github: FakeGitHub,
    runner: IndexingModel,
    parts: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same = "x = 1\n" * 200
    _seed_index({"same.py": (same, "kept summary"), "old.py": (CODE, "a deleted file")})
    repo = FakeRepo(
        {
            "src/app.py": CODE,
            "src/util.py": CODE + "# util\n",
            "same.py": same,
            "tiny.py": "X = 1\n",
            "package.json": json.dumps({"dependencies": {"left-pad": "1.3.0"}}),
            "AGENTS.md": "# Team rules\n\n- Never log a token.\n",
        }
    )
    monkeypatch.setattr(runner_client, "GitHubRepoFetcher", repo.fetcher)
    monkeypatch.setattr(index_jobs, "GitHubRepoFetcher", repo.fetcher)
    monkeypatch.setattr("mira.dashboard.api._build_app_auth", lambda: app_auth)
    monkeypatch.setattr(runner_client, "_PART_BYTES", 1500)

    assert await repos.trigger_index(OWNER, REPO, _ADMIN, full=False) == {  # type: ignore[arg-type]
        "status": "indexing",
        "full": False,
    }
    job = _only_job(db)
    assert job.task == IndexRepoTask()
    record = db.get_repo(OWNER, REPO)
    assert record is not None and record.status == "indexing"
    assert [j.status for j in tracker.get_all()] == ["indexing"]
    [dispatch] = github.calls("POST", "/dispatches")
    assert dispatch is not None and dispatch["inputs"]["job_id"] == job.id

    assert await _run(server, job.id) == 0

    done = db.get_review_job(job.id)
    assert done is not None and done.status is JobStatus.SUCCEEDED
    # The unchanged file kept its summary and was never sent to the model.
    assert _summaries() == {
        "src/app.py": "new summary of src/app.py",
        "src/util.py": "new summary of src/util.py",
        "same.py": "kept summary",
        "tiny.py": "",
        "package.json": "",
    }
    assert sorted(runner.asked) == ["src/app.py", "src/util.py"]
    store = IndexStore.open(OWNER, REPO)
    try:
        app = store.get_summary("src/app.py")
        assert app is not None and [s.name for s in app.symbols] == ["handler"]
        directory = store.get_directory_summary("src")
        assert directory is not None and directory.summary == "dir src"
        assert [p.name for p in store.list_manifest_packages()] == ["left-pad"]
    finally:
        store.close()
    record = db.get_repo(OWNER, REPO)
    assert record is not None
    assert (record.status, record.files_indexed) == ("ready", 5)
    assert record.last_indexed_at > 0 and "Never log a token" in record.conventions
    [tracked] = tracker.get_all()
    assert (tracked.status, tracked.files_done) == ("completed", 4)
    assert not _pending()
    assert parts == list(range(len(parts))) and len(parts) > 1
    assert db.list_job_part_seqs(job.id) == []
    assert [permissions for _, _, permissions in app_auth.minted] == [{"contents": "read"}]
    assert repo.tokens == ["ghs_job_token_1"]


async def test_a_push_is_indexed_incrementally_on_the_runner(
    server: FastAPI,  # noqa: F811
    db: AppDatabase,  # noqa: F811
    app_auth,  # noqa: F811
    github: FakeGitHub,
    runner: IndexingModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_index(
        {
            "src/app.py": (CODE, "old summary of src/app.py"),
            "src/util.py": (CODE, "old summary of src/util.py"),
            "docs/gone.py": (CODE, "a file the push removes"),
        }
    )
    db.set_repo_status(OWNER, REPO, "ready", files_indexed=3)
    repo = FakeRepo({"src/app.py": CODE + "# changed\n", "src/util.py": CODE})
    monkeypatch.setattr(runner_client, "GitHubRepoFetcher", repo.fetcher)
    payload = {
        "ref": "refs/heads/main",
        "installation": {"id": INSTALLATION},
        "repository": {"owner": {"login": OWNER}, "name": REPO, "default_branch": "main"},
        "commits": [{"modified": ["src/app.py"], "removed": ["docs/gone.py"]}],
    }

    await handle_push_index(payload, app_auth, "mira-bot")

    job = _only_job(db)
    assert job.task == IndexDiffTask(
        branch="main", changed_paths=("src/app.py",), removed_paths=("docs/gone.py",)
    )
    assert tracker.get_all() == []

    assert await _run(server, job.id) == 0

    done = db.get_review_job(job.id)
    assert done is not None and done.status is JobStatus.SUCCEEDED
    assert _summaries() == {
        "src/app.py": "new summary of src/app.py",
        "src/util.py": "old summary of src/util.py",
    }
    assert runner.asked == ["src/app.py"]
    store = IndexStore.open(OWNER, REPO)
    try:
        directory = store.get_directory_summary("src")
        assert directory is not None and directory.summary == "src after the push"
    finally:
        store.close()
    record = db.get_repo(OWNER, REPO)
    assert record is not None
    assert (record.status, record.files_indexed) == ("ready", 2)
    assert record.last_indexed_at > 0
    assert not _pending()


async def test_a_failed_rebuild_leaves_the_server_index_untouched(
    server: FastAPI,  # noqa: F811
    db: AppDatabase,  # noqa: F811
    github: FakeGitHub,
    runner: IndexingModel,
    parts: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {"src/app.py": (CODE, "summary before"), "src/util.py": (CODE, "util before")}
    _seed_index(before)
    repo = FakeRepo({"src/app.py": CODE + "# v2\n", "src/util.py": CODE, "tiny.py": "X = 1\n"})
    monkeypatch.setattr(runner_client, "GitHubRepoFetcher", repo.fetcher)
    runner.hang = True
    provider = GitHubProvider("ghs_app_token")
    await dispatch_index_job(
        provider,
        db,
        OWNER,
        REPO,
        IndexRepoTask(full=True, reset=True),
        default_branch="main",
        installation_id=INSTALLATION,
        runs_on=("ubuntu-latest",),
        server_url=SERVER,
    )
    job = _only_job(db)

    # The runner drops the whole index, then never gets its summaries.
    assert await _run(server, job.id, timeout=timedelta(seconds=0.5)) == 1

    done = db.get_review_job(job.id)
    assert done is not None and done.status is JobStatus.FAILED
    assert _summaries() == {p: summary for p, (_, summary) in before.items()}
    assert parts == []
    record = db.get_repo(OWNER, REPO)
    assert record is not None and record.status == "ready" and "timed out" in record.error
    [tracked] = tracker.get_all()
    assert tracked.status == "failed"
    assert _pending()


async def test_a_newer_index_job_supersedes_and_covers_an_older_one(
    server: FastAPI,  # noqa: F811
    db: AppDatabase,  # noqa: F811
    github: FakeGitHub,
    runner: IndexingModel,
) -> None:
    async def push(*changed: str) -> None:
        task = IndexDiffTask(branch="main", changed_paths=changed, removed_paths=())
        await dispatch_index_job(
            GitHubProvider("ghs_app_token"),
            db,
            OWNER,
            REPO,
            task,
            default_branch="main",
            installation_id=INSTALLATION,
            runs_on=("ubuntu-latest",),
            server_url=SERVER,
        )

    await push("a.py")
    first = _only_job(db)
    await push("b.py")
    second = _only_job(db)

    old = db.get_review_job(first.id)
    assert old is not None and old.status is JobStatus.SUPERSEDED
    assert len(github.calls("POST", f"/actions/runs/{first.run_id}/cancel")) == 1
    # The first push's paths are lost with its job, so the second reconciles.
    assert second.task == IndexRepoTask(reconcile=True)
    # The superseded run finds nothing to claim and indexes nothing.
    assert await _run(server, first.id) == 0
    assert runner.asked == []


def _stage(
    client: TestClient, part: IndexPart, seq: int = 0, job_id: str = "job-1"
) -> httpx.Response:
    return client.post(
        f"/runner/jobs/{job_id}/index/{seq}",
        content=part.model_dump_json(),
        headers={**auth(), "content-type": "application/json"},
    )


def _result(client: TestClient, report: IndexReport | None) -> httpx.Response:
    bundle = ResultBundle(job_id="job-1", outcome="succeeded", index=report)
    return client.post(
        "/runner/jobs/job-1/result",
        content=bundle.model_dump_json(),
        headers={**auth(), "content-type": "application/json"},
    )


@pytest.fixture
def client(server: FastAPI) -> TestClient:  # noqa: F811
    return TestClient(server)


def _claimed_index_job(db: AppDatabase, client: TestClient) -> None:  # noqa: F811
    store = IndexStore.open(OWNER, REPO)
    try:
        store.claim_index_run("job-1")
    finally:
        store.close()
    seed_job(db, pr_number=0, head_sha="", check_run_id=None, task=IndexRepoTask())
    assert client.post("/runner/jobs/job-1/claim", headers=auth()).status_code == 200


_REMOVE_ALL = IndexPart(writes=[RemoveIndexedPaths(paths=["src/app.py"])])


def test_a_result_after_a_newer_index_run_applies_nothing(
    db: AppDatabase,  # noqa: F811
    client: TestClient,
    runner: IndexingModel,
) -> None:
    _seed_index({"src/app.py": (CODE, "kept")})
    _claimed_index_job(db, client)
    assert _stage(client, _REMOVE_ALL).status_code == 200
    # A local index run or a newer job claims the index meanwhile.
    store = IndexStore.open(OWNER, REPO)
    try:
        store.claim_index_run()
    finally:
        store.close()

    resp = _result(client, IndexReport(end=IndexEnd.COMPLETE, parts=1))

    assert resp.json() == {"status": "succeeded", "applied": False}
    assert _summaries() == {"src/app.py": "kept"}
    assert db.list_job_part_seqs("job-1") == []


def test_a_result_must_name_exactly_the_staged_parts(
    db: AppDatabase,  # noqa: F811
    client: TestClient,
    runner: IndexingModel,
) -> None:
    _seed_index({"src/app.py": (CODE, "kept")})
    _claimed_index_job(db, client)
    assert _stage(client, _REMOVE_ALL).status_code == 200

    assert _result(client, IndexReport(end=IndexEnd.COMPLETE, parts=2)).status_code == 422
    assert _summaries() == {"src/app.py": "kept"}
    assert db.get_review_job("job-1").status is JobStatus.CLAIMED  # type: ignore[union-attr]


def test_a_part_is_refused_unless_its_job_is_a_claimed_index_job(
    db: AppDatabase,  # noqa: F811
    client: TestClient,
    runner: IndexingModel,
) -> None:
    seed_job(db)  # a review
    seed_job(db, job_id="job-2", pr_number=0, head_sha="", check_run_id=None, task=IndexRepoTask())

    assert _stage(client, _REMOVE_ALL).status_code == 404
    assert _stage(client, _REMOVE_ALL, job_id="job-2").status_code == 409
    assert db.list_job_part_seqs("job-2") == []


def test_a_manifest_part_cannot_list_another_files_packages(
    db: AppDatabase,  # noqa: F811
    client: TestClient,
    runner: IndexingModel,
) -> None:
    _claimed_index_job(db, client)
    package = ManifestPackage(
        name="x", kind="npm", version="1", file_path="other/package.json", is_dev=False
    )
    part = IndexPart(writes=[ReplaceManifestPackages(path="package.json", packages=[package])])

    assert _stage(client, part).status_code == 422
    assert _stage(client, _REMOVE_ALL, seq=256).status_code == 422
    assert db.list_job_part_seqs("job-1") == []


async def test_the_sweep_keeps_an_ended_jobs_parts_for_a_while(
    db: AppDatabase,  # noqa: F811
    client: TestClient,
    runner: IndexingModel,
    app_auth: Any,  # noqa: F811
) -> None:
    _claimed_index_job(db, client)
    assert _stage(client, _REMOVE_ALL).status_code == 200
    # Another replica took the result and is applying the parts.
    assert db.transition_review_job("job-1", JobStatus.CLAIMED, JobStatus.SUCCEEDED)
    ended = db.get_review_job("job-1").finished_at  # type: ignore[union-attr]

    await expire_stale_jobs(db, app_auth, now=ended + 60, claim_timeout_minutes=30)
    assert db.list_job_part_seqs("job-1") == [0]
    await expire_stale_jobs(db, app_auth, now=ended + PARTS_KEPT_S + 1, claim_timeout_minutes=30)
    assert db.list_job_part_seqs("job-1") == []


def test_index_parts_split_in_order_and_refuse_an_oversized_write() -> None:
    writes = [RemoveIndexedPaths(paths=[f"f{i}.py"]) for i in range(20)]

    bodies = index_parts(writes, max_bytes=200)

    assert len(bodies) > 1 and all(len(body) <= 200 for body in bodies)
    assert [w for b in bodies for w in IndexPart.model_validate_json(b).writes] == writes
    with pytest.raises(ValueError, match="does not fit"):
        index_parts([RemoveIndexedPaths(paths=["x" * 300])], max_bytes=200)
