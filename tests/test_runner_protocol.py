"""RemoteStateStore answers like LocalStateStore, and its buffered writes,
applied on the server, leave the rows LocalStateStore writes directly. A
learning job's rules land as the server's own synthesis would write them."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from mira.config import MiraConfig
from mira.core.state_store import FeedbackRecord, LocalStateStore, ReviewRecord
from mira.dashboard.db import AppDatabase, PRReviewProgress, RepoRecord, ReviewedVersion
from mira.index.store import IndexStore, ReviewContext
from mira.models import PRFingerprint
from mira.runner import LearnTask, ReviewJob, ThreadReply, ThreadReplyTask
from mira.runner_protocol import (
    LearningSnapshot,
    PRKey,
    RecordFeedback,
    RecordReview,
    RemoteLearningStore,
    RemoteStateStore,
    RepoKey,
    ResultBundle,
    SetLastReviewedSha,
    StateSnapshot,
    UpsertLearnedRule,
    UsageReport,
    apply_writes,
    build_learning_snapshot,
    build_snapshot,
    runner_config,
)

PR = PRKey(platform="github", owner="o", repo="r", pr_number=7)
# A review job on PR, whose writes the tests apply.
JOB = ReviewJob(
    id="j",
    owner="o",
    repo="r",
    pr_number=7,
    head_sha="sha-1",
    installation_id=1,
    runs_on=("self-hosted", "sora"),
    run_url="https://github.com/o/r/actions/runs/1",
)

REPLY_JOB = replace(
    JOB,
    task=ThreadReplyTask(reply=ThreadReply(comment_id=1, body="no"), may_resolve=True),
)


@pytest.mark.parametrize(
    ("job", "write"),
    [
        # A question or thread-reply job must not rewrite the PR's review state.
        (
            replace(
                JOB,
                task=ThreadReplyTask(reply=ThreadReply(comment_id=1, body="x"), may_resolve=False),
            ),
            "sha",
        ),
        (REPLY_JOB, "review"),
        # A review job does not record thread feedback.
        (JOB, "feedback"),
    ],
)
def test_each_job_kind_may_send_only_its_own_writes(
    local: LocalStateStore, job: ReviewJob, write: str
) -> None:
    writes = {
        "sha": SetLastReviewedSha(sha="sha-2"),
        "review": RecordReview(review=_review()),
        "feedback": RecordFeedback(feedback=_feedback()),
    }
    with pytest.raises(ValueError, match="cannot send"):
        apply_writes(local, job, [writes[write]])


def _review(pr_number: int = 7) -> ReviewRecord:
    return ReviewRecord(
        pr_number=pr_number,
        pr_title="Add billing report",
        pr_url=f"https://github.com/o/r/pull/{pr_number}",
        author="dev",
        author_avatar_url="https://avatars/dev",
        comments_posted=1,
        blockers=1,
        warnings=0,
        suggestions=0,
        files_reviewed=1,
        lines_changed=12,
        tokens_used=900,
        duration_ms=4000,
        categories="bug",
        reviewed_paths=["billing/report.py"],
        comments=[
            {
                "path": "billing/report.py",
                "line": 3,
                "severity": "blocker",
                "category": "bug",
                "title": "Cents shown as dollars",
                "body": "Divide by 100.",
                "github_comment_id": 55,
            }
        ],
    )


def _snapshot(**overrides) -> StateSnapshot:
    fields: dict = {
        "pr": PR,
        "repo_record": RepoRecord(owner="o", repo="r", conventions="Use cents.", private=True),
        "global_rules": ["Style: no prints"],
        "learned_rules": ["Prefer Decimal"],
        "review_context": [ReviewContext(id=1, title="Money", content="Amounts are cents.")],
        "manifest_package_names": ["httpx", "pydantic"],
        "last_reviewed_sha": "sha-0",
        "review_progress": None,
        "fingerprints": [PRFingerprint(pr_number=3, head_sha="s3", title="t", body="")],
    }
    fields.update(overrides)
    return StateSnapshot(**fields)


def _progress(chunk_index: int) -> PRReviewProgress:
    return PRReviewProgress(
        owner="o",
        repo="r",
        pr_number=7,
        total_paths=["a.py", "b.py"],
        reviewed_paths=["a.py"],
        skipped_paths=["b.py"],
        chunk_index=chunk_index,
    )


def _review_round(store, sha: str) -> None:
    """The store calls of one review round, in the order PR1's recording store
    pinned for review_pr, plus an overlap fingerprint."""
    store.get_last_reviewed("o", "r", 7, platform="github")
    store.get_repo("o", "r")
    store.get_learned_rules("o", "r", platform="github")
    store.get_global_rules()
    store.upsert_pr_fingerprint(
        "o", "r", PRFingerprint(pr_number=7, head_sha=sha, title="t", body=""), platform="github"
    )
    store.list_pr_fingerprints("o", "r", platform="github")
    store.get_repo("o", "r")
    store.record_review("o", "r", _review(), platform="github")
    prior = store.get_pr_review_progress("o", "r", 7, platform="github")
    store.upsert_pr_review_progress(
        _progress((prior.chunk_index + 1) if prior else 1), platform="github"
    )
    store.set_last_reviewed("o", "r", 7, ReviewedVersion(sha, f"base-{sha}"), platform="github")


def test_reads_come_from_the_snapshot_for_its_pr_only() -> None:
    snap = _snapshot()
    store = RemoteStateStore(snap)

    assert store.get_repo("o", "r") == snap.repo_record
    assert store.get_global_rules() == ["Style: no prints"]
    assert store.get_learned_rules("o", "r", platform="github") == ["Prefer Decimal"]
    assert store.list_review_context("o", "r", platform="github") == snap.review_context
    assert store.list_manifest_package_names("o", "r", platform="github") == ["httpx", "pydantic"]
    assert store.get_last_reviewed("o", "r", 7, platform="github") == ReviewedVersion("sha-0")
    assert [fp.pr_number for fp in store.list_pr_fingerprints("o", "r", platform="github")] == [3]
    # Anything outside the snapshot reads as unknown, like an empty database.
    assert store.get_repo("o", "dependent") is None
    assert store.get_last_reviewed("o", "r", 8, platform="github") == ReviewedVersion()
    assert store.get_last_reviewed("o", "r", 7, platform="gitlab") == ReviewedVersion()
    assert store.get_learned_rules("o", "other", platform="github") == []
    assert store.list_pr_fingerprints("o", "other", platform="github") == []
    assert store.list_review_context("o", "other", platform="github") == []
    assert store.list_manifest_package_names("o", "other", platform="github") == []


def test_reads_after_writes_see_the_written_values() -> None:
    store = RemoteStateStore(_snapshot())
    _review_round(store, "sha-1")
    _review_round(store, "sha-2")

    assert store.get_last_reviewed("o", "r", 7, platform="github") == ReviewedVersion(
        "sha-2", "base-sha-2"
    )
    progress = store.get_pr_review_progress("o", "r", 7, platform="github")
    assert progress is not None and progress.chunk_index == 2
    fps = {
        fp.pr_number: fp.head_sha for fp in store.list_pr_fingerprints("o", "r", platform="github")
    }
    assert fps == {3: "s3", 7: "sha-2"}
    assert [w.op for w in store.writes] == [
        "upsert_pr_fingerprint",
        "record_review",
        "upsert_pr_review_progress",
        "set_last_reviewed_sha",
    ] * 2


def test_a_write_for_another_pr_is_refused() -> None:
    store = RemoteStateStore(_snapshot())
    with pytest.raises(ValueError):
        store.set_last_reviewed("o", "r", 8, ReviewedVersion("x"), platform="github")
    with pytest.raises(ValueError):
        store.record_review("o", "other", _review(), platform="github")
    with pytest.raises(ValueError):
        store.upsert_pr_fingerprint(
            "o",
            "other",
            PRFingerprint(pr_number=1, head_sha="", title="", body=""),
            platform="github",
        )
    assert store.writes == []


def test_bundle_round_trips_through_json() -> None:
    store = RemoteStateStore(_snapshot())
    _review_round(store, "sha-1")
    usage = UsageReport(harness="claude-code", model="opus", input_tokens=10, cost_usd=0.5)
    bundle = store.bundle("job-1", "succeeded", usage=usage)

    parsed = ResultBundle.model_validate_json(bundle.model_dump_json())
    assert parsed == bundle
    assert isinstance(parsed.writes[1], RecordReview)
    assert parsed.writes[1].review == _review()


@pytest.fixture
def local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalStateStore:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    monkeypatch.setattr("mira.dashboard.api._app_db", AppDatabase(url="", admin_password="pw"))
    return LocalStateStore()


_NOT_STATE = {"id", "created_at", "runner", "run_url"}


def state_rows(pr: PRKey = PR) -> dict:
    """The review state LocalStateStore persists for ``pr``, minus ids, times
    and where each review ran."""
    from mira.dashboard import api

    store = IndexStore.open(pr.owner, pr.repo)
    try:
        events = [
            {k: v for k, v in asdict(e).items() if k not in _NOT_STATE}
            for e in store.list_review_events_for_pr(pr.pr_number)
        ]
        comments = [
            {k: v for k, v in asdict(c).items() if k not in {"id", "review_id", "created_at"}}
            for c in store.list_review_comments(pr.pr_number)
        ]
        fingerprints = sorted(
            (fp.pr_number, fp.head_sha, tuple(fp.paths), tuple(fp.symbols))
            for fp in store.list_pr_fingerprints()
        )
    finally:
        store.close()
    db = api._app_db
    progress = db.get_pr_review_progress(pr.owner, pr.repo, pr.pr_number, platform=pr.platform)
    last = db.get_last_reviewed(pr.owner, pr.repo, pr.pr_number, platform=pr.platform)
    return {
        "sha": last.head_sha,
        "base_sha": last.base_sha,
        "progress": progress and {k: v for k, v in asdict(progress).items() if k != "updated_at"},
        "events": events,
        "comments": comments,
        "fingerprints": fingerprints,
    }


def test_applied_writes_equal_direct_local_writes(
    local: LocalStateStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _review_round(local, "sha-1")
    _review_round(local, "sha-2")
    direct = state_rows()

    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path / "applied"))
    monkeypatch.setattr("mira.dashboard.api._app_db", AppDatabase(url="", admin_password="pw"))
    for sha in ("sha-1", "sha-2"):
        remote = RemoteStateStore(build_snapshot(local, PR))
        _review_round(remote, sha)
        bundle = ResultBundle.model_validate_json(remote.bundle("j", "succeeded").model_dump_json())
        apply_writes(local, JOB, bundle.writes)

    assert state_rows() == direct
    assert direct["sha"] == "sha-2" and len(direct["events"]) == 2
    assert direct["base_sha"] == "base-sha-2"


def test_snapshot_carries_the_dashboards_review_context_and_manifest_packages(
    local: LocalStateStore,
) -> None:
    def pkg(name: str, path: str) -> dict:
        return {"name": name, "kind": "pypi", "version": "", "file_path": path}

    store = IndexStore.open("o", "r")
    try:
        store.upsert_review_context("Money", "Amounts are cents.")
        store.upsert_review_context("Errors", "Never swallow exceptions.")
        store.replace_manifest_packages(
            "pyproject.toml", [pkg("pydantic", "pyproject.toml"), pkg("httpx", "pyproject.toml")]
        )
        store.replace_manifest_packages(
            "tools/pyproject.toml", [pkg("httpx", "tools/pyproject.toml")]
        )
    finally:
        store.close()

    wire = build_snapshot(local, PR).model_dump_json()
    remote = RemoteStateStore(StateSnapshot.model_validate_json(wire))

    rules = remote.list_review_context("o", "r", platform="github")
    assert sorted((c.title, c.content) for c in rules) == [
        ("Errors", "Never swallow exceptions."),
        ("Money", "Amounts are cents."),
    ]
    assert remote.list_manifest_package_names("o", "r", platform="github") == ["httpx", "pydantic"]


def test_apply_refuses_a_review_of_another_pr_before_writing(local: LocalStateStore) -> None:
    from mira.runner_protocol import SetLastReviewedSha

    with pytest.raises(ValueError):
        apply_writes(local, JOB, [SetLastReviewedSha(sha="x"), RecordReview(review=_review(8))])
    assert local.get_last_reviewed("o", "r", 7, platform="github") == ReviewedVersion()


def _feedback(pr_number: int = 7) -> FeedbackRecord:
    return FeedbackRecord(
        pr_number=pr_number,
        pr_url=f"https://github.com/o/r/pull/{pr_number}",
        comment_path="billing.py",
        comment_line=4,
        signal="rejected",
        actor="dev",
    )


def _feedback_rows() -> list[dict]:
    store = IndexStore.open("o", "r")
    try:
        return [
            {k: v for k, v in asdict(e).items() if k not in {"id", "created_at"}}
            for e in store.list_feedback()
        ]
    finally:
        store.close()


def test_applied_feedback_equals_direct_local_feedback(
    local: LocalStateStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local.record_feedback("o", "r", _feedback(), platform="github")
    direct = _feedback_rows()

    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path / "applied"))
    remote = RemoteStateStore(_snapshot())
    remote.record_feedback("o", "r", _feedback(), platform="github")
    bundle = ResultBundle.model_validate_json(remote.bundle("j", "succeeded").model_dump_json())
    apply_writes(local, REPLY_JOB, bundle.writes)

    assert _feedback_rows() == direct
    assert [(r["signal"], r["comment_path"], r["actor"]) for r in direct] == [
        ("rejected", "billing.py", "dev")
    ]


def test_feedback_for_another_pr_is_refused(local: LocalStateStore) -> None:
    with pytest.raises(ValueError):
        RemoteStateStore(_snapshot()).record_feedback("o", "r", _feedback(8), platform="github")
    with pytest.raises(ValueError):
        apply_writes(local, REPLY_JOB, [RecordFeedback(feedback=_feedback(8))])
    assert _feedback_rows() == []


def test_runner_config_drops_the_servers_own_settings() -> None:
    config = MiraConfig.model_validate(
        {
            "llm": {"provider": "codex", "model": "m", "codex_home": "/srv/codex"},
            "review": {"runner": "actions", "runs_on": ["self-hosted"], "max_chunks_per_review": 2},
            "commands": {"allowed_users": ["root"]},
            "database": {"url": "postgresql://u:secret@db/mira", "admin_password": "pw"},
        }
    )
    data = runner_config(config)

    assert "database" not in data and "commands" not in data
    assert "provider" not in data["llm"] and "codex_home" not in data["llm"]
    assert "runner" not in data["review"] and "runs_on" not in data["review"]
    assert (data["llm"]["model"], data["review"]["max_chunks_per_review"]) == ("m", 2)
    assert MiraConfig.model_validate(data).review.max_chunks_per_review == 2


def _ran_on() -> list[tuple[str, str]]:
    store = IndexStore.open("o", "r")
    try:
        return [(e.runner, e.run_url) for e in store.list_review_events_for_pr(7)]
    finally:
        store.close()


def test_an_applied_review_records_the_jobs_runner_not_the_runners_claim(
    local: LocalStateStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local.record_review("o", "r", _review(), platform="github")
    assert _ran_on() == [("local", "")]

    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path / "applied"))
    claimed = ReviewRecord(**{**asdict(_review()), "runner": "local", "run_url": "https://evil"})
    apply_writes(local, JOB, [RecordReview(review=claimed)])

    assert _ran_on() == [("actions:self-hosted,sora", "https://github.com/o/r/actions/runs/1")]


def _learning_job() -> ReviewJob:
    return ReviewJob(
        id="l",
        owner="o",
        repo="r",
        pr_number=7,
        head_sha="sha-1",
        installation_id=1,
        runs_on=("ubuntu-latest",),
        task=LearnTask(),
    )


def _rule() -> UpsertLearnedRule:
    return UpsertLearnedRule(
        rule_text="Avoid raw SQL.", path_pattern="__llm_pattern_0__", sample_count=3
    )


def test_only_a_learning_job_learns_rules_and_it_writes_nothing_else(
    local: LocalStateStore,
) -> None:
    with pytest.raises(ValueError, match="review job cannot send upsert_learned_rule"):
        apply_writes(local, JOB, [_rule()])
    with pytest.raises(ValueError, match="learn job cannot send record_feedback"):
        apply_writes(local, _learning_job(), [RecordFeedback(feedback=_feedback())])
    store = IndexStore.open("o", "r")
    try:
        assert store.list_learned_rules() == []
    finally:
        store.close()


def test_learned_rules_round_trip_into_pending_human_pattern_rules(
    local: LocalStateStore,
) -> None:
    store = IndexStore.open("o", "r")
    try:
        for i, signal in enumerate(["human_review", "accepted", "human_review"]):
            store.record_feedback(
                pr_number=i,
                pr_url="",
                comment_path="db.py",
                comment_line=i,
                comment_category="human_review",
                comment_severity="",
                comment_title=f"comment {i}",
                signal=signal,
                actor="alice",
            )
        wire = build_learning_snapshot(store, RepoKey(platform="github", owner="o", repo="r"))
    finally:
        store.close()

    remote = RemoteLearningStore(LearningSnapshot.model_validate_json(wire.model_dump_json()))
    assert [e.comment_title for e in remote.list_feedback()] == ["comment 2", "comment 0"]
    with pytest.raises(ValueError):
        remote.upsert_learned_rule("x", "rejected", "bug", "__llm_pattern_0__", 1)
    remote.upsert_learned_rule(
        "Avoid raw SQL.", "human_pattern", "human_review", "__llm_pattern_0__", 3
    )
    bundle = ResultBundle.model_validate_json(remote.bundle("l", "succeeded").model_dump_json())
    assert bundle.writes == [_rule()]
    apply_writes(local, _learning_job(), bundle.writes)

    store = IndexStore.open("o", "r")
    try:
        [rule] = store.list_learned_rules()
    finally:
        store.close()
    assert (rule.rule_text, rule.source_signal, rule.category, rule.status, rule.sample_count) == (
        "Avoid raw SQL.",
        "human_pattern",
        "human_review",
        "pending",
        3,
    )


def test_a_learned_rule_cannot_name_another_rule() -> None:
    with pytest.raises(ValueError):
        UpsertLearnedRule(rule_text="x", path_pattern="src/*", sample_count=1)
