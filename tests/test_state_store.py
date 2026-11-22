"""LocalStateStore keeps writing the rows the engine wrote directly before."""

from __future__ import annotations

import pytest

from mira.core.state_store import LocalStateStore, ReviewRecord
from mira.dashboard.db import AppDatabase, PRReviewProgress, ReviewedVersion
from mira.index.store import IndexStore
from mira.models import PRFingerprint


@pytest.fixture
def local(tmp_path, monkeypatch) -> LocalStateStore:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    monkeypatch.setattr("mira.dashboard.api._app_db", AppDatabase(url="", admin_password="pw"))
    return LocalStateStore()


def _review(**overrides) -> ReviewRecord:
    fields: dict = {
        "pr_number": 7,
        "pr_title": "Add billing report",
        "pr_url": "https://github.com/o/r/pull/7",
        "author": "dev",
        "author_avatar_url": "https://avatars/dev",
        "comments_posted": 1,
        "blockers": 1,
        "warnings": 0,
        "suggestions": 0,
        "files_reviewed": 1,
        "lines_changed": 12,
        "tokens_used": 900,
        "duration_ms": 4000,
        "categories": "bug",
        "reviewed_paths": ["billing/report.py"],
        "comments": [
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
    }
    fields.update(overrides)
    return ReviewRecord(**fields)


def test_record_review_writes_event_and_comment_rows(local: LocalStateStore) -> None:
    local.record_review("o", "r", _review(), platform="github")

    store = IndexStore.open("o", "r")
    try:
        [event] = store.list_review_events_for_pr(7)
        [comment] = store.list_review_comments(7)
    finally:
        store.close()
    assert (event.blockers, event.tokens_used, event.author) == (1, 900, "dev")
    assert event.reviewed_paths == '["billing/report.py"]'
    assert comment.review_id == event.id
    assert (comment.path, comment.line, comment.github_comment_id) == ("billing/report.py", 3, 55)


def test_review_state_round_trips_through_app_db(local: LocalStateStore) -> None:
    assert local.get_last_reviewed("o", "r", 7, platform="github") == ReviewedVersion()
    version = ReviewedVersion(head_sha="abc123", base_sha="base9")
    local.set_last_reviewed("o", "r", 7, version, platform="github")
    assert local.get_last_reviewed("o", "r", 7, platform="github") == version
    assert local.get_last_reviewed("o", "r", 7, platform="gitlab") == ReviewedVersion()

    progress = PRReviewProgress(
        owner="o",
        repo="r",
        pr_number=7,
        total_paths=["a.py", "b.py"],
        reviewed_paths=["a.py"],
        skipped_paths=["b.py"],
        chunk_index=1,
    )
    local.upsert_pr_review_progress(progress, platform="github")
    stored = local.get_pr_review_progress("o", "r", 7, platform="github")
    assert stored is not None
    assert (stored.skipped_paths, stored.chunk_index) == (["b.py"], 1)


def test_fingerprints_round_trip_per_repo(local: LocalStateStore) -> None:
    fp = PRFingerprint(pr_number=3, head_sha="s", title="t", body="", paths=["a.py"])
    local.upsert_pr_fingerprint("o", "r", fp, platform="github")

    [stored] = local.list_pr_fingerprints("o", "r", platform="github")
    assert (stored.pr_number, stored.paths) == (3, ["a.py"])
    assert local.list_pr_fingerprints("o", "other", platform="github") == []


def test_global_rules_are_empty_without_an_app_db(local, monkeypatch) -> None:
    monkeypatch.setattr("mira.dashboard.api._app_db", None)
    assert local.get_global_rules() == []
