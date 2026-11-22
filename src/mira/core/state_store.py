"""Persistent review state the engine reads and writes outside the Git host.

``ReviewEngine.review_pr`` goes through a ``ReviewStateStore`` instead of the
dashboard database and the per-repo index store, so a review can run where
neither exists. Every argument and return value is plain data.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from mira.dashboard.db import AppDatabase, PRReviewProgress, RepoRecord
from mira.index.store import IndexStore
from mira.models import PRFingerprint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReviewRecord:
    """One finished review pass and the comments it posted."""

    pr_number: int
    pr_title: str
    pr_url: str
    author: str
    author_avatar_url: str
    comments_posted: int
    blockers: int
    warnings: int
    suggestions: int
    files_reviewed: int
    lines_changed: int
    tokens_used: int
    duration_ms: int
    categories: str
    reviewed_paths: list[str] = field(default_factory=list)
    # path, line, severity, category, title, body, github_comment_id
    comments: list[dict[str, Any]] = field(default_factory=list)


class ReviewStateStore(Protocol):
    def get_repo(self, owner: str, repo: str) -> RepoRecord | None: ...

    def get_global_rules(self) -> list[str]: ...

    def get_learned_rules(self, owner: str, repo: str, *, platform: str) -> list[str]: ...

    def get_last_reviewed_sha(
        self, owner: str, repo: str, pr_number: int, *, platform: str
    ) -> str: ...

    def set_last_reviewed_sha(
        self, owner: str, repo: str, pr_number: int, sha: str, *, platform: str
    ) -> None: ...

    def get_pr_review_progress(
        self, owner: str, repo: str, pr_number: int, *, platform: str
    ) -> PRReviewProgress | None: ...

    def upsert_pr_review_progress(self, progress: PRReviewProgress, *, platform: str) -> None: ...

    def record_review(
        self, owner: str, repo: str, review: ReviewRecord, *, platform: str
    ) -> None: ...

    def list_pr_fingerprints(
        self, owner: str, repo: str, *, platform: str
    ) -> list[PRFingerprint]: ...

    def upsert_pr_fingerprint(
        self, owner: str, repo: str, fingerprint: PRFingerprint, *, platform: str
    ) -> None: ...


@contextmanager
def _index(owner: str, repo: str, platform: str) -> Iterator[IndexStore]:
    store = IndexStore.open(owner, repo, platform=platform)
    try:
        yield store
    finally:
        store.close()


def _app_db() -> AppDatabase:
    # Looked up per call: importing mira.dashboard.api opens the database, and
    # tests swap the module attribute.
    from mira.dashboard import api

    return api._app_db


class LocalStateStore:
    """The server's state: the dashboard database plus each repo's index store."""

    def get_repo(self, owner: str, repo: str) -> RepoRecord | None:
        return _app_db().get_repo(owner, repo)

    def get_global_rules(self) -> list[str]:
        db: AppDatabase | None = _app_db()
        return [] if db is None else db.get_global_rules_text()

    def get_learned_rules(self, owner: str, repo: str, *, platform: str) -> list[str]:
        with _index(owner, repo, platform) as store:
            return store.get_learned_rules_text()

    def get_last_reviewed_sha(self, owner: str, repo: str, pr_number: int, *, platform: str) -> str:
        return _app_db().get_last_reviewed_sha(owner, repo, pr_number, platform=platform)

    def set_last_reviewed_sha(
        self, owner: str, repo: str, pr_number: int, sha: str, *, platform: str
    ) -> None:
        _app_db().set_last_reviewed_sha(owner, repo, pr_number, sha, platform=platform)

    def get_pr_review_progress(
        self, owner: str, repo: str, pr_number: int, *, platform: str
    ) -> PRReviewProgress | None:
        return _app_db().get_pr_review_progress(owner, repo, pr_number, platform=platform)

    def upsert_pr_review_progress(self, progress: PRReviewProgress, *, platform: str) -> None:
        _app_db().upsert_pr_review_progress(progress, platform=platform)

    def record_review(self, owner: str, repo: str, review: ReviewRecord, *, platform: str) -> None:
        from mira.analysis.feedback import synthesize_rules

        with _index(owner, repo, platform) as store:
            event = store.record_review(
                pr_number=review.pr_number,
                pr_title=review.pr_title,
                pr_url=review.pr_url,
                comments_posted=review.comments_posted,
                blockers=review.blockers,
                warnings=review.warnings,
                suggestions=review.suggestions,
                files_reviewed=review.files_reviewed,
                lines_changed=review.lines_changed,
                tokens_used=review.tokens_used,
                duration_ms=review.duration_ms,
                categories=review.categories,
                author=review.author,
                author_avatar_url=review.author_avatar_url,
                reviewed_paths=json.dumps(review.reviewed_paths),
            )
            store.add_review_comments(event.id, review.pr_number, review.pr_url, review.comments)
            try:
                synthesize_rules(store)
            except Exception as exc:
                logger.debug("Feedback synthesis failed: %s", exc)

    def list_pr_fingerprints(self, owner: str, repo: str, *, platform: str) -> list[PRFingerprint]:
        with _index(owner, repo, platform) as store:
            return store.list_pr_fingerprints()

    def upsert_pr_fingerprint(
        self, owner: str, repo: str, fingerprint: PRFingerprint, *, platform: str
    ) -> None:
        with _index(owner, repo, platform) as store:
            store.upsert_pr_fingerprint(fingerprint)
