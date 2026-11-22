"""The wire protocol between the Mira server and an Actions review runner.

The runner claims its job and receives a :class:`JobSpec`: a snapshot of
every piece of review state ``engine.review_pr`` reads for the PR, plus a
repo-scoped GitHub token. It reviews through a :class:`RemoteStateStore`,
which answers reads from the snapshot and buffers writes, and sends the
writes back in one :class:`ResultBundle`. The server applies them with
:func:`apply_writes`.

Writes carry no owner, repo or PR number. The server applies them to the
job's own PR, so a runner cannot write state for a repo it was not sent to.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, SecretStr

from mira.config import MiraConfig, _strip_deployment_only_settings
from mira.core.state_store import ReviewRecord, ReviewStateStore
from mira.dashboard.db import PRReviewProgress, RepoRecord
from mira.models import PRFingerprint

# A secret that serializes to its value on the wire but never shows in a repr
# or a log line.
WireSecret = Annotated[
    SecretStr, PlainSerializer(lambda s: s.get_secret_value(), return_type=str, when_used="json")
]


class _Wire(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PRKey(_Wire):
    platform: str
    owner: str
    repo: str
    pr_number: int


class StateSnapshot(_Wire):
    """Every read ``review_pr`` makes through its ReviewStateStore for one PR.

    ``repo_record`` is the reviewed repo only. Blast radius also looks up the
    dependent repos' visibility, which a runner cannot see, so those read as
    unknown and are hidden from the review, as for an unsynced repo.
    """

    pr: PRKey
    repo_record: RepoRecord | None
    global_rules: list[str]
    learned_rules: list[str]
    last_reviewed_sha: str
    review_progress: PRReviewProgress | None
    fingerprints: list[PRFingerprint]


class SetLastReviewedSha(_Wire):
    op: Literal["set_last_reviewed_sha"] = "set_last_reviewed_sha"
    sha: str


class UpsertPRReviewProgress(_Wire):
    op: Literal["upsert_pr_review_progress"] = "upsert_pr_review_progress"
    total_paths: list[str]
    reviewed_paths: list[str]
    skipped_paths: list[str]
    chunk_index: int


class RecordReview(_Wire):
    op: Literal["record_review"] = "record_review"
    review: ReviewRecord


class UpsertPRFingerprint(_Wire):
    """Fingerprints are per repo: overlap detection caches other open PRs' too."""

    op: Literal["upsert_pr_fingerprint"] = "upsert_pr_fingerprint"
    fingerprint: PRFingerprint


StateWrite = Annotated[
    SetLastReviewedSha | UpsertPRReviewProgress | RecordReview | UpsertPRFingerprint,
    Field(discriminator="op"),
]


class JobSpec(_Wire):
    job_id: str
    pr_url: str
    head_sha: str
    platform: Literal["github"]
    bot_name: str
    check_run_id: int | None
    # MiraConfig as plain data, without the settings that describe the
    # server's own machine (see runner_config).
    config: dict[str, Any]
    snapshot: StateSnapshot
    github_token: WireSecret
    token_expires_at: datetime


class TokenGrant(_Wire):
    github_token: WireSecret
    token_expires_at: datetime


class UsageReport(_Wire):
    harness: str | None = None
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class ResultBundle(_Wire):
    job_id: str
    outcome: Literal["succeeded", "failed"]
    error: str = Field(default="", max_length=4000)
    writes: list[StateWrite] = Field(default_factory=list)
    usage: UsageReport = Field(default_factory=UsageReport)


def runner_config(config: MiraConfig) -> dict[str, Any]:
    """The review config a runner gets. The LLM execution settings, the runner
    choice and command authorization are dropped, as they are from a repo's
    ``.mira.yaml``. They describe the server's machine, not the runner's.
    ``database`` is dropped too: it holds the server's credentials."""
    data = _strip_deployment_only_settings(config.model_dump(mode="json"))
    data.pop("database", None)
    return data


def build_snapshot(store: ReviewStateStore, pr: PRKey) -> StateSnapshot:
    owner, repo, number, platform = pr.owner, pr.repo, pr.pr_number, pr.platform
    return StateSnapshot(
        pr=pr,
        repo_record=store.get_repo(owner, repo),
        global_rules=store.get_global_rules(),
        learned_rules=store.get_learned_rules(owner, repo, platform=platform),
        last_reviewed_sha=store.get_last_reviewed_sha(owner, repo, number, platform=platform),
        review_progress=store.get_pr_review_progress(owner, repo, number, platform=platform),
        fingerprints=store.list_pr_fingerprints(owner, repo, platform=platform),
    )


class RemoteStateStore:
    """A ReviewStateStore for one PR, read from a snapshot.

    Writes are buffered in order for :meth:`bundle`, and later reads see them,
    as they would through LocalStateStore.
    """

    def __init__(self, snapshot: StateSnapshot) -> None:
        self._snapshot = snapshot
        self._last_sha = snapshot.last_reviewed_sha
        self._progress = snapshot.review_progress
        self._fingerprints = {fp.pr_number: fp for fp in snapshot.fingerprints}
        self.writes: list[StateWrite] = []

    def _is_repo(self, owner: str, repo: str, platform: str | None = None) -> bool:
        pr = self._snapshot.pr
        return (owner, repo) == (pr.owner, pr.repo) and platform in (None, pr.platform)

    def _is_pr(self, owner: str, repo: str, pr_number: int, platform: str) -> bool:
        return self._is_repo(owner, repo, platform) and pr_number == self._snapshot.pr.pr_number

    def _check_repo(self, owner: str, repo: str, platform: str) -> None:
        if not self._is_repo(owner, repo, platform):
            raise ValueError(f"this runner holds state for {self._snapshot.pr}, not {owner}/{repo}")

    def _check_pr(self, owner: str, repo: str, pr_number: int, platform: str) -> None:
        if not self._is_pr(owner, repo, pr_number, platform):
            raise ValueError(
                f"this runner holds state for {self._snapshot.pr}, not {owner}/{repo}#{pr_number}"
            )

    def get_repo(self, owner: str, repo: str) -> RepoRecord | None:
        return self._snapshot.repo_record if self._is_repo(owner, repo) else None

    def get_global_rules(self) -> list[str]:
        return list(self._snapshot.global_rules)

    def get_learned_rules(self, owner: str, repo: str, *, platform: str) -> list[str]:
        return list(self._snapshot.learned_rules) if self._is_repo(owner, repo, platform) else []

    def get_last_reviewed_sha(self, owner: str, repo: str, pr_number: int, *, platform: str) -> str:
        return self._last_sha if self._is_pr(owner, repo, pr_number, platform) else ""

    def set_last_reviewed_sha(
        self, owner: str, repo: str, pr_number: int, sha: str, *, platform: str
    ) -> None:
        self._check_pr(owner, repo, pr_number, platform)
        self._last_sha = sha
        self.writes.append(SetLastReviewedSha(sha=sha))

    def get_pr_review_progress(
        self, owner: str, repo: str, pr_number: int, *, platform: str
    ) -> PRReviewProgress | None:
        return self._progress if self._is_pr(owner, repo, pr_number, platform) else None

    def upsert_pr_review_progress(self, progress: PRReviewProgress, *, platform: str) -> None:
        self._check_pr(progress.owner, progress.repo, progress.pr_number, platform)
        self._progress = progress
        self.writes.append(
            UpsertPRReviewProgress(
                total_paths=progress.total_paths,
                reviewed_paths=progress.reviewed_paths,
                skipped_paths=progress.skipped_paths,
                chunk_index=progress.chunk_index,
            )
        )

    def record_review(self, owner: str, repo: str, review: ReviewRecord, *, platform: str) -> None:
        self._check_pr(owner, repo, review.pr_number, platform)
        self.writes.append(RecordReview(review=review))

    def list_pr_fingerprints(self, owner: str, repo: str, *, platform: str) -> list[PRFingerprint]:
        if not self._is_repo(owner, repo, platform):
            return []
        return list(self._fingerprints.values())

    def upsert_pr_fingerprint(
        self, owner: str, repo: str, fingerprint: PRFingerprint, *, platform: str
    ) -> None:
        self._check_repo(owner, repo, platform)
        self._fingerprints[fingerprint.pr_number] = fingerprint
        self.writes.append(UpsertPRFingerprint(fingerprint=fingerprint))

    def bundle(
        self,
        job_id: str,
        outcome: Literal["succeeded", "failed"],
        *,
        error: str = "",
        usage: UsageReport | None = None,
    ) -> ResultBundle:
        return ResultBundle(
            job_id=job_id,
            outcome=outcome,
            error=error,
            writes=list(self.writes),
            usage=usage or UsageReport(),
        )


def check_writes(pr: PRKey, writes: Sequence[StateWrite]) -> None:
    """Raise ValueError when a write does not belong to ``pr``. Only a review
    record names its PR; every other write takes the job's."""
    for write in writes:
        if isinstance(write, RecordReview) and write.review.pr_number != pr.pr_number:
            raise ValueError(f"review record for PR {write.review.pr_number}, job is {pr}")


def apply_writes(store: ReviewStateStore, pr: PRKey, writes: Sequence[StateWrite]) -> None:
    """Apply a runner's buffered writes to ``pr``, in order, after
    :func:`check_writes` passes for all of them."""
    check_writes(pr, writes)
    owner, repo, number, platform = pr.owner, pr.repo, pr.pr_number, pr.platform
    for write in writes:
        match write:
            case SetLastReviewedSha(sha=sha):
                store.set_last_reviewed_sha(owner, repo, number, sha, platform=platform)
            case UpsertPRReviewProgress():
                progress = PRReviewProgress(
                    owner=owner,
                    repo=repo,
                    pr_number=number,
                    total_paths=write.total_paths,
                    reviewed_paths=write.reviewed_paths,
                    skipped_paths=write.skipped_paths,
                    chunk_index=write.chunk_index,
                )
                store.upsert_pr_review_progress(progress, platform=platform)
            case RecordReview(review=review):
                store.record_review(owner, repo, review, platform=platform)
            case UpsertPRFingerprint(fingerprint=fingerprint):
                store.upsert_pr_fingerprint(owner, repo, fingerprint, platform=platform)
