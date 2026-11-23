"""The wire protocol between the Mira server and an Actions review runner.

The runner claims its job and receives a :class:`JobSpec`: the task to run,
a snapshot of every piece of review state ``engine.review_pr`` reads for the
PR, and a repo-scoped GitHub token. It runs the task through a
:class:`RemoteStateStore`, which answers reads from the snapshot and buffers
writes, and sends the writes back in one :class:`ResultBundle`. The server
applies them with :func:`apply_writes`.

Writes carry no owner or repo, and only review and feedback records name a
PR. The server applies them to the job's own PR, so a runner cannot write
state for a repo it was not sent to. The server records where a review ran
from the job, not from the runner.

An index job gets an :class:`IndexSnapshot` instead and indexes through a
:class:`RemoteIndexStore`. Its :data:`IndexWrite` s can outgrow one request,
so the runner uploads them as :class:`IndexPart` s before its result, and
the server applies every part to the job's repo in one transaction when the
result lands (:func:`apply_index_writes`).

A learning job gets a :class:`LearningSnapshot` of the repo's recent human
review comments and learns through a :class:`RemoteLearningStore`. Its
rules come back as :class:`UpsertLearnedRule` writes, which no other job
may send.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    SecretStr,
    TypeAdapter,
    model_validator,
)

from mira.analysis.feedback import HUMAN_PATTERN, HUMAN_REVIEW, recent_human_reviews
from mira.config import MiraConfig, _strip_deployment_only_settings
from mira.core.state_store import (
    FeedbackRecord,
    LocalStateStore,
    ReviewRecord,
    ReviewStateStore,
)
from mira.dashboard.db import PRReviewProgress, RepoRecord, ReviewedVersion
from mira.index.store import (
    DirectorySummary,
    FeedbackEventRow,
    FileDigest,
    FileSummary,
    IndexRun,
    IndexStore,
    IndexWriter,
    ReviewContext,
)
from mira.models import PRFingerprint
from mira.runner import JobKind, JobTask, ReviewJob, ReviewTask, RunnerKind, RunnerTarget

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


def job_pr(job: ReviewJob) -> PRKey:
    return PRKey(platform="github", owner=job.owner, repo=job.repo, pr_number=job.pr_number)


class StateSnapshot(_Wire):
    """Every read ``review_pr`` makes through its ReviewStateStore for one PR.

    ``repo_record`` is the reviewed repo only. Blast radius also looks up the
    dependent repos' visibility, which a runner cannot see, so those read as
    unknown and are hidden from the review, as for an unsynced repo.

    The index's code summaries and symbols are not here: the runner reads
    code from its checkout of the PR instead.
    """

    pr: PRKey
    repo_record: RepoRecord | None
    global_rules: list[str]
    learned_rules: list[str]
    review_context: list[ReviewContext]
    manifest_package_names: list[str]
    last_reviewed_sha: str
    last_reviewed_base_sha: str = ""
    review_progress: PRReviewProgress | None
    fingerprints: list[PRFingerprint]


class SetLastReviewedSha(_Wire):
    op: Literal["set_last_reviewed_sha"] = "set_last_reviewed_sha"
    sha: str
    base_sha: str = ""


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


class RecordFeedback(_Wire):
    op: Literal["record_feedback"] = "record_feedback"
    feedback: FeedbackRecord


class UpsertLearnedRule(_Wire):
    """A rule the model learned from human reviews. It can only be one of
    those rules: the server applies it as a pending ``human_pattern`` rule."""

    op: Literal["upsert_learned_rule"] = "upsert_learned_rule"
    rule_text: str = Field(min_length=1, max_length=4000)
    path_pattern: str = Field(pattern=r"^__llm_pattern_\d+__$")
    sample_count: int = Field(ge=0)


StateWrite = Annotated[
    SetLastReviewedSha
    | UpsertPRReviewProgress
    | RecordReview
    | UpsertPRFingerprint
    | RecordFeedback
    | UpsertLearnedRule,
    Field(discriminator="op"),
]


class RepoKey(_Wire):
    platform: str
    owner: str
    repo: str


class IndexSnapshot(_Wire):
    """What indexing reads back from the repo's index: each indexed file's
    content hash and summary."""

    repo: RepoKey
    files: list[FileDigest]


class LearningSnapshot(_Wire):
    """What learning from human reviews reads: the repo's recent human
    review comments, as ``recent_human_reviews`` picks them."""

    repo: RepoKey
    human_reviews: list[FeedbackEventRow]


class RemoveIndexedPaths(_Wire):
    op: Literal["remove_paths"] = "remove_paths"
    paths: list[str]


class UpsertFileSummary(_Wire):
    op: Literal["upsert_summary"] = "upsert_summary"
    summary: FileSummary


class UpsertDirectorySummary(_Wire):
    op: Literal["upsert_directory"] = "upsert_directory"
    directory: DirectorySummary


class ManifestPackage(_Wire):
    name: str
    kind: str
    version: str
    file_path: str
    is_dev: bool
    scan_eligible: bool = True


class ReplaceManifestPackages(_Wire):
    op: Literal["replace_manifest_packages"] = "replace_manifest_packages"
    path: str
    packages: list[ManifestPackage]


class ClearMissingManifests(_Wire):
    """Drop the packages of every manifest not in ``live_paths``."""

    op: Literal["clear_missing_manifests"] = "clear_missing_manifests"
    live_paths: list[str]


class SetConventions(_Wire):
    op: Literal["set_conventions"] = "set_conventions"
    conventions: str


IndexWrite = Annotated[
    RemoveIndexedPaths
    | UpsertFileSummary
    | UpsertDirectorySummary
    | ReplaceManifestPackages
    | ClearMissingManifests
    | SetConventions,
    Field(discriminator="op"),
]
_INDEX_WRITE: TypeAdapter[IndexWrite] = TypeAdapter(IndexWrite)

MAX_INDEX_PART_BYTES = 4 * 1024 * 1024
# 128 MiB in all: room for a large repo's index, and the most one job can
# make the server store.
MAX_INDEX_PARTS = 32
# Paths per RemoveIndexedPaths, so a reset of a large repo still fits a part.
_PATHS_PER_WRITE = 1000


class IndexPart(_Wire):
    writes: list[IndexWrite]


class IndexEnd(StrEnum):
    """How an index run that left its writes fit to apply ended."""

    # The run finished, so the index is current.
    COMPLETE = "complete"
    # Some files were missed. The writes stand, and the next run reconciles.
    INCOMPLETE = "incomplete"
    # The repo has no commits yet.
    EMPTY = "empty"


class IndexReport(_Wire):
    """An index job's outcome: how its run ended and how many parts it
    uploaded. A job that reports none has no writes to apply."""

    end: IndexEnd
    parts: int = Field(ge=0, le=MAX_INDEX_PARTS)


class JobSpec(_Wire):
    job_id: str
    # A spec from a server older than job kinds carries no task: a review.
    task: JobTask = ReviewTask()
    pr_url: str
    head_sha: str
    platform: Literal["github"]
    bot_name: str
    check_run_id: int | None
    # MiraConfig as plain data, without the settings that describe the
    # server's own machine (see runner_config).
    config: dict[str, Any]
    # An IndexSnapshot for an index job, which is on the repo: pr_url is
    # then the repo's URL and head_sha is empty. A LearningSnapshot for a
    # learning job.
    snapshot: StateSnapshot | IndexSnapshot | LearningSnapshot
    github_token: WireSecret
    token_expires_at: datetime
    # Model credentials from the server (review.runner_credentials), by env
    # var name, and the model to use when the repo names none.
    credentials: dict[str, WireSecret] = Field(default_factory=dict)
    llm_provider: str = ""
    llm_model: str = ""

    @model_validator(mode="after")
    def _snapshot_fits_task(self) -> JobSpec:
        if type(self.snapshot) is not snapshot_type(self.task.kind):
            raise ValueError(
                f"a {self.task.kind} job cannot run on a {type(self.snapshot).__name__}"
            )
        return self


def snapshot_type(kind: JobKind) -> type[StateSnapshot | IndexSnapshot | LearningSnapshot]:
    if kind.indexes:
        return IndexSnapshot
    return LearningSnapshot if kind is JobKind.LEARN else StateSnapshot


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
    index: IndexReport | None = None


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
    last = store.get_last_reviewed(owner, repo, number, platform=platform)
    return StateSnapshot(
        pr=pr,
        repo_record=store.get_repo(owner, repo),
        global_rules=store.get_global_rules(),
        learned_rules=store.get_learned_rules(owner, repo, platform=platform),
        review_context=store.list_review_context(owner, repo, platform=platform),
        manifest_package_names=store.list_manifest_package_names(owner, repo, platform=platform),
        last_reviewed_sha=last.head_sha,
        last_reviewed_base_sha=last.base_sha,
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
        self._last = ReviewedVersion(snapshot.last_reviewed_sha, snapshot.last_reviewed_base_sha)
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

    def list_review_context(self, owner: str, repo: str, *, platform: str) -> list[ReviewContext]:
        return list(self._snapshot.review_context) if self._is_repo(owner, repo, platform) else []

    def list_manifest_package_names(self, owner: str, repo: str, *, platform: str) -> list[str]:
        if not self._is_repo(owner, repo, platform):
            return []
        return list(self._snapshot.manifest_package_names)

    def get_last_reviewed(
        self, owner: str, repo: str, pr_number: int, *, platform: str
    ) -> ReviewedVersion:
        return self._last if self._is_pr(owner, repo, pr_number, platform) else ReviewedVersion()

    def set_last_reviewed(
        self, owner: str, repo: str, pr_number: int, version: ReviewedVersion, *, platform: str
    ) -> None:
        self._check_pr(owner, repo, pr_number, platform)
        self._last = version
        self.writes.append(SetLastReviewedSha(sha=version.head_sha, base_sha=version.base_sha))

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

    def record_feedback(
        self, owner: str, repo: str, feedback: FeedbackRecord, *, platform: str
    ) -> None:
        self._check_pr(owner, repo, feedback.pr_number, platform)
        self.writes.append(RecordFeedback(feedback=feedback))

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


def build_learning_snapshot(store: IndexStore, repo: RepoKey) -> LearningSnapshot:
    return LearningSnapshot(repo=repo, human_reviews=recent_human_reviews(store))


class RemoteLearningStore:
    """The repo's human reviews for a runner's learning job, read from a
    snapshot. The rules learned are buffered for :meth:`bundle`."""

    def __init__(self, snapshot: LearningSnapshot) -> None:
        self._human_reviews = snapshot.human_reviews
        self.writes: list[StateWrite] = []

    def list_feedback(self, limit: int = 500) -> list[FeedbackEventRow]:
        return self._human_reviews[:limit]

    def upsert_learned_rule(
        self,
        rule_text: str,
        source_signal: str,
        category: str,
        path_pattern: str,
        sample_count: int,
    ) -> None:
        if (source_signal, category) != (HUMAN_PATTERN, HUMAN_REVIEW):
            raise ValueError(f"a runner learns only {HUMAN_PATTERN} rules, not {source_signal}")
        self.writes.append(
            UpsertLearnedRule(
                rule_text=rule_text, path_pattern=path_pattern, sample_count=sample_count
            )
        )

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


_REVIEW_WRITES = (SetLastReviewedSha, UpsertPRReviewProgress, RecordReview, UpsertPRFingerprint)

# The writes each kind of job may send. Index jobs send theirs as index parts.
_ALLOWED_WRITES: dict[JobKind, tuple[type[BaseModel], ...]] = {
    JobKind.REVIEW: _REVIEW_WRITES,
    JobKind.REVIEW_REST: _REVIEW_WRITES,
    JobKind.ASK: (),
    JobKind.THREAD_REPLY: (RecordFeedback,),
    JobKind.LEARN: (UpsertLearnedRule,),
    JobKind.INDEX_REPO: (),
    JobKind.INDEX_DIFF: (),
}


def check_writes(job: ReviewJob, writes: Sequence[StateWrite]) -> None:
    """Raise ValueError when a write does not belong to ``job``: each kind
    of job may send only its own writes (:data:`_ALLOWED_WRITES`). Only
    review and feedback records name their PR; every other write takes the
    job's."""
    pr = job_pr(job)
    allowed = _ALLOWED_WRITES[job.kind]
    for write in writes:
        if not isinstance(write, allowed):
            raise ValueError(f"a {job.kind} job cannot send {write.op}")
        match write:
            case (
                RecordReview(review=ReviewRecord(pr_number=number))
                | RecordFeedback(feedback=FeedbackRecord(pr_number=number))
            ) if number != pr.pr_number:
                raise ValueError(f"{write.op} for PR {number}, job is {pr}")


def apply_writes(store: LocalStateStore, job: ReviewJob, writes: Sequence[StateWrite]) -> None:
    """Apply a runner's buffered writes to the job's PR, in order, after
    :func:`check_writes` passes for all of them. A review is recorded as run
    on the job's runner and workflow run."""
    check_writes(job, writes)
    pr = job_pr(job)
    owner, repo, number, platform = pr.owner, pr.repo, pr.pr_number, pr.platform
    runner = str(RunnerTarget(RunnerKind.ACTIONS, job.runs_on))
    for write in writes:
        match write:
            case SetLastReviewedSha(sha=sha, base_sha=base_sha):
                version = ReviewedVersion(head_sha=sha, base_sha=base_sha)
                store.set_last_reviewed(owner, repo, number, version, platform=platform)
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
                ran = replace(review, runner=runner, run_url=job.run_url)
                store.record_review(owner, repo, ran, platform=platform)
            case UpsertPRFingerprint(fingerprint=fingerprint):
                store.upsert_pr_fingerprint(owner, repo, fingerprint, platform=platform)
            case RecordFeedback(feedback=feedback):
                store.record_feedback(owner, repo, feedback, platform=platform)
            case UpsertLearnedRule():
                store.upsert_learned_rule(
                    owner,
                    repo,
                    write.rule_text,
                    write.path_pattern,
                    write.sample_count,
                    platform=platform,
                )


class RemoteIndexStore:
    """The repo's index for a runner's index job, read from a snapshot.

    Writes are buffered in order, and later reads see them. ``get_summary``
    returns only the content hash and summary, which is all indexing reads
    back. ``index_write`` fences nothing here: the server applies the writes
    under the job's own index run. ``finished`` says whether indexing
    finished its run, and ``report`` is set once the writes are uploaded.
    """

    def __init__(self, snapshot: IndexSnapshot, *, reconcile: bool) -> None:
        self.repo = snapshot.repo
        self._reconcile = reconcile
        self._files = {digest.path: digest for digest in snapshot.files}
        self.writes: list[IndexWrite] = []
        self.finished = False
        self.report: IndexReport | None = None

    def claim_index_run(self, token: str | None = None) -> IndexRun:
        return IndexRun(token or "runner", self._reconcile)

    @contextmanager
    def index_write(self, run: IndexRun) -> Iterator[None]:
        yield

    def finish_index_run(self, run: IndexRun) -> None:
        self.finished = True

    def get_summary(self, path: str) -> FileSummary | None:
        digest = self._files.get(path)
        if digest is None:
            return None
        return FileSummary(
            path=path, language="", summary=digest.summary, content_hash=digest.content_hash
        )

    def all_paths(self) -> set[str]:
        return set(self._files)

    def upsert_summary(self, summary: FileSummary) -> None:
        self._files[summary.path] = FileDigest(summary.path, summary.content_hash, summary.summary)
        self.writes.append(UpsertFileSummary(summary=summary))

    def upsert_directory(self, summary: DirectorySummary) -> None:
        self.writes.append(UpsertDirectorySummary(directory=summary))

    def remove_paths(self, paths: list[str]) -> None:
        for path in paths:
            self._files.pop(path, None)
        for i in range(0, len(paths), _PATHS_PER_WRITE):
            self.writes.append(RemoveIndexedPaths(paths=paths[i : i + _PATHS_PER_WRITE]))

    def replace_manifest_packages(self, file_path: str, packages: list[dict]) -> int:
        manifest = [ManifestPackage.model_validate(package) for package in packages]
        self.writes.append(ReplaceManifestPackages(path=file_path, packages=manifest))
        return len(packages)

    def clear_manifest_packages_for_missing_files(self, live_paths: set[str]) -> int:
        self.writes.append(ClearMissingManifests(live_paths=sorted(live_paths)))
        return 0

    def set_repo_conventions(
        self, owner: str, repo: str, conventions: str, *, platform: str
    ) -> None:
        if (platform, owner, repo) != (self.repo.platform, self.repo.owner, self.repo.repo):
            raise ValueError(f"this runner indexes {self.repo}, not {owner}/{repo}")
        self.writes.append(SetConventions(conventions=conventions))

    def close(self) -> None:
        pass

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
            usage=usage or UsageReport(),
            index=self.report,
        )


def index_parts(writes: Sequence[IndexWrite], max_bytes: int = MAX_INDEX_PART_BYTES) -> list[bytes]:
    """``writes`` in order as serialized :class:`IndexPart` bodies of at most
    ``max_bytes`` each. Raises ValueError when one write alone is larger, or
    when there are more than :data:`MAX_INDEX_PARTS` parts."""
    head, tail = b'{"writes":[', b"]}"
    parts: list[list[bytes]] = []
    size = max_bytes
    for write in writes:
        item = _INDEX_WRITE.dump_json(write)
        if len(head) + len(item) + len(tail) > max_bytes:
            raise ValueError(f"a {write.op} write of {len(item)} bytes does not fit in a part")
        if size + 1 + len(item) > max_bytes:
            parts.append([])
            size = len(head) + len(tail) - 1
        parts[-1].append(item)
        size += 1 + len(item)
    if len(parts) > MAX_INDEX_PARTS:
        raise ValueError(f"the index is {len(parts)} parts, over the limit of {MAX_INDEX_PARTS}")
    return [head + b",".join(items) + tail for items in parts]


def check_index_writes(writes: Sequence[IndexWrite]) -> None:
    """Raise ValueError when a manifest write lists a package of another file."""
    for write in writes:
        if isinstance(write, ReplaceManifestPackages) and any(
            package.file_path != write.path for package in write.packages
        ):
            raise ValueError(f"packages of another file in the manifest write for {write.path}")


def apply_index_writes(store: IndexWriter, repo: RepoKey, writes: Sequence[IndexWrite]) -> int:
    """Apply a runner's index writes to ``repo``'s index, in order, inside
    the caller's ``index_write``. Returns the file summaries written."""
    check_index_writes(writes)
    summaries = 0
    for write in writes:
        match write:
            case RemoveIndexedPaths(paths=paths):
                store.remove_paths(paths)
            case UpsertFileSummary(summary=summary):
                store.upsert_summary(summary)
                summaries += 1
            case UpsertDirectorySummary(directory=directory):
                store.upsert_directory(directory)
            case ReplaceManifestPackages(path=path, packages=packages):
                store.replace_manifest_packages(path, [p.model_dump() for p in packages])
            case ClearMissingManifests(live_paths=live_paths):
                store.clear_manifest_packages_for_missing_files(set(live_paths))
            case SetConventions(conventions=conventions):
                store.set_repo_conventions(
                    repo.owner, repo.repo, conventions, platform=repo.platform
                )
    return summaries
