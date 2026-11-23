"""Where a review runs, and the job record for work run outside the server.

``local`` runs every model call in the server process. ``actions``
dispatches the repo's ``mira.yml`` workflow, whose ``runs-on`` takes the
target's labels, and tracks the run as a :class:`ReviewJob`. A job runs one
:data:`JobTask`: a review, an answer to an @mention, an index of the repo, or
the model pass of learning from a merged PR.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

if TYPE_CHECKING:
    from mira.config import ReviewConfig


class RunnerKind(StrEnum):
    LOCAL = "local"
    ACTIONS = "actions"


@dataclass(frozen=True)
class RunnerTarget:
    kind: RunnerKind
    runs_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.kind is RunnerKind.ACTIONS) != bool(self.runs_on):
            raise ValueError(f"runs_on must be set exactly for actions runners, got {self!r}")

    def __str__(self) -> str:
        """The form :func:`parse_runner_override` reads."""
        if self.kind is RunnerKind.LOCAL:
            return RunnerKind.LOCAL.value
        return f"{RunnerKind.ACTIONS.value}:{','.join(self.runs_on)}"


def parse_runner_override(value: str) -> RunnerTarget | None:
    """Parse ``RepoRecord.runner``. ``''`` means inherit and returns None.

    The other forms are ``local`` and ``actions:<label>[,<label>...]``, for
    example ``actions:self-hosted,sora``. Runner labels cannot contain commas
    (``config.sh --labels`` splits on them), so a comma is a safe separator.
    """
    if not value:
        return None
    if value == RunnerKind.LOCAL:
        return RunnerTarget(RunnerKind.LOCAL)
    kind, sep, labels = value.partition(":")
    runs_on = tuple(label.strip() for label in labels.split(","))
    if kind != RunnerKind.ACTIONS or not sep or not all(runs_on):
        raise ValueError(f"invalid runner override {value!r}")
    return RunnerTarget(RunnerKind.ACTIONS, runs_on)


@dataclass(frozen=True)
class RunnerLLMChoice:
    """The model a runner reviews with: a provider (an agent CLI or a
    ``providers.json`` profile) and a model id. An empty model means the
    provider's default."""

    provider: str
    model: str = ""

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def parse_runner_llm(value: str) -> RunnerLLMChoice | None:
    """Parse ``RepoRecord.runner_llm``: ``''`` inherits and returns None,
    else ``<provider>:<model>``, e.g. ``zai-coding:glm-5.3``."""
    if not value:
        return None
    provider, sep, model = value.partition(":")
    provider, model = provider.strip(), model.strip()
    if not sep or not provider or any(c.isspace() for c in provider + model):
        raise ValueError(f"invalid runner model {value!r}, expected '<provider>:<model>'")
    return RunnerLLMChoice(provider, model)


def resolve_runner_target(override: str, review: ReviewConfig) -> RunnerTarget:
    """The repo's override when it has one, else the deployment default."""
    parsed = parse_runner_override(override)
    if parsed is not None:
        return parsed
    if review.runner is RunnerKind.LOCAL:
        return RunnerTarget(RunnerKind.LOCAL)
    return RunnerTarget(RunnerKind.ACTIONS, tuple(review.runs_on))


class JobStatus(StrEnum):
    DISPATCHED = "dispatched"
    CLAIMED = "claimed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


# A job succeeds only through a claim. Every terminal status is final.
_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.DISPATCHED: frozenset(
        {JobStatus.CLAIMED, JobStatus.FAILED, JobStatus.SUPERSEDED, JobStatus.EXPIRED}
    ),
    JobStatus.CLAIMED: frozenset(
        {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.SUPERSEDED, JobStatus.EXPIRED}
    ),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.SUPERSEDED: frozenset(),
    JobStatus.EXPIRED: frozenset(),
}
ACTIVE_JOB_STATUSES = (JobStatus.DISPATCHED, JobStatus.CLAIMED)


class IllegalJobTransition(ValueError):
    pass


def check_transition(old: JobStatus, new: JobStatus) -> None:
    if new not in _TRANSITIONS[old]:
        raise IllegalJobTransition(f"a job cannot go from {old} to {new}")


class JobKind(StrEnum):
    REVIEW = "review"
    REVIEW_REST = "review_rest"
    ASK = "ask"
    THREAD_REPLY = "thread_reply"
    INDEX_REPO = "index_repo"
    INDEX_DIFF = "index_diff"
    LEARN = "learn"

    @property
    def label(self) -> str:
        """The kind as people read it, in a job's errors."""
        return _LABELS[self]

    @property
    def reviews_pr(self) -> bool:
        """Whether the job reviews the PR and writes its review state. A PR
        has one such job active at a time. The others only post a reply."""
        return self in (JobKind.REVIEW, JobKind.REVIEW_REST)

    @property
    def indexes(self) -> bool:
        """Whether the job writes the repo's index. Such a job is on the
        repo, not a PR, and a repo has one active at a time."""
        return self in (JobKind.INDEX_REPO, JobKind.INDEX_DIFF)

    @property
    def posts(self) -> bool:
        """Whether the job posts to its PR. The others only read, so their
        GitHub token can only read."""
        return not (self.indexes or self is JobKind.LEARN)


_LABELS = {
    JobKind.REVIEW: "review",
    JobKind.REVIEW_REST: "review",
    JobKind.ASK: "question",
    JobKind.THREAD_REPLY: "thread reply",
    JobKind.INDEX_REPO: "index",
    JobKind.INDEX_DIFF: "index",
    JobKind.LEARN: "learning",
}


class _Task(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReviewTask(_Task):
    kind: Literal[JobKind.REVIEW] = JobKind.REVIEW


class ReviewRestTask(_Task):
    """Review only ``paths``, the files an earlier pass skipped."""

    kind: Literal[JobKind.REVIEW_REST] = JobKind.REVIEW_REST
    paths: tuple[str, ...]


class AskTask(_Task):
    """Answer a question ``actor`` asked in a PR comment."""

    kind: Literal[JobKind.ASK] = JobKind.ASK
    question: str
    actor: str


class ThreadReply(_Task):
    """A human's reply in one of Mira's review threads. ``body`` has the
    mention stripped. A thread is named by ``thread_id`` or, on GitHub, by
    the reply's ``comment_node_id``."""

    comment_id: int
    body: str
    original_suggestion: str = ""
    thread_id: str | None = None
    comment_node_id: str | None = None
    path: str = ""
    line: int = 0
    actor: str = ""


class ThreadReplyTask(_Task):
    """Answer a thread reply. A disagreement also resolves the thread, which
    needs the reject permission: ``may_resolve`` is the server's verdict."""

    kind: Literal[JobKind.THREAD_REPLY] = JobKind.THREAD_REPLY
    reply: ThreadReply
    may_resolve: bool


class IndexRepoTask(_Task):
    """Index the repo's default branch. ``full`` summarizes unchanged files
    again and ``reset`` first drops the whole index. ``reconcile`` is set at
    dispatch when an earlier index run did not finish: the run must then
    reach every file, or it leaves the index pending."""

    kind: Literal[JobKind.INDEX_REPO] = JobKind.INDEX_REPO
    full: bool = False
    reset: bool = False
    reconcile: bool = False


class IndexDiffTask(_Task):
    """Index a push to the default branch ``branch``."""

    kind: Literal[JobKind.INDEX_DIFF] = JobKind.INDEX_DIFF
    branch: str
    changed_paths: tuple[str, ...]
    removed_paths: tuple[str, ...]


class LearnTask(_Task):
    """Learn review rules with the model from the repo's recent human review
    comments, after a merge recorded them. The job is on the merged PR."""

    kind: Literal[JobKind.LEARN] = JobKind.LEARN


IndexTask = IndexRepoTask | IndexDiffTask
JobTask = Annotated[
    ReviewTask
    | ReviewRestTask
    | AskTask
    | ThreadReplyTask
    | IndexRepoTask
    | IndexDiffTask
    | LearnTask,
    Field(discriminator="kind"),
]
_JOB_TASK: TypeAdapter[JobTask] = TypeAdapter(JobTask)


def task_payload(task: JobTask) -> str:
    """The task as stored beside its kind: JSON without the kind."""
    return task.model_dump_json(exclude={"kind"})


def parse_task(kind: str, payload: str) -> JobTask:
    return _JOB_TASK.validate_python({**json.loads(payload), "kind": kind})


# The pr_number of a job on the whole repo. GitHub numbers PRs from 1.
REPO_JOB = 0


@dataclass
class ReviewJob:
    """One dispatched job. ``id`` is the unguessable workflow input the
    runner later presents to claim the job. An index job is on the whole
    repo: its ``pr_number`` is :data:`REPO_JOB` and its ``head_sha`` empty."""

    id: str
    owner: str
    repo: str
    pr_number: int
    head_sha: str
    installation_id: int
    runs_on: tuple[str, ...]
    task: JobTask = ReviewTask()
    status: JobStatus = JobStatus.DISPATCHED
    check_run_id: int | None = None
    run_id: int | None = None
    run_url: str = ""
    error: str = ""
    created_at: float = 0.0
    claimed_at: float = 0.0
    finished_at: float = 0.0
    # The runner's UsageReport as plain data. None until its result lands.
    usage: dict[str, Any] | None = None

    @property
    def kind(self) -> JobKind:
        return self.task.kind
