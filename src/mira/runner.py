"""Where a review runs, and the job record for reviews run outside the server.

``local`` reviews in the server process. ``actions`` dispatches the repo's
``mira.yml`` workflow, whose ``runs-on`` takes the target's labels, and tracks
the run as a :class:`ReviewJob`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

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
        raise IllegalJobTransition(f"review job cannot go from {old} to {new}")


@dataclass
class ReviewJob:
    """One dispatched review. ``id`` is the unguessable workflow input the
    runner later presents to claim the job."""

    id: str
    owner: str
    repo: str
    pr_number: int
    head_sha: str
    installation_id: int
    runs_on: tuple[str, ...]
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
