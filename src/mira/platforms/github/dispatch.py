"""Hand a PR review to the repo's Mira workflow on GitHub Actions.

The server records a :class:`ReviewJob`, shows a queued check, and dispatches
``.github/workflows/mira.yml`` with the job id, the runs-on labels and the
URL the runner reports back to. A newer
head SHA supersedes the PR's older job and cancels its run.
"""

from __future__ import annotations

import json
import logging
import secrets
import time

from mira.dashboard.db import AppDatabase
from mira.exceptions import WorkflowDispatchError
from mira.models import PRInfo
from mira.providers.base import CheckConclusion
from mira.providers.github import GitHubProvider
from mira.runner import JobStatus, ReviewJob

logger = logging.getLogger(__name__)

WORKFLOW_FILE = "mira.yml"

# What the check run tells the PR author when GitHub refuses the dispatch.
_DISPATCH_FAILURES: dict[int | None, str] = {
    404: (
        f"Mira runner workflow .github/workflows/{WORKFLOW_FILE} not found in this "
        "repo's default branch. Add it, or set this repo's runner to local."
    ),
    403: "The Mira GitHub App lacks the Actions: write permission on this repo.",
    422: (
        f".github/workflows/{WORKFLOW_FILE} does not accept the workflow_dispatch "
        "inputs job_id, runs_on and server_url."
    ),
}
_NO_SERVER_URL = (
    "This repo reviews on GitHub Actions, but the Mira server has no "
    "review.runner_server_url for its runner to report to."
)
_DISPATCH_FAILED = "Mira could not start its runner workflow .github/workflows/{workflow} ({why})."


class RunnerDispatchError(Exception):
    """The review was not handed to a runner. The job is already FAILED and
    its check completed with this message."""


def _failure_message(exc: Exception) -> str:
    status = exc.status_code if isinstance(exc, WorkflowDispatchError) else None
    if status in _DISPATCH_FAILURES:
        return _DISPATCH_FAILURES[status]
    why = f"GitHub answered {status}" if status else "the request to GitHub failed"
    return _DISPATCH_FAILED.format(workflow=WORKFLOW_FILE, why=why)


async def finish_check(
    provider: GitHubProvider,
    pr_info: PRInfo,
    check_run_id: int | None,
    conclusion: CheckConclusion,
    title: str,
    summary: str,
) -> None:
    try:
        await provider.finish_check_run(
            pr_info, check_run_id, conclusion, title=title, summary=summary
        )
    except Exception as exc:
        logger.warning("Failed to finish check run %s on %s: %s", check_run_id, pr_info.url, exc)


async def _retire(provider: GitHubProvider, job: ReviewJob, pr_info: PRInfo) -> None:
    """Best effort: cancel a superseded job's run and close its check."""
    if job.run_id is None:
        logger.warning("Superseded review job %s has no workflow run to cancel", job.id)
    else:
        try:
            await provider.cancel_workflow_run(job.owner, job.repo, job.run_id)
            logger.info("Cancelled workflow run %s of superseded job %s", job.run_id, job.id)
        except Exception as exc:
            logger.warning(
                "Could not cancel workflow run %s of job %s: %s", job.run_id, job.id, exc
            )
    await finish_check(
        provider,
        pr_info,
        job.check_run_id,
        "neutral",
        "Superseded",
        "A newer commit replaced this review.",
    )


async def dispatch_review(
    provider: GitHubProvider,
    db: AppDatabase,
    pr_info: PRInfo,
    *,
    default_branch: str,
    installation_id: int,
    runs_on: tuple[str, ...],
    server_url: str,
) -> None:
    """Dispatch a review of ``pr_info.head_sha``, superseding older jobs.

    Every DB write that decides ownership (supersede, create) runs before
    the first await, so two webhooks for one PR on the event loop cannot
    both dispatch. Raises :class:`RunnerDispatchError` when GitHub refuses.
    """
    owner, repo, number = pr_info.owner, pr_info.repo, pr_info.number
    active = db.list_active_review_jobs(owner, repo, number)
    if any(job.head_sha == pr_info.head_sha for job in active):
        logger.info("Review of %s at %s already dispatched", pr_info.url, pr_info.head_sha[:12])
        return
    superseded = [
        job
        for job in active
        if db.transition_review_job(
            job.id, job.status, JobStatus.SUPERSEDED, error=f"superseded by {pr_info.head_sha}"
        )
    ]
    job = ReviewJob(
        id=secrets.token_urlsafe(32),
        owner=owner,
        repo=repo,
        pr_number=number,
        head_sha=pr_info.head_sha,
        installation_id=installation_id,
        runs_on=runs_on,
        created_at=time.time(),
    )
    db.create_review_job(job)

    for old in superseded:
        await _retire(provider, old, pr_info)

    try:
        check_run_id = await provider.start_check_run(
            pr_info,
            title="Code Review Queued",
            summary="Waiting for a Mira runner (" + ", ".join(runs_on) + ").",
            details_url=pr_info.url,
            status="queued",
        )
    except Exception as exc:
        logger.warning("Failed to start check run for %s: %s", pr_info.url, exc)
        check_run_id = None
    db.set_review_job_check_run(job.id, check_run_id)

    async def fail(message: str) -> RunnerDispatchError:
        db.transition_review_job(job.id, JobStatus.DISPATCHED, JobStatus.FAILED, error=message)
        await finish_check(
            provider, pr_info, check_run_id, "failure", "Mira runner not started", message
        )
        return RunnerDispatchError(message)

    if not server_url:
        raise await fail(_NO_SERVER_URL)
    try:
        run = await provider.dispatch_workflow(
            owner,
            repo,
            WORKFLOW_FILE,
            ref=default_branch,
            inputs={
                "job_id": job.id,
                "runs_on": json.dumps(list(runs_on)),
                "server_url": server_url,
            },
        )
    except Exception as exc:
        raise await fail(_failure_message(exc)) from exc

    db.set_review_job_run(job.id, run.id, run.html_url)
    current = db.get_review_job(job.id)
    if current is not None and current.status is JobStatus.SUPERSEDED:
        # A newer push superseded this job while its dispatch was in flight,
        # before there was a run to cancel.
        await _retire(provider, current, pr_info)
        return
    logger.info(
        "Dispatched review of %s to %s (job %s, run %s)", pr_info.url, runs_on, job.id, run.id
    )
