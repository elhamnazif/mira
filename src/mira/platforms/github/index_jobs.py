"""Index jobs: a repo that reviews on GitHub Actions is indexed there too.

The server dispatches an index job through the repo's ``mira.yml``, as it
does a review, and calls no model. At dispatch the job claims the repo's
index run with its own id as the token, so the newest job owns the index.
An older job's result then no longer applies, and a job that never reports
leaves the index pending, so the next job reconciles it.

The runner uploads its writes in parts before its result. When the result
lands, :func:`finish_index_job` applies every part in one index transaction
under the job's run, and while the run still owns the index records how the
job ended on the repo's row and in the dashboard's indexing tracker.
"""

from __future__ import annotations

import logging
import secrets
import time
from functools import partial

from mira.config import load_config
from mira.dashboard.db import AppDatabase, RepoRecord
from mira.index.indexer import _queue_vulnerability_scan
from mira.index.status import tracker
from mira.index.store import IndexingSuperseded, IndexRun, IndexStore
from mira.platforms.fetch import EmptyRepoError, GitHubRepoFetcher
from mira.platforms.github.auth import GitHubAppAuth
from mira.platforms.github.dispatch import cancel_run, start_workflow, superseded_in_flight
from mira.providers.github import GitHubProvider
from mira.runner import (
    REPO_JOB,
    IndexDiffTask,
    IndexRepoTask,
    IndexTask,
    JobStatus,
    JobTask,
    ReviewJob,
    RunnerKind,
    resolve_runner_target,
)
from mira.runner_protocol import (
    ClearMissingManifests,
    IndexEnd,
    IndexPart,
    IndexReport,
    ReplaceManifestPackages,
    RepoKey,
    apply_index_writes,
)

logger = logging.getLogger(__name__)

_CANCELLED = "Indexing cancelled from the dashboard"


def _covering(task: IndexTask, superseded: list[JobTask], *, reconcile: bool) -> IndexTask:
    """The task that covers ``task`` and the work of the jobs it superseded,
    which is lost. A push becomes a whole-repo index when the index must be
    reconciled or a job was lost, keeping any full or reset asked for."""
    if isinstance(task, IndexDiffTask) and not reconcile and not superseded:
        return task
    asked = [t for t in (task, *superseded) if isinstance(t, IndexRepoTask)]
    return IndexRepoTask(
        full=any(t.full for t in asked), reset=any(t.reset for t in asked), reconcile=reconcile
    )


async def dispatch_index_job(
    provider: GitHubProvider,
    db: AppDatabase,
    owner: str,
    repo: str,
    task: IndexTask,
    *,
    default_branch: str,
    installation_id: int,
    runs_on: tuple[str, ...],
    server_url: str,
) -> str:
    """Dispatch ``task`` on the repo, superseding its active index jobs.
    Returns why the runner was not started, or "" when it was.

    Every write that decides ownership (supersede, claim, create) runs
    before the first await, as in :func:`~mira.platforms.github.dispatch.dispatch_job`.
    """
    superseded = [
        job
        for job in db.list_active_review_jobs(owner, repo, REPO_JOB)
        if job.kind.indexes
        and db.transition_review_job(
            job.id, job.status, JobStatus.SUPERSEDED, error="superseded by a newer index job"
        )
    ]
    job_id = secrets.token_urlsafe(32)
    store = IndexStore.open(owner, repo, allow_fallback=False)
    try:
        run = store.claim_index_run(job_id)
        files = len(store.all_paths())
    finally:
        store.close()
    job = ReviewJob(
        id=job_id,
        owner=owner,
        repo=repo,
        pr_number=REPO_JOB,
        head_sha="",
        installation_id=installation_id,
        runs_on=runs_on,
        task=_covering(task, [j.task for j in superseded], reconcile=run.needs_reconciliation),
        created_at=time.time(),
    )
    db.create_review_job(job)
    if isinstance(job.task, IndexRepoTask):
        db.set_repo_status(owner, repo, "indexing", files_indexed=files)
        tracker.start(f"{owner}/{repo}")

    for old in superseded:
        await cancel_run(provider, old)
    failure = await start_workflow(
        provider, db, job, default_branch=default_branch, server_url=server_url
    )
    if failure:
        if db.transition_review_job(job.id, JobStatus.DISPATCHED, JobStatus.FAILED, error=failure):
            finish_index_job(db, job, None, failure)
        return failure
    if (current := superseded_in_flight(db, job)) is not None:
        await cancel_run(provider, current)
        return ""
    logger.info("Dispatched %s of %s/%s to %s (job %s)", job.kind, owner, repo, runs_on, job.id)
    return ""


def actions_runs_on(record: RepoRecord) -> tuple[str, ...]:
    """The runs-on labels when this repo indexes on Actions, else ()."""
    if record.platform != "github":
        return ()
    target = resolve_runner_target(record.runner, load_config().review)
    return target.runs_on if target.kind is RunnerKind.ACTIONS else ()


async def dispatch_repo_index(
    app_auth: GitHubAppAuth,
    db: AppDatabase,
    record: RepoRecord,
    task: IndexRepoTask,
    runs_on: tuple[str, ...],
) -> str:
    """Dispatch a whole-repo index asked for from the dashboard. Returns why
    the runner was not started, or ""."""
    token = partial(app_auth.get_installation_token, record.installation_id)
    branch = await GitHubRepoFetcher(token).default_branch(record.owner, record.repo)
    return await dispatch_index_job(
        GitHubProvider(token),
        db,
        record.owner,
        record.repo,
        task,
        default_branch=branch,
        installation_id=record.installation_id,
        runs_on=runs_on,
        server_url=load_config().review.runner_server_url,
    )


async def cancel_index_jobs(
    app_auth: GitHubAppAuth, db: AppDatabase, owner: str, repo: str
) -> bool:
    """Fail the repo's active index jobs and cancel their runs. The index
    keeps what it had and stays pending, so the next run reconciles.
    Returns whether there was a job to cancel."""
    cancelled = [
        job
        for job in db.list_active_review_jobs(owner, repo, REPO_JOB)
        if job.kind.indexes
        and db.transition_review_job(job.id, job.status, JobStatus.FAILED, error=_CANCELLED)
    ]
    for job in cancelled:
        finish_index_job(db, job, None, _CANCELLED)
        tracker.cancel(f"{owner}/{repo}", 0)
    for job in cancelled:
        provider = GitHubProvider(partial(app_auth.get_installation_token, job.installation_id))
        await cancel_run(provider, job)
    return bool(cancelled)


def finish_index_job(
    db: AppDatabase, job: ReviewJob, report: IndexReport | None, error: str
) -> int | None:
    """Apply an ended index job's uploaded writes, when it reports any, and
    record how it ended. Returns the file summaries applied, or None when a
    newer index run owns the repo, so the job changed nothing.

    The writes land in one index transaction under the job's index run,
    and the repo's status is set inside it, so a job that no longer owns the
    index sets neither. Only a complete run clears the index's pending mark.
    """
    key = RepoKey(platform="github", owner=job.owner, repo=job.repo)
    run = IndexRun(job.id, needs_reconciliation=False)
    end = report.end if report is not None else None
    written = 0
    manifests = False
    store = IndexStore.open(job.owner, job.repo, allow_fallback=False)
    try:
        with store.index_write(run):
            for seq in range(report.parts if report is not None else 0):
                part = IndexPart.model_validate_json(db.get_job_part(job.id, seq))
                written += apply_index_writes(store, key, part.writes)
                manifests = manifests or any(
                    isinstance(w, ReplaceManifestPackages | ClearMissingManifests)
                    for w in part.writes
                )
            _record_end(db, job, end, written, len(store.all_paths()), error)
        if end is IndexEnd.COMPLETE:
            store.finish_index_run(run)
    except IndexingSuperseded:
        logger.info(
            "A newer index run owns %s/%s; job %s applied nothing", job.owner, job.repo, job.id
        )
        return None
    finally:
        store.close()
    if manifests:
        _queue_vulnerability_scan(job.owner, job.repo)
    logger.info(
        "Index job %s on %s/%s ended %s: %d file summaries",
        job.id,
        job.owner,
        job.repo,
        end or "with no writes",
        written,
    )
    return written


def _record_end(
    db: AppDatabase, job: ReviewJob, end: IndexEnd | None, written: int, files: int, error: str
) -> None:
    owner, repo = job.owner, job.repo
    name = f"{owner}/{repo}"
    tracked = isinstance(job.task, IndexRepoTask)
    if end is IndexEnd.COMPLETE:
        db.set_repo_status(
            owner, repo, "ready", files_indexed=files, bump_last_indexed=tracked or written > 0
        )
        if tracked:
            tracker.complete(name, written)
    elif end is IndexEnd.EMPTY:
        db.set_repo_status(owner, repo, "empty", error=str(EmptyRepoError(owner, repo)))
        if tracked:
            tracker.complete(name, 0)
    else:
        # The index keeps what it had plus what an incomplete run wrote. A
        # repo with an index stays ready, so pushes keep indexing it and the
        # next run reconciles.
        db.set_repo_status(owner, repo, "ready" if files else "failed", files, error=error)
        if tracked:
            tracker.fail(name, error)
