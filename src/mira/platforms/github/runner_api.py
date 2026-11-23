"""The runner API: an Actions runner claims its job, refreshes its GitHub
token, and reports its result.

Every route authenticates with the run's GitHub Actions OIDC token only
(``Authorization: Bearer``), never a dashboard session. A periodic sweep
expires jobs no runner picked up or finished.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from functools import partial

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import SecretStr, ValidationError

from mira.config import load_config
from mira.core.state_store import LocalStateStore
from mira.dashboard.db import AppDatabase
from mira.models import PRInfo
from mira.platforms.github.auth import GitHubAppAuth, InstallationToken
from mira.platforms.github.dispatch import finish_check
from mira.platforms.github.oidc import (
    GitHubOIDCVerifier,
    InvalidRunnerToken,
    JWKSUnavailable,
    binding_error,
)
from mira.providers.base import CheckConclusion
from mira.providers.github import GitHubProvider
from mira.runner import JobStatus, ReviewJob, RunnerLLMChoice, parse_runner_llm
from mira.runner_protocol import (
    JobSpec,
    PRKey,
    ResultBundle,
    TokenGrant,
    apply_writes,
    build_snapshot,
    check_writes,
    runner_config,
)

logger = logging.getLogger(__name__)

# What a review needs: read the code, post comments and reviews, resolve
# threads, and complete its check run.
RUNNER_TOKEN_PERMISSIONS = {
    "contents": "read",
    "pull_requests": "write",
    "issues": "write",
    "checks": "write",
}
MAX_RESULT_BYTES = 8 * 1024 * 1024
# GitHub cancels a job that waits for a self-hosted runner for 24 hours.
DISPATCHED_TTL_S = 25 * 3600
SWEEP_INTERVAL_S = 600


def _db() -> AppDatabase:
    # Looked up per call, as LocalStateStore does: tests swap the module attribute.
    from mira.dashboard import api

    return api._app_db


def _pr_key(job: ReviewJob) -> PRKey:
    return PRKey(platform="github", owner=job.owner, repo=job.repo, pr_number=job.pr_number)


def _pr_url(job: ReviewJob) -> str:
    return f"https://github.com/{job.owner}/{job.repo}/pull/{job.pr_number}"


def _pr_info(job: ReviewJob) -> PRInfo:
    return PRInfo(
        title="",
        description="",
        base_branch="",
        head_branch="",
        url=_pr_url(job),
        number=job.pr_number,
        owner=job.owner,
        repo=job.repo,
        head_sha=job.head_sha,
    )


async def _finish_job_check(
    app_auth: GitHubAppAuth, job: ReviewJob, conclusion: CheckConclusion, title: str, summary: str
) -> None:
    provider = GitHubProvider(partial(app_auth.get_installation_token, job.installation_id))
    await finish_check(provider, _pr_info(job), job.check_run_id, conclusion, title, summary)


def _conflict(job: ReviewJob) -> JSONResponse:
    return JSONResponse(
        status_code=409, content={"error": f"job is {job.status}", "status": job.status}
    )


def _bearer(request: Request) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="unauthorized")
    return token.strip()


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail="result too large")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(status_code=413, detail="result too large")
    return bytes(body)


def create_runner_router(
    app_auth: GitHubAppAuth,
    bot_name: str,
    *,
    verifier: GitHubOIDCVerifier | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/runner")
    oidc = verifier or GitHubOIDCVerifier()

    async def authorize(request: Request, job_id: str) -> ReviewJob:
        """The job, once the request's OIDC token is shown to be its run's.
        Reasons are logged, never returned."""
        token = _bearer(request)
        review = load_config().review
        try:
            claims = await asyncio.to_thread(
                oidc.verify, token, audience=review.runner_oidc_audience
            )
        except JWKSUnavailable as exc:
            logger.warning("Runner auth for job %s: GitHub JWKS unavailable: %s", job_id, exc)
            raise HTTPException(status_code=503, detail="try again") from None
        except InvalidRunnerToken as exc:
            logger.warning("Runner auth for job %s: invalid OIDC token: %s", job_id, exc)
            raise HTTPException(status_code=401, detail="unauthorized") from None
        job = _db().get_review_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        reason = binding_error(claims, job, review.runner_workflow_refs)
        if reason is not None:
            logger.warning("Runner refused for job %s (run %s): %s", job_id, claims.run_id, reason)
            raise HTTPException(status_code=403, detail="forbidden")
        return job

    async def mint(job: ReviewJob) -> InstallationToken:
        try:
            return await app_auth.mint_repo_token(
                job.installation_id, job.repo, RUNNER_TOKEN_PERMISSIONS
            )
        except Exception as exc:
            logger.warning("Could not mint a runner token for job %s: %s", job.id, exc)
            raise HTTPException(status_code=502, detail="could not mint a GitHub token") from None

    @router.post("/jobs/{job_id}/claim")
    async def claim(job_id: str, request: Request) -> Response:
        job = await authorize(request, job_id)
        db = _db()
        if job.status is JobStatus.DISPATCHED:
            # Whoever wins the compare-and-set, the job's status now says
            # what this run may do.
            db.transition_review_job(job.id, JobStatus.DISPATCHED, JobStatus.CLAIMED)
            job = db.get_review_job(job.id) or job
        # A CLAIMED job claims again: the binding already proved this is the
        # job's run, retrying after it lost the first answer.
        if job.status is not JobStatus.CLAIMED:
            return _conflict(job)
        snapshot = build_snapshot(LocalStateStore(), _pr_key(job))
        grant = await mint(job)
        config = load_config()
        review = config.review
        record = db.get_repo(job.owner, job.repo)
        llm = parse_runner_llm(record.runner_llm if record else "") or RunnerLLMChoice(
            review.runner_llm_provider, review.runner_model
        )
        credentials = {
            name: SecretStr(value)
            for name in review.runner_credentials
            if (value := os.environ.get(name))
        }
        spec = JobSpec(
            job_id=job.id,
            task=job.task,
            pr_url=_pr_url(job),
            head_sha=job.head_sha,
            platform="github",
            bot_name=bot_name,
            check_run_id=job.check_run_id,
            config=runner_config(config),
            snapshot=snapshot,
            github_token=SecretStr(grant.token),
            token_expires_at=grant.expires_at,
            credentials=credentials,
            llm_provider=llm.provider,
            llm_model=llm.model,
        )
        logger.info("Run %s claimed %s job %s for %s", job.run_id, job.kind, job.id, _pr_url(job))
        return Response(content=spec.model_dump_json(), media_type="application/json")

    @router.post("/jobs/{job_id}/token")
    async def refresh_token(job_id: str, request: Request) -> Response:
        job = await authorize(request, job_id)
        if job.status is not JobStatus.CLAIMED:
            return _conflict(job)
        grant = await mint(job)
        body = TokenGrant(github_token=SecretStr(grant.token), token_expires_at=grant.expires_at)
        return Response(content=body.model_dump_json(), media_type="application/json")

    @router.post("/jobs/{job_id}/result")
    async def result(job_id: str, request: Request) -> Response:
        job = await authorize(request, job_id)
        raw = await _read_body(request, MAX_RESULT_BYTES)
        try:
            bundle = ResultBundle.model_validate_json(raw)
            if bundle.job_id != job.id:
                raise ValueError(f"bundle is for job {bundle.job_id}")
            check_writes(_pr_key(job), bundle.writes)
        except (ValidationError, ValueError) as exc:
            logger.warning("Rejected the result for job %s: %s", job.id, exc)
            raise HTTPException(status_code=422, detail="invalid result bundle") from None

        # From here to the end of _accept_result nothing awaits, so no other
        # request on this server moves the job in between.
        current = _db().get_review_job(job.id) or job
        status, applied = _accept_result(_db(), current, bundle)
        if status is None:
            return _conflict(current)
        if applied and status is JobStatus.FAILED:
            await _finish_job_check(
                app_auth, current, "failure", "Mira review failed", bundle.error or "Unknown error"
            )
        return JSONResponse(content={"status": status, "applied": applied})

    return router


def _accept_result(
    db: AppDatabase, job: ReviewJob, bundle: ResultBundle
) -> tuple[JobStatus | None, bool]:
    """Record a runner's result. Returns the job's status afterwards and
    whether this call applied the writes, or None when the job cannot take a
    result.

    A job superseded or expired after its claim keeps its status but still
    gets its writes: they describe comments the runner really posted. The
    usage column is written
    once, first, so a retried result is never applied twice. A crash after
    it loses that result's writes rather than doubling them.
    """
    if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
        return job.status, False
    was_claimed = job.status is JobStatus.CLAIMED
    ran_late = job.status in (JobStatus.SUPERSEDED, JobStatus.EXPIRED) and job.claimed_at
    if not (was_claimed or ran_late):
        return None, False
    if not db.record_review_job_result(job.id, bundle.usage.model_dump(mode="json")):
        return job.status, False
    status = job.status
    if was_claimed:
        new = JobStatus.SUCCEEDED if bundle.outcome == "succeeded" else JobStatus.FAILED
        error = bundle.error if new is JobStatus.FAILED else ""
        if db.transition_review_job(job.id, JobStatus.CLAIMED, new, error=error):
            status = new
        else:
            moved = db.get_review_job(job.id)
            status = moved.status if moved else status
            logger.warning("Review job %s moved to %s while its result arrived", job.id, status)
    apply_writes(LocalStateStore(), _pr_key(job), bundle.writes)
    logger.info(
        "Applied %d state writes from job %s (%s)", len(bundle.writes), job.id, bundle.outcome
    )
    return status, True


# How an expired job's check run ends, by the status it expired from.
_EXPIRY: dict[JobStatus, tuple[CheckConclusion, str]] = {
    JobStatus.DISPATCHED: (
        "neutral",
        f"No Mira runner picked up this review within {DISPATCHED_TTL_S // 3600} hours.",
    ),
    JobStatus.CLAIMED: (
        "failure",
        "The Mira runner did not report a result within {minutes} minutes.",
    ),
}


async def expire_stale_jobs(
    db: AppDatabase, app_auth: GitHubAppAuth, *, now: float, claim_timeout_minutes: int
) -> list[ReviewJob]:
    """Expire dispatched jobs older than 25 hours and claimed jobs past the
    claim timeout, and close their check runs. Returns the jobs expired."""
    stale = db.list_stale_review_jobs(
        dispatched_before=now - DISPATCHED_TTL_S,
        claimed_before=now - claim_timeout_minutes * 60,
    )
    expired: list[tuple[ReviewJob, CheckConclusion, str]] = []
    for job in stale:
        conclusion, template = _EXPIRY[job.status]
        reason = template.format(minutes=claim_timeout_minutes)
        if db.transition_review_job(job.id, job.status, JobStatus.EXPIRED, error=reason):
            expired.append((job, conclusion, reason))
    for job, conclusion, reason in expired:
        logger.info("Expired review job %s (%s): %s", job.id, job.status, reason)
        await _finish_job_check(app_auth, job, conclusion, "Review expired", reason)
    return [job for job, _, _ in expired]


async def run_expiry_sweeper(app_auth: GitHubAppAuth, interval_s: float = SWEEP_INTERVAL_S) -> None:
    while True:
        try:
            await expire_stale_jobs(
                _db(),
                app_auth,
                now=time.time(),
                claim_timeout_minutes=load_config().review.runner_claim_timeout_minutes,
            )
        except Exception:
            logger.exception("Review job expiry sweep failed")
        await asyncio.sleep(interval_s)
