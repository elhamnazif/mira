"""Verify the GitHub Actions OIDC token a review runner presents.

A runner proves which workflow run it is with the job's OIDC token
(``id-token: write``). The server checks GitHub's signature and the standard
claims, then binds the token to the job: the dispatched repo, the run the
dispatch started, a ``workflow_dispatch`` event, and an allowed reusable
workflow.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jwt
from jwt import PyJWKClient, PyJWKClientConnectionError

from mira.runner import ReviewJob

ISSUER = "https://token.actions.githubusercontent.com"
JWKS_URL = f"{ISSUER}/.well-known/jwks"
_LEEWAY_S = 60
_REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud", "repository", "run_id", "event_name"]

# Maps a token to the public key that should have signed it.
SigningKeyResolver = Callable[[str], Any]


class InvalidRunnerToken(Exception):
    """Not a valid GitHub Actions OIDC token for this server."""


class JWKSUnavailable(Exception):
    """GitHub's signing keys could not be fetched."""


@dataclass(frozen=True)
class RunnerClaims:
    repository: str
    run_id: int
    event_name: str
    job_workflow_ref: str


def _github_jwks() -> SigningKeyResolver:
    client = PyJWKClient(JWKS_URL, cache_jwk_set=True, lifespan=600, timeout=10)
    return lambda token: client.get_signing_key_from_jwt(token).key


class GitHubOIDCVerifier:
    def __init__(self, signing_key: SigningKeyResolver | None = None) -> None:
        self._signing_key = signing_key or _github_jwks()

    def verify(self, token: str, *, audience: str) -> RunnerClaims:
        try:
            claims = jwt.decode(
                token,
                self._signing_key(token),
                algorithms=["RS256"],
                audience=audience,
                issuer=ISSUER,
                leeway=_LEEWAY_S,
                options={"require": _REQUIRED_CLAIMS},
            )
        except PyJWKClientConnectionError as exc:
            raise JWKSUnavailable(str(exc)) from exc
        except jwt.PyJWTError as exc:
            raise InvalidRunnerToken(str(exc)) from exc
        try:
            return RunnerClaims(
                repository=str(claims["repository"]),
                run_id=int(claims["run_id"]),
                event_name=str(claims["event_name"]),
                job_workflow_ref=str(claims.get("job_workflow_ref", "")),
            )
        except ValueError as exc:
            raise InvalidRunnerToken(f"malformed run_id claim: {exc}") from exc


def binding_error(
    claims: RunnerClaims, job: ReviewJob, allowed_workflow_refs: Sequence[str]
) -> str | None:
    """Why this token may not act for ``job``, or None when it may.

    Workflow refs match exactly. A glob such as ``review.yml@*`` would also
    admit any commit of the fork network the ref can name, including one
    from a pull request, so the server would run code nobody pinned.
    """
    if not allowed_workflow_refs:
        return "review.runner_workflow_refs is empty, so the server accepts no runner"
    if claims.repository.lower() != f"{job.owner}/{job.repo}".lower():
        return f"token is for {claims.repository}"
    if job.run_id is None or claims.run_id != job.run_id:
        return f"token is for run {claims.run_id}, the job's run is {job.run_id}"
    if claims.event_name != "workflow_dispatch":
        return f"token is from a {claims.event_name} event"
    if claims.job_workflow_ref not in allowed_workflow_refs:
        return f"workflow {claims.job_workflow_ref!r} is not in review.runner_workflow_refs"
    return None
