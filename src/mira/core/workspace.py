"""Throwaway checkouts of a pull request head for workspace-mode review agents."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from mira.exceptions import WorkspaceError

_GIT_TIMEOUT_SECONDS = 300
_MAX_STDERR_DETAIL = 500


@dataclass(frozen=True)
class CheckoutSource:
    """Where to fetch a PR head from, and the commit it must resolve to."""

    clone_url: str
    sha: str
    #: Full HTTP ``Authorization`` header value, or None for public/local remotes.
    auth_header: str | None = field(default=None, repr=False)


@asynccontextmanager
async def pr_workspace(source: CheckoutSource) -> AsyncIterator[Path]:
    """Check ``source.sha`` out into a fresh directory, yield it, then delete it."""
    # A short root: small models mangle long absolute paths.
    root = Path(tempfile.gettempdir()) / "mira-ws"
    root.mkdir(mode=0o700, exist_ok=True)
    if os.name == "posix" and (root.is_symlink() or root.stat().st_uid != os.getuid()):
        raise WorkspaceError(f"{root} is not a directory owned by this user")
    workspace = Path(tempfile.mkdtemp(prefix="", dir=root))
    try:
        await _checkout(workspace, source)
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


async def _checkout(workspace: Path, source: CheckoutSource) -> None:
    # Inherited GIT_DIR, GIT_TRACE and the like would redirect or log the fetch.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    # Host git config could add hooks, URL rewrites or credential helpers.
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    fetch_env = dict(env)
    if source.auth_header:
        # Through the environment only, so the header never reaches argv or .git/config.
        fetch_env["GIT_CONFIG_COUNT"] = "1"
        fetch_env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
        fetch_env["GIT_CONFIG_VALUE_0"] = f"Authorization: {source.auth_header}"

    await _git(workspace, env, "init", "-q")
    await _git(workspace, env, "config", "core.symlinks", "false")
    await _git(
        workspace, fetch_env, "fetch", "-q", "--depth=1", "--no-tags", source.clone_url, source.sha
    )
    await _git(workspace, env, "checkout", "-q", "--detach", "FETCH_HEAD")
    head = (await _git(workspace, env, "rev-parse", "HEAD")).strip()
    if head != source.sha:
        raise WorkspaceError(f"Checkout resolved to {head}, expected {source.sha}")


async def _git(cwd: Path, env: dict[str, str], *args: str) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise WorkspaceError("Git is required to check out the pull request.") from exc
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), _GIT_TIMEOUT_SECONDS)
    except BaseException as exc:
        with suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        if isinstance(exc, TimeoutError):
            raise WorkspaceError(f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS}s") from None
        raise
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[:_MAX_STDERR_DETAIL]
        raise WorkspaceError(f"git {args[0]} failed: {detail or f'exit {proc.returncode}'}")
    return stdout.decode("utf-8", errors="replace")
