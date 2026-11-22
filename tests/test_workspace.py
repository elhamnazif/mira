"""pr_workspace checks a PR head out of a real (local) Git remote."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from mira.core.workspace import CheckoutSource, pr_workspace
from mira.exceptions import WorkspaceError

_HEADER = "Basic c2VjcmV0LXRva2VuLXZhbHVl"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Mira Tests",
            "-c",
            "user.email=mira@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def remote(tmp_path: Path) -> tuple[str, str, str]:
    """A bare repo with two commits; returns (file URL, old SHA, head SHA)."""
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "--initial-branch=main", "--template=")
    (work / "app.py").write_text("VERSION = 1\n")
    _git(work, "add", ".")
    _git(work, "commit", "-qm", "first")
    old = _git(work, "rev-parse", "HEAD")
    (work / "app.py").write_text("VERSION = 2\n")
    (work / "link.py").symlink_to("/etc/passwd")
    _git(work, "add", ".")
    _git(work, "commit", "-qm", "second")
    head = _git(work, "rev-parse", "HEAD")
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(work), str(bare))
    return bare.as_uri(), old, head


@pytest.mark.asyncio
async def test_checks_out_requested_sha_with_symlinks_as_files(remote):
    url, _, head = remote
    async with pr_workspace(CheckoutSource(url, head, _HEADER)) as ws:
        assert _git(ws, "rev-parse", "HEAD") == head
        assert (ws / "app.py").read_text() == "VERSION = 2\n"
        link = ws / "link.py"
        assert not link.is_symlink()
        assert link.read_text() == "/etc/passwd"
        assert _HEADER not in (ws / ".git" / "config").read_text()
    assert not ws.exists()


@pytest.mark.asyncio
async def test_checks_out_a_non_tip_commit(remote):
    url, old, _ = remote
    async with pr_workspace(CheckoutSource(url, old)) as ws:
        assert (ws / "app.py").read_text() == "VERSION = 1\n"
        assert not (ws / "link.py").exists()


@pytest.mark.asyncio
async def test_directory_removed_when_body_raises(remote):
    url, _, head = remote
    with pytest.raises(RuntimeError, match="boom"):
        async with pr_workspace(CheckoutSource(url, head)) as ws:
            assert ws.is_dir()
            raise RuntimeError("boom")
    assert not ws.exists()


@pytest.mark.asyncio
async def test_unknown_sha_raises_and_cleans_up(remote):
    url, _, _ = remote
    root = Path(tempfile.gettempdir()) / "mira-ws"
    before = set(root.iterdir()) if root.exists() else set()
    with pytest.raises(WorkspaceError) as excinfo:
        async with pr_workspace(CheckoutSource(url, "0" * 40, _HEADER)):
            pytest.fail("workspace should not be yielded")
    assert _HEADER not in str(excinfo.value)
    assert set(root.iterdir()) == before


@pytest.mark.asyncio
async def test_fetched_commit_must_match_requested_sha(remote):
    url, _, _ = remote
    with pytest.raises(WorkspaceError, match="expected main"):
        async with pr_workspace(CheckoutSource(url, "main")):
            pytest.fail("workspace should not be yielded")


def test_auth_header_not_in_repr():
    assert _HEADER not in repr(CheckoutSource("https://example.invalid/r.git", "a" * 40, _HEADER))
